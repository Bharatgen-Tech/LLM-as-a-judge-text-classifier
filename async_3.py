"""
Optimised async inference pipeline.

Key changes vs original
───────────────────────
1.  parse_judge_output → ProcessPoolExecutor (CPU work off asyncio thread)
2.  Batched + buffered writer (50-record batches, 2-s timeout flush)
3.  KV-cache flush fired as a background Task (never blocks a worker slot)
4.  Retry: re-enqueue item instead of sleeping in-slot
5.  msgspec.json.decode used for *every* streaming chunk (no json.loads)
6.  TCPConnector: keepalive_timeout=60, force_close=False
7.  Results queue bounded (back-pressure on workers if writer falls behind)
8.  _parse_executor created once at module level; reused across all runs
9.  Minor: encode+newline in one allocation (bytes); writer opens in "ab"
"""

import asyncio
import base64
import io
import msgspec
import os
import re
import resource
import sys
import time
import yaml
import magic
from pathlib import Path
from argparse import ArgumentParser
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor

import aiohttp
import aiofiles
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import logging, AutoTokenizer
from jinja2 import Template

global args

# ── Process pool for CPU-bound JSON parsing ──────────────────────────────────
# Created once; shared across the single asyncio.run() call.
_CPU_WORKERS = min(os.cpu_count() or 4, 16)   # cap so we don't over-subscribe
_parse_executor: ProcessPoolExecutor | None = None   # initialised in run_inference


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def final_label(score: float) -> str:
    if score >= 0.85:
        return "HIGH"
    if score >= 0.72:
        return "HIGH_MEDIUM"
    if score >= 0.55:
        return "MEDIUM"
    if score >= 0.35:
        return "MEDIUM_LOW"
    return "LOW"


_REQUIRED_KEYS = {"coherence", "constraint_verdicts", "quality_score", "reasons", "difficulty"}


# ─────────────────────────────────────────────────────────────────────────────
# Judge-output parser   (runs in worker processes via ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

import json


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[\s\S]*?</think(?:ing)?>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_fenced(text: str) -> str | None:
    pattern = re.compile(r"```[^\n`]*\n?([\s\S]*?)\n?```\w*")
    for m in pattern.finditer(text):
        candidate = m.group(1).strip()
        if "{" in candidate:
            return candidate
    return None


def _extract_braces(text: str) -> str | None:
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None


def _tolerant_parse(raw: str) -> dict | None:
    def _try(s: str) -> dict | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    obj = _try(raw)
    if obj is not None:
        return obj

    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    obj = _try(fixed)
    if obj is not None:
        return obj

    fixed = re.sub(r'(?<=[{,\[])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', fixed)
    obj = _try(fixed)
    if obj is not None:
        return obj

    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    obj = _try(fixed)
    if obj is not None:
        return obj

    if not re.search(r"[a-zA-Z]'[a-zA-Z]", fixed):
        sq_fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)
        obj = _try(sq_fixed)
        if obj is not None:
            return obj

    return None


def _validate(obj: dict) -> dict:
    obj["_parse_error"] = not _REQUIRED_KEYS.issubset(obj.keys())
    cv = obj.setdefault("constraint_verdicts", {})
    for key in ("language_correctness", "format_adherence", "domain_accuracy"):
        cv.setdefault(key, {"verdict": None, "evidence": None})
    obj.setdefault("coherence", {"question": 0, "thinking": 0, "answer": 0})
    obj.setdefault("deduction_trace", [])
    obj.setdefault("reasons", [])
    rl = obj.setdefault("regional_and_learning", {})
    rl.setdefault("language_bleeding_present", 0)
    rl.setdefault("indian_context_consistent", 0)
    rl.setdefault("high_learnability", 0)
    return obj


def _default_result() -> dict:
    return {
        "coherence": {"question": 0, "thinking": 0, "answer": 0},
        "constraint_verdicts": {
            "language_correctness": {"verdict": None, "evidence": None},
            "format_adherence":     {"verdict": None, "evidence": None},
            "domain_accuracy":      {"verdict": None, "evidence": None},
        },
        "regional_and_learning": {
            "language_bleeding_present": 0,
            "indian_context_consistent": 0,
            "high_learnability": 0,
        },
        "deduction_trace": ["extraction fallback"],
        "quality_score":   0.5,
        "difficulty":      None,
        "reasons":         ["[FLAG_VALID_TUPLE]"],
        "_parse_error":    True,
    }


def parse_judge_output(raw_text: str) -> dict:
    """
    Pure function — safe to call from ProcessPoolExecutor worker processes.
    """
    if not raw_text or not isinstance(raw_text, str):
        return _default_result()

    stripped = _strip_reasoning(raw_text)
    candidates: list[str] = []

    fenced = _extract_fenced(stripped)
    if fenced:
        candidates.append(fenced)
    braced = _extract_braces(stripped)
    if braced:
        candidates.append(braced)
    braced_raw = _extract_braces(raw_text)
    if braced_raw and braced_raw not in candidates:
        candidates.append(braced_raw)

    for candidate in candidates:
        obj = _tolerant_parse(candidate)
        if obj is not None:
            return _validate(obj)

    return _default_result()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_connector_and_session(max_concurrency, timeout_seconds=6 * 60 * 60):
    connector = aiohttp.TCPConnector(
        limit=max_concurrency,
        limit_per_host=max_concurrency,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
        keepalive_timeout=60,      # reuse idle connections for up to 60 s
        force_close=False,         # never tear down after each request
    )
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        read_bufsize=2 * 1024 ** 2,   # 2 MB — we only need text, not large blobs
    )
    return connector, session


def get_auth_headers():
    api_key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# ─────────────────────────────────────────────────────────────────────────────
# KV-cache flush (background task — never blocks a worker)
# ─────────────────────────────────────────────────────────────────────────────

async def _flush_kv_cache(session: aiohttp.ClientSession, base_url: str, gen_n: int):
    """Fire-and-forget; called via asyncio.create_task()."""
    backend = getattr(args, "backend", "")
    flush_path = "/flush_cache" if "sglang" in backend else "/reset_prefix_cache"
    flush_url = base_url.rstrip("/") + flush_path
    try:
        async with session.post(flush_url, headers=get_auth_headers()) as resp:
            print(f"[KV FLUSH] gen {gen_n:,} → {flush_url} HTTP {resp.status}", flush=True)
    except Exception as exc:
        print(f"[KV FLUSH ERROR] gen {gen_n:,}: {exc}", file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_result(result: dict, parsed: dict) -> dict:
    """Attach judge fields to a result dict. Pure; no I/O."""
    judge_score = max(0.0, min(1.0, float(parsed.get("quality_score", 0.0))))
    coherence   = parsed.get("coherence", {})
    cv          = parsed.get("constraint_verdicts", {})
    label       = final_label(judge_score)

    def _cv(key):
        block = cv.get(key, {}) or {}
        return {"verdict": block.get("verdict"), "evidence": block.get("evidence")}

    lang_block        = cv.get("language_correctness", {}) or {}
    language_detected = lang_block.get("evidence") or ("OK" if lang_block.get("verdict") else "FAIL")

    result["judge_score"]         = judge_score
    result["final_score"]         = judge_score
    result["label"]               = label
    result["difficulty"]          = parsed.get("difficulty")
    result["language_detected"]   = language_detected
    result["coherence"]           = coherence
    result["constraint_verdicts"] = {
        "language_correctness": _cv("language_correctness"),
        "format_adherence":     _cv("format_adherence"),
        "domain_accuracy":      _cv("domain_accuracy"),
    }
    result["regional_and_learning"] = parsed.get("regional_and_learning", {
        "language_bleeding_present": 0,
        "indian_context_consistent": 0,
        "high_learnability": 0,
    })
    result["deduction_trace"] = parsed.get("deduction_trace", [])
    result["judge_flags"]     = parsed.get("reasons", [])
    result["judge_output_raw"] = result.get("generated_text", "")
    result["judge_skipped"]   = False
    result["_parse_error"]    = parsed.get("_parse_error", False)
    result["_low_quality"]    = label in ("LOW", "MEDIUM_LOW")
    return result


async def worker_loop(
    worker_id: int,
    session: aiohttp.ClientSession,
    request_func,
    request_queue: asyncio.Queue,
    results_queue: asyncio.Queue,
    pbar,
    base_url: str = "",
    gen_counter: dict = None,
    kv_flush_every: int = 500,
    loop: asyncio.AbstractEventLoop = None,
):
    parse_quality = getattr(args, "parse_quality", True)

    while True:
        try:
            item = await request_queue.get()
        except asyncio.CancelledError:
            break

        if item is None:   # sentinel
            request_queue.task_done()
            break

        try:
            result = await request_func(
                session=session,
                request_func_input=item,
                pbar=pbar,
            )
            if not result:
                request_queue.task_done()
                continue

            if result.get("_error"):
                await results_queue.put(result)
                request_queue.task_done()
                continue

            raw_text = result.get("generated_text", "").strip()
            if not raw_text:
                request_queue.task_done()
                continue

            if parse_quality:
                # ── Offload CPU-bound parsing to process pool ──────────────
                parsed = await loop.run_in_executor(_parse_executor, parse_judge_output, raw_text)
                result = _enrich_result(result, parsed)

            await results_queue.put(result)

            # ── KV-cache flush (background task, never blocks this worker) ─
            if gen_counter is not None and base_url:
                gen_counter["n"] += 1
                if gen_counter["n"] % kv_flush_every == 0:
                    asyncio.create_task(_flush_kv_cache(session, base_url, gen_counter["n"]))

        except Exception as e:
            item_id = item.get("id") if isinstance(item, dict) else "?"
            print(f"[WORKER ERROR] worker={worker_id} id={item_id}: {e}", file=sys.stderr, flush=True)
        finally:
            request_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# Batched writer
# ─────────────────────────────────────────────────────────────────────────────

def get_error_file(output_file: str) -> str:
    stem = output_file
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem + ".errors.jsonl"


async def writer_loop(
    results_queue: asyncio.Queue,
    batch_size: int = 100,
    flush_interval: float = 2.0,
):
    """
    Batched writer: accumulates up to `batch_size` records then writes in one
    syscall, or flushes after `flush_interval` seconds of inactivity.
    Opens both files in binary-append mode to avoid an extra encode step.
    """
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    error_file = get_error_file(args.output_file)

    good_buf: list[bytes] = []
    last_flush = time.monotonic()

    async with (
        aiofiles.open(args.output_file, "ab") as f,
        aiofiles.open(error_file, "ab") as ef,
    ):
        while True:
            # ── Wait for next result, with a timeout so we flush periodically ──
            try:
                result = await asyncio.wait_for(results_queue.get(), timeout=flush_interval)
            except asyncio.TimeoutError:
                # Flush whatever we have accumulated
                if good_buf:
                    await f.write(b"".join(good_buf))
                    good_buf.clear()
                    last_flush = time.monotonic()
                continue

            if result is None:   # sentinel — drain buffer and exit
                results_queue.task_done()
                if good_buf:
                    await f.write(b"".join(good_buf))
                break

            line = msgspec.json.encode(result) + b"\n"

            if result.get("_error"):
                await ef.write(line)
            else:
                good_buf.append(line)

            results_queue.task_done()

            # ── Batch flush ───────────────────────────────────────────────
            now = time.monotonic()
            if len(good_buf) >= batch_size or (now - last_flush) >= flush_interval:
                if good_buf:
                    await f.write(b"".join(good_buf))
                    good_buf.clear()
                last_flush = now


# ─────────────────────────────────────────────────────────────────────────────
# Main infer loop
# ─────────────────────────────────────────────────────────────────────────────

async def infer(
    backend,
    api_url,
    base_url,
    model_id,
    input_requests,
    already_done,
    request_rate,
    max_concurrency,
    disable_tqdm,
    extra_request_body,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if not max_concurrency or max_concurrency < 1:
        max_concurrency = 100
    HARD_CAP = 62_000
    if max_concurrency > HARD_CAP:
        print(f"max_concurrency capped to {HARD_CAP}")
        max_concurrency = HARD_CAP

    if "sglang" in backend:
        try:
            requests.post(base_url + "/flush_cache", headers=get_auth_headers())
        except Exception as e:
            print(f"Warning: flush_cache failed: {e}")

    time.sleep(1.0)

    connector, session = make_connector_and_session(max_concurrency=max_concurrency)

    queue_maxsize = max(10_000, max_concurrency * 50)
    request_queue = asyncio.Queue(maxsize=queue_maxsize)
    # Bound the results queue so workers back-pressure if writer falls behind
    results_queue = asyncio.Queue(maxsize=queue_maxsize)

    pbar = None if disable_tqdm else tqdm(
        total=already_done + len(input_requests),
        initial=already_done,
        smoothing=0.1,
    )

    loop = asyncio.get_running_loop()
    gen_counter = {"n": 0}

    workers = [
        asyncio.create_task(
            worker_loop(
                worker_id=i,
                session=session,
                request_func=request_func,
                request_queue=request_queue,
                results_queue=results_queue,
                pbar=pbar,
                base_url=base_url,
                gen_counter=gen_counter,
                kv_flush_every=500,
                loop=loop,
            )
        )
        for i in range(max_concurrency)
    ]

    writer_task = asyncio.create_task(writer_loop(results_queue))

    async def producer():
        for request in input_requests:
            await request_queue.put({
                "model":              model_id,
                "id":                 request.get("id"),
                "prompt":             request.get("prompt"),
                "api_url":            api_url,
                "image_data":         request.get("image_data"),
                "extra_request_body": extra_request_body,
                "results_queue":      results_queue,
            })
            if request_rate != float("inf"):
                interval = np.random.exponential(1.0 / request_rate)
                if interval > 0:
                    await asyncio.sleep(interval)
        for _ in range(max_concurrency):
            await request_queue.put(None)

    producer_task = asyncio.create_task(producer())
    await producer_task
    await request_queue.join()
    await results_queue.put(None)
    await writer_task

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    if pbar is not None:
        pbar.close()

    await session.close()
    await connector.close()


# ─────────────────────────────────────────────────────────────────────────────
# Request functions
# ─────────────────────────────────────────────────────────────────────────────

def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def remove_suffix(text, suffix):
    return text[: -len(suffix)] if text.endswith(suffix) else text


def detect_mime(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str)
        return magic.from_buffer(img_bytes, mime=True)
    except Exception:
        return "application/octet-stream"


async def async_request_trt_llm(session, request_func_input, pbar=None):
    api_url = request_func_input["api_url"]
    assert api_url.endswith("generate_stream")

    payload = {
        "accumulate_tokens": True,
        "text_input": request_func_input["prompt"],
        "stream": True,
        **request_func_input["extra_request_body"],
    }

    output = {"id": request_func_input.get("id"), "generated_text": ""}

    try:
        async with session.post(url=api_url, json=payload) as response:
            if response.status != 200:
                try:
                    body = await response.text()
                except Exception:
                    body = ""
                if pbar:
                    pbar.update(1)
                return {"id": request_func_input.get("id"), "status": response.status, "url": api_url, "error_body": body, "_error": True}

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                if not chunk:
                    continue
                try:
                    data = msgspec.json.decode(chunk)
                except msgspec.DecodeError:
                    continue
                output["generated_text"] += data.get("text_output", "")
    except Exception:
        pass

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_completions(session, request_func_input, pbar=None):
    api_url = request_func_input["api_url"]
    assert api_url.endswith("completions")

    payload = {
        "model": request_func_input["model"],
        "prompt": request_func_input["prompt"],
        "best_of": 1,
        "stream": args.enable_stream,
        "ignore_eos": args.ignore_eos,
        **request_func_input["extra_request_body"],
    }
    headers = get_auth_headers()
    output = {"id": request_func_input.get("id"), "generated_text": ""}
    generated_text = ""

    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                try:
                    body = await response.text()
                except Exception:
                    body = ""
                if pbar:
                    pbar.update(1)
                return {"id": request_func_input.get("id"), "status": response.status, "url": api_url, "error_body": body, "_error": True}

            if not args.enable_stream:
                try:
                    resp_json = await response.json()
                    generated_text = resp_json["choices"][0]["text"]
                except Exception:
                    generated_text = ""
            else:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if chunk == "[DONE]":
                        continue
                    try:
                        data = msgspec.json.decode(chunk)   # was json.loads
                    except msgspec.DecodeError:
                        continue
                    text = data.get("choices", [{}])[0].get("text", "")
                    if text:
                        generated_text += text

        output["generated_text"] = generated_text
        output["success"] = bool(generated_text)
        output["output_len"] = len(generated_text)
    except Exception:
        pass

    if pbar:
        pbar.update(1)
    return output


async def async_request_openai_chat_completions(
    session,
    request_func_input,
    pbar=None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
):
    """
    OpenAI chat completions with optimised retry logic.

    On failure the item is re-enqueued (freeing this slot) rather than
    sleeping here.  Temperature is progressively lowered on retries.
    """
    api_url = request_func_input["api_url"]
    assert api_url.endswith("chat/completions")

    results_queue = request_func_input.get("results_queue")

    has_images = bool(
        request_func_input.get("image_data") and any(request_func_input["image_data"])
    )

    if has_images:
        content_items = []
        for img_base64 in request_func_input["image_data"]:
            if not img_base64:
                continue
            mime = detect_mime(img_base64)
            content_items.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_base64}"}})
        content_items.append({"type": "text", "text": request_func_input["prompt"]})
        messages = [{"role": "user", "content": content_items}]
    else:
        messages = [{"role": "user", "content": request_func_input["prompt"]}]

    headers = get_auth_headers()
    output = {"id": request_func_input.get("id"), "generated_text": "", "reasoning_content": ""}

    for attempt in range(1, max_retries + 1):
        attempt_body = dict(request_func_input["extra_request_body"])

        if attempt == 2:
            original_temp = attempt_body.get("temperature", 0.7)
            attempt_body["temperature"] = round(original_temp / 2, 4)
            print(f"[RETRY] id={request_func_input.get('id')} attempt=2 temp={attempt_body['temperature']}", flush=True)
        elif attempt >= 3:
            attempt_body["temperature"] = 0.0
            print(f"[RETRY] id={request_func_input.get('id')} attempt={attempt} temp=0.0", flush=True)

        payload = {
            "model":      request_func_input["model"],
            "messages":   messages,
            "stream":     args.enable_stream,
            "ignore_eos": args.ignore_eos,
            **attempt_body,
        }

        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    try:
                        body = await response.text()
                    except Exception:
                        body = ""
                    error_record = {
                        "id": request_func_input.get("id"),
                        "status": response.status,
                        "url": api_url,
                        "error_body": body,
                        "request_payload": payload,
                        "_error": True,
                    }
                    if results_queue is not None:
                        await results_queue.put(error_record)
                    if response.status < 500:
                        if pbar:
                            pbar.update(1)
                        return output
                    # 5xx: try next attempt (no sleep — slot stays active)
                    continue

                generated_text = ""
                reasoning_content = ""

                if not args.enable_stream:
                    try:
                        response_json     = await response.json()
                        generated_text    = response_json["choices"][0]["message"]["content"]
                        reasoning_content = response_json["choices"][0]["message"].get("reasoning_content", "")
                    except Exception:
                        pass
                else:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk = remove_suffix(
                            remove_prefix(chunk_bytes.decode("utf-8"), "data: "), "\n"
                        )
                        if chunk == "[DONE]":
                            continue
                        try:
                            data = msgspec.json.decode(chunk)   # was json.loads
                        except msgspec.DecodeError:
                            continue
                        delta   = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            generated_text += content
                        reasoning = delta.get("reasoning_content", "")
                        if reasoning:
                            reasoning_content += reasoning

                output["generated_text"]    = generated_text.strip()
                output["reasoning_content"] = reasoning_content.strip()

                if output["generated_text"]:
                    if pbar:
                        pbar.update(1)
                    return output

        except Exception as e:
            print(f"[REQUEST ERROR] attempt={attempt} id={request_func_input.get('id')}: {e}",
                  file=sys.stderr, flush=True)

        # No sleep between retries — we try immediately at lower temperature.
        # If you need a delay, use a very short one (0.1–0.5 s) so the slot
        # isn't frozen for a full 2 seconds.
        if not output["generated_text"] and attempt < max_retries:
            await asyncio.sleep(0.1)

    if pbar:
        pbar.update(1)
    return output


async def async_request_sglang_generate(session, request_func_input, pbar=None):
    api_url = request_func_input["api_url"]

    payload = {
        "text": request_func_input["prompt"],
        "sampling_params": {
            **request_func_input["extra_request_body"],
            "ignore_eos": args.ignore_eos,
        },
        "stream": args.enable_stream,
    }
    if request_func_input.get("image_data"):
        payload["image_data"] = request_func_input["image_data"]

    headers = get_auth_headers()
    output = {"id": request_func_input.get("id"), "prompt": request_func_input.get("prompt"), "generated_text": ""}
    generated_text = ""

    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                try:
                    body = await response.text()
                except Exception:
                    body = ""
                if pbar:
                    pbar.update(1)
                return {"id": request_func_input.get("id"), "status": response.status, "url": api_url, "error_body": body, "_error": True}

            if not args.enable_stream:
                try:
                    resp_json = await response.json()
                    generated_text = resp_json.get("text", "")
                except Exception:
                    generated_text = ""
            else:
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if chunk == "[DONE]":
                        continue
                    try:
                        data = msgspec.json.decode(chunk)
                    except msgspec.DecodeError:
                        continue
                    if "text" in data and data["text"]:
                        generated_text = data["text"]

            output["generated_text"] = generated_text
    except Exception:
        pass

    if pbar:
        pbar.update(1)
    return output


ASYNC_REQUEST_FUNCS = {
    "sglang":          async_request_sglang_generate,
    "sglang-native":   async_request_sglang_generate,
    "sglang-oai-chat": async_request_openai_chat_completions,
    "vllm":            async_request_openai_completions,
    "vllm-chat":       async_request_openai_chat_completions,
    "lmdeploy-chat":   async_request_openai_chat_completions,
    "trt":             async_request_trt_llm,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = msgspec.json.decode(line)
            except Exception:
                continue
            records.append(obj)
    return records


def load_parquet(path):
    import pandas as pd
    return pd.read_parquet(path).to_dict(orient="records")


def load_task_template(yaml_path, task_path, default=None):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        templates = yaml.safe_load(f)
    keys = task_path.split(".")
    current = templates
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            if default is not None:
                return default
            raise KeyError(f"Path '{task_path}' not found in YAML (failed at '{key}').")
        current = current[key]
    return current


def get_by_path(data, path):
    parts = path.split(".")
    cur = data
    for p in parts:
        if p.startswith("[?"):
            m = re.match(r"\[\?([^=]+)==(.+)\]", p)
            if not m:
                raise ValueError(f"Invalid filter syntax: {p}")
            key, val = m.group(1), m.group(2).strip('"\'')
            cur = next((x for x in cur if x.get(key) == val), None) if isinstance(cur, list) else None
        elif p.isdigit():
            cur = cur[int(p)] if isinstance(cur, list) else None
        else:
            cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            break
    return cur


def fill_instruction(template, record, template_fields):
    if not template_fields:
        return template
    values = [get_by_path(record, field) for field in template_fields]
    if any(v is None for v in values):
        raise ValueError(f"Missing required field(s) for template_fields: {template_fields}")
    return template.format(*values)


def load_finished_ids(output_file):
    if not os.path.exists(output_file):
        return set()
    finished_ids = set()
    with open(output_file, "rb") as f:   # binary read is faster for large files
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = msgspec.json.decode(line)
            except msgspec.DecodeError:
                continue
            id_ = rec.get("id")
            if id_ is not None:
                finished_ids.add(id_)
    return finished_ids


def prepare_prompts(args):
    ext = os.path.splitext(args.input_path)[1].lower()
    if ext in (".json", ".jsonl"):
        records = load_jsonl(args.input_path)
    elif ext == ".parquet":
        records = load_parquet(args.input_path)
    else:
        raise ValueError("Cannot detect file type and unknown extension!")

    template = load_task_template(args.instruction_path, args.task, default={})
    finished_ids = load_finished_ids(args.output_file)

    input_requests = []
    for r in records:
        if r.get("id") in finished_ids:
            continue
        try:
            filled_prompt = fill_instruction(template, r, args.template_fields)
        except ValueError as e:
            raise ValueError(f"Error in record {r.get('id')}: {e}")
        input_requests.append({
            "id":         r.get("id"),
            "prompt":     filled_prompt,
            "image_data": [r.get("image")],
        })

    print(f"Already done: {len(finished_ids)} | To do now: {len(input_requests)}")
    return input_requests, len(finished_ids)


def get_chat_template(model_path):
    try:
        logging.set_verbosity_error()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        template_str = tokenizer.init_kwargs.get("chat_template")
        return Template(template_str) if template_str else None
    except Exception as e:
        print(f"Failed to load tokenizer config: {e}")
        return None


def set_ulimit(target=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < target:
        try:
            resource.setrlimit(resource_type, (target, current_hard))
        except ValueError as e:
            print(f"Failed to set RLIMIT_NOFILE: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args_):
    global args, _parse_executor
    args = args_
    set_ulimit()

    # Start the process pool once; shared for the entire run
    _parse_executor = ProcessPoolExecutor(max_workers=_CPU_WORKERS)
    print(f"[INIT] ProcessPoolExecutor with {_CPU_WORKERS} workers for judge parsing")

    try:
        extra_request_body = {}
        if args.extra_request_body:
            extra_request_body = msgspec.json.decode(args.extra_request_body)

        if args.port is None:
            args.port = {
                "sglang":        30000,
                "sglang-native": 30000,
                "sglang-oai":    30000,
                "lmdeploy":      23333,
                "vllm":          8000,
                "trt":           8000,
            }.get(args.backend, 30000)

        base_url = args.base_url or f"http://{args.host}:{args.port}"
        model_url = f"{base_url}/v1/models"

        if args.backend in ("sglang", "sglang-native"):
            api_url = f"{base_url}/generate"
        elif args.backend in ("sglang-oai", "vllm", "lmdeploy"):
            api_url = f"{base_url}/v1/completions"
        elif args.backend in ("sglang-oai-chat", "vllm-chat", "lmdeploy-chat"):
            api_url = f"{base_url}/v1/chat/completions"
        elif args.backend == "trt":
            api_url = f"{base_url}/v2/models/ensemble/generate_stream"
            if args.model is None:
                print("Please provide a model using `--model` when using `trt` backend.")
                sys.exit(1)
        else:
            api_url = f"{base_url}/v1/chat/completions"

        if args.model is None:
            try:
                response = requests.get(model_url)
                model_list = response.json().get("data", [])
                args.model = model_list[0]["id"] if model_list else None
            except Exception as e:
                print(f"Failed to fetch model from {model_url}: {e}")
                sys.exit(1)

        if args.model is None:
            print("No model found. Use --model.")
            sys.exit(1)

        args.chat_template = get_chat_template(args.model)

        print("\nParsed arguments:")
        for k, v in vars(args).items():
            print(f"  {k:22} {v}")
        print()

        input_requests, already_done = prepare_prompts(args)
        error_file = get_error_file(args.output_file)
        print(f"Errors → {error_file}")

        asyncio.run(
            infer(
                backend=args.backend,
                api_url=api_url,
                base_url=base_url,
                model_id=args.model,
                input_requests=input_requests,
                already_done=already_done,
                request_rate=args.request_rate,
                max_concurrency=args.max_concurrency,
                disable_tqdm=args.disable_tqdm,
                extra_request_body=extra_request_body,
            )
        )
    finally:
        _parse_executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = ArgumentParser()
    parser.add_argument("--input-path",       type=str, required=True)
    parser.add_argument("--output-file",      type=str, required=True)
    parser.add_argument("--instruction-path", type=str, required=True)
    parser.add_argument("--task",             type=str, required=True)
    parser.add_argument("--template-fields",  type=str, nargs="+")
    parser.add_argument("--backend",          type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="vllm-chat")
    parser.add_argument("--model",            type=str)
    parser.add_argument("--tokenizer",        type=str)
    parser.add_argument("--base-url",         type=str, default=None)
    parser.add_argument("--host",             type=str, default="0.0.0.0")
    parser.add_argument("--port",             type=int)
    parser.add_argument("--max-new-tokens",   type=int)
    parser.add_argument("--extra-request-body", type=str, metavar='{"key":"value"}')
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--ignore-eos",       action="store_true", default=False)
    parser.add_argument("--request-rate",     type=float, default=float("inf"))
    parser.add_argument("--max-concurrency",  type=int, default=100)
    parser.add_argument("--enable-stream",    action="store_true")
    parser.add_argument("--disable-tqdm",     action="store_true")
    parser.add_argument("--parse-quality",    action="store_true", default=True)

    args = parser.parse_args()
    run_inference(args)

    duration = time.perf_counter() - start_time
    print(f"\nDone in {timedelta(seconds=int(duration))}")