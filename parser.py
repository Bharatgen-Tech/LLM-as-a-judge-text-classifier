"""
parse_judge_outputs.py
─────────────────────
Re-parse the `generated_text` field of every record in a JSONL file using
the robust judge-output parser.  Records that pass are written to OUTPUT_FILE;
records where `_parse_error` is True are written to OUTPUT_FILE.errors.jsonl.

Usage:
    python parse_judge_outputs.py \
        --input  scored.jsonl \
        --output reparsed.jsonl \
        [--text-key generated_text]   # default: generated_text
"""

import argparse
import re
import sys
from pathlib import Path

import msgspec


# ─────────────────────────── parse helpers ────────────────────────────────────

# Only `quality_score` and `constraint_verdicts` are truly non-defaultable;
# `coherence` and `reasons` are filled by _normalise_schema / _validate.
_REQUIRED_KEYS = {"quality_score", "constraint_verdicts", "difficulty"}


def _strip_reasoning(text: str) -> str:
    text = re.sub(
        r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", "", text, flags=re.IGNORECASE
    )
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
            obj = msgspec.json.decode(s)
            return obj if isinstance(obj, dict) else None
        except msgspec.DecodeError:
            return None

    obj = _try(raw)
    if obj is not None:
        return obj

    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    obj = _try(fixed)
    if obj is not None:
        return obj

    fixed = re.sub(
        r'(?<=[{,\[])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'"\1":', fixed
    )
    obj = _try(fixed)
    if obj is not None:
        return obj

    for src, dst in [("True", "true"), ("False", "false"), ("None", "null")]:
        fixed = fixed.replace(src, dst)
    obj = _try(fixed)
    if obj is not None:
        return obj

    if not re.search(r"[a-zA-Z]'[a-zA-Z]", fixed):
        sq = re.sub(r"'([^']*)'", r'"\1"', fixed)
        obj = _try(sq)
        if obj is not None:
            return obj

    return None


def _normalise_schema(obj: dict) -> dict:
    """
    Bridge common schema variants to the canonical format expected by _validate.

    Known variants
    ──────────────
    • `score`            → `quality_score`
    • `explanation`      → mapped into `reasons` list (if `reasons` absent)
    • missing `coherence`→ filled with zeros
    • missing `constraint_verdicts` → filled with null blocks
    """
    # score → quality_score
    if "quality_score" not in obj and "score" in obj:
        obj["quality_score"] = obj.pop("score")

    # explanation → reasons
    if "reasons" not in obj and "explanation" in obj:
        explanation = obj.pop("explanation")
        obj["reasons"] = [explanation] if explanation else []

    return obj


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


_DEFAULT = lambda: {
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
    if not raw_text or not isinstance(raw_text, str):
        return _DEFAULT()

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
            obj = _normalise_schema(obj)
            return _validate(obj)

    return _DEFAULT()


# ──────────────────────────── scoring helpers ─────────────────────────────────

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


def enrich(record: dict, parsed: dict, raw_text: str) -> dict:
    """Merge parsed judge fields back into the original record."""
    judge_score = max(0.0, min(1.0, float(parsed.get("quality_score", 0.0))))
    cv = parsed.get("constraint_verdicts", {})

    def _cv(key):
        block = cv.get(key, {}) or {}
        return {"verdict": block.get("verdict"), "evidence": block.get("evidence")}

    lang_block = cv.get("language_correctness", {}) or {}
    language_detected = (
        lang_block.get("evidence")
        or ("OK" if lang_block.get("verdict") else "FAIL")
    )

    label = final_label(judge_score)

    record.update(
        judge_score         = judge_score,
        final_score         = judge_score,
        label               = label,
        difficulty          = parsed.get("difficulty"),
        language_detected   = language_detected,
        coherence           = parsed.get("coherence", {}),
        constraint_verdicts = {
            "language_correctness": _cv("language_correctness"),
            "format_adherence":     _cv("format_adherence"),
            "domain_accuracy":      _cv("domain_accuracy"),
        },
        regional_and_learning = parsed.get("regional_and_learning", {
            "language_bleeding_present": 0,
            "indian_context_consistent": 0,
            "high_learnability": 0,
        }),
        deduction_trace     = parsed.get("deduction_trace", []),
        judge_flags         = parsed.get("reasons", []),
        judge_output_raw    = raw_text,
        judge_skipped       = False,
        _parse_error        = parsed.get("_parse_error", False),
        _low_quality        = label in ("LOW", "MEDIUM_LOW"),
    )
    return record


# ───────────────────────────────── main ───────────────────────────────────────

def get_error_path(output_path: str) -> str:
    stem = output_path
    for ext in (".jsonl", ".json"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem + ".errors.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Re-parse judge outputs from a JSONL file.")
    parser.add_argument("--input",    required=True,  help="Input JSONL file")
    parser.add_argument("--output",   required=True,  help="Output JSONL file")
    parser.add_argument("--text-key", default="generated_text",
                        help="Key in each record that holds the raw LLM output (default: generated_text)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    error_path  = Path(get_error_path(args.output))

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = ok = errors = skipped = 0

    with (
        input_path.open("r", encoding="utf-8") as fin,
        output_path.open("w", encoding="utf-8") as fout,
        error_path.open("w", encoding="utf-8") as ferr,
    ):
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = msgspec.json.decode(line)
            except msgspec.DecodeError as exc:
                print(f"[WARN] line {lineno}: JSON decode error — {exc}", file=sys.stderr)
                skipped += 1
                continue

            raw_text = record.get(args.text_key, "")
            if not raw_text:
                print(f"[WARN] line {lineno}: key '{args.text_key}' is empty or missing", file=sys.stderr)
                skipped += 1
                continue

            parsed = parse_judge_output(raw_text)
            record = enrich(record, parsed, raw_text)

            if record.get("_parse_error"):
                errors += 1
                ferr.write(msgspec.json.encode(record).decode("utf-8") + "\n")
            else:
                ok += 1
                fout.write(msgspec.json.encode(record).decode("utf-8") + "\n")

    print(
        f"\nDone. {total} records processed.\n"
        f"  ✓  {ok:>6} written to  {output_path}\n"
        f"  ✗  {errors:>6} errors →   {error_path}\n"
        f"  -  {skipped:>6} skipped (bad JSON or missing key)"
    )


if __name__ == "__main__":
    main()