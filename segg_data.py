#!/usr/bin/env python3
"""
segregate_by_quality.py  (EC2 / msgspec edition)
-------------------------------------------------
Joins one input JSONL (evaluation results) with one data JSONL (source data)
on the `id` field, then segregates the merged records by the `label` field.

Output:  <output_dir>/<input_stem>_<LABEL>.jsonl
         <output_dir>/<input_stem>_UNMATCHED.jsonl   (data records with no input match)

Speed design
  • msgspec.json.Encoder / Decoder  – fastest pure-Python-callable JSON on CPython;
    single reusable Encoder/Decoder avoids per-call object construction
  • mmap                             – OS-level zero-copy read; no extra buffer copies
  • streaming writes per bucket      – no in-memory list accumulation, no giant join();
    records go straight to disk as they arrive
  • 4 MiB BufferedWriter per bucket  – amortises syscall cost across many small writes
  • single pass over input file      – O(N) time, O(data) memory

Usage:
    python segregate_by_quality.py --input eval.jsonl --data source.jsonl
    python segregate_by_quality.py --input eval.jsonl --data source.jsonl --output-dir ./out
    python segregate_by_quality.py --input eval.jsonl --data source.jsonl \
        --label-field quality_label
"""

from __future__ import annotations

import argparse
import mmap
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

import msgspec.json as mj

# ── Config ────────────────────────────────────────────────────────────────────

LABEL_FIELD    = "label"
ID_FIELD       = "id"
FALLBACK_LABEL = "UNKNOWN"
WRITE_BUF      = 16 * 1024 * 1024   # 4 MiB write buffer per output file

# Single shared encoder / decoder – construction cost paid once
_dec = mj.Decoder()   # decodes to plain dict / list / str / int / float / None
_enc = mj.Encoder()


# ── Fast line iterator via mmap ───────────────────────────────────────────────

def _iterlines(path: Path):
    """
    Yield raw line bytes (no trailing \\n) via mmap.
    The OS handles read-ahead; we do zero extra buffer copies.
    Falls back gracefully for empty files.
    """
    size = path.stat().st_size
    if size == 0:
        return
    with path.open("rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            while start < size:
                end = mm.find(b"\n", start)
                if end == -1:
                    end = size
                chunk = mm[start:end]
                if chunk and chunk != b"\r":
                    yield chunk.rstrip(b"\r")
                start = end + 1


# ── Core logic ────────────────────────────────────────────────────────────────

def load_data_records(data_path: Path) -> dict[str, dict]:
    """Load data JSONL into {id: record}."""
    records: dict[str, dict] = {}
    warn = 0
    for lineno, raw in enumerate(_iterlines(data_path), 1):
        try:
            obj: dict = _dec.decode(raw)
        except Exception as exc:
            warn += 1
            print(f"  [WARN] {data_path.name}:{lineno} parse error: {exc}", file=sys.stderr)
            continue
        rid = obj.get(ID_FIELD)
        if rid is None:
            warn += 1
            print(f"  [WARN] {data_path.name}:{lineno} missing '{ID_FIELD}', skipping",
                  file=sys.stderr)
            continue
        records[str(rid)] = obj
    if warn:
        print(f"  {warn:,} warnings in {data_path.name}", file=sys.stderr)
    return records


def process(
    input_path:  Path,
    data_path:   Path,
    output_dir:  Path,
    label_field: str = LABEL_FIELD,
) -> None:
    stem   = input_path.stem
    errors = 0

    # ── 1. Load data lookup ────────────────────────────────────────────────
    print(f"Loading data file : {data_path}")
    t_load = time.perf_counter()
    data_lookup = load_data_records(data_path)
    print(f"  {len(data_lookup):,} records  ({time.perf_counter()-t_load:.2f}s)")

    # ── 2. Prepare output dir & lazy bucket file-handles ──────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, BinaryIO]  = {}
    counts:  dict[str, int]       = defaultdict(int)
    NEWLINE = b"\n"

    def get_handle(label: str) -> BinaryIO:
        if label not in handles:
            out = output_dir / f"{stem}_{label}.jsonl"
            handles[label] = open(out, "wb", buffering=WRITE_BUF)
        return handles[label]

    matched_ids: set[str] = set()

    # ── 3. Single-pass: decode → merge → encode → write ───────────────────
    print(f"Processing input  : {input_path}")
    t_proc = time.perf_counter()

    for lineno, raw in enumerate(_iterlines(input_path), 1):
        try:
            obj: dict = _dec.decode(raw)
        except Exception as exc:
            errors += 1
            print(f"  [WARN] {input_path.name}:{lineno} parse error: {exc}", file=sys.stderr)
            continue

        rid = str(obj.get(ID_FIELD, ""))
        if rid and rid in data_lookup:
            merged = data_lookup[rid] | obj   # dict union (3.9+); input wins on conflict
            matched_ids.add(rid)
        else:
            continue 

        label = str(merged.get(label_field) or FALLBACK_LABEL).upper()
        fh = get_handle(label)
        fh.write(_enc.encode(merged))
        fh.write(NEWLINE)
        counts[label] += 1

    # ── 4. Unmatched data records ──────────────────────────────────────────
    unmatched_count = 0
    if len(matched_ids) < len(data_lookup):
        fh_unmatched = get_handle("UNMATCHED")
        for rid, rec in data_lookup.items():
            if rid not in matched_ids:
                fh_unmatched.write(_enc.encode(rec))
                fh_unmatched.write(NEWLINE)
                unmatched_count += 1
        counts["UNMATCHED"] = unmatched_count

    # ── 5. Flush & close all handles ──────────────────────────────────────
    for fh in handles.values():
        fh.close()

    elapsed_proc = time.perf_counter() - t_proc

    # ── 6. Report ─────────────────────────────────────────────────────────
    print(f"\nOutput files → {output_dir.resolve()}")
    for label, count in sorted(counts.items()):
        tag = "  ← unmatched data records" if label == "UNMATCHED" else ""
        print(f"  {stem}_{label}.jsonl  →  {count:,} records{tag}")

    total_input = sum(v for k, v in counts.items() if k != "UNMATCHED")
    print(f"\n  Total input records  : {total_input:,}")
    print(f"  Matched to data      : {len(matched_ids):,}")
    print(f"  Unmatched data recs  : {unmatched_count:,}")
    if errors:
        print(f"  Parse errors         : {errors:,}")
    print(f"  Processing time      : {elapsed_proc:.2f}s")
    if total_input:
        rate = total_input / elapsed_proc if elapsed_proc > 0 else float("inf")
        print(f"  Throughput           : {rate:,.0f} records/s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join input JSONL with data JSONL on `id`, segregate by `label`."
    )
    parser.add_argument("--input",  "-i", type=Path, required=True,
                        help="Input JSONL (evaluation results; contains `label`).")
    parser.add_argument("--data",   "-d", type=Path, required=True,
                        help="Data JSONL (source data; contains `id`).")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Output directory (default: same dir as --input).")
    parser.add_argument("--label-field", default=LABEL_FIELD,
                        help=f"JSON key to segregate on (default: '{LABEL_FIELD}').")

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file not found: {args.input}")
    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}")

    output_dir = args.output_dir or args.input.parent

    t0 = time.perf_counter()
    process(args.input, args.data, output_dir, label_field=args.label_field)
    print(f"\n  ⏱  Total elapsed: {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()