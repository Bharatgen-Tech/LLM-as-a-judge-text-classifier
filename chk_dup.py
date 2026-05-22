#!/usr/bin/env python3
"""
check_duplicate_ids.py
----------------------
Finds duplicate IDs in a JSONL file.
Uses msgspec + mmap for maximum throughput.

Usage:
    python check_duplicate_ids.py <file.jsonl>
    python check_duplicate_ids.py <file.jsonl> --field id
    python check_duplicate_ids.py <file.jsonl> --show-dupes 20
"""

import argparse
import mmap
import sys
import time
from collections import Counter
from pathlib import Path

import msgspec.json as mj

_dec = mj.Decoder()


def _iterlines(path: Path):
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
                chunk = mm[start:end].rstrip(b"\r")
                if chunk:
                    yield chunk
                start = end + 1


def check_duplicates(path: Path, field: str = "id", show_dupes: int = 10) -> None:
    counts: Counter = Counter()
    total = 0
    null_count = 0
    errors = 0

    t0 = time.perf_counter()

    for raw in _iterlines(path):
        try:
            obj: dict = _dec.decode(raw)
        except Exception:
            errors += 1
            continue
        total += 1
        val = obj.get(field)
        if val is None:
            null_count += 1
            continue
        counts[str(val)] += 1

    elapsed = time.perf_counter() - t0
    rate = total / elapsed if elapsed > 0 else float("inf")

    unique       = sum(1 for v in counts.values() if v == 1)
    dupes        = {k: v for k, v in counts.items() if v > 1}
    dupe_ids     = len(dupes)
    dupe_records = sum(v - 1 for v in dupes.values())  # extra copies

    print(f"File              : {path}")
    print(f"Field checked     : '{field}'")
    print(f"Total records     : {total:,}")
    print(f"Null/missing      : {null_count:,}")
    print(f"Unique IDs        : {unique:,}")
    print(f"Duplicate IDs     : {dupe_ids:,}  (extra copies: {dupe_records:,})")
    if errors:
        print(f"Parse errors      : {errors:,}")
    print(f"Elapsed           : {elapsed:.2f}s  ({rate:,.0f} records/s)")

    if dupes and show_dupes > 0:
        print(f"\nTop {min(show_dupes, len(dupes))} most-duplicated IDs:")
        for kid, cnt in sorted(dupes.items(), key=lambda x: -x[1])[:show_dupes]:
            print(f"  {cnt:>6}x  {kid}")


def main():
    parser = argparse.ArgumentParser(
        description="Find duplicate IDs in a JSONL file."
    )
    parser.add_argument("file", type=Path)
    parser.add_argument("--field", default="id",
                        help="Field to check for duplicates (default: id)")
    parser.add_argument("--show-dupes", type=int, default=10,
                        help="How many duplicate IDs to print (default: 10, 0 to suppress)")
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    check_duplicates(args.file, args.field, args.show_dupes)


if __name__ == "__main__":
    main()