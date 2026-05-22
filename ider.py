#!/usr/bin/env python3
"""
Add a UUID to any JSONL record that lacks an "id" key.
Reads from stdin (or a file), writes to stdout.

Usage:
    python add_uuid.py < input.jsonl > output.jsonl
    python add_uuid.py input.jsonl > output.jsonl
    python add_uuid.py input.jsonl output.jsonl
"""

import sys
import uuid
import msgspec.json


def process(in_stream, out_stream) -> None:
    decoder = msgspec.json.Decoder(dict)
    encoder = msgspec.json.Encoder()

    for raw_line in in_stream.buffer:
        line = raw_line.rstrip(b"\n")
        if not line:
            continue

        record: dict = decoder.decode(line)

        if "id" not in record:
            # Insert id as the first key for readability
            record = {"id": str(uuid.uuid4()), **record}

        out_stream.buffer.write(encoder.encode(record))
        out_stream.buffer.write(b"\n")


def main() -> None:
    args = sys.argv[1:]

    if len(args) == 0:
        process(sys.stdin, sys.stdout)

    elif len(args) == 1:
        with open(args[0], "rb") as f:

            class _Wrap:
                buffer = f

            process(_Wrap(), sys.stdout)

    elif len(args) == 2:
        with open(args[0], "rb") as f_in, open(args[1], "wb") as f_out:

            class _InWrap:
                buffer = f_in

            class _OutWrap:
                buffer = f_out

            process(_InWrap(), _OutWrap())

    else:
        print("Usage: add_uuid.py [input.jsonl [output.jsonl]]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()