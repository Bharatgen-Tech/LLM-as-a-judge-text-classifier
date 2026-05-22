#!/usr/bin/env python3
import sys
import uuid
from collections import Counter
import msgspec.json


def process(in_stream, out_stream) -> None:
    decoder = msgspec.json.Decoder(dict)
    encoder = msgspec.json.Encoder()

    # First pass: count ID occurrences
    lines = list(in_stream.buffer)
    records = []
    ids = []
    for raw_line in lines:
        line = raw_line.rstrip(b"\n")
        if not line:
            records.append(None)
            ids.append(None)
            continue
        record = decoder.decode(line)
        records.append(record)
        ids.append(record.get("id"))

    dupes = {id_ for id_, count in Counter(ids).items() if count > 1 and id_ is not None}
    print(f"Found {len(dupes)} duplicate IDs across {sum(Counter(ids)[d] for d in dupes)} records", file=sys.stderr)

    # Second pass: write, appending UUID only to duplicates
    for record in records:
        if record is None:
            out_stream.buffer.write(b"\n")
            continue
        id_ = record.get("id")
        if id_ in dupes:
            record["id"] = f"{id_}_{uuid.uuid4()}"
        elif "id" not in record:
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