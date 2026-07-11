#!/usr/bin/env python3
"""Rebuild WAVs from PCMDUMP base64 lines in a captured ESPHome log.

The firmware's full_duplex_audio::dump_pcm_b64() logs the mic ring buffer as:
    PCMDUMP <id> BEGIN rate=<hz> bytes=<n> lines=<n>
    PCMDUMP <id> <seq> <base64>
    PCMDUMP <id> END

Usage:  python3 decode_pcm_log.py game.log [outdir]
Writes <outdir>/dump_<id>.wav (default outdir: ./clips) and reports any
missing/corrupt lines so you know whether the serial capture was clean.
"""

import base64
import re
import struct
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
BEGIN = re.compile(r"PCMDUMP (\d+) BEGIN rate=(\d+) bytes=(\d+) lines=(\d+)")
DATA = re.compile(r"PCMDUMP (\d+) (\d+) ([A-Za-z0-9+/=]+)\s*$")
END = re.compile(r"PCMDUMP (\d+) END")


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data" + struct.pack("<I", len(pcm))
    )
    path.write_bytes(header + pcm)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    log_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "clips"
    outdir.mkdir(exist_ok=True)

    dumps: dict[str, dict] = {}  # id -> {rate, lines, chunks: {seq: bytes}}
    for raw in log_path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if (m := BEGIN.search(line)):
            dumps[m[1]] = {"rate": int(m[2]), "bytes": int(m[3]), "lines": int(m[4]), "chunks": {}}
        elif (m := DATA.search(line)) and m[1] in dumps:
            try:
                dumps[m[1]]["chunks"][int(m[2])] = base64.b64decode(m[3])
            except ValueError:
                print(f"dump {m[1]}: corrupt base64 at line {m[2]}")

    if not dumps:
        sys.exit("no PCMDUMP blocks found in log")

    for dump_id, d in dumps.items():
        missing = [i for i in range(d["lines"]) if i not in d["chunks"]]
        pcm = b"".join(d["chunks"].get(i, b"") for i in range(d["lines"]))
        path = outdir / f"dump_{dump_id}.wav"
        write_wav(path, pcm, d["rate"])
        status = "OK" if not missing and len(pcm) == d["bytes"] else f"INCOMPLETE (missing lines: {missing[:10]}{'...' if len(missing) > 10 else ''})"
        print(f"{path}  {len(pcm)} bytes @ {d['rate']} Hz  {status}")


if __name__ == "__main__":
    main()
