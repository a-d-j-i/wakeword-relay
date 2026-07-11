#!/usr/bin/env python3
"""Receive mic clips POSTed by the full_duplex_audio component and save them as WAVs.

The firmware POSTs the last 3 s of exactly the PCM bytes it feeds micro_wake_word
(Content-Type: audio/wav, plus X-Wake-Word / X-Probability / X-Tier headers).

Usage:  python3 clip_server.py [port]     # default 5000
Clips land in ./clips/ as <timestamp>_<wake_word>_p<prob>.wav
"""

import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

CLIP_DIR = Path(__file__).parent / "clips"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        wake_word = self.headers.get("X-Wake-Word", "unknown")
        prob = self.headers.get("X-Probability", "0")
        tier = self.headers.get("X-Tier", "0")

        CLIP_DIR.mkdir(exist_ok=True)
        name = f"{time.strftime('%H%M%S')}_{wake_word}_p{prob}_t{tier}.wav"
        path = CLIP_DIR / name
        path.write_bytes(body)
        print(f"saved {path}  ({length} bytes)")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # quiet the default per-request noise
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"listening on 0.0.0.0:{port}, saving to {CLIP_DIR}/")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()