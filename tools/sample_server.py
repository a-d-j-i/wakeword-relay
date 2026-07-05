#!/usr/bin/env python3
"""
Receives audio clips from the AI-VOX3 frío/tibio/caliente game and saves them
for model training.  Clips arrive as HTTP POST with a WAV body and metadata headers.

Usage:
    pip install flask
    python tools/sample_server.py

Configure the ESP32 to POST to http://<this-machine>:5000/upload
(set sample_upload_url in firmware/aivox3/secrets.yaml).

Saved files:
    samples/<wake_word>/tier<N>_p<probability>_<timestamp>.wav

These can be fed directly into the training pipeline as positive examples or
used for evaluation.  Sort by tier to separate WIN clips from weak/frío clips.
"""

import os
from datetime import datetime

from flask import Flask, request

app = Flask(__name__)
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


@app.post("/upload")
def upload():
    wake_word = request.headers.get("X-Wake-Word", "unknown")
    probability = request.headers.get("X-Probability", "0.000")
    tier = request.headers.get("X-Tier", "0")

    subdir = os.path.join(SAVE_DIR, wake_word)
    os.makedirs(subdir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"tier{tier}_p{float(probability):.3f}_{timestamp}.wav"
    path = os.path.join(subdir, filename)

    with open(path, "wb") as f:
        f.write(request.data)

    size_kb = len(request.data) / 1024
    print(f"  [{wake_word}] tier={tier} p={probability}  {size_kb:.0f} KB  -> {filename}")
    return "", 200


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Sample server listening on :5000  ->  {os.path.abspath(SAVE_DIR)}/")
    app.run(host="0.0.0.0", port=5000)
