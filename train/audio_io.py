"""Shared audio I/O for the wake-word pipeline (our own module).

Lives at the train/ root — NOT inside the vendored microwakeword tree — so
our logic stays in our own files and the vendored code only decorates it
(see microwakeword/audio/clips.py). Keeping a single canonical loader avoids
divergence bugs such as one consumer accepting a sample rate another rejects,
or two consumers resampling with different filters.

Deliberately TF-free (numpy + scipy only) so both the student feature pipeline
(clips.py) and the distillation teacher (train_teacher.py, which must not
import TensorFlow) can import it.
"""

import numpy as np
import scipy.io.wavfile
import scipy.signal

TARGET_SR = 16000


def load_wav_16k(path) -> np.ndarray:
    """Load a WAV as mono float32 in [-1, 1], resampled to 16 kHz.

    Handles any input sample rate (some Piper voices emit 22050 Hz) and
    multi-channel WAVs. scipy.io.wavfile is used instead of soundfile because
    libsndfile segfaults when called after TensorFlow installs its custom
    malloc hooks.
    """
    sr, data = scipy.io.wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    if sr != TARGET_SR:
        data = scipy.signal.resample(data, round(len(data) * TARGET_SR / sr))
    return data.astype(np.float32)
