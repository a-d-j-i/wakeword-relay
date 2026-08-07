"""Teacher inference pipeline (openWakeWord-style) via onnxruntime.

Runs the three-stage teacher produced by train_teacher.py:

    audio 16 kHz float [-1,1]
      -> melspectrogram.onnx   (25 ms window / 10 ms hop, 32 mel bins)
      -> embedding_model.onnx  (Google speech_embedding: 76 mel frames -> 96-d,
                                one embedding per 80 ms)
      -> head.onnx             (our trained wake-word head: 16 embeddings -> score)

Depends only on numpy + onnxruntime, so it is importable from both the
training venv (train.py --distill scoring, B2) and the teacher venv
(train_teacher.py export self-check, B1). The same three ONNX files run in
the browser via onnxruntime-web (B5b).
"""

from pathlib import Path

import numpy as np

SR = 16000
MEL_HOP = 160  # samples per mel frame (10 ms)
# openWakeWord's melspectrogram ONNX frames with a ~640-sample effective window
# (measured: nmel ≈ (n − 640)/160 + 1), larger than the 400-sample analysis
# window because of FFT padding. Using 400 here undercounts the required audio
# by one embedding frame (yields 15, not 16), so MIN_SAMPLES uses 640.
MEL_WINDOW_SAMPLES = 640
EMB_MEL_WINDOW = 76  # mel frames consumed per embedding frame
EMB_MEL_STRIDE = 8  # mel frames between embedding frames (80 ms)
HEAD_EMB_WINDOW = 16  # embedding frames consumed by the head (~2 s context)

# Shortest audio that yields one full head window (16 embedding frames = 196 mel
# frames). Verified against the ONNX frontend: 31840 samples ≈ 1.99 s → 16 frames.
MIN_SAMPLES = MEL_WINDOW_SAMPLES + (
    (EMB_MEL_WINDOW + (HEAD_EMB_WINDOW - 1) * EMB_MEL_STRIDE) - 1
) * MEL_HOP


class Teacher:
    """Loads the three teacher ONNX models and scores raw waveforms."""

    def __init__(self, model_dir):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        for name in ("melspectrogram.onnx", "embedding_model.onnx", "head.onnx"):
            if not (model_dir / name).exists():
                raise FileNotFoundError(
                    f"{model_dir / name} not found — train the teacher first: "
                    "python train_teacher.py (see DISTILLATION.md §5 B1)"
                )

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(
            str(model_dir / "melspectrogram.onnx"), opts, providers=providers
        )
        self.emb = ort.InferenceSession(
            str(model_dir / "embedding_model.onnx"), opts, providers=providers
        )
        self.head = ort.InferenceSession(
            str(model_dir / "head.onnx"), opts, providers=providers
        )
        self._mel_in = self.mel.get_inputs()[0].name
        self._emb_in = self.emb.get_inputs()[0].name
        self._head_in = self.head.get_inputs()[0].name

    def melspectrogram(self, audio_f32: np.ndarray) -> np.ndarray:
        """float [-1,1] audio -> [F, 32] mel spectrogram (openWakeWord scaling)."""
        # The ONNX melspectrogram expects int16-range float values; openWakeWord
        # then applies x/10 + 2 before feeding the embedding model.
        x = (np.asarray(audio_f32, dtype=np.float32) * 32767.0)[None, :]
        mel = self.mel.run(None, {self._mel_in: x})[0]
        return np.squeeze(mel) / 10.0 + 2.0

    def embed(self, audio_f32: np.ndarray) -> np.ndarray:
        """float [-1,1] audio -> [N, 96] embedding sequence (one row / 80 ms)."""
        mel = self.melspectrogram(audio_f32)
        if mel.shape[0] < EMB_MEL_WINDOW:
            mel = np.pad(mel, ((EMB_MEL_WINDOW - mel.shape[0], 0), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(
            mel, (EMB_MEL_WINDOW, mel.shape[1])
        )[::EMB_MEL_STRIDE, 0]
        windows = windows[..., None].astype(np.float32)  # [N, 76, 32, 1]
        emb = self.emb.run(None, {self._emb_in: windows})[0]
        return emb.reshape(windows.shape[0], -1)  # [N, 96]

    def score_windows(self, embeddings: np.ndarray) -> np.ndarray:
        """[N, 96] embeddings -> scores for every 16-frame sliding window."""
        if embeddings.shape[0] < HEAD_EMB_WINDOW:
            embeddings = np.pad(
                embeddings, ((HEAD_EMB_WINDOW - embeddings.shape[0], 0), (0, 0))
            )
        windows = np.lib.stride_tricks.sliding_window_view(
            embeddings, (HEAD_EMB_WINDOW, embeddings.shape[1])
        )[:, 0].astype(np.float32)  # [M, 16, 96]
        return self.head.run(None, {self._head_in: windows})[0].reshape(-1)

    def score_clip(self, audio_f32: np.ndarray) -> float:
        """float [-1,1] audio -> max wake-word probability over the clip."""
        audio_f32 = np.asarray(audio_f32, dtype=np.float32)
        if audio_f32.shape[0] < MIN_SAMPLES:
            audio_f32 = np.pad(audio_f32, (MIN_SAMPLES - audio_f32.shape[0], 0))
        return float(self.score_windows(self.embed(audio_f32)).max())
