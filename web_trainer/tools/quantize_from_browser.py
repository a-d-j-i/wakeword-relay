#!/usr/bin/env python3
"""
Quantize browser-exported weights into a streaming INT8 TFLite model.

Reads the wakeword_weights.bin produced by the "Export Weights" button in the
web trainer (milestone 7), builds the same MixedNet in Keras, injects the
float32 weights, converts to a streaming SavedModel, then quantizes to INT8.

Usage (activate train/venv first):
    python web_trainer/tools/quantize_from_browser.py \\
        --weights wakeword_weights.bin \\
        --out_dir /path/to/output_dir

Output:
    <out_dir>/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite

This is a drop-in replacement for the .tflite produced by train.py.
Copy it to firmware/models/ and update the JSON manifest probability_cutoff as
needed (see README).
"""

import argparse
import os
import struct
import sys
from pathlib import Path

# Allow running from any directory; microwakeword lives in train/.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _TOOLS_DIR.parents[1]
_TRAIN_DIR  = _REPO_ROOT / "train"
sys.path.insert(0, str(_TRAIN_DIR))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only

import numpy as np

# ── Architecture constants (must match web_trainer/src/mixednet.h) ──────────

_FIRST_K    = 5    # first conv kernel size
_IN_F       = 40   # spectrogram feature dim
_FIRST_F    = 32   # first conv output channels
_FILTERS    = 64   # MixConv block output channels
_STRIDE     = 3    # first conv temporal stride

# Per-block depthwise kernel sizes and input channel counts
_BLK_DW_KS = [[5], [7, 11], [9, 15], [23]]
_BLK_C_IN  = [_FIRST_F, _FILTERS, _FILTERS, _FILTERS]

def _count_params(dense_in: int) -> int:
    """Total param count for a given dense input dimension."""
    return (
        _FIRST_K * _IN_F * _FIRST_F
        + sum(
            sum(ks * (c_in // len(ks_list)) for ks in ks_list)
            + c_in * _FILTERS
            + 4 * _FILTERS
            for ks_list, c_in in zip(_BLK_DW_KS, _BLK_C_IN)
        )
        + dense_in + 1
    )

_WINDOW_FRAMES = 17
_N_PARAMS_POOLED = _count_params(_FILTERS)          # 24 801 (pooled=1)
_N_PARAMS_FULL   = _count_params(_WINDOW_FRAMES * _FILTERS)  # 25 825 (pooled=0)

# Calibration range observed in training data (float32 spectrograms)
_SPEC_MIN = 0.0
_SPEC_MAX = 26.0


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Browser weights → INT8 TFLite")
    p.add_argument("--weights", required=True,
                   help="Path to wakeword_weights.bin exported from the browser")
    p.add_argument("--out_dir", default="data/browser_model",
                   help="Directory that will receive the TFLite and intermediate files")
    p.add_argument("--calibration_samples", type=int, default=500,
                   help="Number of synthetic spectrograms for INT8 calibration")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for synthetic calibration data")
    return p.parse_args()


# ── Binary parser ────────────────────────────────────────────────────────────

def load_browser_weights(path: str) -> tuple[np.ndarray, int]:
    """Parse wakeword_weights.bin and return (weights, pooled_flag).

    File format (little-endian):
        4 bytes  magic  "MWWW"
        4 bytes  uint32 n_params
        n_params × 4 bytes  float32 weights

    pooled is auto-detected from n_params:
        24 801 → pooled=1 (global-avg-pool + Dense(64→1))
        25 825 → pooled=0 (flatten(17×64) + Dense(1088→1))
    """
    data = Path(path).read_bytes()
    if data[:4] != b"MWWW":
        raise ValueError(f"Bad magic {data[:4]!r} — is this a wakeword_weights.bin?")
    (n_params,) = struct.unpack_from("<I", data, 4)
    weights = np.frombuffer(data, dtype="<f4", offset=8)
    if len(weights) != n_params:
        raise ValueError(f"Truncated file: expected {n_params} params, got {len(weights)}")
    if n_params == _N_PARAMS_POOLED:
        pooled = 1
    elif n_params == _N_PARAMS_FULL:
        pooled = 0
    else:
        raise ValueError(
            f"Unknown param count {n_params} "
            f"(expected {_N_PARAMS_POOLED} for pooled or {_N_PARAMS_FULL} for full-window)"
        )
    print(f"Loaded {n_params} float32 params from {path} — pooled={pooled}")
    return weights.copy(), pooled


# ── Model construction ───────────────────────────────────────────────────────

def build_keras_model(pooled: int = 1):
    """Build the training-mode MixedNet matching the WASM architecture.

    pooled=1: global-avg-pool + Dense(64→1)   — 24 801 params
    pooled=0: flatten(17×64) + Dense(1088→1)  — 25 825 params (train.py default)

    Returns (model, spec_len) where spec_len is the expected input spectrogram
    length (number of frontend frames fed to the training model).
    """
    import argparse as _ap
    from microwakeword import mixednet

    flags = _ap.Namespace(
        pointwise_filters="64,64,64,64",
        repeat_in_block="1, 1, 1, 1",
        mixconv_kernel_sizes="[5], [7,11], [9,15], [23]",
        residual_connection="0,0,0,0",
        first_conv_filters=_FIRST_F,
        first_conv_kernel_size=_FIRST_K,
        stride=_STRIDE,
        spatial_attention=0,
        pooled=pooled,
        max_pool=0,
    )

    # Mirror load_config() from model_train_eval.py
    clip_ms     = 1500
    rate        = 16000
    window_ms   = 30
    step_ms     = 10

    desired      = int(rate * clip_ms   / 1000)                  # 24 000 samples
    window       = int(rate * window_ms / 1000)                   # 480
    step         = int(_STRIDE * rate * step_ms / 1000)           # 480
    spec_final   = 1 + int((desired - window) / step)             # 50
    spec_dropped = mixednet.spectrogram_slices_dropped(flags)     # 154
    spec_len     = spec_final + spec_dropped                       # 204

    print(f"spectrogram_length = {spec_len}  (final={spec_final} + dropped={spec_dropped})")

    model = mixednet.model(flags, shape=(spec_len, _IN_F), batch_size=1)
    return model, spec_len


# ── Weight injection ─────────────────────────────────────────────────────────

def inject_weights(model, weights: np.ndarray, pooled: int = 1) -> None:
    """Map the flat C++ weight array into the Keras model's layers.

    C++ get_params() layout (see web_trainer/src/mixednet.cpp):
        first_conv [kT, C_in, C_out]
        for each block:
            dw[g]   [kT, C_g]  for each group g
            pw      [C_in, filters]
            bn_gamma, bn_beta, bn_mean, bn_var  [filters] each
        dense_w [filters]           (pooled=1) or [_WINDOW_FRAMES * filters] (pooled=0)
        dense_b  scalar

    Keras weight shapes:
        Conv2D kernel:          [kT, 1, C_in, C_out]
        DepthwiseConv2D kernel: [kT, 1, C_g, 1]
        Conv2D(1×1) kernel:     [1, 1, C_in, filters]
        BatchNorm:              [gamma, beta, moving_mean, moving_var]
        Dense (pooled=1):       kernel [filters, 1], bias [1]
        Dense (pooled=0):       kernel [_WINDOW_FRAMES*filters, 1], bias [1]
    """
    import tensorflow as tf

    offset = 0

    def take(n: int) -> np.ndarray:
        nonlocal offset
        v = weights[offset:offset + n].copy()
        offset += n
        return v

    from microwakeword.layers import stream as _stream_mod

    # The first Conv2D is the cell inside the first Stream layer, not a top-level
    # model layer. All other Conv2D layers (pointwise 1×1) are top-level.
    first_stream = next(l for l in model.layers if isinstance(l, _stream_mod.Stream)
                        and isinstance(l.cell, tf.keras.layers.Conv2D))
    first_conv   = first_stream.cell
    pw_layers    = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    dw_layers    = [l for l in model.layers if isinstance(l, tf.keras.layers.DepthwiseConv2D)]
    bn_layers    = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

    _check("pointwise Conv2D", len(pw_layers),  len(_BLK_DW_KS))
    _check("DepthwiseConv2D", len(dw_layers),    sum(len(ks) for ks in _BLK_DW_KS))
    _check("BatchNorm",       len(bn_layers),    len(_BLK_DW_KS))
    _check("Dense",           len(dense_layers), 1)

    # first conv: [kT, C_in, C_out] → [kT, 1, C_in, C_out]
    fc = take(_FIRST_K * _IN_F * _FIRST_F).reshape(_FIRST_K, _IN_F, _FIRST_F)
    first_conv.set_weights([fc[:, np.newaxis, :, :]])

    dw_idx = 0
    for b, (ks_list, C_in) in enumerate(zip(_BLK_DW_KS, _BLK_C_IN)):
        C_g = C_in // len(ks_list)

        for ks in ks_list:
            dw = take(ks * C_g).reshape(ks, C_g)
            # Keras DepthwiseConv2D has a bias by default; C++ has none (BN beta absorbs it)
            dw_layers[dw_idx].set_weights([dw[:, np.newaxis, :, np.newaxis],
                                           np.zeros(C_g, dtype=np.float32)])
            dw_idx += 1

        pw = take(C_in * _FILTERS).reshape(C_in, _FILTERS)
        pw_layers[b].set_weights([pw[np.newaxis, np.newaxis, :, :]])

        gamma = take(_FILTERS)
        beta  = take(_FILTERS)
        mean  = take(_FILTERS)
        var   = take(_FILTERS)
        bn_layers[b].set_weights([gamma, beta, mean, var])

    dense_in = _FILTERS if pooled else _WINDOW_FRAMES * _FILTERS
    dense_w  = take(dense_in).reshape(dense_in, 1)
    dense_b  = take(1)
    dense_layers[0].set_weights([dense_w, dense_b])

    assert offset == len(weights), f"Consumed {offset}/{len(weights)} params — layout mismatch"
    print(f"Injected {offset} params into Keras model (pooled={pooled})")


def _check(name, got, expected):
    if got != expected:
        raise RuntimeError(f"Expected {expected} {name} layers, found {got} — "
                           "model architecture mismatch")


# ── Synthetic calibration data ───────────────────────────────────────────────

def make_representative_dataset(spec_len: int, n_samples: int, seed: int):
    """Return a callable that yields (1, stride, 40) slices for INT8 calibration.

    We use synthetic uniform noise in [_SPEC_MIN, _SPEC_MAX] rather than real
    training data so this script has no dependency on the 10 GB dataset.  The
    first sample guarantees one pixel at each extreme so the quantisation range
    covers the full observed spectrogram value range.
    """
    rng = np.random.default_rng(seed)

    def generator():
        for i in range(n_samples):
            spec = rng.uniform(_SPEC_MIN, _SPEC_MAX, (spec_len, _IN_F)).astype(np.float32)
            if i == 0:
                spec[0, 0] = _SPEC_MIN
                spec[0, 1] = _SPEC_MAX
            for t in range(0, spec.shape[0] - _STRIDE, _STRIDE):
                yield [spec[t : t + _STRIDE][np.newaxis]]  # (1, stride, 40)

    return generator


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    raw_weights, pooled = load_browser_weights(args.weights)

    import tensorflow as tf
    print(f"TensorFlow {tf.__version__}")

    model, spec_len = build_keras_model(pooled)
    inject_weights(model, raw_weights, pooled)

    weights_h5 = out_dir / "best_weights.weights.h5"
    model.save_weights(str(weights_h5))
    print(f"Saved Keras weights → {weights_h5}")

    config = {
        "train_dir": str(out_dir),
        "summaries_dir": str(out_dir / "logs"),
        "stride": _STRIDE,
        "window_step_ms": 10,
        "spectrogram_length": spec_len,
    }

    from microwakeword import utils
    from microwakeword.layers import modes

    print("Converting to streaming SavedModel …")
    utils.convert_model_saved(
        model, config,
        "stream_state_internal",
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
    )

    saved_model_path = str(out_dir / "stream_state_internal")
    print("Quantizing to INT8 TFLite …")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = {tf.lite.Optimize.DEFAULT}
    converter._experimental_variable_quantization = True
    converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(
        make_representative_dataset(spec_len, args.calibration_samples, args.seed)
    )

    tflite_dir  = out_dir / "tflite_stream_state_internal_quant"
    tflite_dir.mkdir(exist_ok=True)
    tflite_path = tflite_dir / "stream_state_internal_quant.tflite"
    tflite_path.write_bytes(converter.convert())

    size_kb = tflite_path.stat().st_size // 1024
    print(f"\nDone. TFLite model ({size_kb} KB) → {tflite_path}")
    print("Copy to firmware/models/ and tune probability_cutoff in the JSON manifest.")


if __name__ == "__main__":
    main()
