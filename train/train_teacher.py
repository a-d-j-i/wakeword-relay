#!/usr/bin/env python3
"""Train an openWakeWord-style teacher classifier for one wake phrase (B1 of
DISTILLATION.md).

Frozen Google speech_embedding backbone (openWakeWord's ONNX re-exports,
Apache-2.0) + a small dense head trained on our Piper-generated positives vs
local negative audio (FMA music + AudioSet ambient). Positives are augmented
with the exact same vendored Augmentation the student pipeline uses, so the
teacher sees the same domain it will later score in B2.

Runs in the same venv as train.py (its onnx/onnxruntime deps are in
train/requirements.txt — onnxruntime is needed by --distill scoring anyway):

    cd train
    source venv/bin/activate
    python train_teacher.py \
        --positives_dir data/generated_samples \
        --negatives_dirs data/fma_16k data/audioset_16k \
        --impulse_dirs data/mit_rirs \
        --output_dir data/teacher

Output: <output_dir>/{melspectrogram,embedding_model,head}.onnx + teacher.json
Then distill into the student: python train.py ... --distill --teacher_model <output_dir>
"""

import argparse
import json
import random
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
from tqdm import tqdm

# train/ dir — for teacher_infer and the vendored (TF-free) augmentation module.
_TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TRAIN_DIR))

from teacher_infer import HEAD_EMB_WINDOW, MIN_SAMPLES, SR, Teacher  # noqa: E402
from audio_io import load_wav_16k  # noqa: E402

_BACKBONE_URLS = {
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train the openWakeWord-style teacher (B1)")
    p.add_argument("--positives_dir", default="data/generated_samples",
                   help="Directory of positive wake-phrase WAVs (Piper output)")
    p.add_argument("--negatives_dirs", nargs="+",
                   default=["data/fma_16k", "data/audioset_16k"],
                   help="Directories of 16 kHz negative audio WAVs")
    p.add_argument("--impulse_dirs", nargs="+", default=["data/mit_rirs"],
                   help="Room impulse response WAVs for positive augmentation")
    p.add_argument("--output_dir", default="data/teacher",
                   help="Where the teacher ONNX files + metadata land")
    p.add_argument("--n_aug", type=int, default=2,
                   help="Augmented copies per positive clip, on top of the clean copy (default: 2)")
    p.add_argument("--max_pos", type=int, default=0,
                   help="Cap on positive clips (0 = use all)")
    p.add_argument("--neg_windows", type=int, default=100000,
                   help="Target number of negative training windows (default: 100000)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regen_features", action="store_true",
                   help="Recompute cached embeddings even if present")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print the resolved plan and exit — no download, no training")
    return p.parse_args()


def ensure_backbone(output_dir: Path) -> None:
    for name, url in _BACKBONE_URLS.items():
        dest = output_dir / name
        if dest.exists() and dest.stat().st_size > 100_000:
            continue
        print(f"[teacher] Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, str(dest))
        except Exception as e:
            raise RuntimeError(
                f"Failed to download {url}: {e}\n"
                f"Download it manually into {output_dir}/ (also shipped inside the "
                "openwakeword PyPI package under openwakeword/resources/models/)."
            ) from e


# Shared with the student pipeline (microwakeword/audio/clips.py) so teacher and
# student never diverge on sample-rate handling / resampling. The teacher's SR
# must match the shared loader's 16 kHz target.
assert SR == 16000, f"teacher SR {SR} must match load_wav_16k target (16000)"
load_wav = load_wav_16k


def build_augmenter(args):
    """Same augmentation config as train.py generate_augmented_features()."""
    from microwakeword.audio.augmentation import Augmentation

    return Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5,
        },
        impulse_paths=[d for d in args.impulse_dirs if Path(d).is_dir()],
        background_paths=[d for d in args.negatives_dirs if Path(d).is_dir()],
        background_min_snr_db=-5,
        background_max_snr_db=10,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def positive_features(teacher, wavs, augmenter, n_aug):
    """Embed clean + augmented copies; one [16, 96] feature per copy.

    The phrase sits at the end of every clip (augmentation right-pads only
    ~0.2 s of jitter), so the last HEAD_EMB_WINDOW embedding frames cover it.
    Returns (features [N,16,96], clip_index [N]) — clip_index ties augmented
    copies to their source clip so train/val splitting can't leak.
    """
    feats, clip_idx = [], []
    for i, path in enumerate(tqdm(wavs, desc="embed positives")):
        clean = load_wav(path)
        copies = [clean] + [augmenter.augment_clip(clean) for _ in range(n_aug)]
        for audio in copies:
            # Pad/trim the AUDIO to exactly MIN_SAMPLES so this matches
            # Teacher.score_clip() bit-for-bit: the head must train on the same
            # representation it scores at inference. Audio-domain silence embeds
            # to real (non-zero) frames; padding in the embedding domain instead
            # trains short clips on literal-zero rows that never occur at
            # inference — the cause of low clean-clip self-check scores. Clips
            # >= MIN_SAMPLES (all augmented copies) are unchanged. Yields a fixed
            # 16 embedding frames, so np.stack below is safe.
            if audio.shape[0] > MIN_SAMPLES:
                audio = audio[-MIN_SAMPLES:]
            elif audio.shape[0] < MIN_SAMPLES:
                audio = np.pad(audio, (MIN_SAMPLES - audio.shape[0], 0))
            emb = teacher.embed(audio)
            feats.append(emb[-HEAD_EMB_WINDOW:])
            clip_idx.append(i)
    return np.stack(feats).astype(np.float32), np.array(clip_idx)


def negative_features(teacher, neg_dirs, target_windows, rng):
    """Slice negative audio into [16, 96] windows (stride 8 frames = 0.64 s).

    Returns (features [N,16,96], file_index [N]) for leak-free splitting.
    """
    files = []
    for d in neg_dirs:
        files.extend(sorted(Path(d).glob("*.wav")))
    if not files:
        raise FileNotFoundError(f"No negative WAVs found in {neg_dirs}")
    rng.shuffle(files)

    feats, file_idx = [], []
    pbar = tqdm(total=target_windows, desc="embed negatives")
    for i, path in enumerate(files):
        if len(feats) >= target_windows:
            break
        try:
            audio = load_wav(path)
        except Exception:
            continue
        if audio.shape[0] < MIN_SAMPLES:
            continue
        emb = teacher.embed(audio)
        if emb.shape[0] < HEAD_EMB_WINDOW:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            emb, (HEAD_EMB_WINDOW, emb.shape[1])
        )[::8, 0]
        for w in windows:
            feats.append(w.astype(np.float32))
            file_idx.append(i)
        pbar.update(len(windows))
    pbar.close()
    return np.stack(feats)[:target_windows], np.array(file_idx)[:target_windows]


def split_by_group(features, groups, val_frac, rng):
    """Train/val split keeping all rows of a group on the same side."""
    unique = list(np.unique(groups))
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_frac))
    val_groups = set(unique[:n_val])
    val_mask = np.isin(groups, list(val_groups))
    return features[~val_mask], features[val_mask]


def train_head(pos_tr, neg_tr, pos_val, neg_val, args):
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)

    head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(HEAD_EMB_WINDOW * 96, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(),
        nn.Linear(64, 1),
    )

    def to_xy(pos, neg):
        x = torch.from_numpy(np.concatenate([pos, neg]))
        y = torch.zeros(len(x), 1)
        y[: len(pos)] = 1.0
        return x, y

    x_tr, y_tr = to_xy(pos_tr, neg_tr)
    x_val, y_val = to_xy(pos_val, neg_val)

    pos_weight = torch.tensor([min(len(neg_tr) / max(len(pos_tr), 1), 20.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    best_val, best_state, patience = float("inf"), None, 0
    n = len(x_tr)
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(head(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_loss = loss_fn(head(x_val), y_val).item()
        print(f"[teacher] epoch {epoch + 1:3d}  train_loss {total / n:.4f}  val_loss {val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                print("[teacher] early stop (no val improvement for 5 epochs)")
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head


def val_metrics(head, pos_val, neg_val):
    import torch

    with torch.no_grad():
        ps = torch.sigmoid(head(torch.from_numpy(pos_val))).numpy().reshape(-1)
        ns = torch.sigmoid(head(torch.from_numpy(neg_val))).numpy().reshape(-1)

    # Rank-based AUC.
    order = np.argsort(np.concatenate([ps, ns]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[: len(ps)].sum() - len(ps) * (len(ps) + 1) / 2) / (len(ps) * len(ns))

    metrics = {"auc": float(auc), "n_pos_val": len(ps), "n_neg_val": len(ns)}
    ns_sorted = np.sort(ns)
    for fpr in (1e-2, 1e-3):
        k = int(np.ceil(len(ns) * fpr))
        thr = float(ns_sorted[-k - 1]) if k < len(ns) else 0.0
        metrics[f"recall_at_fpr_{fpr:g}"] = float((ps > thr).mean())
        metrics[f"threshold_at_fpr_{fpr:g}"] = thr
    return metrics


def export_onnx(head, output_dir: Path):
    import torch
    import torch.nn as nn

    wrapper = nn.Sequential(head, nn.Sigmoid()).eval()
    dummy = torch.zeros(1, HEAD_EMB_WINDOW, 96)
    torch.onnx.export(
        wrapper, dummy, str(output_dir / "head.onnx"),
        input_names=["input"], output_names=["score"],
        dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
        # Use the legacy TorchScript exporter; the new torch.export/dynamo path
        # (default in torch 2.9+) pulls in onnxscript, which we don't ship. This
        # head is a tiny MLP the legacy exporter handles fine.
        dynamo=False,
    )

    # Parity check: exported ONNX must match torch within float tolerance.
    import onnxruntime as ort

    x = np.random.default_rng(0).normal(0, 1, (64, HEAD_EMB_WINDOW, 96)).astype(np.float32)
    sess = ort.InferenceSession(str(output_dir / "head.onnx"),
                                providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": x})[0].reshape(-1)
    with torch.no_grad():
        torch_out = wrapper(torch.from_numpy(x)).numpy().reshape(-1)
    max_diff = float(np.abs(onnx_out - torch_out).max())
    print(f"[teacher] ONNX export parity: max |torch - onnx| = {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError("head.onnx does not match the torch model")


def self_check(output_dir: Path, pos_wavs, rng):
    """End-to-end sanity: score raw positive WAVs through the full ONNX chain."""
    teacher = Teacher(output_dir)
    sample = rng.sample(pos_wavs, min(20, len(pos_wavs)))
    scores = [teacher.score_clip(load_wav(p)) for p in sample]
    det = sum(s >= 0.5 for s in scores)
    print(f"[teacher] self-check on {len(sample)} clean positives: "
          f"mean score {np.mean(scores):.3f}, {det}/{len(sample)} >= 0.5")
    return {"clean_pos_mean_score": float(np.mean(scores)),
            "clean_pos_detected_at_0.5": f"{det}/{len(sample)}"}


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.dry_run:
        pdir = Path(args.positives_dir)
        n_pos = len(list(pdir.glob("*.wav"))) if pdir.is_dir() else 0
        print("DRY RUN — teacher plan (nothing downloaded or trained):")
        print(f"  positives_dir : {args.positives_dir} "
              f"({n_pos} wavs{'' if pdir.is_dir() else ' — dir not created yet'})")
        print(f"  negatives     : {args.negatives_dirs}")
        print(f"  impulses      : {args.impulse_dirs}")
        print(f"  output_dir    : {args.output_dir}")
        print(f"  n_aug / epochs: {args.n_aug} / {args.epochs}")
        print(f"  neg_windows   : {args.neg_windows}")
        sys.exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_backbone(output_dir)

    pos_wavs = sorted(Path(args.positives_dir).glob("*.wav"))
    if not pos_wavs:
        raise FileNotFoundError(f"No positive WAVs in {args.positives_dir} — "
                                "run train.py first to generate Piper samples.")
    if args.max_pos:
        rng.shuffle(pos_wavs)
        pos_wavs = pos_wavs[: args.max_pos]
    print(f"[teacher] {len(pos_wavs)} positive clips, +{args.n_aug} augmented copies each")

    pos_cache = output_dir / "pos_features.npz"
    neg_cache = output_dir / "neg_features.npz"
    backbone = None

    def get_backbone():
        nonlocal backbone
        if backbone is None:
            backbone = _BackboneTeacher(output_dir)
        return backbone

    if pos_cache.exists() and not args.regen_features:
        d = np.load(pos_cache)
        pos_feats, pos_groups = d["features"], d["groups"]
        print(f"[teacher] loaded cached positive features: {pos_feats.shape}")
    else:
        augmenter = build_augmenter(args)
        pos_feats, pos_groups = positive_features(
            get_backbone(), pos_wavs, augmenter, args.n_aug
        )
        np.savez_compressed(pos_cache, features=pos_feats, groups=pos_groups)

    if neg_cache.exists() and not args.regen_features:
        d = np.load(neg_cache)
        neg_feats, neg_groups = d["features"], d["groups"]
        print(f"[teacher] loaded cached negative features: {neg_feats.shape}")
    else:
        neg_feats, neg_groups = negative_features(
            get_backbone(), args.negatives_dirs, args.neg_windows, rng
        )
        np.savez_compressed(neg_cache, features=neg_feats, groups=neg_groups)

    print(f"[teacher] features: {len(pos_feats)} positive, {len(neg_feats)} negative")

    split_rng = random.Random(args.seed + 1)
    pos_tr, pos_val = split_by_group(pos_feats, pos_groups, args.val_frac, split_rng)
    neg_tr, neg_val = split_by_group(neg_feats, neg_groups, args.val_frac, split_rng)
    print(f"[teacher] split: train {len(pos_tr)}p/{len(neg_tr)}n, "
          f"val {len(pos_val)}p/{len(neg_val)}n")

    head = train_head(pos_tr, neg_tr, pos_val, neg_val, args)
    metrics = val_metrics(head, pos_val, neg_val)
    print("[teacher] validation:", json.dumps(metrics, indent=2))

    export_onnx(head, output_dir)
    check = self_check(output_dir, pos_wavs, rng)

    meta = {
        "type": "openwakeword_head",
        "created": date.today().isoformat(),
        "emb_window": HEAD_EMB_WINDOW,
        "positives_dir": str(args.positives_dir),
        "negatives_dirs": [str(d) for d in args.negatives_dirs],
        "n_pos_features": int(len(pos_feats)),
        "n_neg_features": int(len(neg_feats)),
        "n_aug": args.n_aug,
        "seed": args.seed,
        "val_metrics": metrics,
        "self_check": check,
    }
    (output_dir / "teacher.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\n[teacher] Done. Teacher model: {output_dir}/")
    print("[teacher] Next: sanity-gate it on the real-voice eval set (B5) before "
          "distilling; then: python train.py ... --distill --teacher_model "
          f"{output_dir}")


class _BackboneTeacher(Teacher):
    """Teacher with only the two backbone models (no head yet)."""

    def __init__(self, model_dir):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(
            str(model_dir / "melspectrogram.onnx"), opts, providers=providers
        )
        self.emb = ort.InferenceSession(
            str(model_dir / "embedding_model.onnx"), opts, providers=providers
        )
        self._mel_in = self.mel.get_inputs()[0].name
        self._emb_in = self.emb.get_inputs()[0].name


if __name__ == "__main__":
    main()
