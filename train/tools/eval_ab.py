#!/usr/bin/env python3
"""Automated A/B eval for wake-word .tflite models (e.g. baseline vs distill).

Two subcommands:

  synth   Build an eval clip bank with the SAME vendored Piper voices as training:
            eval_pos/      full wake phrase              (positives  -> recall)
            eval_partial/  partials + confusables        (hard negs  -> must reject)
            eval_slow_g<gap>/  "<w1> <gap> <w2>" concatenated, one dir per gap
                               length (out-of-spec positives; traces fall-off)
          Open-set negatives (real Spanish speech, music, ambient) are NOT synthed
          here -- point `run --neg` at your existing $DL/es_speech_16k etc.

  run     Stream every clip through each model with microWakeWord's own streaming
          interpreter (state reset per clip, step_ms=10, PCAN warm-up pad), then
          report, per model and at each operating cutoff:
            - positive recall (full phrase)
            - PARTIAL false-accept rate  <- the "fires on 'chispa' alone" test
            - slow/gapped recall         <- the "long pause between words" test
            - open-set false-accepts/hour (refractory-aware, like the firmware)
          plus an ROC (FA/hour vs false-reject) overlaying all models. Writes
          results.json and a self-contained report.html (inline SVG, no deps).

Runs in train/venv (needs ai_edge_litert + tensorflow + the vendored microwakeword).
Models live wherever run.sh put them, e.g.
  <phrase>_sa/model_baseline/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
  <phrase>_sa/model_distill/ tflite_stream_state_internal_quant/stream_state_internal_quant.tflite

Example
-------
  # (activate the train venv first: source train/venv/bin/activate)
  # 1) build the clip bank (once)
  python train/tools/eval_ab.py synth \
      --voices_dir /vms2/work_tmp/download/piper_voices \
      --out_root   /vms2/work_tmp/eval_chispa \
      --phrase "chispa magica" --w1 chispa --w2 magica

  # 2) A/B the two students on it
  python train/tools/eval_ab.py run \
      --model baseline=/vms2/work_tmp/chispa_magica_sa/model_baseline/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
      --model distill=/vms2/work_tmp/chispa_magica_sa/model_distill/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
      --pos     /vms2/work_tmp/eval_chispa/eval_pos \
      --partial /vms2/work_tmp/eval_chispa/eval_partial \
      --slow    /vms2/work_tmp/eval_chispa/eval_slow_g* \
      --neg     /vms2/work_tmp/download/es_speech_16k \
      --out_dir /vms2/work_tmp/eval_chispa/ab
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_TRAIN_DIR = Path(__file__).resolve().parents[1]  # train/ (this file is train/tools/)
sys.path.insert(0, str(_TRAIN_DIR))  # microwakeword + audio_io live here

STEP_MS = 10  # 20 (the default) floors streaming scores; 10 matches how we test.


# --------------------------------------------------------------------------- #
# synth: build the eval clip bank with Piper                                   #
# --------------------------------------------------------------------------- #
def _piper(voice: Path, phrase: str, n: int, out_dir: Path):
    """Synthesize `n` clips of `phrase` with one voice into out_dir (as 0.wav ...)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TRAIN_DIR) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    subprocess.run(
        [
            sys.executable, "-m", "piper_sample_generator", phrase,
            "--model", str(voice), "--max-samples", str(n),
            "--batch-size", "50", "--output-dir", str(out_dir),
        ],
        check=True, env=env,
    )


def cmd_synth(args):
    from audio_io import load_wav_16k  # vendored canonical 16 kHz mono loader

    voices = sorted(Path(args.voices_dir).glob("*.onnx"))
    if not voices:
        raise SystemExit(f"No *.onnx voices in {args.voices_dir}")

    root = Path(args.out_root)
    pos_dir = root / "eval_pos"
    par_dir = root / "eval_partial"
    for d in (pos_dir, par_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Hard negatives: partials of the phrase + phonetic confusables. Keep the
    # FULL phrase out of here. Defaults are Spanish; override with --partials.
    partials = args.partials or [
        args.w1, args.w2,                        # each word alone
        "mi mamá me ama", "mi mamá me mima",     # open-vowel confusables
        "una chispa", "chispa magia", "mágica chispa",
        "buenas", "buenos días", "cómo estás", "enciende la luz",
    ]

    print(f"[synth] {len(voices)} voices -> {root}")
    # positives (full phrase)
    for vi, v in enumerate(voices):
        tmp = pos_dir / "_tmp"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        _piper(v, args.phrase, args.per_phrase, tmp)
        for w in sorted(tmp.glob("*.wav"), key=lambda p: int(p.stem)):
            w.rename(pos_dir / f"pos_{vi:02d}_{w.stem}.wav")
        shutil.rmtree(tmp)

    # partials / confusables
    for pi, phrase in enumerate(partials):
        for vi, v in enumerate(voices):
            tmp = par_dir / "_tmp"; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
            _piper(v, phrase, args.per_partial, tmp)
            for w in sorted(tmp.glob("*.wav"), key=lambda p: int(p.stem)):
                w.rename(par_dir / f"par{pi:02d}_{vi:02d}_{w.stem}.wav")
            shutil.rmtree(tmp)

    # slow / gapped positives: "<w1> <silence gap> <w2>" per voice, one dir per
    # gap length so `run` can trace where both models fall off the phrase.
    from scipy.io import wavfile
    slow_dirs = []
    for g in args.gap_sweep:
        d = root / f"eval_slow_g{g:.2f}"
        d.mkdir(parents=True, exist_ok=True)
        slow_dirs.append((g, d))
    for vi, v in enumerate(voices):
        t1 = root / "_w1"; t2 = root / "_w2"
        for t in (t1, t2):
            shutil.rmtree(t, ignore_errors=True); t.mkdir()
        _piper(v, args.w1, 3, t1)
        _piper(v, args.w2, 3, t2)
        w1s = sorted(t1.glob("*.wav")); w2s = sorted(t2.glob("*.wav"))
        pairs = [((load_wav_16k(str(a)) * 32767).astype(np.int16),
                  (load_wav_16k(str(b)) * 32767).astype(np.int16))
                 for a, b in zip(w1s, w2s)]
        for g, d in slow_dirs:
            gap = np.zeros(int(16000 * g), dtype=np.int16)
            for k, (aw, bw) in enumerate(pairs):
                joined = np.concatenate([aw, gap, bw])
                wavfile.write(d / f"slow_{vi:02d}_{k}.wav", 16000, joined)
        shutil.rmtree(t1); shutil.rmtree(t2)

    for d in (pos_dir, par_dir, *[d for _, d in slow_dirs]):
        print(f"  {d.name:16s} {len(list(d.glob('*.wav')))} clips")


# --------------------------------------------------------------------------- #
# run: stream clips through each model + metrics                               #
# --------------------------------------------------------------------------- #
def _load_clips(d: Path):
    from audio_io import load_wav_16k
    return [load_wav_16k(str(p)) for p in sorted(Path(d).glob("*.wav"))]


def _gap_label(d):
    """'.../eval_slow_g0.30' -> '0.30s'; fall back to the bare dir name."""
    name = Path(d).name
    return f"{name.split('_g')[-1]}s" if "_g" in name else name


def _gap_num(label):
    try:
        return float(label.rstrip("s"))
    except ValueError:
        return float("inf")


def _stream(model, wav_f32, pad_ms):
    """Reset streaming state, warm PCAN with silence, return per-frame probs."""
    # Reset internal (variable) state so clips don't contaminate each other
    # (presence checked once at load in cmd_run).
    model.model.reset_all_variables()
    if pad_ms:
        pad = np.zeros(int(16000 * pad_ms / 1000.0), dtype=np.float32)
        wav_f32 = np.concatenate([pad, wav_f32.astype(np.float32)])
    return np.asarray(model.predict_clip(wav_f32, step_ms=STEP_MS), dtype=np.float32)


def cmd_run(args):
    from microwakeword.inference import Model
    from microwakeword.test import compute_false_accepts_per_hour

    models = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model must be name=path, got {spec!r}")
        models[name] = Model(path)
        if not hasattr(models[name].model, "reset_all_variables"):
            raise SystemExit(
                "interpreter has no reset_all_variables(); streaming state would "
                "bleed between clips and invalidate the A/B. Update ai_edge_litert, "
                "or evaluate one clip per interpreter instance."
            )
        print(f"[model] {name}: stride={models[name].stride} <- {path}")

    pos = _load_clips(args.pos) if args.pos else []
    partial = _load_clips(args.partial) if args.partial else []
    # one (gap-label, clips) entry per gapped dir, sorted by gap length
    slow_sets = sorted(
        ((_gap_label(d), _load_clips(d)) for d in (args.slow or [])),
        key=lambda kv: (_gap_num(kv[0]), kv[0]),
    )
    negs = []
    for d in args.neg or []:
        negs += _load_clips(d)
    print(f"[clips] pos={len(pos)} partial={len(partial)} "
          f"slow={{{', '.join(f'{lbl}:{len(c)}' for lbl, c in slow_sets)}}} neg={len(negs)}")

    cutoffs = np.round(np.linspace(0.05, 0.99, 95), 4)
    report_cutoffs = sorted(set([round(c, 2) for c in args.cutoff]))
    results = {"cutoffs": cutoffs.tolist(), "report_cutoffs": report_cutoffs, "models": {}}

    for name, model in models.items():
        step_s = STEP_MS / 1000.0
        refractory_slices = max(1, int(round(args.refractory_s / (model.stride * step_s))))

        # clip-level peak score per category (positives / partials / slow)
        def peaks(clips):
            return np.array([float(_stream(model, w, args.pad_ms).max()) if len(w) else 0.0
                             for w in clips]) if clips else np.zeros(0)

        pos_peak = peaks(pos)
        par_peak = peaks(partial)

        # open-set negatives: keep full per-clip probability tracks for FA/hour
        neg_tracks = [_stream(model, w, args.pad_ms) for w in negs]
        faph = compute_false_accepts_per_hour(
            neg_tracks, cutoffs,
            ignore_slices_after_accept=refractory_slices,
            stride=model.stride, step_s=step_s,
        ) if neg_tracks else np.zeros_like(cutoffs)

        # sweep-derived rates (fractions in [0,1])
        pos_recall = np.array([(pos_peak > c).mean() if len(pos_peak) else float("nan")
                               for c in cutoffs])
        par_fa = np.array([(par_peak > c).mean() if len(par_peak) else float("nan")
                           for c in cutoffs])
        # gapped-positive recall, one curve per gap length
        slow_recall = {}
        for label, clips in slow_sets:
            sp = peaks(clips)
            slow_recall[label] = [float((sp > c).mean()) if len(sp) else float("nan")
                                  for c in cutoffs]

        results["models"][name] = {
            "stride": int(model.stride),
            "peaks": {"pos": pos_peak.tolist(), "partial": par_peak.tolist()},
            "pos_recall": pos_recall.tolist(),
            "partial_fa_rate": par_fa.tolist(),
            "slow_recall": slow_recall,  # {gap_label: recall-vs-cutoff}
            "faph": faph.tolist(),
        }

    _print_tables(results, report_cutoffs)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "report.html").write_text(_html_report(results))
    print(f"\n[out] {out/'results.json'}\n[out] {out/'report.html'}")


def _at(cutoffs, arr, c):
    i = int(np.argmin(np.abs(np.asarray(cutoffs) - c)))
    return arr[i]


def _slow_labels(results):
    """Union of gap labels across models, ordered by gap length."""
    labels = set()
    for m in results["models"].values():
        labels.update(m.get("slow_recall", {}).keys())
    return sorted(labels, key=_gap_num)


def _print_tables(results, report_cutoffs):
    cutoffs = results["cutoffs"]
    for c in report_cutoffs:
        print(f"\n==== operating cutoff = {c:.2f} "
              f"(higher partial-FA / lower recall = worse) ====")
        hdr = f"{'model':<12}{'pos recall':>12}{'partial FA':>12}{'FA/hour':>10}"
        print(hdr); print("-" * len(hdr))
        for name, m in results["models"].items():
            pr = _at(cutoffs, m["pos_recall"], c)
            pf = _at(cutoffs, m["partial_fa_rate"], c)
            fa = _at(cutoffs, m["faph"], c)
            print(f"{name:<12}{pr:>11.1%}{pf:>12.1%}{fa:>10.2f}")

    # gapped-phrase recall vs gap length (where both models fall off the phrase)
    gaps = _slow_labels(results)
    if gaps:
        for c in report_cutoffs:
            print(f"\n==== gapped-phrase recall @ cutoff {c:.2f} "
                  f"(gap = silence between the two words) ====")
            hdr = f"{'model':<12}" + "".join(f"{g:>9}" for g in gaps)
            print(hdr); print("-" * len(hdr))
            for name, m in results["models"].items():
                cells = "".join(f"{_at(cutoffs, m['slow_recall'][g], c):>8.0%} " if g in
                                m["slow_recall"] else f"{'-':>9}" for g in gaps)
                print(f"{name:<12}{cells}")

    # margin: mean peak on positives minus mean peak on partials (bigger = sharper)
    print("\n==== peak-score margins (mean peak: higher pos & lower partial = sharper) ====")
    for name, m in results["models"].items():
        pp = np.array(m["peaks"]["pos"]); pa = np.array(m["peaks"]["partial"])
        mp = pp.mean() if len(pp) else float("nan")
        ma = pa.mean() if len(pa) else float("nan")
        print(f"{name:<12} pos_peak={mp:.3f}  partial_peak={ma:.3f}  margin={mp-ma:+.3f}")


# --------------------------------------------------------------------------- #
# self-contained HTML (inline SVG ROC, no plotting deps)                       #
# --------------------------------------------------------------------------- #
def _html_report(results):
    cutoffs = np.array(results["cutoffs"])
    colors = ["#4f8cff", "#ff5c8a", "#38c793", "#f0a020", "#a06cf0"]
    # ROC: x = FA/hour (clip to [0, xmax]), y = false-reject = 1 - pos_recall
    xmax = 5.0
    W, H, P = 560, 360, 48
    def sx(v): return P + (min(v, xmax) / xmax) * (W - 2 * P)
    def sy(v): return H - P - v * (H - 2 * P)
    paths, legend = [], []
    for i, (name, m) in enumerate(results["models"].items()):
        col = colors[i % len(colors)]
        faph = np.array(m["faph"]); fr = 1 - np.array(m["pos_recall"])
        order = np.argsort(faph)
        pts = " ".join(f"{sx(faph[j]):.1f},{sy(fr[j]):.1f}" for j in order)
        paths.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts}"/>')
        legend.append(f'<span style="color:{col}">&#9632;</span>&nbsp;{name}')
    grid = "".join(
        f'<line x1="{sx(g)}" y1="{sy(0)}" x2="{sx(g)}" y2="{sy(1)}" stroke="#8884" '
        f'stroke-dasharray="3 3"/><text x="{sx(g)}" y="{H-P+16}" font-size="11" '
        f'text-anchor="middle" fill="#888">{g}</text>'
        for g in (0, 1, 2, 3, 4, 5)
    )
    yticks = "".join(
        f'<line x1="{sx(0)}" y1="{sy(g)}" x2="{sx(xmax)}" y2="{sy(g)}" stroke="#8884" '
        f'stroke-dasharray="3 3"/><text x="{P-8}" y="{sy(g)+4}" font-size="11" '
        f'text-anchor="end" fill="#888">{int(g*100)}%</text>'
        for g in (0, 0.25, 0.5, 0.75, 1.0)
    )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px">'
        f'{grid}{yticks}{"".join(paths)}'
        f'<text x="{W/2}" y="{H-8}" font-size="12" text-anchor="middle" fill="#aaa">'
        f'false accepts / hour (open-set speech)</text>'
        f'<text x="14" y="{H/2}" font-size="12" text-anchor="middle" fill="#aaa" '
        f'transform="rotate(-90 14 {H/2})">false-reject rate (positives)</text></svg>'
    )
    # tables at report cutoffs
    tabs = []
    for c in results["report_cutoffs"]:
        rows = ""
        for name, m in results["models"].items():
            pr = _at(cutoffs, m["pos_recall"], c); pf = _at(cutoffs, m["partial_fa_rate"], c)
            fa = _at(cutoffs, m["faph"], c)
            rows += (f"<tr><td>{name}</td><td>{pr:.1%}</td>"
                     f"<td>{pf:.1%}</td><td>{fa:.2f}</td></tr>")
        tabs.append(
            f"<h3>cutoff {c:.2f}</h3><table><tr><th>model</th><th>pos recall</th>"
            f"<th>partial FA</th><th>FA/hour</th></tr>{rows}</table>")

    # gapped-phrase recall table (at the first report cutoff)
    slow_html = ""
    gaps = _slow_labels(results)
    if gaps:
        c = results["report_cutoffs"][0]
        head = "".join(f"<th>{g}</th>" for g in gaps)
        rows = ""
        for name, m in results["models"].items():
            cells = "".join(
                (f"<td>{_at(cutoffs, m['slow_recall'][g], c):.0%}</td>"
                 if g in m["slow_recall"] else "<td>-</td>") for g in gaps)
            rows += f"<tr><td>{name}</td>{cells}</tr>"
        slow_html = (
            f"<h2>Gapped-phrase recall @ cutoff {c:.2f}</h2>"
            "<p style='color:#888'>Silence inserted between the two words. Recall "
            "falling toward 0 as the gap grows is the fixed streaming receptive "
            "field — expected to affect both models alike.</p>"
            f"<table><tr><th>model</th>{head}</tr>{rows}</table>")

    return (
        "<style>body{font:14px system-ui;max-width:720px;margin:24px auto;padding:0 16px}"
        "table{border-collapse:collapse;margin:8px 0 20px}"
        "td,th{border:1px solid #8883;padding:4px 12px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}"
        "@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}}</style>"
        "<h1>Wake-word A/B: baseline vs distill</h1>"
        f"<p>{' &nbsp; '.join(legend)}</p>{svg}"
        "<p style='color:#888'>Lower-left is better (few false accepts, few misses). "
        "A model whose curve is inside another's dominates it.</p>"
        + "".join(tabs) + slow_html
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="build the eval clip bank with Piper")
    s.add_argument("--voices_dir", required=True)
    s.add_argument("--out_root", required=True)
    s.add_argument("--phrase", required=True, help='full wake phrase, e.g. "chispa magica"')
    s.add_argument("--w1", required=True, help="first word (for slow/gapped test)")
    s.add_argument("--w2", required=True, help="second word")
    s.add_argument("--per_phrase", type=int, default=30, help="positive clips per voice")
    s.add_argument("--per_partial", type=int, default=10, help="clips per (partial, voice)")
    s.add_argument("--gap_sweep", type=float, nargs="*", default=[0.0, 0.3, 0.5, 0.7, 1.0],
                   help="silence gaps (s) between w1/w2 -> one eval_slow_g<gap>/ dir each")
    s.add_argument("--partials", nargs="*", default=None, help="override partial phrases")
    s.set_defaults(func=cmd_synth)

    r = sub.add_parser("run", help="stream clips through models + report")
    r.add_argument("--model", action="append", required=True, metavar="name=path.tflite")
    r.add_argument("--pos")
    r.add_argument("--partial")
    r.add_argument("--slow", nargs="*", help="one or more eval_slow_g<gap>/ dirs")
    r.add_argument("--neg", nargs="*")
    r.add_argument("--out_dir", default="./ab_eval")
    r.add_argument("--pad_ms", type=int, default=500, help="silence pad to warm PCAN")
    r.add_argument("--refractory_s", type=float, default=2.0, help="post-accept cooldown")
    r.add_argument("--cutoff", type=float, nargs="*", default=[0.5, 0.7, 0.9, 0.95])
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
