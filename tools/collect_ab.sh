#!/usr/bin/env bash
# Collect the distillation A/B files into one flat directory for loading in the
# emulator (docs/emulator/index.html). Per phrase you get the two students
# (baseline = lambda 0, distill = lambda>0) plus the phrase-specific teacher head;
# the openWakeWord backbone (melspectrogram + embedding_model) is identical across
# phrases and copied once.
#
# Usage:  bash tools/collect_ab.sh [OUT_DIR]   (default: ./ab_models)
# Then in the emulator: "+ Add model" the .tflite files, "Load teacher ONNX" the
# melspectrogram + embedding_model + matching <phrase>_head.onnx.
set -euo pipefail

SRC=${SRC:-/vms2/work_tmp}     # override with SRC=... if your WORKDIR differs
OUT=${1:-./ab_models}
PHRASES=(chispa_magica buenas_noches)

mkdir -p "$OUT"
TF=tflite_stream_state_internal_quant/stream_state_internal_quant.tflite

for p in "${PHRASES[@]}"; do
    d="$SRC/${p}_sa"
    cp "$d/model_baseline/$TF" "$OUT/${p}_baseline.tflite"
    cp "$d/model_distill/$TF"  "$OUT/${p}_distill.tflite"
    cp "$d/teacher/head.onnx"  "$OUT/${p}_head.onnx"     # phrase-specific
done

# Shared openWakeWord backbone — identical across phrases, copy once.
cp "$SRC/${PHRASES[0]}_sa/teacher/melspectrogram.onnx"  "$OUT/melspectrogram.onnx"
cp "$SRC/${PHRASES[0]}_sa/teacher/embedding_model.onnx" "$OUT/embedding_model.onnx"

# Normalize perms so the copies are yours to read/move regardless of source modes.
chmod -R u+rwX "$OUT"

echo "Copied to $OUT:"
ls -la "$OUT"
