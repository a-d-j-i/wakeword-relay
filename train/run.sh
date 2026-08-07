#!/usr/bin/env bash
# Retrain both Spanish wake words and deploy them (SpecAugment is on in train.py).
#   chispa magica -> firmware/models/turn_on.tflite
#   buenas noches -> firmware/models/turn_off.tflite
#
# Config comes from train/.env (see .env.example); a real environment variable
# still overrides a value in .env. Two modes, via DISTILL:
#   DISTILL=0  (default) plain training — one train.py per phrase, as before
#   DISTILL=1  knowledge distillation (DISTILLATION.md Option B): per phrase
#              train a baseline (λ=0, also generates the data), then an
#              openWakeWord teacher on those samples, then the deployed student
#              with soft labels (λ=DISTILL_WEIGHT). The baseline is kept for the
#              A/B comparison; only the distilled model is deployed.
#
# One working dir holds everything (shared dataset cache + per-phrase runs):
#   $WORKDIR/download              datasets (~10GB first run) + teacher backbone
#   $WORKDIR/<phrase>_sa           samples, features, models (+ teacher/ if DISTILL=1)
#
# Usage (activate the env first: source venv/bin/activate):
#   ./run.sh                     reads train/.env
#   ./run.sh --dry-run           preview (passes through to train.py/train_teacher.py)
#   DISTILL=1 ./run.sh           one-off distillation run
#   WORKDIR=/tmp/d SAMPLES=200 STEPS=300 ./run.sh     fast smoke test
set -e
cd "$(dirname "$0")"

# Load train/.env as defaults — only for vars not already set, so an explicit
# environment variable (or `VAR=… ./run.sh`) wins. Simple KEY=VALUE lines;
# lines starting with # and blanks are ignored.
if [ -f .env ]; then
  while IFS='=' read -r k v; do
    [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    v="${v%$'\r'}"                          # tolerate CRLF
    [ -z "${!k+x}" ] && export "$k=$v"
  done < .env
fi

WORKDIR="${WORKDIR:-data}"                   # one working dir; defaults under train/data/
DL="$WORKDIR/download"                        # shared dataset + backbone cache
SAMPLES="${SAMPLES:-5000}"
STEPS="${STEPS:-45000}"
NEG_CLASS_WEIGHT="${NEG_CLASS_WEIGHT:-22}"
DISTILL="${DISTILL:-0}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-0.5}"
LANG_CODE="${LANG_CODE:-es}"                  # language of the speech negatives (train/tools/fetch_speech.py --lang)
SPEECH_NEG_DIR="$DL/${LANG_CODE}_speech_16k"  # real-speech negatives for this language

# Teacher negative audio (DISTILL=1 only). Music + ambient ALONE makes the teacher
# fire on any speech — it never learns that non-wake speech is a negative (observed:
# fires on "mi mamá me mima"). Add SPEECH negatives so it learns to reject it:
#   $DL/piper_negatives_16k       — Piper hard negatives (train/tools/gen_piper_negatives.py):
#                                   confusables/partials, same voices as positives
#   $DL/<lang>_speech_16k         — real speech (train/tools/fetch_speech.py, FLEURS/VoxPopuli):
#                                   generalization beyond the TTS domain
# All are auto-included when present. Override the whole list with
# TEACHER_NEG_DIRS="dirA dirB".
if [ -n "${TEACHER_NEG_DIRS:-}" ]; then
  read -r -a TEACHER_NEG_DIRS <<< "$TEACHER_NEG_DIRS"
else
  TEACHER_NEG_DIRS=("$DL/fma_16k" "$DL/audioset_16k")
  [ -d "$DL/piper_negatives_16k" ] && TEACHER_NEG_DIRS+=("$DL/piper_negatives_16k")
  [ -d "$SPEECH_NEG_DIR" ] && TEACHER_NEG_DIRS+=("$SPEECH_NEG_DIR")
fi

# STUDENT extra negatives: the SPEECH negatives above, spectrogram-ized and added to
# the MixedNet's negatives (train.py --student_neg_audio_dirs). The stock HF negative
# spectrograms are English-only speech, so this patches the target-language gap that
# makes non-English students false-fire on ordinary speech. Music/ambient stay as the
# augmenter's background; only the speech dirs go here. Set STUDENT_NEG=0 to disable.
STUDENT_NEG="${STUDENT_NEG:-1}"
STUDENT_NEG_WEIGHT="${STUDENT_NEG_WEIGHT:-5.0}"
STUDENT_NEG_ARGS=()
if [ "$STUDENT_NEG" = "1" ]; then
  _sneg=()
  [ -d "$DL/piper_negatives_16k" ] && _sneg+=("$DL/piper_negatives_16k")
  [ -d "$SPEECH_NEG_DIR" ] && _sneg+=("$SPEECH_NEG_DIR")
  if [ "${#_sneg[@]}" -gt 0 ]; then
    STUDENT_NEG_ARGS=(--student_neg_audio_dirs "${_sneg[@]}" --student_neg_weight "$STUDENT_NEG_WEIGHT")
  fi
fi

VOICES=(
  es_AR-daniela-high     # AR female, high
  es_ES-carlfm-x_low     # ES male, x_low
  es_ES-davefx-medium    # ES male, medium
  es_ES-sharvard-medium  # ES male, medium
  es_MX-ald-x_low        # MX male, x_low
  es_MX-claude-high      # MX female, high
)
# BROKEN or STRANGE
#  es_ES-mls_9972-low     # ES male, low
#  es_ES-mls_10246-low    # ES female, low
#  es_MX-ald-medium       # MX male, medium

# run_phrase <phrase> <phonetic> <firmware_tflite>
# Pass "" for phonetic to skip --phonetic. Extra args ("$@") flow to train.py.
run_phrase() {
  local phrase="$1" phonetic="$2" deploy="$3"; shift 3
  local d="$WORKDIR/$(echo "$phrase" | tr ' ' '_')_sa"
  local phon=()
  [ -n "$phonetic" ] && phon=(--phonetic "$phonetic")

  echo "══════════════════════════════════════════════════════════════"
  echo "  $phrase  →  $deploy   (workdir: $d, distill: $DISTILL)"
  echo "══════════════════════════════════════════════════════════════"

  if [ "$DISTILL" != "1" ]; then
    python train.py --phrase "$phrase" "${phon[@]}" \
      --piper_model "${VOICES[@]}" --samples "$SAMPLES" --steps "$STEPS" \
      --neg_class_weight "$NEG_CLASS_WEIGHT" \
      --downloads_dir "$DL" --data_dir "$d" \
      "${STUDENT_NEG_ARGS[@]}" \
      --copy-to "$deploy" "$@"
    return
  fi

  echo "── phase 1/3: baseline (λ=0, also generates samples + negatives) ──"
  python train.py --phrase "$phrase" "${phon[@]}" \
    --piper_model "${VOICES[@]}" --samples "$SAMPLES" --steps "$STEPS" \
    --neg_class_weight "$NEG_CLASS_WEIGHT" \
    --downloads_dir "$DL" --data_dir "$d" \
    "${STUDENT_NEG_ARGS[@]}" \
    --output_dir "$d/model_baseline" "$@"

  echo "── phase 2/3: teacher (openWakeWord head) ──"
  echo "   teacher negatives: ${TEACHER_NEG_DIRS[*]}"
  python train_teacher.py \
    --positives_dir "$d/generated_samples" \
    --negatives_dirs "${TEACHER_NEG_DIRS[@]}" \
    --impulse_dirs "$DL/mit_rirs" \
    --output_dir "$d/teacher" "$@"

  echo "── phase 3/3: distilled student (λ=$DISTILL_WEIGHT) → $deploy ──"
  python train.py --phrase "$phrase" "${phon[@]}" \
    --piper_model "${VOICES[@]}" --samples "$SAMPLES" --steps "$STEPS" \
    --neg_class_weight "$NEG_CLASS_WEIGHT" \
    --downloads_dir "$DL" --data_dir "$d" \
    --output_dir "$d/model_distill" \
    --distill --teacher_model "$d/teacher" --distill_weight "$DISTILL_WEIGHT" \
    "${STUDENT_NEG_ARGS[@]}" \
    --regen_features \
    --copy-to "$deploy" "$@"
}

run_phrase "chispa magica" "chispa mágica" ../firmware/models/turn_on.tflite  "$@"
run_phrase "buenas noches" ""              ../firmware/models/turn_off.tflite "$@"

if [ "$DISTILL" = "1" ]; then
  echo
  echo "Done (distilled). For the A/B, load both into the emulator:"
  echo "  baseline: $WORKDIR/<phrase>_sa/model_baseline/.../stream_state_internal_quant.tflite"
  echo "  distill:  $WORKDIR/<phrase>_sa/model_distill/.../stream_state_internal_quant.tflite"
fi
