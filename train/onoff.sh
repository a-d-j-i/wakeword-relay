#!/usr/bin/env bash
# Retrain both Spanish wake words and deploy them (SpecAugment is on in train.py).
#   chispa magica -> firmware/models/turn_on.tflite
#   buenas noches -> firmware/models/turn_off.tflite
#
# Pass ONE working dir; the dataset cache and per-phrase runs are derived under it:
#   $WORKDIR/download            (datasets, ~10GB on first run)
#   $WORKDIR/chispa_magica_sa    (chispa run)
#   $WORKDIR/buenas_noches_sa    (buenas run)
# Override on your machine, e.g.:  WORKDIR=/path/to/cache ./onoff.sh
# Preview first:  ./onoff.sh --dry-run     (passes through to both train.py calls)
# Real run:       ./onoff.sh
# (activate the training env first:  source venv/bin/activate)
set -e
cd "$(dirname "$0")"

WORKDIR="${WORKDIR:-data}"        # one working dir; defaults under train/data/
DL="$WORKDIR/download"            # shared dataset cache
VOICES=(
  es_AR-daniela-high     # AR female, high
  es_MX-ald-medium       # MX male, medium
  es_MX-ald-x_low        # MX male, x_low
  es_MX-claude-high      # MX female, high
  es_ES-carlfm-x_low     # ES male, x_low
  es_ES-davefx-medium    # ES male, medium
  es_ES-sharvard-medium  # ES male, medium
  es_ES-mls_10246-low    # ES female, low
  es_ES-mls_9972-low     # ES male, low
)

python train.py --phrase "chispa magica" --phonetic "chispa mágica" \
  --piper_model "${VOICES[@]}" --samples 5000 --steps 45000 --neg_class_weight 15 \
  --downloads_dir "$DL" --data_dir "$WORKDIR/chispa_magica_sa" \
  --copy-to ../firmware/models/turn_on.tflite "$@"

python train.py --phrase "buenas noches" \
  --piper_model "${VOICES[@]}" --samples 5000 --steps 45000 --neg_class_weight 15 \
  --downloads_dir "$DL" --data_dir "$WORKDIR/buenas_noches_sa" \
  --copy-to ../firmware/models/turn_off.tflite "$@"
