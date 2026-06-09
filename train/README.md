# train/

CLI pipeline for training custom INT8 TFLite wake word models for ESP32-S3.

```bash
python3.10 -m venv .venv-mww && source .venv-mww/bin/activate
pip install -r requirements.txt
python train.py --phrase "hey jarvis"
```

Output: `data/trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite`

See the [project README](../README.md) for full documentation and hyperparameter reference, and [`../firmware/train.md`](../firmware/train.md) for detailed training guides (Colab, CUDA GPU, CPU-only).