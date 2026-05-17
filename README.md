# Arabic Audio Understanding and Retrieval System

A compact Arabic ASR project with two approaches:

- CNN-LSTM (trainable model)
- Whisper (pretrained baseline)

## Structure

- cnn_lstm/ - training, evaluation, inference
- whisper_baseline/ - Whisper evaluation + single-file inference
- demo/ - landing-page demo + minimal API
- report/ - results summary
- data/ - Common Voice Arabic splits
- results/ - metrics.json and per-sample outputs

## Quick start (local)

### 1) Train CNN-LSTM

```bash
pip install -r cnn_lstm/requirements.txt
python -m cnn_lstm.train --data-dir data/raw --epochs 20 --batch-size 32 --limit 10000
```

### 2) Evaluate Whisper + CNN-LSTM

```bash
pip install -r whisper_baseline/requirements.txt
python whisper_baseline/whisper_eval.py
```

This generates results/metrics.json for the demo.

### 3) Run the demo

```bash
pip install -r demo/requirements.txt
python demo/app.py
```

Open http://localhost:7860

## Environment overrides

- WHISPER_MODEL: default "medium"
- WHISPER_DEVICE: "cuda" | "mps" | "cpu"
- PORT: demo server port (default 7860)

## Notes

Whisper is a baseline (no training). The CNN-LSTM is the trainable model.
