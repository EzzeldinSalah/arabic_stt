# Arabic Audio Understanding and Retrieval System

An intelligent Arabic speech-to-text pipeline with AI summarization and search.

**Pipeline:** Arabic Audio → Whisper Transcription → T5 Arabic Summary → Mamba Storytelling (English)

## Features

- **Whisper (Large)** Arabic speech recognition
- **Arabic T5** text summarization (`malmarjeh/t5-arabic-text-summarization`)
- **Mamba SSM** English storytelling via `state-spaces/mamba-130m-hf`
- **Browser recording** with in-browser WebM→WAV conversion
- **Search & history** powered by local SQLite

## Structure

```
cnn_lstm/        CNN-LSTM model architecture, training, and evaluation
whisper_baseline/  Whisper evaluation notebooks
demo/            Web app (FastAPI + vanilla JS)
results/         Evaluation metrics
data/            Common Voice Arabic splits (gitignored)
```

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r demo/requirements.txt
```

### 2. Run the demo

```bash
python3 demo/app.py
```

Open **http://localhost:8000**

### 3. Train CNN-LSTM (optional)

```bash
pip install -r cnn_lstm/requirements.txt
python3 main.py
```

### 4. Evaluate models (optional)

```bash
pip install -r whisper_baseline/requirements.txt
python3 main.py
```

## Environment Variables

| Variable           | Default     | Description                   |
| ------------------ | ----------- | ----------------------------- |
| `WHISPER_MODEL`  | `large`   | Whisper model size            |
| `WHISPER_DEVICE` | auto-detect | `cuda`, `mps`, or `cpu` |

## Notes

- The Whisper `large` model requires ~5GB RAM. Use `medium` or `small` if constrained.
- Mamba storytelling translates Arabic→English via Whisper, then retells it creatively.
- The T5 summarizer generates concise Arabic summaries of transcriptions.
