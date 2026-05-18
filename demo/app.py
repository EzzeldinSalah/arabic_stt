import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import whisper
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from demo.db import init_db, save_record, search_records, get_recent_records

try:
    import miniaudio
    HAS_MINIAUDIO = True
except ImportError:
    HAS_MINIAUDIO = False
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

STATIC_DIR = BASE_DIR / "static"
RESULTS_PATH = BASE_DIR.parent / "results" / "metrics.json"
MODEL_NAME = os.getenv("WHISPER_MODEL", "large")
DEVICE = os.getenv("WHISPER_DEVICE")
if not DEVICE:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Initialize DB
init_db()

# Load Whisper
print(f"Loading Whisper model '{MODEL_NAME}' on {DEVICE}...")
whisper_model = whisper.load_model(MODEL_NAME, device=DEVICE)

# Load Arabic Summarizer (T5)
print(f"Loading Arabic Summarizer (T5) on {DEVICE}...")
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    sum_tokenizer = AutoTokenizer.from_pretrained("malmarjeh/t5-arabic-text-summarization")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("malmarjeh/t5-arabic-text-summarization").to(DEVICE)
    summarizer_loaded = True
except Exception as e:
    print(f"Warning: Failed to load T5 summarizer: {e}")
    summarizer_loaded = False

# Load Mamba (English Storytelling)
print(f"Loading Mamba (English) on {DEVICE}...")
try:
    from transformers import pipeline
    mamba_summarizer = pipeline("text-generation", model="state-spaces/mamba-130m-hf", device=DEVICE)
except Exception as e:
    print(f"Warning: Failed to load Mamba summarizer: {e}")
    mamba_summarizer = None


def _load_audio_numpy(audio_path, target_sr=16000):
    if HAS_MINIAUDIO:
        try:
            decoded = miniaudio.decode_file(
                audio_path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
                sample_rate=target_sr,
            )
            return np.frombuffer(decoded.samples, dtype=np.float32).copy(), None
        except Exception as exc:
            return None, f"miniaudio failed: {exc}"
    if HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=target_sr
                )
            return waveform.squeeze(0).numpy().astype(np.float32), None
        except Exception as exc:
            return None, f"torchaudio failed: {exc}"
    return None, "No audio backend available (install miniaudio or torchaudio)."


app = FastAPI(title="Arabic ASR Demo", version="1.0")
app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/metrics")
def metrics():
    if not RESULTS_PATH.exists():
        return JSONResponse(
            {"status": "missing", "message": "Run evaluation first."},
            status_code=404,
        )
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "message": str(exc)}, status_code=500
        )


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), mamba_analysis: str = Form("false")):
    if not file.filename:
        return JSONResponse({"error": "No file provided."}, status_code=400)
    suffix = Path(file.filename).suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        audio_np, error = _load_audio_numpy(tmp_path)
        if audio_np is None:
            return JSONResponse(
                {"error": f"Audio load failed: {error}"}, status_code=400
            )

        # 1. Transcribe Arabic
        result = whisper_model.transcribe(
            audio_np,
            fp16=False,
            language="ar",
            task="transcribe",
            without_timestamps=True,
        )
        whisper_text = result["text"].strip()

        # 2. Arabic T5 Summary
        summary_text = ""
        if summarizer_loaded and whisper_text:
            try:
                input_length = len(whisper_text.split())
                max_len = min(128, max(10, input_length))
                inputs = sum_tokenizer(whisper_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
                outputs = sum_model.generate(**inputs, max_length=max_len, min_length=5, do_sample=False)
                summary_text = sum_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"Summarization error: {e}")
                summary_text = "Failed to generate summary."

        # 3. Mamba Storytelling (optional)
        mamba_story = ""
        is_mamba_requested = (mamba_analysis.lower() == "true")
        if is_mamba_requested:
            try:
                translation_result = whisper_model.transcribe(
                    audio_np,
                    fp16=False,
                    task="translate",
                    without_timestamps=True,
                )
                english_text = translation_result["text"].strip()
            except Exception as e:
                print(f"Translation error: {e}")
                english_text = ""

            if mamba_summarizer and english_text:
                try:
                    prompt = f"Once upon a time, {english_text.lower()} "
                    gen = mamba_summarizer(
                        prompt,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.6,
                        top_k=50,
                        repetition_penalty=1.4,
                    )
                    mamba_story = gen[0]["generated_text"].strip()
                except Exception as e:
                    print(f"Mamba storytelling error: {e}")
                    mamba_story = "Failed to generate story."

        # Save to DB
        save_record(whisper_text, summary_text, mamba_story)

        return {
            "text": whisper_text,
            "summary": summary_text,
            "mamba_story": mamba_story,
            "model": f"Whisper ({MODEL_NAME}) + Arabic T5" + (" + Mamba" if is_mamba_requested else "")
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/search")
async def search_history(q: str = ""):
    if q.strip():
        results = search_records(q)
    else:
        results = get_recent_records()
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("demo.app:app", host="0.0.0.0", port=8000, reload=True)
