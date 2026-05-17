import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent))

from cnn_lstm.model import CNNLSTM, LegacyASRModel
from cnn_lstm.utils import CharacterEncoder, greedy_decoder
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
BASE_DIR = Path(__file__).resolve().parent
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
_model = None
_cnn_model = None
_char_encoder = None
_mel_transform = None
_db_transform = None

def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_NAME, device=DEVICE)
    return _model

def _get_cnn_model():
    global _cnn_model, _char_encoder, _mel_transform, _db_transform
    if _cnn_model is None:
        _char_encoder = CharacterEncoder()
        model_path = BASE_DIR.parent / "saved_models" / "asr_model.pth"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=DEVICE)
            if "cnn.0.bias" in state_dict:
                _cnn_model = LegacyASRModel(input_dim=80, num_classes=_char_encoder.vocab_size).to(DEVICE)
            else:
                _cnn_model = CNNLSTM(input_dim=80, num_classes=_char_encoder.vocab_size).to(DEVICE)
            _cnn_model.load_state_dict(state_dict)
        else:
            _cnn_model = CNNLSTM(input_dim=80, num_classes=_char_encoder.vocab_size).to(DEVICE)
        _cnn_model.eval()
        _mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
        _db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
    return _cnn_model, _char_encoder, _mel_transform, _db_transform

def transcribe_cnn(audio_np):
    model, encoder, mel_trans, db_trans = _get_cnn_model()
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    mel = mel_trans(waveform)
    mel_db = db_trans(mel)
    spectrogram = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    spectrogram = spectrogram.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(spectrogram)
        preds = greedy_decoder(outputs, encoder)
    return preds[0] if preds else ""

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
            {
                "status": "missing",
                "message": "Run python whisper_baseline/whisper_eval.py",
            },
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
async def transcribe(file: UploadFile = File(...)):
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
        result = _get_model().transcribe(
            audio_np,
            language="ar",
            task="transcribe",
            fp16=False,
            without_timestamps=True,
        )
        whisper_text = result["text"].strip()
        return {"text": whisper_text, "model": f"Whisper ({MODEL_NAME})"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
