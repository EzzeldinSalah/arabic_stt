import os
import json
import csv
import time
from datetime import datetime, timezone
from typing import Optional
import pandas as pd
import numpy as np
import torch
import whisper
from torch.utils.data import DataLoader
from cnn_lstm.dataset import ASRDataset, collate_fn
from cnn_lstm.utils import CharacterEncoder, calculate_cer, calculate_wer, clean_arabic_text, greedy_decoder
from cnn_lstm.model import CNNLSTM
from cnn_lstm.train import train as train_cnn_model
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

ACTION = "train_cnn"
MODELS_TO_EVALUATE = "cnn_lstm"
WHISPER_MODEL_SIZE = "small"
EVALUATE_LIMIT = None
DATA_DIR = "data/raw"
SAVE_DIR = "saved_models"
RESULTS_DIR = "results"
AUDIO_FILE_TO_TRANSCRIBE = "data/raw/clips/example.mp3"

def _load_audio_numpy(audio_path, target_sr=16000) -> Optional[np.ndarray]:
    if HAS_MINIAUDIO:
        try:
            decoded = miniaudio.decode_file(
                audio_path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
                sample_rate=target_sr,
            )
            return np.frombuffer(decoded.samples, dtype=np.float32).copy()
        except Exception:
            pass
    if HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=target_sr
                )
            return waveform.squeeze(0).numpy().astype(np.float32)
        except Exception:
            pass
    print(f"Could not decode audio file: '{audio_path}'")
    return None

def transcribe_audio_whisper(whisper_model, audio_path):
    audio_np = _load_audio_numpy(audio_path)
    if audio_np is None:
        return None
    try:
        result = whisper_model.transcribe(
            audio_np,
            language="ar",
            task="transcribe",
            fp16=False,
            beam_size=5,
            without_timestamps=True,
        )
        return clean_arabic_text(result["text"])
    except Exception as e:
        print(f"Whisper Failed to transcribe {audio_path}: {e}")
        return ""

def evaluate_cnn_lstm(device):
    print("[CNN-LSTM] Starting evaluation...")
    clips_dir = os.path.join(DATA_DIR, "clips")
    test_tsv = os.path.join(DATA_DIR, "test.tsv")
    char_encoder = CharacterEncoder()
    test_dataset = ASRDataset(test_tsv, clips_dir, limit=EVALUATE_LIMIT)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    model_path = os.path.join(SAVE_DIR, "asr_model.pth")
    if not os.path.exists(model_path):
        print(f"[CNN-LSTM] Model checkpoint not found at '{model_path}'")
        return None, None
    model = CNNLSTM(input_dim=80, num_classes=char_encoder.vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    from tqdm import tqdm
    all_preds, all_truths = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating CNN-LSTM"):
            if batch is None:
                continue
            specs, labels, _, lbl_lens = batch
            specs = specs.to(device)
            outputs = model(specs)
            preds = greedy_decoder(outputs, char_encoder)
            all_preds.extend(preds)
            for lbl, length in zip(labels, lbl_lens):
                all_truths.append(char_encoder.decode(lbl[:length].tolist()))
    all_truths = [clean_arabic_text(t) for t in all_truths]
    print("\n--- CNN-LSTM Predictions Sample ---")
    for i in range(min(5, len(all_truths))):
        print(f"GT: {all_truths[i]}")
        print(f"PR: {all_preds[i]}")
        print("-" * 30)
    wers = [calculate_wer(t, p) for t, p in zip(all_truths, all_preds)]
    cers = [calculate_cer(t, p) for t, p in zip(all_truths, all_preds)]
    avg_wer = sum(wers) / max(len(wers), 1)
    avg_cer = sum(cers) / max(len(cers), 1)
    print(f"CNN-LSTM Average WER: {avg_wer:.4f}")
    print(f"CNN-LSTM Average CER: {avg_cer:.4f}\n")
    return avg_wer, avg_cer

def evaluate_whisper(device):
    print(f"[Whisper] Loading '{WHISPER_MODEL_SIZE}' model...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
    clips_dir = os.path.join(DATA_DIR, "clips")
    test_tsv = os.path.join(DATA_DIR, "test.tsv")
    df = pd.read_csv(test_tsv, sep="\t")
    df["sentence"] = df["sentence"].apply(clean_arabic_text)
    df = df[df["sentence"].str.len() > 0].reset_index(drop=True)
    if EVALUATE_LIMIT is not None:
        df = df.head(int(EVALUATE_LIMIT))
    n_samples = len(df)
    print(f"Loaded {n_samples} testing samples.")
    wer_per_sample, cer_per_sample = [], []
    references, hypotheses, filenames = [], [], []
    start = time.time()
    from tqdm import tqdm
    for i, row in tqdm(df.iterrows(), total=n_samples, desc=f"Evaluating Whisper ({WHISPER_MODEL_SIZE})"):
        audio_path = os.path.join(clips_dir, row["path"])
        reference = row["sentence"]
        if not os.path.exists(audio_path):
            continue
        hyp = transcribe_audio_whisper(whisper_model, audio_path)
        if hyp is None:
            continue
        wer = calculate_wer(reference, hyp)
        cer = calculate_cer(reference, hyp)
        references.append(reference)
        hypotheses.append(hyp)
        filenames.append(row["path"])
        wer_per_sample.append(wer)
        cer_per_sample.append(cer)
    print("\n--- Whisper Predictions Sample ---")
    for i in range(min(5, len(references))):
        print(f"GT: {references[i]}")
        print(f"PR: {hypotheses[i]}")
        print("-" * 30)
    whisper_wer = sum(wer_per_sample) / max(len(wer_per_sample), 1)
    whisper_cer = sum(cer_per_sample) / max(len(cer_per_sample), 1)
    print(f"Whisper ({WHISPER_MODEL_SIZE}) Average WER: {whisper_wer:.4f}")
    print(f"Whisper ({WHISPER_MODEL_SIZE}) Average CER: {whisper_cer:.4f}\n")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "whisper_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "reference", "whisper_hypothesis", "wer"])
        for fname, ref, hyp, wer in zip(filenames, references, hypotheses, wer_per_sample):
            writer.writerow([fname, ref, hyp, f"{wer:.4f}"])
    return whisper_wer, whisper_cer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware selected: {device}")
    if ACTION == "train_cnn":
        print("Starting CNN-LSTM Training...")
        class TrainArgs:
            data_dir = DATA_DIR
            save_dir = SAVE_DIR
            batch_size = 32
            num_workers = 2
            epochs = 20
            lr = 1e-3
            limit = None
            dev_limit = None
            val_every = 2
            log_every = None
        train_cnn_model(TrainArgs())
        print("Training Complete!")
    elif ACTION == "transcribe":
        if not os.path.exists(AUDIO_FILE_TO_TRANSCRIBE):
            print(f"Error: Could not find audio file at {AUDIO_FILE_TO_TRANSCRIBE}")
            return
        print(f"Transcribing {AUDIO_FILE_TO_TRANSCRIBE} with Whisper...")
        w_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        text = transcribe_audio_whisper(w_model, AUDIO_FILE_TO_TRANSCRIBE)
        print(f"\nTranscript: {text}\n")
    elif ACTION == "evaluate":
        metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except json.JSONDecodeError:
                pass
        
        metrics["generated_at"] = datetime.now(timezone.utc).isoformat()
        
        if MODELS_TO_EVALUATE in ["whisper", "both"]:
            w_wer, w_cer = evaluate_whisper(device)
            metrics["whisper"] = {"model": WHISPER_MODEL_SIZE, "wer": w_wer, "cer": w_cer}
        if MODELS_TO_EVALUATE in ["cnn_lstm", "both"]:
            c_wer, c_cer = evaluate_cnn_lstm(device)
            if c_wer is not None:
                metrics["cnn_lstm"] = {"wer": c_wer, "cer": c_cer}
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation results saved to '{metrics_path}'")
    else:
        print("Invalid ACTION configured.")

if __name__ == "__main__":
    main()
