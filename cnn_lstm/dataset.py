import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from .utils import CharacterEncoder, clean_arabic_text
try:
    import soundfile as sf
except ImportError:
    sf = None
class ASRDataset(Dataset):
    def __init__(self, tsv_path, clips_dir, limit=None, is_train=False):
        df = pd.read_csv(tsv_path, sep="\t")
        df["sentence"] = df["sentence"].apply(clean_arabic_text)
        df = df[df["sentence"].str.len() > 0].reset_index(drop=True)
        if limit is not None:
            df = df.head(limit)
        self.clips_dir = clips_dir
        self.df = df
        self.encoder = CharacterEncoder()
        self.resamplers = {}
        self.use_torchaudio_loader = True
        self.is_train = is_train
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
        if self.is_train:
            self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=30)
            self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        file_path = os.path.join(self.clips_dir, self.df["path"][idx])
        label = self.df["sentence"][idx]
        
        waveform = None
        sample_rate = 16000
        
        try:
            import miniaudio
            decoded = miniaudio.decode_file(
                file_path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
                sample_rate=16000,
            )
            waveform_np = np.frombuffer(decoded.samples, dtype=np.float32).copy()
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
        except Exception:
            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            except Exception:
                return None
                
        if waveform is None:
            return None

        try:
            mel = self.mel_transform(waveform)
            mel_db = self.db_transform(mel)
            spectrogram = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
            if self.is_train:
                spectrogram = self.time_masking(spectrogram)
                spectrogram = self.freq_masking(spectrogram)
        except Exception:
            return None
            
        label_tensor = torch.tensor(self.encoder.encode(label), dtype=torch.long)
        if label_tensor.numel() == 0:
            return None
            
        return spectrogram, label_tensor
def collate_fn(batch):
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    spectrograms, labels = zip(*batch)
    input_lengths = torch.tensor([spec.shape[2] for spec in spectrograms], dtype=torch.long)
    label_lengths = torch.tensor([lbl.size(0) for lbl in labels], dtype=torch.long)
    max_time = input_lengths.max()
    padded_spectrograms = torch.zeros((len(spectrograms), 1, 80, max_time))
    for i, spec in enumerate(spectrograms):
        padded_spectrograms[i, :, :, : spec.shape[2]] = spec
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_spectrograms, padded_labels, input_lengths, label_lengths
