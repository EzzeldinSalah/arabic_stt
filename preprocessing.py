import os
import re
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    sf = None

def clean_arabic_text(text):
    """Clean Arabic text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'[يى]', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^ا-ي\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_vocabulary():
    """Create vocabulary with blank at index 0."""
    arabic_chars = [chr(i) for i in range(0x0621, 0x064A + 1)]
    valid_chars = sorted(list(set(clean_arabic_text("".join(arabic_chars)))))
    return ['<BLANK>'] + valid_chars + [' ']

class CharacterEncoder:
    """Encode and decode labels."""
    def __init__(self):
        self.vocab = get_vocabulary()
        self.char_to_num = {char: idx for idx, char in enumerate(self.vocab)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.blank_id = 0

    def encode(self, text):
        return [self.char_to_num[c] for c in text if c in self.char_to_num]

    def decode(self, indices):
        return ''.join([self.num_to_char.get(idx, '') for idx in indices if idx != self.blank_id])

class ASRDataset(Dataset):
    """Load ASR samples."""
    def __init__(self, tsv_path, clips_dir, limit=None):
        df = pd.read_csv(tsv_path, sep='\t')
        df['sentence'] = df['sentence'].apply(clean_arabic_text)
        df = df[df['sentence'].str.len() > 0].reset_index(drop=True)

        if limit is not None:
            df = df.head(limit)

        self.clips_dir = clips_dir
        self.df = df
        self.encoder = CharacterEncoder()
        self.resamplers = {}
        self.use_torchaudio_loader = True

        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = os.path.join(self.clips_dir, self.df['path'][idx])
        label = self.df['sentence'][idx]

        if self.use_torchaudio_loader:
            try:
                waveform, sample_rate = torchaudio.load(file_path)
            except Exception:
                self.use_torchaudio_loader = False

        if not self.use_torchaudio_loader:
            if sf is None:
                return None
            try:
                waveform_np, sample_rate = sf.read(file_path, always_2d=True, dtype='float32')
                waveform = torch.from_numpy(waveform_np.T).contiguous()
            except Exception:
                return None

        try:
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                if sample_rate not in self.resamplers:
                    self.resamplers[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000,
                    )
                waveform = self.resamplers[sample_rate](waveform)

            mel = self.mel_transform(waveform)
            mel_db = self.db_transform(mel)
            spectrogram = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        except Exception:
            return None
            
        label_tensor = torch.tensor(self.encoder.encode(label), dtype=torch.long)
        if label_tensor.numel() == 0:
            return None
        return spectrogram, label_tensor

def collate_fn(batch):
    """Pad variable-length batch."""
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None

    spectrograms, labels = zip(*batch)

    input_lengths = torch.tensor([spec.shape[2] for spec in spectrograms], dtype=torch.long)
    label_lengths = torch.tensor([lbl.size(0) for lbl in labels], dtype=torch.long)

    max_time = input_lengths.max()
    padded_spectrograms = torch.zeros((len(spectrograms), 1, 80, max_time))
    for i, spec in enumerate(spectrograms):
        padded_spectrograms[i, :, :, :spec.shape[2]] = spec

    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_spectrograms, padded_labels, input_lengths, label_lengths
