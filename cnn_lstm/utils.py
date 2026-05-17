import re
import torch
def clean_arabic_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[يى]", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ـ", "", text)
    text = re.sub(r"[^\u0621-\u064A\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
def get_vocabulary():
    arabic_chars = [chr(i) for i in range(0x0621, 0x064A + 1)]
    valid_chars = sorted(list(set(clean_arabic_text("".join(arabic_chars)))))
    return ["<BLANK>"] + valid_chars + [" "]
class CharacterEncoder:
    def __init__(self):
        self.vocab = get_vocabulary()
        self.char_to_num = {char: idx for idx, char in enumerate(self.vocab)}
        self.num_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.blank_id = 0
    def encode(self, text):
        return [self.char_to_num[c] for c in text if c in self.char_to_num]
    def decode(self, indices):
        return "".join(self.num_to_char.get(idx, "") for idx in indices if idx != self.blank_id)
def greedy_decoder(outputs, char_encoder):
    best_paths = torch.argmax(outputs, dim=2)
    decoded_strings = []
    for sequence in best_paths:
        decoded_path = []
        previous_char_id = -1
        for char_tensor in sequence:
            char_id = char_tensor.item()
            if char_id != char_encoder.blank_id and char_id != previous_char_id:
                decoded_path.append(char_id)
            previous_char_id = char_id
        decoded_strings.append(char_encoder.decode(decoded_path))
    return decoded_strings
def calculate_wer(reference, hypothesis):
    import jiwer
    if not reference:
        return 1.0 if hypothesis else 0.0
    if not hypothesis:
        return 1.0
    return jiwer.wer(reference, hypothesis)
def calculate_cer(reference, hypothesis):
    import jiwer
    if not reference:
        return 1.0 if hypothesis else 0.0
    if not hypothesis:
        return 1.0
    return jiwer.cer(reference, hypothesis)
