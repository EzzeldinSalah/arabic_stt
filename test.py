import os
import torch
from torch.utils.data import DataLoader
from preprocessing import ASRDataset, CharacterEncoder, collate_fn
from model import ASRModel


def calculate_wer(reference, hypothesis):
    """Calculate WER"""
    try:
        import jiwer
        return jiwer.wer(reference, hypothesis)
    except ImportError:
        from difflib import SequenceMatcher
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        sm = SequenceMatcher(None, ref_words, hyp_words)
        matches = sum(trip.size for trip in sm.get_matching_blocks())
        errors = max(len(ref_words), len(hyp_words)) - matches
        return errors / max(len(ref_words), 1)


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

        decoded_text = char_encoder.decode(decoded_path)
        decoded_strings.append(decoded_text)

    return decoded_strings


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    clips_dir = os.path.join(args.data_dir, 'clips')
    test_tsv = os.path.join(args.data_dir, 'test.tsv')
    char_encoder = CharacterEncoder()

    test_data = ASRDataset(test_tsv, clips_dir)
    workers = max(0, int(getattr(args, 'num_workers', min(4, os.cpu_count() or 1))))
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=workers,
        **loader_kwargs,
    )

    model = ASRModel(input_dim=80, num_classes=char_encoder.vocab_size).to(device)
    model_path = os.path.join(args.save_dir, 'asr_model.pth')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Model not found at {model_path}")
        return

    all_preds, all_truths = [], []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            specs, labels, in_lens, lbl_lens = batch
            specs = specs.to(device)

            outputs = model(specs)
            preds = greedy_decoder(outputs, char_encoder)
            all_preds.extend(preds)

            for label, length in zip(labels, lbl_lens):
                truth_text = char_encoder.decode(label[:length].tolist())
                all_truths.append(truth_text)

    total_wer = sum(calculate_wer(t, p) for t, p in zip(all_truths, all_preds) if t)
    avg_wer = total_wer / max(len(all_truths), 1)

    print(f"WER: {avg_wer:.4f}")
