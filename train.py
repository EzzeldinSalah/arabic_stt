import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import ASRDataset, CharacterEncoder, collate_fn
from model import ASRModel


def compute_ctc_loss(outputs, labels, pooled_lengths, lbl_lens, criterion, device):
    outputs = outputs.permute(1, 0, 2)
    if device.type == 'mps':
        return criterion(
            outputs.to('cpu'),
            labels.to('cpu'),
            pooled_lengths.to('cpu'),
            lbl_lens.to('cpu'),
        )
    return criterion(outputs, labels, pooled_lengths, lbl_lens)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    clips_dir = os.path.join(args.data_dir, 'clips')
    train_tsv = os.path.join(args.data_dir, 'train.tsv')
    dev_tsv = os.path.join(args.data_dir, 'dev.tsv')

    char_encoder = CharacterEncoder()
    limit = getattr(args, 'limit', None)
    train_data = ASRDataset(train_tsv, clips_dir, limit=limit)

    dev_limit = getattr(args, 'dev_limit', (limit // 5) if limit else None)
    dev_data = ASRDataset(dev_tsv, clips_dir, limit=dev_limit)

    workers = max(0, int(getattr(args, 'num_workers', min(4, os.cpu_count() or 1))))
    loader_kwargs = {}
    if workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=workers,
        **loader_kwargs,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=workers,
        **loader_kwargs,
    )

    print("Building Neural Network...")
    model = ASRModel(input_dim=80, num_classes=char_encoder.vocab_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, 'asr_model.pth')

    best_val_loss = float('inf')
    patience_triggered = 0
    val_every = max(1, int(getattr(args, 'val_every', 1)))
    log_every = max(1, int(getattr(args, 'log_every', max(1, len(train_loader) // 2))))

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            specs, labels, in_lens, lbl_lens = batch

            pooled_lengths = in_lens // 4
            valid_mask = pooled_lengths >= lbl_lens
            if not torch.any(valid_mask):
                continue

            specs = specs[valid_mask].to(device)
            labels = labels[valid_mask].to(device)
            pooled_lengths = pooled_lengths[valid_mask]
            lbl_lens = lbl_lens[valid_mask]

            optimizer.zero_grad()

            outputs = model(specs)
            loss = compute_ctc_loss(outputs, labels, pooled_lengths, lbl_lens, criterion, device)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()
            total_train_loss += loss.item()
            train_steps += 1

            if batch_idx % log_every == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        if train_steps == 0:
            break

        avg_train_loss = total_train_loss / train_steps
        if ((epoch + 1) % val_every) != 0 and (epoch + 1) != args.epochs:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f}")
            continue

        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in dev_loader:
                if batch is None:
                    continue

                specs, labels, in_lens, lbl_lens = batch

                pooled_lengths = in_lens // 4
                valid_mask = pooled_lengths >= lbl_lens
                if not torch.any(valid_mask):
                    continue

                specs = specs[valid_mask].to(device)
                labels = labels[valid_mask].to(device)
                pooled_lengths = pooled_lengths[valid_mask]
                lbl_lens = lbl_lens[valid_mask]

                outputs = model(specs)
                loss = compute_ctc_loss(outputs, labels, pooled_lengths, lbl_lens, criterion, device)

                total_val_loss += loss.item()
                val_steps += 1

        if val_steps == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f}")
            continue

        avg_val_loss = total_val_loss / val_steps
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_triggered = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_triggered += 1
            if patience_triggered > 10:
                break
