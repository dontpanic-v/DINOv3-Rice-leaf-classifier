"""Shared per-epoch train/validate batch loop."""
import torch


def _run_batches(model, loader, criterion, device, use_amp, optimizer=None):
    running_loss = 0.0
    correct_count = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        total_count += labels.size(0)

    return running_loss / total_count, correct_count / total_count


# bf16 autocast: backbone was trained in bf16 (paper Section C).
# GradScaler not needed for bf16, only for fp16.
# Gradient clipping at 1.0 prevents spikes during fine-tuning.
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=True):
    model.train()
    return _run_batches(model, loader, criterion, device, use_amp, optimizer=optimizer)


def validate(model, loader, criterion, device, use_amp=True):
    model.eval()
    with torch.no_grad():
        return _run_batches(model, loader, criterion, device, use_amp, optimizer=None)
