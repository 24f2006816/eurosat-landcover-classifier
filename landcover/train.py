"""Training loop with validation-based checkpointing, and evaluation on the held-out test set."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn


def run_epoch(model, loader, optimizer=None) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1 if train else 0.0)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, epochs: int, lr: float, log=print) -> dict:
    """AdamW + cosine schedule; keeps the weights from the epoch with the best validation accuracy."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc, best_state, history = -1.0, None, []
    start = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                        "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)})
        log(f"epoch {epoch + 1}/{epochs} train {train_acc:.3f} val {val_acc:.3f} ({time.time() - start:.0f}s)")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"best_val_acc": round(best_acc, 4), "train_seconds": round(time.time() - start), "history": history}


@torch.no_grad()
def evaluate(model, loader, class_names: list[str]) -> dict:
    model.eval()
    preds, labels = [], []
    for images, y in loader:
        preds.append(model(images).argmax(1).numpy())
        labels.append(y.numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    per_class = f1_score(labels, preds, average=None, labels=range(len(class_names)))
    return {
        "accuracy": round(float((preds == labels).mean()), 4),
        "macro_f1": round(float(per_class.mean()), 4),
        "per_class_f1": {c: round(float(f), 4) for c, f in zip(class_names, per_class, strict=True)},
        "confusion_matrix": confusion_matrix(labels, preds, labels=range(len(class_names))).tolist(),
    }


@torch.no_grad()
def latency_ms(model, size: int, runs: int = 50) -> float:
    model.eval()
    x = torch.randn(1, 3, size, size)
    for _ in range(5):
        model(x)
    start = time.perf_counter()
    for _ in range(runs):
        model(x)
    return round((time.perf_counter() - start) * 1000 / runs, 2)
