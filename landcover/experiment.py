"""Train and compare the three models; write results/ and export the best model.

Usage: python -m landcover.experiment [--fast] [--export models/landcover.pt]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .data import ROOT, load_eurosat, make_loaders
from .models import build, count_parameters
from .train import evaluate, fit, latency_ms

# name, image size, epochs, learning rate
PLAN = [
    ("cnn-scratch", 64, 15, 2e-3),
    ("mobilenet-frozen", 128, 5, 2e-3),
    ("mobilenet-finetune", 128, 8, 1e-3),
]
LABELS = {
    "cnn-scratch": "Small CNN from scratch",
    "mobilenet-frozen": "MobileNetV3-Small, frozen backbone (linear probe)",
    "mobilenet-finetune": "MobileNetV3-Small, fine-tuned",
}


def plot_confusion(matrix, classes, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=8,
                        color="white" if v > max(map(max, matrix)) / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def to_markdown(rows: list[dict]) -> str:
    lines = ["| Model | Input | Trainable params | Test accuracy | Macro-F1 | ms / image (CPU) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['size']} px | {r['trainable_params']:,} | {r['accuracy']:.1%} | "
                     f"{r['macro_f1']:.3f} | {r['ms_per_image']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="1 epoch each (smoke test)")
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 2)
    base = load_eurosat()
    classes = list(base.classes)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, models = [], {}
    for name, size, epochs, lr in PLAN:
        train_loader, val_loader, test_loader = make_loaders(base, size=size)
        model = build(name, num_classes=len(classes))
        info = fit(model, train_loader, val_loader, epochs=1 if args.fast else epochs, lr=lr)
        metrics = evaluate(model, test_loader, classes)
        total, trainable = count_parameters(model)
        row = {"model": name, "label": LABELS[name], "size": size, "params": total, "trainable_params": trainable,
               "ms_per_image": latency_ms(model, size), **info, **metrics}
        rows.append(row)
        models[name] = model
        print(json.dumps({k: v for k, v in row.items() if k not in ("history", "confusion_matrix")}))

    best = max(rows, key=lambda r: r["accuracy"])
    plot_confusion(best["confusion_matrix"], classes, args.out / "confusion_matrix.png",
                   f"{best['label']}: test confusion matrix")
    report = {"dataset": "EuroSAT RGB", "images": len(base), "classes": classes, "split": "70/15/15 stratified",
              "best_model": best["model"], "results": rows}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "metrics.md").write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best["model"], "size": best["size"], "classes": classes,
                    "state_dict": models[best["model"]].state_dict()}, args.export)
        save_examples(base, args.out / "examples")


def save_examples(base, folder: Path) -> None:
    """One test-set tile per class, upscaled for display, used as demo examples."""
    from .data import stratified_split

    _, _, test_idx = stratified_split(list(base.targets))
    folder.mkdir(parents=True, exist_ok=True)
    seen = set()
    for i in test_idx:
        image, label = base[i]
        name = base.classes[label]
        if name not in seen:
            image.resize((256, 256)).save(folder / f"{name}.png")
            seen.add(name)


if __name__ == "__main__":
    main()
