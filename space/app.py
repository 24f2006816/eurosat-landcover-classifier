"""Hugging Face Space for eurosat-landcover-classifier (free CPU tier)."""

import io
import urllib.request
import zipfile
from pathlib import Path

import gradio as gr
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

RELEASE = "https://github.com/24f2006816/eurosat-landcover-classifier/releases/download/model-v1"


def fetch(name: str) -> bytes:
    with urllib.request.urlopen(f"{RELEASE}/{name}", timeout=120) as response:  # noqa: S310 - fixed URL
        return response.read()


checkpoint = torch.load(io.BytesIO(fetch("landcover.pt")), map_location="cpu", weights_only=False)
classes, size = checkpoint["classes"], checkpoint["size"]
if not checkpoint["model"].startswith("mobilenet"):
    raise RuntimeError(f"demo expects a MobileNetV3 checkpoint, got {checkpoint['model']}")
model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

examples_dir = Path("examples")
if not examples_dir.exists():
    zipfile.ZipFile(io.BytesIO(fetch("examples.zip"))).extractall(examples_dir)

preprocess = transforms.Compose([
    transforms.Resize((size, size)), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
NICE = {"AnnualCrop": "Annual crop", "HerbaceousVegetation": "Herbaceous vegetation",
        "PermanentCrop": "Permanent crop", "SeaLake": "Sea or lake"}


def classify(image):
    if image is None:
        return {}
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(image.convert("RGB")).unsqueeze(0)), dim=-1)[0]
    return {NICE.get(c, c): float(p) for c, p in zip(classes, probs, strict=True)}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Satellite tile"),
    outputs=gr.Label(num_top_classes=3, label="Land cover"),
    title="EuroSAT Land-Cover Classifier",
    description=("Classifies Sentinel-2 satellite tiles into 10 land-cover types with a fine-tuned MobileNetV3. "
                 "Try the example tiles from the held-out test set. "
                 "[Code, training workflow and results](https://github.com/24f2006816/eurosat-landcover-classifier)"),
    examples=sorted(str(p) for p in examples_dir.glob("*.png")),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
