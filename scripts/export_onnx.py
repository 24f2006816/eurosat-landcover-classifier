"""Export the released PyTorch checkpoint to ONNX for in-browser inference, and check both agree.

Usage: python scripts/export_onnx.py landcover.pt web/landcover.onnx
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
ckpt = torch.load(src, map_location="cpu", weights_only=False)
model = mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(ckpt["classes"]))
model.load_state_dict(ckpt["state_dict"])
model.eval()

size = ckpt["size"]
torch.onnx.export(model, torch.randn(1, 3, size, size), dst, input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
(dst.parent / "classes.json").write_text(json.dumps(ckpt["classes"]))

x = torch.randn(8, 3, size, size)
diff = np.abs(model(x).detach().numpy() - ort.InferenceSession(str(dst)).run(None, {"image": x.numpy()})[0]).max()
print(f"exported {dst} ({dst.stat().st_size / 1e6:.1f} MB); max |torch - onnx| = {diff:.2e}")
assert diff < 1e-4
