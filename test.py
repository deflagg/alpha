import torch
from torch import nn
from PIL import Image
import pandas as pd
import numpy as np

# --- Device setup for GPU acceleration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# For reproducibility across CPU/GPU
torch.manual_seed(0)
if device.type == "cuda":
    torch.cuda.manual_seed_all(0)

# --- Load and preprocess image (no resizing) ---
img_path = "myimg.jpg"
img = Image.open(img_path).convert("L")          # grayscale, original resolution
width, height = img.size                         # PIL returns (width, height)
print(f"Original image size: {width}x{height}")

x_np = np.array(img)
x = torch.from_numpy(x_np).float() / 255.0       # [height, width] in [0,1]
x = x.flatten().unsqueeze(0).to(device)          # [1, height*width] → move to GPU

input_features = x.shape[1]
print(f"Flattened input features: {input_features}")

# --- Linear projection down to 50x50 ---
# Dynamically create the linear layer based on actual input dimension
proj = nn.Linear(input_features, 128*128, bias=True).to(device)  # projects down (or up if input < 2500)
y = proj(x)                                      # [1, 2500] on GPU
y2d = y.view(128, 128)                             # [50, 50] (drops batch dim)

# --- Convert to a sparse SDR (k-winners-take-all) ---
k = 50  # ~2% sparsity of 2500 bits
flat = y2d.flatten()
topk_idx = torch.topk(flat, k=k, largest=True).indices

sdr = torch.zeros_like(flat, dtype=torch.int32)
sdr[topk_idx] = 1
sdr2d = sdr.view(128, 128)

# --- Show results ---
active = int(sdr.sum().item())
density = active / sdr.numel()

print(f"Input flattened features: {input_features}")
print(f"Projected to: 128x128 ({sdr.numel()} bits)")
print(f"SDR k-winners active bits: {active} (density {density:.2%})")