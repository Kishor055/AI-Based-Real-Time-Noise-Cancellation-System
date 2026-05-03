import torch
import torch.nn as nn
import numpy as np
import os

EPS = 1e-8

# ---------------- MODEL ----------------
class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------
model = DenoiseNet().to(DEVICE)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "denoise_model.pth")

if os.path.exists(WEIGHTS_PATH):
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    # Handle both raw state_dict and checkpoint dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("✅ Model weights loaded")
else:
    raise FileNotFoundError("❌ Trained model weights not found. Run train.py first.")

model.eval()


# ---------------- INFERENCE FUNCTION ----------------
def denoise_model(signal, chunk_size=1024, overlap=256):
    """
    Input: 1D numpy array
    Output: denoised signal (same length)
    """

    signal = np.asarray(signal, dtype=np.float32)

    # Normalize
    max_val = np.max(np.abs(signal)) + EPS
    signal_norm = signal / max_val

    step = chunk_size - overlap
    output = np.zeros(len(signal_norm), dtype=np.float32)
    weight = np.zeros(len(signal_norm), dtype=np.float32)

    chunks = []
    indices = []

    # ----------- CREATE CHUNKS -----------
    for i in range(0, len(signal_norm), step):
        chunk = signal_norm[i:i + chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        chunks.append(chunk)
        indices.append(i)

    # Convert to tensor batch
    batch = torch.tensor(np.array(chunks), dtype=torch.float32).to(DEVICE)

    # ----------- MODEL INFERENCE -----------
    with torch.no_grad():
        preds = model(batch).cpu().numpy()

    # ----------- OVERLAP-ADD RECONSTRUCTION -----------
    for i, pred in zip(indices, preds):
        end = i + chunk_size

        valid_len = min(chunk_size, len(signal_norm) - i)

        output[i:i + valid_len] += pred[:valid_len]
        weight[i:i + valid_len] += 1.0

    # Avoid division by zero
    weight[weight == 0] = 1.0
    output = output / weight

    # Restore scale
    output = output * max_val

    return output
