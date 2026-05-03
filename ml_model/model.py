import torch
import torch.nn as nn
import numpy as np
import os

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

# ---------------- LOAD MODEL ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DenoiseNet().to(DEVICE)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "denoise_model.pth")

if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    print("✅ Model weights loaded")
else:
    print("⚠️ Warning: No trained weights found. Using untrained model.")

model.eval()

# ---------------- INFERENCE FUNCTION ----------------
def denoise_model(signal):
    """
    Input: 1D numpy array (any length)
    Output: denoised signal (same length)
    """

    signal = np.array(signal)

    # Normalize
    max_val = np.max(np.abs(signal)) + 1e-8
    signal = signal / max_val

    # Process in chunks
    chunk_size = 1024
    output = []

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]

        # Pad if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        x = torch.tensor(chunk, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            y = model(x)

        y = y.cpu().numpy()

        # Remove padding
        y = y[:len(signal[i:i+chunk_size])]

        output.extend(y)

    output = np.array(output)

    # Restore scale
    output = output * max_val

    return output
