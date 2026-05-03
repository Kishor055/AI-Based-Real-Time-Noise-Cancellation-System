import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from model import DenoiseNet

# ---------------- CONFIG ----------------
DATA_PATH = "../data/clean"
EPOCHS = 20
BATCH_SIZE = 16
LR = 0.001
CHUNK_SIZE = 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATASET ----------------
class AudioDataset(Dataset):
    def __init__(self, folder):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        signal, _ = sf.read(self.files[idx])

        # Mono
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        # Normalize
        signal = signal / np.max(np.abs(signal))

        # Random chunk
        if len(signal) > CHUNK_SIZE:
            start = np.random.randint(0, len(signal) - CHUNK_SIZE)
            clean = signal[start:start+CHUNK_SIZE]
        else:
            clean = np.pad(signal, (0, CHUNK_SIZE - len(signal)))

        # Add noise (realistic training)
        noise = np.random.randn(CHUNK_SIZE) * 0.05
        noisy = clean + noise

        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)

# ---------------- TRAIN FUNCTION ----------------
def train():
    dataset = AudioDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DenoiseNet().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {DEVICE}")
    print(f"Total samples: {len(dataset)}")

    for epoch in range(EPOCHS):
        total_loss = 0

        for noisy, clean in loader:
            noisy = noisy.to(DEVICE)
            clean = clean.to(DEVICE)

            output = model(noisy)

            loss = criterion(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

    # Save model
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/denoise_model.pth")

    print("✅ Model saved to weights/denoise_model.pth")

# ---------------- RUN ----------------
if __name__ == "__main__":
    train()
