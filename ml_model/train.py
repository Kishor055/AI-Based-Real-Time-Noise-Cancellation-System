import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import DenoiseNet

# ---------------- CONFIG ----------------
DATA_PATH = "../data/clean"
WEIGHTS_PATH = "weights/denoise_model.pth"

EPOCHS = 20
BATCH_SIZE = 16
LR = 0.001
CHUNK_SIZE = 1024

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("weights", exist_ok=True)


# ---------------- DATA LOADER ----------------
def load_dataset(path):
    import soundfile as sf

    chunks_clean = []
    chunks_noisy = []

    for file in os.listdir(path):
        if not file.endswith(".wav"):
            continue

        signal, _ = sf.read(os.path.join(path, file))

        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        signal = signal / (np.max(np.abs(signal)) + 1e-8)

        # Create noisy version
        noise = np.random.randn(len(signal)) * 0.02
        noisy = signal + noise

        # Split into chunks
        for i in range(0, len(signal), CHUNK_SIZE):
            clean_chunk = signal[i:i+CHUNK_SIZE]
            noisy_chunk = noisy[i:i+CHUNK_SIZE]

            if len(clean_chunk) < CHUNK_SIZE:
                pad = CHUNK_SIZE - len(clean_chunk)
                clean_chunk = np.pad(clean_chunk, (0, pad))
                noisy_chunk = np.pad(noisy_chunk, (0, pad))

            chunks_clean.append(clean_chunk)
            chunks_noisy.append(noisy_chunk)

    return np.array(chunks_noisy), np.array(chunks_clean)


# ---------------- TRAIN ----------------
def train():
    print("🚀 Loading dataset...")

    X, Y = load_dataset(DATA_PATH)

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)

    dataset_size = X.shape[0]

    model = DenoiseNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    # Resume training if exists
    if os.path.exists(WEIGHTS_PATH):
        print("🔄 Loading existing model...")
        checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        # Shuffle indices
        indices = np.random.permutation(dataset_size)

        for i in tqdm(range(0, dataset_size, BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_idx = indices[i:i+BATCH_SIZE]

            batch_x = X[batch_idx]
            batch_y = Y[batch_idx]

            optimizer.zero_grad()

            output = model(batch_x)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (dataset_size / BATCH_SIZE)

        print(f"📉 Epoch {epoch+1} Loss: {avg_loss:.6f}")

        # ---------------- SAVE BEST MODEL ----------------
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "epoch": epoch
            }, WEIGHTS_PATH)

            print("💾 Saved best model")

    print("✅ Training complete!")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    train()
