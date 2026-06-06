import os
import sys
import uuid
import shutil
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Force UTF-8 on Windows
import io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add root directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters.lms import lms_filter
from filters.nlms import nlms_filter
from filters.wiener import wiener_filter
from ml_model.model import denoise_model, DenoiseNet, WEIGHTS_PATH
from utils.metrics import evaluate_all
from utils.audio import normalize_audio

app = FastAPI(title="SoundShield AI Backend API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

# Global training state
training_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0,
    "losses": []
}

# ---------------- HELPERS ----------------
def generate_noise(signal, noise_type, length):
    if noise_type == "White Gaussian Noise":
        return np.random.randn(length).astype(np.float32)
    elif noise_type == "Pink Noise (1/f)":
        uneven = length % 2
        X = np.random.randn(length // 2 + 1 + uneven) + 1j * np.random.randn(length // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1)
        y = (np.fft.irfft(X / S))[:length]
        return y / (np.max(np.abs(y)) + 1e-8)
    elif noise_type == "AC Hum (50 Hz)":
        t = np.arange(length) / 16000
        hum = (
            np.sin(2 * np.pi * 50 * t) +
            0.5 * np.sin(2 * np.pi * 100 * t) +
            0.25 * np.sin(2 * np.pi * 150 * t)
        )
        return hum / (np.max(np.abs(hum)) + 1e-8)
    return np.random.randn(length).astype(np.float32) * 1e-4

def mix_noise(clean_signal, noise_signal, target_snr_db):
    s_power = np.mean(clean_signal ** 2)
    n_power = np.mean(noise_signal ** 2)
    if s_power < 1e-8:
        return clean_signal + noise_signal, noise_signal
    target_n_power = s_power / (10 ** (target_snr_db / 10))
    scale = np.sqrt(target_n_power / (n_power + 1e-8))
    scaled_noise = noise_signal * scale
    return clean_signal + scaled_noise, scaled_noise

# ---------------- ENDPOINTS ----------------

@app.get("/api/status")
def get_status():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    has_weights = os.path.exists(WEIGHTS_PATH)
    
    # Mock system metrics for display
    cpu_usage = int(np.random.randint(10, 45))
    gpu_usage = int(np.random.randint(5, 25)) if device == "cuda" else 0
    gpu_mem = "1.2 GB / 8.0 GB" if device == "cuda" else "N/A"

    return {
        "status": "online",
        "device": device.upper(),
        "model_loaded": has_weights,
        "cpu_usage": cpu_usage,
        "gpu_usage": gpu_usage,
        "gpu_memory": gpu_mem,
        "training": training_state
    }

@app.post("/api/process-file")
async def process_file(
    file: UploadFile = File(...),
    filter_mode: str = Form("NLMS"),
    filter_order: int = Form(64),
    mu: float = Form(0.08),
    alpha: float = Form(1.0),
    inject_noise: bool = Form(False),
    noise_type: str = Form("White Gaussian Noise"),
    snr_level: float = Form(10.0)
):
    try:
        # Save uploaded file
        file_id = str(uuid.uuid4())
        input_filename = f"{file_id}_in.wav"
        input_path = os.path.join(TEMP_DIR, input_filename)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load audio
        signal, sr = sf.read(input_path)
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        
        signal = normalize_audio(signal)
        
        # Prepare signals
        if inject_noise:
            noise_ref = generate_noise(signal, noise_type, len(signal))
            noisy, noise_final = mix_noise(signal, noise_ref, snr_level)
        else:
            noisy = signal.copy()
            noise_final = np.random.randn(len(signal)).astype(np.float32) * 1e-4

        # Denoising
        config = {
            "lms": {"filter_order": filter_order, "mu": mu},
            "nlms": {"filter_order": filter_order, "mu": mu},
            "wiener": {"alpha": alpha}
        }
        
        if filter_mode == "LMS":
            denoised = lms_filter(noisy, noise_final, M=filter_order, mu=mu)
        elif filter_mode == "NLMS":
            denoised = nlms_filter(noisy, noise_final, M=filter_order, mu=mu)
        elif filter_mode == "Wiener":
            denoised = wiener_filter(noisy, noise_est=noise_final, alpha=alpha)
        elif filter_mode == "DL":
            denoised = denoise_model(noisy)
        else:
            denoised = noisy.copy()

        # Compute metrics
        metrics = evaluate_all(signal, denoised)
        noisy_metrics = evaluate_all(signal, noisy)
        
        # Save denoised
        output_filename = f"{file_id}_out.wav"
        output_path = os.path.join(TEMP_DIR, output_filename)
        sf.write(output_path, np.clip(denoised, -1.0, 1.0), sr)
        
        # Also save noisy file for playback
        noisy_filename = f"{file_id}_noisy.wav"
        noisy_path = os.path.join(TEMP_DIR, noisy_filename)
        sf.write(noisy_path, np.clip(noisy, -1.0, 1.0), sr)

        return {
            "success": True,
            "file_id": file_id,
            "metrics": {
                "snr": metrics["SNR"],
                "mse": metrics["MSE"],
                "psnr": metrics["PSNR"],
                "rmse": metrics["RMSE"],
                "snr_improvement": metrics["SNR"] - noisy_metrics["SNR"]
            },
            "audio_details": {
                "filename": file.filename,
                "sample_rate": sr,
                "duration": len(signal) / sr,
                "size_bytes": os.path.getsize(input_path)
            },
            "urls": {
                "noisy": f"/api/audio/{noisy_filename}",
                "denoised": f"/api/audio/{output_filename}"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="File not found")

# ---------------- TRAINING BACKGROUND TASK ----------------
def run_training(epochs: int, lr: float, batch_size: int):
    global training_state
    try:
        from ml_model.train import load_dataset
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "data", "clean")
        
        if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
            training_state = {
                "is_training": False,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_loss": 0.0,
                "losses": [],
                "error": f"Dataset path '{data_path}' is empty or does not exist."
            }
            return

        X, Y = load_dataset(data_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
        
        model = DenoiseNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        training_state["is_training"] = True
        training_state["total_epochs"] = epochs
        training_state["losses"] = []
        
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            indices = np.random.permutation(dataset_size)
            
            for i in range(0, dataset_size, batch_size):
                batch_idx = indices[i:i+batch_size]
                bx = X_t[batch_idx]
                by = Y_t[batch_idx]
                
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / (dataset_size / batch_size)
            training_state["current_epoch"] = epoch + 1
            training_state["current_loss"] = avg_loss
            training_state["losses"].append(avg_loss)
            
        # Save model weights
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": training_state["current_loss"],
            "epoch": epochs - 1
        }, WEIGHTS_PATH)
        
        training_state["is_training"] = False
        
        # Reload model inside model.py
        import ml_model.model
        ml_model.model.MODEL_AVAILABLE = True
        ml_model.model.model.load_state_dict(model.state_dict())
        ml_model.model.model.eval()
        
    except Exception as e:
        training_state = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "current_loss": 0.0,
            "losses": [],
            "error": str(e)
        }

@app.post("/api/train")
def start_train(
    background_tasks: BackgroundTasks,
    epochs: int = Form(15),
    lr: float = Form(0.001),
    batch_size: int = Form(16)
):
    if training_state["is_training"]:
        raise HTTPException(status_code=400, detail="Training is already in progress")
        
    background_tasks.add_task(run_training, epochs, lr, batch_size)
    return {"message": "Training started in background."}

@app.get("/api/train/status")
def get_train_status():
    return training_state
