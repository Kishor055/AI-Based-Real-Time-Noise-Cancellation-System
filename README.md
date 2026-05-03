# AI-Based-Real-Time-Noise-Cancellation-System
Built an AI-powered real-time noise cancellation system integrating adaptive filtering (LMS/NLMS), Wiener filtering, and deep learning-based denoising with a live Streamlit dashboard for performance monitoring (SNR, MSE).

---

# 🎧 AI Noise Cancellation System

**DSP + Deep Learning based Audio Denoising Pipeline**

---

## 📌 Overview

This project implements a **complete audio noise reduction system** using both:

* 🎚 **Digital Signal Processing (DSP)** techniques
* 🤖 **Deep Learning-based denoising**

The system supports multiple filtering approaches and provides **performance evaluation, real-time processing, and a web interface**.

---

## 🚀 Key Features

* 🔊 Audio denoising using:

  * LMS (Least Mean Squares)
  * NLMS (Normalized LMS)
  * Wiener Filter
  * Deep Learning Model (PyTorch)

* 🎤 Real-time microphone processing (low latency)

* 📊 Performance metrics:

  * SNR (Signal-to-Noise Ratio)
  * MSE / RMSE
  * PSNR

* 🌐 Streamlit web app for interactive testing

* ⚙️ Config-driven pipeline

* 📦 Modular and scalable architecture

---

## 🧠 System Architecture

```
Input Audio
     │
     ▼
Noise Addition / Real Noise Input
     │
     ▼
Filter Selection
(LMS / NLMS / Wiener / DL)
     │
     ▼
Processed Audio Output
     │
     ▼
Performance Metrics (SNR, MSE, PSNR)
```

---

## 📂 Project Structure

```
project/
│
├── run.py                 # Main pipeline
├── config.yaml           # Configuration file
├── requirements.txt
│
├── data/                 # Input audio
│   └── clean/
│       └── sample1.wav
│
├── output/               # Processed audio output
│
├── filters/              # DSP filters
│   ├── lms.py
│   ├── nlms.py
│   └── wiener.py
│
├── ml_model/             # Deep learning model
│   ├── model.py
│   ├── train.py
│   └── weights/
│       └── denoise_model.pth
│
├── utils/
│   ├── audio.py
│   ├── metrics.py
│   └── config.py
│
├── core/
│   └── processor.py
│
└── app/
    └── app.py            # Streamlit UI
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-noise-cancellation.git
cd ai-noise-cancellation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run Main Pipeline

```bash
python run.py
```

### 🔹 Run Web App

```bash
streamlit run app/app.py
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
audio:
  input:
    file: "data/clean/sample1.wav"
  output:
    file: "output/filtered_audio.wav"

filters:
  active: "NLMS"

  lms:
    filter_order: 32
    mu: 0.01

  nlms:
    filter_order: 64
    mu: 0.008

  wiener:
    alpha: 1.0

metrics:
  enabled: true

output:
  save_audio: true
```

---

## 🤖 Deep Learning Model

* Framework: **PyTorch**
* Architecture: Fully Connected Autoencoder
* Input size: 1024 samples (chunk-based processing)
* Output: Denoised signal

### Train Model

```bash
python ml_model/train.py
```

---

## 📊 Performance Metrics

| Metric | Description                |
| ------ | -------------------------- |
| SNR    | Signal-to-Noise Ratio      |
| MSE    | Mean Squared Error         |
| RMSE   | Root Mean Squared Error    |
| PSNR   | Peak Signal-to-Noise Ratio |

---

## 🎤 Real-Time Processing

```python
from utils.audio import stream_audio
from core.processor import process_audio

stream = stream_audio(lambda x: process_audio(x, mode="NLMS"))
```

---

## ⚠️ Limitations

* LMS/NLMS require a **noise reference signal**
* Wiener filter assumes **stationary noise**
* Deep learning model trained on **synthetic noise**
* Real-world performance depends on dataset quality

---

## 🚀 Future Improvements

* CNN / U-Net based denoising model
* Real-world noise dataset integration
* Dual-microphone adaptive filtering
* Mobile / desktop deployment
* API (FastAPI) integration

---

## 📦 Requirements

```
numpy
scipy
matplotlib
soundfile
sounddevice
torch
tqdm
pyyaml
streamlit
```

---

## 🎯 Resume Highlight

> Developed a full-stack audio denoising system integrating DSP algorithms and deep learning, with real-time processing, performance evaluation, and interactive web deployment.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Kishor Kakde**
GitHub: https://github.com/yourusername

---

## ⭐ Support

If you find this project useful, please ⭐ star the repository!

---
