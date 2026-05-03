import numpy as np

EPS = 1e-10


# ---------------- INTERNAL UTILS ----------------
def _prepare_signals(clean, processed):
    clean = np.asarray(clean, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)

    min_len = min(len(clean), len(processed))
    return clean[:min_len], processed[:min_len]


def _safe_log10(x):
    return np.log10(np.maximum(x, EPS))


# ---------------- BASIC METRICS ----------------
def compute_snr(clean, processed):
    clean, processed = _prepare_signals(clean, processed)

    noise = clean - processed

    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)

    if signal_power < EPS:
        return 0.0

    snr = 10 * _safe_log10((signal_power + EPS) / (noise_power + EPS))

    return float(np.nan_to_num(snr))


def compute_mse(clean, processed):
    clean, processed = _prepare_signals(clean, processed)
    mse = np.mean((clean - processed) ** 2)
    return float(np.nan_to_num(mse))


def compute_rmse(clean, processed):
    return float(np.sqrt(compute_mse(clean, processed)))


def compute_psnr(clean, processed):
    clean, processed = _prepare_signals(clean, processed)

    noise = clean - processed
    mse = np.mean(noise ** 2)

    if mse < EPS:
        return 100.0

    max_val = 1.0  # normalized audio
    psnr = 20 * _safe_log10(max_val / (np.sqrt(mse) + EPS))

    return float(np.nan_to_num(psnr))


# ---------------- FRAME-BASED METRICS ----------------
def _frame_split(signal, frame_size):
    """Split signal into frames including last partial frame"""
    num_frames = int(np.ceil(len(signal) / frame_size))

    padded = np.pad(signal, (0, num_frames * frame_size - len(signal)))
    return padded.reshape(num_frames, frame_size)


def frame_snr(clean, processed, frame_size=1024):
    clean, processed = _prepare_signals(clean, processed)

    clean_frames = _frame_split(clean, frame_size)
    proc_frames = _frame_split(processed, frame_size)

    snr_values = [
        compute_snr(c, p) for c, p in zip(clean_frames, proc_frames)
    ]

    return np.array(snr_values, dtype=np.float32)


def frame_mse(clean, processed, frame_size=1024):
    clean, processed = _prepare_signals(clean, processed)

    clean_frames = _frame_split(clean, frame_size)
    proc_frames = _frame_split(processed, frame_size)

    mse_values = [
        compute_mse(c, p) for c, p in zip(clean_frames, proc_frames)
    ]

    return np.array(mse_values, dtype=np.float32)


# ---------------- REAL-TIME SAFE METRICS ----------------
def safe_snr(clean, processed):
    clean, processed = _prepare_signals(clean, processed)

    if np.all(np.abs(clean) < EPS):
        return 0.0

    return compute_snr(clean, processed)


def safe_mse(clean, processed):
    return compute_mse(clean, processed)


# ---------------- SUMMARY FUNCTION ----------------
def evaluate_all(clean, processed):
    return {
        "SNR": compute_snr(clean, processed),
        "MSE": compute_mse(clean, processed),
        "RMSE": compute_rmse(clean, processed),
        "PSNR": compute_psnr(clean, processed),
    }
