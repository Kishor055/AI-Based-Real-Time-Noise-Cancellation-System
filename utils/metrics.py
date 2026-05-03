import numpy as np

EPS = 1e-10


# ---------------- INTERNAL UTILS ----------------

def _prepare_signals(clean, processed):
    """
    Ensure both signals are numpy arrays and same length
    """
    clean = np.asarray(clean, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)

    min_len = min(len(clean), len(processed))

    return clean[:min_len], processed[:min_len]


# ---------------- BASIC METRICS ----------------

def compute_snr(clean, processed):
    clean, processed = _prepare_signals(clean, processed)

    noise = clean - processed

    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)

    if signal_power < EPS:
        return 0.0

    return 10 * np.log10((signal_power + EPS) / (noise_power + EPS))


def compute_mse(clean, processed):
    clean, processed = _prepare_signals(clean, processed)
    return np.mean((clean - processed) ** 2)


def compute_rmse(clean, processed):
    return np.sqrt(compute_mse(clean, processed))


def compute_psnr(clean, processed):
    clean, processed = _prepare_signals(clean, processed)

    mse = compute_mse(clean, processed)

    if mse < EPS:
        return 100.0  # near perfect signal

    max_val = 1.0  # assume normalized audio

    return 20 * np.log10(max_val / (np.sqrt(mse) + EPS))


# ---------------- FRAME-BASED METRICS ----------------

def frame_snr(clean, processed, frame_size=1024):
    clean, processed = _prepare_signals(clean, processed)

    num_frames = len(clean) // frame_size
    snr_values = []

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        c = clean[start:end]
        p = processed[start:end]

        snr_values.append(compute_snr(c, p))

    return np.array(snr_values)


def frame_mse(clean, processed, frame_size=1024):
    clean, processed = _prepare_signals(clean, processed)

    num_frames = len(clean) // frame_size
    mse_values = []

    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size

        c = clean[start:end]
        p = processed[start:end]

        mse_values.append(compute_mse(c, p))

    return np.array(mse_values)


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
