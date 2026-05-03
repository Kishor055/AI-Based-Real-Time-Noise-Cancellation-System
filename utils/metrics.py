import numpy as np

EPS = 1e-10  # to avoid division by zero


# ---------------- BASIC METRICS ----------------

def compute_snr(clean, processed):
    """
    Signal-to-Noise Ratio (dB)
    """
    clean = np.array(clean)
    processed = np.array(processed)

    noise = clean - processed

    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2) + EPS

    return 10 * np.log10(signal_power / noise_power)


def compute_mse(clean, processed):
    """
    Mean Squared Error
    """
    clean = np.array(clean)
    processed = np.array(processed)

    return np.mean((clean - processed) ** 2)


def compute_rmse(clean, processed):
    """
    Root Mean Squared Error
    """
    return np.sqrt(compute_mse(clean, processed))


def compute_psnr(clean, processed):
    """
    Peak Signal-to-Noise Ratio (dB)
    """
    mse = compute_mse(clean, processed)

    max_val = np.max(np.abs(clean)) + EPS

    return 20 * np.log10(max_val / (np.sqrt(mse) + EPS))


# ---------------- FRAME-BASED METRICS ----------------

def frame_snr(clean, processed, frame_size=1024):
    """
    Frame-wise SNR (useful for real-time dashboard)
    """
    clean = np.array(clean)
    processed = np.array(processed)

    snr_values = []

    for i in range(0, len(clean), frame_size):
        c = clean[i:i+frame_size]
        p = processed[i:i+frame_size]

        if len(c) == 0:
            continue

        snr_values.append(compute_snr(c, p))

    return np.array(snr_values)


def frame_mse(clean, processed, frame_size=1024):
    """
    Frame-wise MSE
    """
    clean = np.array(clean)
    processed = np.array(processed)

    mse_values = []

    for i in range(0, len(clean), frame_size):
        c = clean[i:i+frame_size]
        p = processed[i:i+frame_size]

        if len(c) == 0:
            continue

        mse_values.append(compute_mse(c, p))

    return np.array(mse_values)


# ---------------- REAL-TIME SAFE METRICS ----------------

def safe_snr(clean, processed):
    """
    Safe SNR for real-time (handles silent signals)
    """
    if np.all(clean == 0):
        return 0.0

    return compute_snr(clean, processed)


def safe_mse(clean, processed):
    """
    Safe MSE
    """
    return compute_mse(clean, processed)


# ---------------- SUMMARY FUNCTION ----------------

def evaluate_all(clean, processed):
    """
    Returns all key metrics in one dictionary
    """
    return {
        "SNR": compute_snr(clean, processed),
        "MSE": compute_mse(clean, processed),
        "RMSE": compute_rmse(clean, processed),
        "PSNR": compute_psnr(clean, processed),
    }
