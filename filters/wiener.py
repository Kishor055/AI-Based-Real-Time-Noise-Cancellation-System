import numpy as np

EPS = 1e-8


def wiener_filter(signal, noise_est=None, alpha=1.0):
    """
    Improved Wiener Filter (Frequency Domain)

    signal     : noisy input signal
    noise_est  : optional noise estimate
    alpha      : noise scaling factor

    Returns:
        filtered signal
    """

    signal = np.asarray(signal, dtype=np.float32)
    N = len(signal)

    # ---------------- WINDOWING ----------------
    window = np.hanning(N)
    signal_win = signal * window

    # ---------------- FFT ----------------
    S = np.fft.fft(signal_win)
    S_power = np.abs(S) ** 2

    # ---------------- NOISE ESTIMATION ----------------
    if noise_est is None:
        # Use small segment (more robust than 10%)
        seg_len = min(2048, N)
        noise_sample = signal[:seg_len] * np.hanning(seg_len)

        N_fft = np.fft.fft(noise_sample, n=N)
        N_power = np.abs(N_fft) ** 2

    else:
        noise_est = np.asarray(noise_est, dtype=np.float32)

        # Match length
        if len(noise_est) != N:
            noise_est = np.pad(noise_est, (0, N - len(noise_est)))[:N]

        noise_win = noise_est * window
        N_fft = np.fft.fft(noise_win)
        N_power = np.abs(N_fft) ** 2

    # ---------------- WIENER FILTER ----------------
    H = S_power / (S_power + alpha * N_power + EPS)

    # ---------------- APPLY FILTER ----------------
    S_filtered = H * S

    # ---------------- INVERSE FFT ----------------
    filtered = np.fft.ifft(S_filtered).real

    # Remove window effect
    filtered = filtered / (window + EPS)

    return filtered
