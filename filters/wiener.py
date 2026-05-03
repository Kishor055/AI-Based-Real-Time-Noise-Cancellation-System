import numpy as np

def wiener_filter(signal, noise_est=None, alpha=1.0):
    """
    Wiener Filter (Frequency Domain)

    signal     : noisy input signal
    noise_est  : optional noise estimate (same length)
    alpha      : noise scaling factor

    Returns:
        filtered signal
    """

    signal = np.array(signal)

    # FFT of signal
    S = np.fft.fft(signal)

    # Power spectrum of signal
    S_power = np.abs(S) ** 2

    # Estimate noise power
    if noise_est is None:
        # Estimate noise from first 10% of signal (assumes silence)
        noise_sample = signal[:len(signal)//10]
        N_power = np.mean(np.abs(np.fft.fft(noise_sample)) ** 2)
        N_power = np.ones_like(S_power) * N_power
    else:
        N = np.fft.fft(noise_est)
        N_power = np.abs(N) ** 2

    # Wiener filter formula
    H = S_power / (S_power + alpha * N_power + 1e-8)

    # Apply filter
    S_filtered = H * S

    # Inverse FFT
    filtered = np.fft.ifft(S_filtered).real

    return filtered
