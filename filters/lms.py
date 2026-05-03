import numpy as np

EPS = 1e-8


class LMSFilter:
    """
    LMS Adaptive Filter for noise cancellation
    """

    def __init__(self, filter_order=32, mu=0.01):
        self.M = filter_order
        self.mu = mu
        self.w = np.zeros(self.M, dtype=np.float32)

    def reset(self):
        """Reset filter weights"""
        self.w.fill(0.0)

    def process(self, desired, reference):
        """
        desired  : noisy signal (d[n])
        reference: noise reference (x[n])

        Returns:
            y : estimated noise
            e : filtered output (clean signal)
        """

        desired = np.asarray(desired, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)

        # Ensure same length
        N = min(len(desired), len(reference))
        desired = desired[:N]
        reference = reference[:N]

        y = np.zeros(N, dtype=np.float32)
        e = np.zeros(N, dtype=np.float32)

        for n in range(self.M, N):
            # Build input vector (reversed window)
            x_vec = reference[n - self.M:n][::-1]

            # Estimate noise
            y[n] = np.dot(self.w, x_vec)

            # Error = cleaned signal
            e[n] = desired[n] - y[n]

            # ---- Stable LMS Update ----
            norm = np.dot(x_vec, x_vec) + EPS
            self.w += (self.mu / norm) * e[n] * x_vec

        return y, e


# ---------------- SIMPLE FUNCTION API ----------------
def lms_filter(noisy_signal, noise_ref, M=32, mu=0.01):
    """
    Wrapper function for quick use
    """
    lms = LMSFilter(filter_order=M, mu=mu)
    _, filtered = lms.process(noisy_signal, noise_ref)
    return filtered
