import numpy as np

class NLMSFilter:
    """
    Normalized Least Mean Squares (NLMS) Adaptive Filter
    More stable than LMS (handles varying signal power)
    """

    def __init__(self, filter_order=32, mu=0.5, epsilon=1e-6):
        self.M = filter_order
        self.mu = mu
        self.epsilon = epsilon
        self.w = np.zeros(self.M)

    def reset(self):
        """Reset filter weights"""
        self.w = np.zeros(self.M)

    def process(self, desired, reference):
        """
        desired  : noisy signal (d[n])
        reference: noise reference (x[n])

        Returns:
            y : estimated noise
            e : filtered output (clean signal)
        """

        N = len(desired)
        y = np.zeros(N)
        e = np.zeros(N)

        for n in range(self.M, N):
            x_vec = reference[n:n-self.M:-1]

            if len(x_vec) != self.M:
                continue

            # Estimated noise
            y[n] = np.dot(self.w, x_vec)

            # Error (clean signal)
            e[n] = desired[n] - y[n]

            # Normalization factor (VERY IMPORTANT)
            norm = np.dot(x_vec, x_vec) + self.epsilon

            # NLMS weight update
            self.w += (self.mu / norm) * e[n] * x_vec

        return y, e


# ---------------- SIMPLE FUNCTION API ----------------
def nlms_filter(noisy_signal, noise_ref, M=32, mu=0.5):
    """
    Wrapper function for easy use
    """
    nlms = NLMSFilter(filter_order=M, mu=mu)
    _, filtered = nlms.process(noisy_signal, noise_ref)
    return filtered
