import numpy as np

EPS = 1e-8


class NLMSFilter:
    """
    Normalized LMS Adaptive Filter
    """

    def __init__(self, filter_order=32, mu=0.1):
        self.M = filter_order
        self.mu = mu
        self.w = np.zeros(self.M, dtype=np.float32)

    def reset(self):
        self.w.fill(0.0)

    def process(self, desired, reference):

        desired = np.asarray(desired, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)

        N = min(len(desired), len(reference))

        desired = desired[:N]
        reference = reference[:N]

        y = np.zeros(N, dtype=np.float32)
        e = np.zeros(N, dtype=np.float32)

        for n in range(self.M, N):

            x_vec = reference[n-self.M:n][::-1]

            # Estimated noise
            y[n] = np.dot(self.w, x_vec)

            # Error signal
            e[n] = desired[n] - y[n]

            # NORMALIZATION
            norm = np.dot(x_vec, x_vec) + EPS

            # NLMS UPDATE
            self.w += (self.mu / norm) * e[n] * x_vec

        return y, e


# ---------------- SIMPLE API ----------------
def nlms_filter(noisy_signal, noise_ref, M=32, mu=0.1):

    nlms = NLMSFilter(filter_order=M, mu=mu)
    _, filtered = nlms.process(noisy_signal, noise_ref)

    return filtered