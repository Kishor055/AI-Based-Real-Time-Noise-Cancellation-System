import numpy as np

class LMSFilter:
    """
    LMS Adaptive Filter for noise cancellation
    """

    def __init__(self, filter_order=32, mu=0.01):
        self.M = filter_order
        self.mu = mu
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

            y[n] = np.dot(self.w, x_vec)   # estimated noise
            e[n] = desired[n] - y[n]       # error (clean output)

            # LMS weight update
            self.w += self.mu * e[n] * x_vec

        return y, e


# ---------------- SIMPLE FUNCTION API ----------------
def lms_filter(noisy_signal, noise_ref, M=32, mu=0.01):
    """
    Wrapper function for quick use
    """
    lms = LMSFilter(filter_order=M, mu=mu)
    _, filtered = lms.process(noisy_signal, noise_ref)
    return filtered
