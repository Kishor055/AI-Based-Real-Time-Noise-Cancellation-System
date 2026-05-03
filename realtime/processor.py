import numpy as np
from filters.lms import lms_filter
from filters.nlms import nlms_filter
from filters.wiener import wiener_filter
from ml_model.model import denoise_model

def process_audio(chunk, mode="LMS"):
    noise_ref = np.random.randn(len(chunk)) * 0.01

    if mode == "LMS":
        return lms_filter(chunk, noise_ref)

    elif mode == "NLMS":
        return nlms_filter(chunk, noise_ref)

    elif mode == "Wiener":
        return wiener_filter(chunk)

    elif mode == "DL":
        return denoise_model(chunk)

    return chunk
