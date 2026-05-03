import numpy as np

from filters.lms import lms_filter
from filters.nlms import nlms_filter
from filters.wiener import wiener_filter
from ml_model.model import denoise_model


EPS = 1e-8


def process_audio(
    chunk,
    mode="LMS",
    noise_ref=None,
    config=None
):
    """
    Generic audio processing function

    Parameters:
        chunk      : input signal (1D numpy array)
        mode       : "LMS" | "NLMS" | "Wiener" | "DL"
        noise_ref  : optional noise reference signal
        config     : optional dict for parameters

    Returns:
        processed signal
    """

    # ----------- PREPARE INPUT -----------
    chunk = np.asarray(chunk, dtype=np.float32)

    if len(chunk) == 0:
        return chunk

    # ----------- NOISE REFERENCE -----------
    if noise_ref is None:
        # fallback ONLY for testing
        noise_ref = np.random.randn(len(chunk)).astype(np.float32) * 0.01
    else:
        noise_ref = np.asarray(noise_ref, dtype=np.float32)

        # match length
        if len(noise_ref) != len(chunk):
            noise_ref = np.pad(noise_ref, (0, len(chunk) - len(noise_ref)))[:len(chunk)]

    # ----------- DEFAULT CONFIG -----------
    config = config or {}

    try:
        # ----------- LMS -----------
        if mode == "LMS":
            params = config.get("lms", {})
            return lms_filter(
                chunk,
                noise_ref,
                M=params.get("filter_order", 32),
                mu=params.get("mu", 0.01)
            )

        # ----------- NLMS -----------
        elif mode == "NLMS":
            params = config.get("nlms", {})
            return nlms_filter(
                chunk,
                noise_ref,
                M=params.get("filter_order", 32),
                mu=params.get("mu", 0.01)
            )

        # ----------- WIENER -----------
        elif mode == "Wiener":
            params = config.get("wiener", {})
            return wiener_filter(
                chunk,
                alpha=params.get("alpha", 1.0)
            )

        # ----------- DEEP LEARNING -----------
        elif mode == "DL":
            return denoise_model(chunk)

        else:
            print(f"⚠️ Unknown mode: {mode}")
            return chunk

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return chunk
