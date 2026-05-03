import os
import numpy as np

from utils.config import load_config
from utils.audio import load_audio, save_audio
from utils.metrics import evaluate_all

from filters.lms import lms_filter
from filters.nlms import nlms_filter
from filters.wiener import wiener_filter
from ml_model.model import denoise_model


# ---------------- LOGGER ----------------
def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


# ---------------- FILTER ENGINE ----------------
def process_audio(signal, config):
    mode = config["filters"]["active"]

    log(f"Using filter: {mode}")

    # NOTE: Fake noise reference (for testing only)
    noise_ref = np.random.randn(len(signal)) * 0.01

    try:
        if mode == "LMS":
            params = config["filters"]["lms"]
            return lms_filter(signal, noise_ref,
                              M=params["filter_order"],
                              mu=params["mu"])

        elif mode == "NLMS":
            params = config["filters"]["nlms"]
            return nlms_filter(signal, noise_ref,
                               M=params["filter_order"],
                               mu=params["mu"])

        elif mode == "Wiener":
            params = config["filters"]["wiener"]
            return wiener_filter(signal, alpha=params["alpha"])

        elif mode == "DL":
            return denoise_model(signal)

        else:
            log("Unknown filter mode. Returning original signal.", "WARNING")
            return signal

    except Exception as e:
        log(f"Processing failed: {e}", "ERROR")
        return signal


# ---------------- MAIN PIPELINE ----------------
def main():
    log("Starting AI Noise Cancellation System...")

    # Load config
    try:
        config = load_config()
    except Exception as e:
        log(f"Failed to load config: {e}", "ERROR")
        return

    # Load audio
    input_path = config["audio"]["input"]["file"]

    if not os.path.exists(input_path):
        log(f"Input file not found: {input_path}", "ERROR")
        return

    signal, sr = load_audio(input_path)

    log(f"Loaded audio: {input_path}")
    log(f"Sample rate: {sr}")
    log(f"Signal length: {len(signal)}")

    # Add synthetic noise (for evaluation only)
    noise = np.random.randn(len(signal)) * 0.01
    noisy_signal = signal + noise

    log("Noise added for testing")

    # Process
    filtered = process_audio(noisy_signal, config)

    # Metrics
    if config.get("metrics", {}).get("enabled", False):
        try:
            results = evaluate_all(signal, filtered)

            log("Performance Metrics:")
            for k, v in results.items():
                log(f"{k}: {v:.4f}")

        except Exception as e:
            log(f"Metrics computation failed: {e}", "WARNING")

    # Save output
    if config.get("output", {}).get("save_audio", False):
        output_path = config["audio"]["output"]["file"]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            save_audio(output_path, filtered, sr)
            log(f"Output saved: {output_path}")
        except Exception as e:
            log(f"Failed to save audio: {e}", "ERROR")

    log("Processing complete ✅")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()