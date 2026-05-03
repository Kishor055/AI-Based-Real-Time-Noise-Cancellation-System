import numpy as np

from utils.config import load_config
from utils.audio import load_audio, save_audio
from utils.metrics import evaluate_all

from filters.lms import lms_filter
from filters.nlms import nlms_filter
from filters.wiener import wiener_filter
from ml_model.model import denoise_model


# ---------------- FILTER SELECTOR ----------------
def process_audio(signal, config):
    mode = config["filters"]["active"]

    # Fake noise reference (for testing only)
    noise_ref = np.random.randn(len(signal)) * 0.01

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
        print("⚠️ Unknown mode, returning original signal")
        return signal


# ---------------- MAIN ----------------
def main():
    print("🚀 Starting AI Noise Cancellation System...\n")

    # Load config
    config = load_config()

    # Load audio
    input_path = config["audio"]["input"]["file"]
    signal, sr = load_audio(input_path)

    print(f"📂 Loaded: {input_path}")
    print(f"🔊 Sample Rate: {sr}")
    print(f"📏 Signal Length: {len(signal)}\n")

    # Add synthetic noise (for evaluation only)
    noise = np.random.randn(len(signal)) * 0.01
    noisy_signal = signal + noise

    # Process
    filtered = process_audio(noisy_signal, config)

    # Metrics
    if config["metrics"]["enabled"]:
        results = evaluate_all(signal, filtered)

        print("📊 Performance Metrics:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        print()

    # Save output
    if config["output"]["save_audio"]:
        output_path = config["audio"]["output"]["file"]
        save_audio(output_path, filtered, sr)
        print(f"💾 Saved output to: {output_path}")

    print("\n✅ Processing complete!")


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()