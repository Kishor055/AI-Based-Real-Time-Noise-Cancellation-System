import numpy as np
import os

from utils.audio import load_audio
from utils.metrics import compute_snr, compute_mse
from filters.lms import lms_filter
from filters.nlms import nlms_filter
from filters.wiener import wiener_filter
from ml_model.model import denoise_model


# ---------------- TEST CONFIG ----------------
TEST_FILE = "data/clean/sample1.wav"


# ---------------- TEST AUDIO LOAD ----------------
def test_audio_loading():
    signal, sr = load_audio(TEST_FILE)

    assert signal is not None, "Signal is None"
    assert len(signal) > 0, "Signal is empty"
    assert sr > 0, "Invalid sample rate"

    print("✅ Audio loading test passed")


# ---------------- TEST LMS FILTER ----------------
def test_lms():
    signal, _ = load_audio(TEST_FILE)

    noise = np.random.randn(len(signal)) * 0.01
    noisy = signal + noise

    filtered = lms_filter(noisy, noise)

    assert len(filtered) == len(signal), "LMS output length mismatch"

    print("✅ LMS filter test passed")


# ---------------- TEST NLMS FILTER ----------------
def test_nlms():
    signal, _ = load_audio(TEST_FILE)

    noise = np.random.randn(len(signal)) * 0.01
    noisy = signal + noise

    filtered = nlms_filter(noisy, noise)

    assert len(filtered) == len(signal), "NLMS output length mismatch"

    print("✅ NLMS filter test passed")


# ---------------- TEST WIENER FILTER ----------------
def test_wiener():
    signal, _ = load_audio(TEST_FILE)

    noise = np.random.randn(len(signal)) * 0.01
    noisy = signal + noise

    filtered = wiener_filter(noisy)

    assert len(filtered) == len(signal), "Wiener output length mismatch"

    print("✅ Wiener filter test passed")


# ---------------- TEST ML MODEL ----------------
def test_ml_model():
    signal, _ = load_audio(TEST_FILE)

    noisy = signal + np.random.randn(len(signal)) * 0.01

    filtered = denoise_model(noisy)

    assert len(filtered) == len(signal), "ML output length mismatch"

    print("✅ ML model test passed")


# ---------------- TEST METRICS ----------------
def test_metrics():
    signal, _ = load_audio(TEST_FILE)

    noisy = signal + np.random.randn(len(signal)) * 0.01

    snr = compute_snr(signal, noisy)
    mse = compute_mse(signal, noisy)

    assert isinstance(snr, float), "SNR not float"
    assert isinstance(mse, float), "MSE not float"

    print(f"✅ Metrics test passed | SNR: {snr:.2f}, MSE: {mse:.6f}")


# ---------------- END-TO-END PIPELINE ----------------
def test_full_pipeline():
    signal, _ = load_audio(TEST_FILE)

    noise = np.random.randn(len(signal)) * 0.01
    noisy = signal + noise

    # Apply all filters
    lms_out = lms_filter(noisy, noise)
    nlms_out = nlms_filter(noisy, noise)
    wiener_out = wiener_filter(noisy)
    ml_out = denoise_model(noisy)

    # Compute metrics
    snr_lms = compute_snr(signal, lms_out)
    snr_nlms = compute_snr(signal, nlms_out)
    snr_wiener = compute_snr(signal, wiener_out)
    snr_ml = compute_snr(signal, ml_out)

    print("\n📊 SNR Comparison:")
    print(f"LMS    : {snr_lms:.2f} dB")
    print(f"NLMS   : {snr_nlms:.2f} dB")
    print(f"Wiener : {snr_wiener:.2f} dB")
    print(f"ML     : {snr_ml:.2f} dB")

    assert all(isinstance(x, float) for x in [snr_lms, snr_nlms, snr_wiener, snr_ml]), "Invalid SNR values"

    print("✅ Full pipeline test passed")


# ---------------- RUN ALL TESTS ----------------
if __name__ == "__main__":
    print("🚀 Running Tests...\n")

    test_audio_loading()
    test_lms()
    test_nlms()
    test_wiener()
    test_ml_model()
    test_metrics()
    test_full_pipeline()

    print("\n🎉 ALL TESTS PASSED")