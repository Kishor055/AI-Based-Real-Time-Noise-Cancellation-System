import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy.signal import resample

# ---------------- LOAD AUDIO ----------------

def load_audio(path, target_sr=None):
    """
    Load audio file

    Returns:
        signal (numpy array), sample_rate
    """
    signal, sr = sf.read(path)

    # Convert to mono
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # Normalize
    signal = normalize_audio(signal)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        signal = resample_audio(signal, sr, target_sr)
        sr = target_sr

    return signal, sr


# ---------------- SAVE AUDIO ----------------

def save_audio(path, signal, sr):
    """
    Save audio to file
    """
    signal = np.clip(signal, -1.0, 1.0)
    sf.write(path, signal, sr)


# ---------------- NORMALIZATION ----------------

def normalize_audio(signal):
    """
    Normalize signal to [-1, 1]
    """
    max_val = np.max(np.abs(signal)) + 1e-8
    return signal / max_val


# ---------------- RESAMPLING ----------------

def resample_audio(signal, original_sr, target_sr):
    """
    Resample signal
    """
    duration = len(signal) / original_sr
    target_length = int(duration * target_sr)
    return resample(signal, target_length)


# ---------------- CHUNKING ----------------

def split_chunks(signal, chunk_size=1024):
    """
    Split signal into fixed-size chunks
    """
    chunks = []

    for i in range(0, len(signal), chunk_size):
        chunk = signal[i:i+chunk_size]

        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        chunks.append(chunk)

    return np.array(chunks)


def merge_chunks(chunks, original_length):
    """
    Merge chunks back into signal
    """
    signal = np.concatenate(chunks)
    return signal[:original_length]


# ---------------- PLAYBACK ----------------

def play_audio(signal, sr):
    """
    Play audio signal
    """
    sd.play(signal, sr)
    sd.wait()


# ---------------- RECORD AUDIO ----------------

def record_audio(duration=5, sr=16000):
    """
    Record audio from microphone

    Returns:
        signal (numpy array)
    """
    print("🎤 Recording...")

    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()

    print("✅ Recording complete")

    return recording.flatten()


# ---------------- REAL-TIME STREAM ----------------

def stream_audio(callback, sr=16000, block_size=1024):
    """
    Real-time audio streaming

    callback: function that processes audio chunks
    """

    def audio_callback(indata, frames, time, status):
        chunk = indata[:, 0]
        processed = callback(chunk)
        sd.play(processed, sr)

    stream = sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=sr,
                            blocksize=block_size)

    stream.start()
    return stream
