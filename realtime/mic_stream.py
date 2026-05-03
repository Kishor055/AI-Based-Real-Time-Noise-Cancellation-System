import sounddevice as sd
import queue

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    q.put(indata[:, 0])

def start_stream():
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
    stream.start()
    return stream, q
