import librosa
import matplotlib.pyplot as plt

# Audio file
audio_path = "audio/test_audio.wav"

# Load audio
signal, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=False
)

# Take left channel
if signal.ndim == 2:
    signal = signal[0]

# Create time axis
time = [i / sample_rate for i in range(len(signal))]

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(time, signal)

plt.title("Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()