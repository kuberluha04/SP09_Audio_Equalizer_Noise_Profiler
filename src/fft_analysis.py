import librosa
import numpy as np
import matplotlib.pyplot as plt

# Audio file
audio_path = "audio/test_audio.wav"

# Load audio
signal, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

# Number of samples
N = len(signal)

# Apply FFT
fft_result = np.fft.rfft(signal)

# Magnitude
magnitude = np.abs(fft_result)

# Frequency axis
frequencies = np.fft.rfftfreq(N, d=1/sample_rate)

# Plot
plt.figure(figsize=(12, 5))

plt.plot(frequencies, magnitude)

plt.title("Frequency Spectrum of Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, sample_rate / 2)

plt.grid()

plt.tight_layout()
plt.show()