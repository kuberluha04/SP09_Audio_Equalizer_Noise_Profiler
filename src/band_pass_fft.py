import matplotlib
matplotlib.use('TkAgg')

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Audio file
audio_path = "audio/test_audio.wav"

# Load audio (mono)
signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)

# Band-pass settings (same as band_pass_filter.py)
low_cutoff = 500
high_cutoff = 3000
order = 4

nyquist = sample_rate / 2
low_norm = low_cutoff / nyquist
high_norm = high_cutoff / nyquist

# Design and apply band-pass filter
b, a = butter(order, [low_norm, high_norm], btype='band', analog=False)
filtered_signal = filtfilt(b, a, signal)

# ---- FFT of original signal ----
N = len(signal)
fft_original = np.fft.rfft(signal)
magnitude_original = np.abs(fft_original)
frequencies = np.fft.rfftfreq(N, d=1/sample_rate)

# ---- FFT of filtered signal ----
fft_filtered = np.fft.rfft(filtered_signal)
magnitude_filtered = np.abs(fft_filtered)

# ---- Plot comparison ----
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(frequencies, magnitude_original)
plt.title("Original Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sample_rate / 2)
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(frequencies, magnitude_filtered, color='green')
plt.title(f"Band-Pass Filtered Spectrum ({low_cutoff}-{high_cutoff} Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sample_rate / 2)
plt.grid()

plt.tight_layout()
plt.show()