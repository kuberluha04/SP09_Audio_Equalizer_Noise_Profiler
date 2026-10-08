import matplotlib
print("Current backend:", matplotlib.get_backend())
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Audio file
audio_path = "audio/test_audio.wav"

# Load audio (mono)
signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)

# Band-pass settings
low_cutoff = 500      # lower bound in Hz
high_cutoff = 3000    # upper bound in Hz
order = 4

nyquist = sample_rate / 2

# Normalize both cutoffs
low_norm = low_cutoff / nyquist
high_norm = high_cutoff / nyquist

# Design band-pass filter
b, a = butter(order, [low_norm, high_norm], btype='band', analog=False)

# Apply filter
filtered_signal = filtfilt(b, a, signal)

# Plot comparison
time = np.arange(len(signal)) / sample_rate

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, filtered_signal, color='green')
plt.title(f"Band-Pass Filtered Signal ({low_cutoff}-{high_cutoff} Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()