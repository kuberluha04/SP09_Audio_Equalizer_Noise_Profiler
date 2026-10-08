import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# ==============================
# 1. Load audio
# ==============================

audio_path = "audio/test_audio.wav"

signal, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)


# ==============================
# 2. Low-pass filter parameters
# ==============================

cutoff_frequency = 3000
filter_order = 4


# ==============================
# 3. Design low-pass filter
# ==============================

sos = butter(
    filter_order,
    cutoff_frequency,
    btype="lowpass",
    fs=sample_rate,
    output="sos"
)


# ==============================
# 4. Apply filter
# ==============================

filtered_signal = sosfiltfilt(
    sos,
    signal
)


# ==============================
# 5. Calculate FFT
# ==============================

N = len(signal)

original_fft = np.fft.rfft(signal)
filtered_fft = np.fft.rfft(filtered_signal)

frequencies = np.fft.rfftfreq(
    N,
    d=1 / sample_rate
)


# ==============================
# 6. Calculate magnitude
# ==============================

original_magnitude = np.abs(original_fft)
filtered_magnitude = np.abs(filtered_fft)


# ==============================
# 7. Plot FFT comparison
# ==============================

plt.figure(figsize=(12, 5))

plt.plot(
    frequencies,
    original_magnitude,
    label="Original"
)

plt.plot(
    frequencies,
    filtered_magnitude,
    label="Low-pass Filtered"
)

plt.axvline(
    cutoff_frequency,
    linestyle="--",
    label="Cutoff = 3000 Hz"
)

plt.title("Original vs Low-pass Filtered Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 10000)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()