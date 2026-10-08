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
# 2. Define filter parameters
# ==============================

cutoff_frequency = 3000  # Hz
filter_order = 4


# ==============================
# 3. Design high-pass filter
# ==============================

sos = butter(
    filter_order,
    cutoff_frequency,
    btype="highpass",
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
# 5. Create time axis
# ==============================

time = np.arange(len(signal)) / sample_rate


# ==============================
# 6. Plot original and filtered
# ==============================

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    signal,
    label="Original"
)

plt.plot(
    time,
    filtered_signal,
    label="High-pass Filtered"
)

plt.title("Original vs High-pass Filtered Audio")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.xlim(0, 5)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()