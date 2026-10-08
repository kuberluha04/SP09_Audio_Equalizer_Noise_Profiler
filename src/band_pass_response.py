import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz

# Band-pass settings (same as band_pass_filter.py)
low_cutoff = 500
high_cutoff = 3000
order = 4

sample_rate = 48000
nyquist = sample_rate / 2
low_norm = low_cutoff / nyquist
high_norm = high_cutoff / nyquist

# Design filter
b, a = butter(order, [low_norm, high_norm], btype='band', analog=False)

# Compute frequency response
w, h = freqz(b, a, worN=8000, fs=sample_rate)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(w, abs(h))
plt.title(f"Band-Pass Filter Frequency Response ({low_cutoff}-{high_cutoff} Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain")
plt.xlim(0, 8000)
plt.grid()
plt.tight_layout()
plt.show()