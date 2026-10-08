import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz


# ==============================
# 1. Filter parameters
# ==============================

sample_rate = 48000
cutoff_frequency = 3000
filter_order = 4


# ==============================
# 2. Design low-pass filter
# ==============================

sos = butter(
    filter_order,
    cutoff_frequency,
    btype="lowpass",
    fs=sample_rate,
    output="sos"
)


# ==============================
# 3. Calculate frequency response
# ==============================

frequencies, response = sosfreqz(
    sos,
    worN=2000,
    fs=sample_rate
)


# ==============================
# 4. Convert response to dB
# ==============================

magnitude_db = 20 * np.log10(
    np.maximum(np.abs(response), 1e-10)
)


# ==============================
# 5. Plot frequency response
# ==============================

plt.figure(figsize=(12, 5))

plt.plot(
    frequencies,
    magnitude_db
)

plt.axvline(
    cutoff_frequency,
    linestyle="--",
    label="Cutoff = 3000 Hz"
)

plt.axhline(
    -3,
    linestyle=":",
    label="-3 dB"
)

plt.title("Frequency Response of Low-pass Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(0, 10000)
plt.ylim(-80, 5)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()