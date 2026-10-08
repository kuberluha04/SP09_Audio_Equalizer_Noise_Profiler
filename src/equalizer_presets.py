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

nyquist = sample_rate / 2
filter_order = 4


# ==============================
# 2. Define frequency bands
# ==============================

base_bands = [
    {"name": "Bass",      "low": 20,   "high": 250},
    {"name": "Low-mid",   "low": 250,  "high": 500},
    {"name": "Mid",       "low": 500,  "high": 2000},
    {"name": "High-mid",  "low": 2000, "high": 6000},
    {"name": "Treble",    "low": 6000, "high": 18000},
]


# ==============================
# 3. Define equalizer presets
# ==============================

presets = {

    "Flat": [0, 0, 0, 0, 0],

    "Bass Boost": [6, 3, 0, 0, 0],

    "Vocal": [-3, -1, 4, 3, 0],

    "Rock": [5, 2, 0, 3, 5],

    "Classical": [2, 0, 2, 1, 2]
}


# ==============================
# 4. Function to extract a band
# ==============================

def get_band_signal(signal, low, high):

    low_norm = max(low / nyquist, 0.001)
    high_norm = min(high / nyquist, 0.999)

    sos = butter(
        filter_order,
        [low_norm, high_norm],
        btype="bandpass",
        output="sos"
    )

    return sosfiltfilt(sos, signal)


# ==============================
# 5. Equalizer function
# ==============================

def apply_equalizer(signal, preset_name):

    gains = presets[preset_name]

    output = np.zeros_like(signal)

    for band, gain_db in zip(base_bands, gains):

        band_signal = get_band_signal(
            signal,
            band["low"],
            band["high"]
        )

        gain_linear = 10 ** (gain_db / 20)

        output += gain_linear * band_signal

    return output


# ==============================
# 6. Select preset
# ==============================

selected_preset = "Classical"

equalized_signal = apply_equalizer(
    signal,
    selected_preset
)


# ==============================
# 7. Normalize if necessary
# ==============================

max_amplitude = np.max(np.abs(equalized_signal))

if max_amplitude > 1:
    equalized_signal = equalized_signal / max_amplitude


# ==============================
# 8. Create time axis
# ==============================

time = np.arange(len(signal)) / sample_rate


# ==============================
# 9. Plot result
# ==============================

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)

plt.plot(time, signal)

plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.grid()


plt.subplot(2, 1, 2)

plt.plot(time, equalized_signal)

plt.title(
    f"Equalized Signal - {selected_preset}"
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.grid()

plt.tight_layout()
plt.show()


# ==============================
# 10. Print selected preset
# ==============================

print("===== EQUALIZER PRESET =====")
print("Selected preset:", selected_preset)

for band, gain in zip(base_bands, presets[selected_preset]):

    print(
        f"{band['name']} "
        f"({band['low']}-{band['high']} Hz): "
        f"{gain} dB"
    )

print("Maximum output amplitude:",
      np.max(np.abs(equalized_signal)))