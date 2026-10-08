import matplotlib
matplotlib.use('TkAgg')

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt
print("SCRIPT STARTED")

# Audio file
audio_path = "audio/test_audio.wav"

# Load audio (mono)
signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)
nyquist = sample_rate / 2
order = 4

# ---- Define 5 bands (low, high, gain_db) ----
bands = [
    {"name": "Bass",      "low": 20,   "high": 250,   "gain_db": 6},
    {"name": "Low-mid",   "low": 250,  "high": 500,   "gain_db": 0},
    {"name": "Mid",       "low": 500,  "high": 2000,  "gain_db": 0},
    {"name": "High-mid",  "low": 2000, "high": 6000,  "gain_db": 0},
    {"name": "Treble",    "low": 6000, "high": 18000, "gain_db": 0},
]

def get_band_signal(signal, low, high, order, nyquist):
    low_norm = max(low / nyquist, 0.001)   # safety floor above 0
    high_norm = min(high / nyquist, 0.999)  # safety cap below 1.0
    sos = butter(order, [low_norm, high_norm], btype='band', analog=False, output='sos')
    return sosfiltfilt(sos, signal)

# ---- Build output by summing gained bands ----
output_signal = np.zeros_like(signal)

for band in bands:
    band_signal = get_band_signal(signal, band["low"], band["high"], order, nyquist)

    # Diagnostic check
    if np.any(np.isnan(band_signal)):
        print(f"⚠️ WARNING: {band['name']} band has NaN values!")
    if np.any(np.isinf(band_signal)):
        print(f"⚠️ WARNING: {band['name']} band has Infinity values!")

    gain_linear = 10 ** (band["gain_db"] / 20)
    output_signal += gain_linear * band_signal
    print(f"{band['name']} ({band['low']}-{band['high']} Hz): gain = {band['gain_db']} dB, band max = {np.max(np.abs(band_signal)):.4f}")

# Final diagnostic on combined output
print("Output signal min:", np.min(output_signal))
print("Output signal max:", np.max(output_signal))
print("Any NaN in output?", np.any(np.isnan(output_signal)))
print("Any Inf in output?", np.any(np.isinf(output_signal)))

# ---- Plot comparison ----
time = np.arange(len(signal)) / sample_rate

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, output_signal, color='purple')
plt.title("Equalized Signal (Bass +6dB boost)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

print("SCRIPT FINISHED")