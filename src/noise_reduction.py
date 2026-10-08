import matplotlib
matplotlib.use('TkAgg')

import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Audio file
audio_path = "audio/test_audio.wav"
signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)

# ---- Step 1: Find quiet frames (same as noise_profiler.py) ----
frame_length = 2048
hop_length = 512

rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
rms_db = librosa.amplitude_to_db(rms, ref=np.max)
noise_floor_db = np.percentile(rms_db, 10)
quiet_frame_indices = np.where(rms_db <= noise_floor_db)[0]

# ---- Step 2: Compute STFT of full signal ----
stft_signal = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
magnitude = np.abs(stft_signal)
phase = np.angle(stft_signal)

# ---- Step 3: Estimate average noise spectrum from quiet frames ----
noise_magnitude_frames = magnitude[:, quiet_frame_indices]
noise_spectrum_estimate = np.mean(noise_magnitude_frames, axis=1, keepdims=True)

print("Noise spectrum estimated from", len(quiet_frame_indices), "quiet frames")

# ---- Step 4: Spectral Subtraction ----
oversubtraction_factor = 1.5   # controls how aggressively we subtract
clean_magnitude = magnitude - (oversubtraction_factor * noise_spectrum_estimate)

# Prevent negative magnitudes
clean_magnitude = np.maximum(clean_magnitude, 0)

# ---- Step 5: Reconstruct signal (magnitude + original phase) ----
clean_stft = clean_magnitude * np.exp(1j * phase)
clean_signal = librosa.istft(clean_stft, hop_length=hop_length)

# ---- Save output ----
sf.write("results/denoised_audio.wav", clean_signal, sample_rate)
print("Denoised audio saved to results/denoised_audio.wav")

# ---- Plot comparison ----
time_original = np.arange(len(signal)) / sample_rate
time_clean = np.arange(len(clean_signal)) / sample_rate

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_original, signal)
plt.title("Original (Noisy) Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_clean, clean_signal, color='green')
plt.title("Denoised Signal (Spectral Subtraction)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()