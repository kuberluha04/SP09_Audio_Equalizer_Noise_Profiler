import matplotlib
matplotlib.use('TkAgg')

import librosa
import numpy as np
import matplotlib.pyplot as plt

# Audio file
audio_path = "audio/test_audio.wav"
signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)

# ---- Step 1: Frame-wise RMS to find quiet frames ----
frame_length = 2048
hop_length = 512

rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
rms_db = librosa.amplitude_to_db(rms, ref=np.max)

noise_floor_db = np.percentile(rms_db, 10)
print(f"Estimated noise floor: {noise_floor_db:.2f} dB")

# ---- Step 2: Find frame indices that are "quiet" (likely noise-only) ----
quiet_frame_indices = np.where(rms_db <= noise_floor_db)[0]
print(f"Number of quiet frames found: {len(quiet_frame_indices)}")

# ---- Step 3: Extract actual audio samples from quiet frames ----
noise_samples = []
for idx in quiet_frame_indices:
    start = idx * hop_length
    end = start + frame_length
    if end <= len(signal):
        noise_samples.extend(signal[start:end])

noise_samples = np.array(noise_samples)
print(f"Total noise samples collected: {len(noise_samples)}")

# ---- Step 4: FFT of the collected "noise-only" audio ----
N = len(noise_samples)
fft_noise = np.fft.rfft(noise_samples)
magnitude_noise = np.abs(fft_noise)
frequencies = np.fft.rfftfreq(N, d=1/sample_rate)

# ---- Step 5: Find dominant noise frequency ----
dominant_freq_index = np.argmax(magnitude_noise)
dominant_freq = frequencies[dominant_freq_index]
print(f"Dominant noise frequency: {dominant_freq:.2f} Hz")

# ---- Plot noise spectrum ----
plt.figure(figsize=(12, 5))
plt.plot(frequencies, magnitude_noise)
plt.axvline(dominant_freq, color='red', linestyle='--', label=f'Dominant: {dominant_freq:.0f} Hz')
plt.title("Estimated Noise Spectrum (from quiet segments)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sample_rate / 2)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ---- Summary ----
print("\n===== NOISE PROFILE SUMMARY =====")
print(f"Estimated noise floor: {noise_floor_db:.2f} dB")
print(f"Dominant noise frequency: {dominant_freq:.2f} Hz")