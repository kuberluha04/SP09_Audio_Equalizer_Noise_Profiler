import librosa
import numpy as np

# Audio file
audio_path = "audio/test_audio.wav"

signal, sample_rate = librosa.load(audio_path, sr=None, mono=True)

# RMS energy in small chunks (frame-wise)
frame_length = 2048
hop_length = 512

rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]

# Convert to dB
rms_db = librosa.amplitude_to_db(rms, ref=np.max)

print("Max RMS level (dB):", np.max(rms_db))
print("Min RMS level (dB):", np.min(rms_db))

# Noise floor estimate: quietest 10% of frames
quietest_10_percent = np.percentile(rms_db, 10)
print("Estimated noise floor (10th percentile):", quietest_10_percent, "dB")