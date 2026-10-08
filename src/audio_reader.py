import librosa

audio_path = "audio/test_audio.wav"

signal, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=False
)

print("===== SP09 AUDIO INFORMATION =====")

print("Sample rate:", sample_rate, "Hz")
print("Signal shape:", signal.shape)

if signal.ndim == 1:
    channels = 1
    samples = signal.shape[0]
else:
    channels = signal.shape[0]
    samples = signal.shape[1]

duration = samples / sample_rate

print("Number of channels:", channels)
print("Number of samples per channel:", samples)
print("Duration:", duration, "seconds")