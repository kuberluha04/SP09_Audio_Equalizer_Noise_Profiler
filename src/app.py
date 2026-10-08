import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import soundfile as sf
import io

st.set_page_config(page_title="SP09 Audio Equalizer & Noise Profiler", layout="wide")

st.title("🎚️ Audio Equalizer & Noise Profiler")
st.write("Upload an audio file to analyze and process it.")

# ---- Equalizer setup ----
band_ranges = [
    {"name": "Bass",     "low": 20,   "high": 250},
    {"name": "Low-mid",  "low": 250,  "high": 500},
    {"name": "Mid",      "low": 500,  "high": 2000},
    {"name": "High-mid", "low": 2000, "high": 6000},
    {"name": "Treble",   "low": 6000, "high": 18000},
]

presets = {
    "Flat":       [0, 0, 0, 0, 0],
    "Bass Boost": [8, 4, 0, 0, 0],
    "Vocal":      [-3, 0, 5, 3, -1],
    "Rock":       [5, 2, -2, 3, 4],
    "Classical":  [2, 0, 0, 0, 3],
}

def get_band_signal(signal, low, high, order, nyquist):
    low_norm = max(low / nyquist, 0.001)
    high_norm = min(high / nyquist, 0.999)
    sos = butter(order, [low_norm, high_norm], btype='band', analog=False, output='sos')
    return sosfiltfilt(sos, signal)

def apply_equalizer(signal, preset_name, nyquist, order=4):
    gains_db = presets[preset_name]
    output_signal = np.zeros_like(signal)
    for band, gain_db in zip(band_ranges, gains_db):
        band_signal = get_band_signal(signal, band["low"], band["high"], order, nyquist)
        gain_linear = 10 ** (gain_db / 20)
        output_signal += gain_linear * band_signal
    return output_signal

# ---- Noise profiler + reduction functions ----
def estimate_noise_profile(signal, sample_rate, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    noise_floor_db = np.percentile(rms_db, 10)
    quiet_frame_indices = np.where(rms_db <= noise_floor_db)[0]

    noise_samples = []
    for idx in quiet_frame_indices:
        start = idx * hop_length
        end = start + frame_length
        if end <= len(signal):
            noise_samples.extend(signal[start:end])
    noise_samples = np.array(noise_samples)

    N = len(noise_samples)
    fft_noise = np.fft.rfft(noise_samples)
    magnitude_noise = np.abs(fft_noise)
    frequencies = np.fft.rfftfreq(N, d=1/sample_rate)
    dominant_freq = frequencies[np.argmax(magnitude_noise)]

    return noise_floor_db, dominant_freq, quiet_frame_indices

def reduce_noise(signal, sample_rate, quiet_frame_indices, frame_length=2048, hop_length=512, oversubtraction_factor=1.5):
    stft_signal = librosa.stft(signal, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft_signal)
    phase = np.angle(stft_signal)

    noise_magnitude_frames = magnitude[:, quiet_frame_indices]
    noise_spectrum_estimate = np.mean(noise_magnitude_frames, axis=1, keepdims=True)

    clean_magnitude = magnitude - (oversubtraction_factor * noise_spectrum_estimate)
    clean_magnitude = np.maximum(clean_magnitude, 0)

    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_signal = librosa.istft(clean_stft, hop_length=hop_length)
    return clean_signal

# ---- Plotting helper functions ----
def plot_waveform(signal, sample_rate, title, color='tab:blue'):
    fig, ax = plt.subplots(figsize=(10, 3))
    time = np.arange(len(signal)) / sample_rate
    ax.plot(time, signal, color=color)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid()
    return fig

def plot_fft(signal, sample_rate, title, color='tab:blue'):
    N = len(signal)
    fft_result = np.fft.rfft(signal)
    magnitude = np.abs(fft_result)
    frequencies = np.fft.rfftfreq(N, d=1/sample_rate)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(frequencies, magnitude, color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim(0, sample_rate / 2)
    ax.grid()
    return fig

# ---- File upload ----
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    signal, sample_rate = librosa.load(uploaded_file, sr=None, mono=True)
    nyquist = sample_rate / 2

    st.success(f"Loaded audio: {len(signal)/sample_rate:.2f} seconds, {sample_rate} Hz")

    st.subheader("Original Audio")
    st.audio(uploaded_file)
    st.pyplot(plot_waveform(signal, sample_rate, "Original Waveform"))
    st.pyplot(plot_fft(signal, sample_rate, "Original Frequency Spectrum"))

    # ---- Equalizer section ----
    st.subheader("🎚️ Equalizer")
    selected_preset = st.selectbox("Choose a preset", list(presets.keys()))

    if st.button("Apply Equalizer"):
        eq_signal = apply_equalizer(signal, selected_preset, nyquist)

        max_val = np.max(np.abs(eq_signal))
        if max_val > 1.0:
            eq_signal = eq_signal / max_val

        buffer = io.BytesIO()
        sf.write(buffer, eq_signal, sample_rate, format='WAV')
        buffer.seek(0)

        st.subheader(f"Equalized Audio (Preset: {selected_preset})")
        st.audio(buffer)
        st.pyplot(plot_waveform(eq_signal, sample_rate, f"Equalized Waveform ({selected_preset})", color='purple'))
        st.pyplot(plot_fft(eq_signal, sample_rate, f"Equalized Frequency Spectrum ({selected_preset})", color='purple'))

    # ---- Noise Profiler + Reduction section ----
    st.subheader("🔍 Noise Profiler & Reduction")

    if st.button("Analyze Noise"):
        noise_floor_db, dominant_freq, quiet_frame_indices = estimate_noise_profile(signal, sample_rate)
        st.write(f"**Estimated noise floor:** {noise_floor_db:.2f} dB")
        st.write(f"**Dominant noise frequency:** {dominant_freq:.2f} Hz")

    if st.button("Apply Noise Reduction"):
        noise_floor_db, dominant_freq, quiet_frame_indices = estimate_noise_profile(signal, sample_rate)
        clean_signal = reduce_noise(signal, sample_rate, quiet_frame_indices)

        max_val = np.max(np.abs(clean_signal))
        if max_val > 1.0:
            clean_signal = clean_signal / max_val

        buffer2 = io.BytesIO()
        sf.write(buffer2, clean_signal, sample_rate, format='WAV')
        buffer2.seek(0)

        st.subheader("Denoised Audio")
        st.audio(buffer2)
        st.pyplot(plot_waveform(clean_signal, sample_rate, "Denoised Waveform", color='green'))
        st.pyplot(plot_fft(clean_signal, sample_rate, "Denoised Frequency Spectrum", color='green'))

        st.download_button(
            label="Download Denoised Audio",
            data=buffer2,
            file_name="denoised_audio.wav",
            mime="audio/wav"
        )

else:
    st.info("👆 Please upload an audio file to begin.")

