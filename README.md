# SP09 - Audio Equalizer & Noise Profiler   (SP09 is project code  SP stands for signal processing )

IIT Madras Signal Processing project.
A web app (built with Streamlit) where you can upload an audio file, see its waveform and frequency spectrum, 
apply one of 5 equalizer presets (Flat, Bass Boost, Vocal, Rock, Classical), check the noise level, and reduce noise using spectral subtraction.

## How to run

1. Install the libraries:
   pip install -r requirements.txt

2. Start the app:
   streamlit run src/app.py

3. Upload an audio file from the `audio` folder (test_audio.wav).

## Folders

- `src` - all the Python code
- `audio` - test audio file
- `results` - saved output audio
