import torch
from transformers import pipeline
import librosa

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device
)
print("Model loaded successfully!")


file_name = "2025-2026_Projects_projects_voice2text_test1.ogg"
print(f"Loading and resampling '{file_name}' to 16kHz...")

try:
  audio_waveform, sampling_rate = librosa.load(file_name, sr=16000)
  print(f"File loaded successfully. Waveform shape: {audio_waveform.shape}, Sample Rate: {sampling_rate} Hz")
except Exception as e:
  print(f"Error loading audio file: {e}")


print("\nTranscribing audio...")
result = transcriber(inputs=audio_waveform, return_timestamps=True)

print("\n--- Transcription Result ---")
print(result["text"])