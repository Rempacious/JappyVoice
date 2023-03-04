from deep_translator import GoogleTranslator as Translator
import whisper as Whisper
import os
import torch
import pyaudio
import wave
from pynput.keyboard import Key, Listener
from datetime import datetime

# PyAudio stream variables
chunk = 2048  # Record in chunks of 2048 samples
sample_format = pyaudio.paInt32  # 32 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "tmp_voice_file.wav"

pyaudioIF = pyaudio.PyAudio() # Create PortAudio Interface

voice_stream = pyaudioIF.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)
audio_frames = []

def on_key_press(key):
    if key is not Key.backspace: return
    audio_frames.append(voice_stream.read(chunk))

def on_key_release(key):
    # Stop recording
    voice_stream.stop_stream()

    # Open and save recording to .wave file for whisper
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(pyaudioIF.get_sample_size(sample_format))
    wave_file.setframerate(fs)
    wave_file.writeframes(b''.join(audio_frames))
    wave_file.close()
    clock_start = datetime.now()
    
    if key is not Key.backspace: return
    # Make sure to run on GPU if avalible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get the path of audio file (DEBUG)
    audio_path = os.path.join(os.path.dirname(__file__), filename)
    # Use Whisper (OPENAI) to transcribe from audiofile
    text_model = Whisper.load_model("base").to(device)
    whisper_result = text_model.transcribe(audio_path, language="en", fp16=False)
    # The 'transcribe' function has much more info, we just need text for translation
    whisper_text = whisper_result["text"]
    print(f"English:  {whisper_text}")

    # use MyMemory as a translator. (source=, target=)
    translate_output = Translator("english","japanese").translate(whisper_text)
    print(f"Japanese: {translate_output}")
    direct_translate_output = Translator("japanese","english").translate(translate_output)
    print(f"Direct  : {direct_translate_output}")

    # See how long the translation took
    clock_end = datetime.now()
    clock_delta = clock_end - clock_start
    clock_delta = round(clock_delta.total_seconds() * 1000, 3)
    print(f"ProcTime: {(clock_delta)}ms")

while True:
    with Listener(
        on_press=on_key_press,
        on_release=on_key_release) as listener:
        listener.join()