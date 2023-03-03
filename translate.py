from deep_translator import GoogleTranslator as Translator
import whisper as Whisper
import os
import torch
import pyaudio
import wave
from pynput.keyboard import Key, Listener
from datetime import datetime

def on_key_press(key):
    if key is not Key.home: return
    print("[DEBUG] Backspace pressed") #REMOVE ME
def on_key_release(key):
    clock_start = datetime.now()
    
    if key is not Key.home: return
    print("[DEBUG] Backspace released") #REMOVE ME
    # Make sure to run on GPU if avalible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get the path of audio file (DEBUG)
    audio_path = os.path.join(os.path.dirname(__file__), "Destiny.mp3")
    # Use Whisper (OPENAI) to transcribe from audiofile
    text_model = Whisper.load_model("base").to(device)
    whisper_result = text_model.transcribe(audio_path, language="en", fp16=False)
    # The 'transcribe' function has much more info, we just need text for translation
    whisper_text = whisper_result["text"]
    print(f"English:  {whisper_text}")

    # use MyMemory as a translator. (source=, target=)
    translate_output = Translator("english","japanese").translate(whisper_text)
    print(f"Japanese: {translate_output}")

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