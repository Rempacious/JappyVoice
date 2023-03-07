from deep_translator import GoogleTranslator as Translator
import whisper as Whisper
import os
import torch
import pyaudio
import wave
import keyboard as kb
import time
import voicevox as vvx
import asyncio
import winsound

# PyAudio stream variables
chunk = 2048  # Record in chunks of 2048 samples
sample_format = pyaudio.paInt16  # 32 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
temp_filename = "tmp_rec_file.wav"

pyaudioIF = pyaudio.PyAudio() # Create PortAudio Interface

def func_timer(func):
    def wrapper():
        start = time.time()
        func()
        end = time.time()
        print(f"Function '{func.__name__}' took {(end-start)*1000}ms to complete.")
    return wrapper

def init_recorder():
    global voice_stream
    global audio_frames

    voice_stream = pyaudioIF.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)
    audio_frames = []

async def gen_speech(jp_text):
    async with vvx.Client() as client:
        audio_query = await client.create_audio_query(
            jp_text, speaker=8
        )
        with open("voice.wav", "wb") as f:
            f.write(await audio_query.synthesis())

def record_voice():
    audio_frames.append(voice_stream.read(chunk))
    print(f"Frame count: {len(audio_frames)}")

@func_timer
def process_voice():
    # Stop recording
    voice_stream.stop_stream()
    voice_stream.close()

    # Open and save recording to .wave file for whisper
    wave_file = wave.open(temp_filename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(pyaudioIF.get_sample_size(sample_format))
    wave_file.setframerate(fs)
    wave_file.writeframes(b''.join(audio_frames))
    wave_file.close()
    
    # Make sure to run on GPU if avalible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Get the path of audio file (DEBUG)
    audio_path = os.path.join(os.path.dirname(__file__), temp_filename)
    # Use Whisper (OPENAI) to transcribe from audiofile
    text_model = Whisper.load_model("base").to(device)
    whisper_result = text_model.transcribe(audio_path, language="en", fp16=False)
    # The 'transcribe' function has much more info, we just need text for translation
    eng_text = whisper_result["text"]
    print(f"English:  {eng_text}")

    # Counts and limits punctuations. VoiceVox does not allow over 7. (NOT YET)
    # Converts eng -> jp for VoiceVox; Converts eng -> jp -> eng for understanding what
    # the translator came up with.
    jp_text = Translator("english","japanese").translate(eng_text)
    print(f"Japanese: {jp_text}")
    direct_jp_text = Translator("japanese","english").translate(jp_text)
    print(f"Direct  : {direct_jp_text}")

    # Generate speech .wave file using VoiceVox
    try:
        asyncio.run(gen_speech(jp_text))
    except:
        print("Speech processing failed.")
    init_recorder()

def main():
    print("Program Start")
    init_recorder()
    print("Finished PyAudio initialization.")
    while True:
        # Event = any key press; KEY_DOWN = press; KEY_UP = release
        event = kb.read_event()
        if event.event_type == kb.KEY_DOWN and event.name == 'backspace':
            record_voice()
        if event.event_type == kb.KEY_UP and event.name == 'backspace':
            print("Beginning processing.")
            process_voice()
            print("Finished processing.")
            try:
                winsound.PlaySound(vvx_filename, winsound.SND_FILENAME)
            except:
                print("Speech playback failed.")

if __name__ == "__main__":
    main()