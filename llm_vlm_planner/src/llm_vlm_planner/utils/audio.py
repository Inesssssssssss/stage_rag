import wave
import pyaudio
import threading
#from pynput import keyboard
import ollama
import numpy as np
from faster_whisper import WhisperModel
import time

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
FILENAME = "recorded_audio.wav"

# Shared state
is_recording = False
stop_recording = False
frames = []

def reset_audio_state():
    global is_recording, stop_recording, frames
    is_recording = False
    stop_recording = False
    frames = []


def on_press(key):
    ''' Gère les événements de pression de touche. '''
    global is_recording, stop_recording, frames

    try:
        if key.char == 'b':
            is_recording = not is_recording
            if is_recording:
                print("Début de l'enregistrement.")
                frames = []
            else:
                print("Fin de l'enregistrement.")
                stop_recording = True  # Indique à la boucle d'arrêter
    except AttributeError:
        pass  # Ignore les touches spéciales

def start_listener():
    ''' Démarre un listener pour les événements du clavier. '''
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def record_audio():
    ''' Enregistre l'audio à partir du microphone et le sauvegarde dans un fichier WAV. '''
    global is_recording, stop_recording, frames

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Appuie sur 'b' pour démarrer/arrêter l'enregistrement (Ctrl+C pour quitter)")

    try:
        while not stop_recording:
            if is_recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
    except KeyboardInterrupt:
        print("Interruption manuelle.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

        if frames:
            wf = wave.open(FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            print(f"Fichier sauvegardé : {FILENAME}")
            return FILENAME
        else:
            print("Aucun enregistrement effectué.")
            return None

def getTranscript(audioPath, model_size):
    '''
    Fait la transcription du fichier audio donnée
    
    Args:
        audioPath : str - path to audio file (m4a or mp3)
        model_size : str - name of the model used to process the audio files
        record : bool - if True it writes the data into a file, else just prints it
    
    Returns:
        transcript : str - transcription of the audio
    
    '''

    # model_size = "large-v3"

    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    start = time.time()
    segments, info = model.transcribe(audioPath, beam_size=5)
    end1 = time.time()
    transcribe_t = end1 - start

    transcript = ""
    for segment in segments:
        # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcript += segment.text

    end2 = time.time()
    total_t = end2 - start
    
    print("Transcription of audio took : %.2fs" % (total_t))

    return transcript