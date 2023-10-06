from flask import Flask, request, jsonify
from os import system
from EdgeGPT.EdgeUtils import Query
import speech_recognition as sr
import sys, whisper, warnings, time, openai
import pyttsx3
import asyncio
import threading
import warnings

# Your other imports and code here...

app = Flask(__name__)
# Wake word variables
BING_WAKE_WORD = "bing"
GPT_WAKE_WORD = "darwin"

# Initialize global variables here
listening_for_wake_word = True  # Initialize to True
bing_engine = True  # Initialize to True

# Initialize the OpenAI API
openai.api_key = ""

r = sr.Recognizer()
tiny_model = whisper.load_model('tiny')
base_model = whisper.load_model('base')
source = sr.Microphone()
warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe', lineno=114)

if sys.platform != 'darwin':
    engine = pyttsx3.init()


# Initialize a threading event
stop_speak_event = threading.Event()

# Initialize a variable to store the current speak thread
current_speak_thread = None

def speak(text):
    print("Before text-to-speech")

    def run_tts():
        try:
            if sys.platform == 'darwin':
                ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ")
                clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
                system(f"say '{clean_text}'")
            else:
                engine.say(text)
                engine.runAndWait()
        except Exception as e:
            print("Error in text-to-speech:", e)
        print("After text-to-speech")
        engine.stop()

    # Set the event to stop the previous speak thread
    stop_speak_event.set()

    # Create a new speak thread
    tts_thread = threading.Thread(target=run_tts)
    tts_thread.start()


def listen_for_wake_word(audio):
    global listening_for_wake_word
    global bing_engine
    try:
        with open("wake_detect.wav", "wb") as f:
            f.write(audio.get_wav_data())
        result = tiny_model.transcribe('wake_detect.wav')
        text_input = result['text']
        print("Detected wake word:", text_input)

        if BING_WAKE_WORD in text_input.lower().strip():
            print("Speak your prompt to Bing.")
            speak('Listening')
            print("Simplified prompt: Hello, test.")
            bing_engine = True
            listening_for_wake_word = False
        elif GPT_WAKE_WORD in text_input.lower().strip():
            print("Before speak('Listening')")
            speak('Listening')
            print("After speak('Listening')")
            bing_engine = False
            listening_for_wake_word = False
    except Exception as e:
        print("Error in listen_for_wake_word:", e)


def prompt_bing(audio):
    global listening_for_wake_word
    global bing_engine
    try:
        with open("prompt.wav", "wb") as f:
            f.write(audio.get_wav_data())
        result = base_model.transcribe('prompt.wav')
        prompt_text = result['text']
        print("User input:", prompt_text)
        if len(prompt_text.strip()) == 0:
            print("Empty prompt. Please speak again.")
            speak("Empty prompt. Please speak again.")
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)
            output = Query(prompt_text)
            print('Bing: ' + str(output))
            speak(str(output))
            print('\nSay Ok Bing or Ok GPT to wake me up. \n')
            bing_engine = True
            listening_for_wake_word = True

    except Exception as e:
        print("Prompt error: ", e)


def prompt_gpt(audio):
    global listening_for_wake_word
    global bing_engine
    try:
        with open("prompt.wav", "wb") as f:
            f.write(audio.get_wav_data())
        result = base_model.transcribe('prompt.wav')
        prompt_text = result['text']
        print("User input:", prompt_text)
        if len(prompt_text.strip()) == 0:
            print("Empty prompt. Please speak again.")
            speak("Empty prompt. Please speak again.")
            listening_for_wake_word = True
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content":
                        "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text},
                ],
                temperature=0.5,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                n=1,
                stop=["\nUser:"],
            )
            bot_response = response["choices"][0]["message"]["content"]
            print(":: ", bot_response)
            stop_speak_event.set()
            speak(bot_response)
            listening_for_wake_word = True
            bing_engine = True

    except Exception as e:
        print("Prompt error: ", e)


def callback(recognizer, audio):
    global listening_for_wake_word
    global bing_engine

    if listening_for_wake_word:
        listen_for_wake_word(audio)
    elif bing_engine:
        print("Using Bing engine.")
        prompt_bing(audio)
    elif not bing_engine:
        print("Using GPT engine.")
        prompt_gpt(audio)
async def async_main():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)
    print('\nSay Ok Bing or Ok GPT to wake me up. \n')
    r.listen_in_background(source, callback)
    while True:
        await asyncio.sleep(1)


@app.route('/wake-word', methods=['POST'])
def wake_word():
    global listening_for_wake_word
    global bing_engine
    if request.method == 'POST':
        data = request.get_json()
        audio_data = data.get('audio_data')

        if listening_for_wake_word:
            listen_for_wake_word(audio_data)
        elif bing_engine:
            print("Using Bing engine.")
            prompt_bing(audio_data)
        elif not bing_engine:
            print("Using GPT engine.")
            prompt_gpt(audio_data)

        return jsonify({"message": "Request processed"})

if __name__ == '__main__':
    asyncio.run(async_main())
