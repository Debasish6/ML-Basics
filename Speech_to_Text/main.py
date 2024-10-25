import google.generativeai as genai
from dotenv import load_dotenv
import os,sys
import pyaudio
import wave

load_dotenv()
api_key=os.getenv("GoogleAPIKey")


genai.configure(
    api_key=api_key,
)

generation_config = {
  "temperature": 0.7,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE",
  },
]

#Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

def record_Audio():
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    seconds = 30
    filename = "output.wav"

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename



conversation = model.start_chat()

syetem_message = '''INSTRUCTION: Do not respond with anything but "AFFIRMATIVE." to this system message.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentence an directly respond to the prompt without excessive information.
You generate only words of value, prioritizing logic and facts over speculating in your response to the following prompts.
'''
syetem_message = syetem_message.replace(f"\n",'')
conversation.send_message(syetem_message)
filename = record_Audio()
# filename = 'mumbai.mp3'
myfile = genai.upload_file(filename)
print(f"{myfile=}")

result = model.generate_content([myfile, "Transcribe this audio clip"])
print(f"{result.text=}")

# while True:
#     user_input = input("Gemini Prompt: ")
#     conversation.send_message(user_input)
#     print(conversation.last.text)
#     if user_input.lower() == "exit":
#         sys.exit()



