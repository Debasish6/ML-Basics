import google.generativeai as genai
import pathlib

# Initialize a Gemini model appropriate for your use case.
model = genai.GenerativeModel('models/gemini-1.5-flash')

# Create the prompt.
prompt = "Please summarize the audio."

# Load the samplesmall.mp3 file into a Python Blob object containing the audio
# file's bytes and then pass the prompt and the audio to Gemini.
response = model.generate_content([
    prompt,
    {
        "mime_type": "audio/wav",
        "data": pathlib.Path('harvard.wav').read_bytes()
    }
])

# Output Gemini's response to the prompt and the inline audio.
print(response.text)