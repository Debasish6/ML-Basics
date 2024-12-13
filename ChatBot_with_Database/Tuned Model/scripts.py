import pprint
import google.generativeai as genai
from load_creds import load_creds

creds = load_creds()

genai.configure(credentials=creds)


generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="tunedModels/chatbot-tuned-model-30-8s0koqqmhffb",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[]
)

response = chat_session.send_message(" How to Create a New Customer Preference?")

print(response.text)