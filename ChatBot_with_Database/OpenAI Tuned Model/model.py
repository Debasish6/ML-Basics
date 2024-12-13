from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key =os.getenv("OpenAIKey"))

completion = client.chat.completions.create(
model="ft:gpt-3.5-turbo-0125:expanderp:tunedmodel:AdcIQNWq",
messages=[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is Open Balance Date?"}
],
  temperature=0,
  max_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
print(completion.choices[0].message)