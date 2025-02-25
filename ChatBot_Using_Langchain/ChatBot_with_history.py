import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# Ensure the API key is loaded
api_key = os.getenv("GoogleAPIKey")
if not api_key:
    print("API Key not found. Please set it in the .env file.")
    sys.exit(1)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=api_key)

chat_history = []

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

system_instruction = os.getenv("system_instruction")
if not system_instruction:
    print("System instruction not found. Please set it in the .env file.")
    sys.exit(1)

chat_history.append(system_instruction)  # Appending system message to the chat history

try:
    while True:
        query = input("You: ")
        
        if query.lower() == 'bye':
            break
        chat_history.append(HumanMessage(content=query))

        # Get Response using History
        response = model.invoke(chat_history).content

        chat_history.append(AIMessage(content=response))

        print(f"AI Assistant: {response}")

except Exception as e:
    print(f"An error occurred: {e}")

print("-----Message History-----")
for message in chat_history:
    print(message)
