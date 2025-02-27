import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

class Chatbot:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Ensure the API key is loaded
        self.api_key = os.getenv("GoogleAPIKey")
        if not self.api_key:
            raise ValueError("API Key not found. Please set it in the .env file.")
        
        self.model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=self.api_key)
        self.chat_history = []
        
        # Load system instruction
        self.system_instruction = os.getenv("system_instruction")
        if not self.system_instruction:
            raise ValueError("System instruction not found. Please set it in the .env file.")
        
        # Add system instruction to the history
        self.chat_history.append(SystemMessage(content=self.system_instruction))

    def get_response(self, query):
        """Process the query and get a response from the model."""
        if query.lower() == 'bye':
            return "Goodbye! Have a nice day!"

        # Add the human query to the chat history
        self.chat_history.append(HumanMessage(content=query))

        # Get the response using the chat history
        response = self.model.invoke(self.chat_history).content

        # Append AI response to chat history
        self.chat_history.append(AIMessage(content=response))

        return response

