from celery import shared_task
from .models import ChatSession
from .main import ChatBot
import json
import os

@shared_task
def process_chatbot_request(session_id, prompt):
    chat_session = ChatSession.objects.get(session_id=session_id)
    chat_history = chat_session.chat_history

    chatbot = ChatBot(api_key=os.getenv("GoogleAPIKey"))
    chatbot.start_conversation()
    chatbot.previous_db_results = chat_history

    try:
        response = chatbot.send_prompts(prompt, chat_history)
        return json.loads(response)['text']
    except Exception as e:
        print(f'Error processing request: {e}')
        return f'An error occurred while processing your request. Please try again later.'