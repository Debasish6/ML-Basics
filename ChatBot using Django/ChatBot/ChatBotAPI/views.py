from django.shortcuts import render, HttpResponse
from django.contrib.sessions.models import Session
from .models import ChatSession
from .tasks import process_chatbot_request
from .main import ChatBot
from dotenv import load_dotenv
import os, sys, json

load_dotenv()

def home(request):
    return HttpResponse("<h1>Welcome to Our ChatBot Web Application</h1>")

def chatbot(request):
    if request.method == 'POST':
        prompt = request.POST['prompttext']

        # Get current session or create a new one
        session_id = request.session.session_key
        chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)
        chat_history = chat_session.chat_history

        # Process request asynchronously
        chatbot_response_task = process_chatbot_request.delay(session_id, prompt)

        # Update chat history in the session
        if prompt.lower() != 'bye':
            chat_history.append({'user': prompt})
        chat_session.chat_history = chat_history
        chat_session.save()

        # Render with a placeholder message while waiting for the async response
        context = {
            'history': chat_history,
            'ai_data': 'Processing...',
        }
        return render(request, "index.html", context)

    # Render the template with chat history on GET requests
    session_id = request.session.session_key
    chat_session = ChatSession.objects.get_or_create(session_id=session_id)[0]
    chat_history = chat_session.chat_history
    context = {
        'history': chat_history,
        'ai_data': '',  # Clear AI response on new GET request
    }
    return render(request, "index.html", context)