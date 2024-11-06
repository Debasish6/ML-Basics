from django.shortcuts import render,HttpResponse
from .main import ChatBot
from dotenv import load_dotenv
import os,sys

# Create your views here.
def home(request):
    return HttpResponse("<h1>Welcome to Our ChatBot Web Application</h1>")

def chatbot(request):
    if request.method == 'POST':
        prompt = request.POST['prompttext']
        ai_response = prompt
        chatbot = ChatBot(api_key=os.getenv("GoogleAPIKey"))
        chatbot.start_conversation()
        chatbot.previous_db_results = []

        # while True:
        user_input = prompt
        if user_input.lower() =='bye':
            response = chatbot.send_prompts(user_input,chatbot.previous_db_results)
            print(f"\n{chatbot.CHATBOT_NAME}: {response}")
            sys.exit("............Exiting ChatBot..........")
        try:
            response = chatbot.send_prompts(user_input,chatbot.previous_db_results)
            print(f"\n{chatbot.CHATBOT_NAME}: {response}")
        except Exception as e:
            print(f'Error: {e}')

        return render(request,"index.html",{"context":response})
    return render(request,"index.html")