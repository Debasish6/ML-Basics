from django.shortcuts import render,HttpResponse
from .main import ChatBot
from dotenv import load_dotenv
import os,sys,json

# Create your views here.
def home(request):
    return HttpResponse("<h1>Welcome to Our ChatBot Web Application</h1>")

def chatbot(request):
    if request.method == 'POST':
        prompt = request.POST['prompttext']
        chatbot = ChatBot(api_key=os.getenv("GoogleAPIKey"))
        chatbot.start_conversation()
        chatbot.previous_db_results = []

        # Initialize session history if not present
        if 'chat_history' not in request.session:
            request.session['chat_history'] = []

        user_input = prompt
        print("You : ", user_input)

        if user_input.lower() == 'bye':
            response = chatbot.send_prompts(user_input, chatbot.previous_db_results)
            print(f"\n{chatbot.CHATBOT_NAME}: {response}")
            sys.exit("............Exiting ChatBot..........")

        try:
            response = chatbot.send_prompts(user_input, chatbot.previous_db_results)
            print(f"\n{chatbot.CHATBOT_NAME}: {response}")
            
            # Update session history
            chat_entry = {'role': 'user', 'text': user_input}
            request.session['chat_history'].append(chat_entry)
            request.session['chat_history'].append({'role': 'ai', 'text': json.loads(response)['text']})
            request.session.modified = True

        except Exception as e:
            print(f'Error: {e}')
            request.session['chat_history'].append({'role': 'error', 'text': str(e)})
            request.session.modified = True

        # Load the JSON data for context
        with open(r'C:\Users\edominer\Python Project\ChatBot using Django\ChatBot\chat_history.json', 'r') as file:
            data = json.load(file)
        
        # Pass data to the template
        context = {
            'data': data[2:], 
            'ai_data': json.loads(response)['text'],
            'history': request.session['chat_history']  # Include history in the context
        }

        print(context['data'])
        return render(request, "index.html", context)
    
    return render(request, "index.html", {'history': request.session.get('chat_history', [])})

