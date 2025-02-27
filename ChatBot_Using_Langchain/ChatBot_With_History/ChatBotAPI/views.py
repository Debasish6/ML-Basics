from django.shortcuts import render
from django.http import JsonResponse
from django.views import View

# Create your views here.
def home(request):
    context = {
        'message': 'Hello, Django!',
    }
    return render(request, 'home.html', context)



from .utils import Chatbot

class ChatbotView(View):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chatbot = Chatbot()  # Initialize the chatbot

    def post(self, request, *args, **kwargs):
        query = request.POST.get("query")
        
        if not query:
            return JsonResponse({"error": "Query is required."}, status=400)

        try:
            response = self.chatbot.get_response(query)
            return JsonResponse({"response": response})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
