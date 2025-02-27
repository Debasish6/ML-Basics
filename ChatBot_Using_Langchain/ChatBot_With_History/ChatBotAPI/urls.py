from django.urls import path
from . import views

urlpatterns = [
    path("",views.home,name = 'home'),
    path('chat/', views.ChatbotView.as_view(), name='chatbot')
]
