from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('users/', views.UserView.as_view(), name='users'),
    path('tokens/',  views.TokenView.as_view(), name='tokens'),
    path('chatbot/',  views.AIResponseView.as_view(), name='AI_Response' )
]
