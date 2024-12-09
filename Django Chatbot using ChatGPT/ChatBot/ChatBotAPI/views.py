from django.shortcuts import render,HttpResponse
from rest_framework import views, status
from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny
# internals
from ChatBotAPI.serializers import AIResponseSerializer, UserSerializer, TokenSerializer
from ChatBotAPI.models import AIResponse

# Create your views here.
def home(request):
    return HttpResponse("This is home page.")


class AIResponseView(views.APIView):
    serializer_class = AIResponseSerializer
    authentication_classes = [TokenAuthentication]

    def get(self, request, format=None):
        qs = AIResponse.objects.all()
        serializer = self.serializer_class(qs, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserView(views.APIView):
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

    def get(self, request, format=None):
        qs = User.objects.all()
        serializer = self.serializer_class(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TokenView(ObtainAuthToken):
    serializer_class = TokenSerializer