from django.urls import path
from . import views

urlpatterns = [
    path("", views.chatbot_form, name = "chatbot_form"),
    path("response", views.chatbot_form, name = "chatbot_response")
]