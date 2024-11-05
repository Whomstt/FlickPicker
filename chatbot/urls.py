from django.urls import path
from . import views

urlpatterns = [
    path("", views.OllamaRequestView.as_view(), name="chat"),
]
