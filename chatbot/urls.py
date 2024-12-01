from django.urls import path
from . import views

urlpatterns = [
    path("", views.MovieRecommendationView.as_view(), name="chat"),
]
