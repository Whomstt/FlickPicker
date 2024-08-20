from django.urls import path
from . import views

urlpatterns = [
    path("films/", views.film_list, name="film_list"),
]