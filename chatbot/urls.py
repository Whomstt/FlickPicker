from django.urls import path
from . import views

urlpatterns = [
    path(
        "film-recommendations/",
        views.FilmRecommendationsView.as_view(),
        name="film_recommendations",
    ),
]
