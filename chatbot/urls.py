from django.urls import path
from chatbot.views.film_recommendations import FilmRecommendationsView
from chatbot.views.film_explanation import FilmExplanationView

urlpatterns = [
    path(
        "film-recommendations/",
        FilmRecommendationsView.as_view(),
        name="film_recommendations",
    ),
    path(
        "film-explanation/",
        FilmExplanationView.as_view(),
        name="film_explanation",
    ),
]
