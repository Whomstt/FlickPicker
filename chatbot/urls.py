from django.urls import path
from chatbot.views.film_recommendations import FilmRecommendationsView

urlpatterns = [
    path(
        "film-recommendations/",
        FilmRecommendationsView.as_view(),
        name="film_recommendations",
    ),
]
