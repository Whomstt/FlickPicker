from django.urls import path
from . import views

urlpatterns = [
    path("film-recommendations/", views.FilmRecommendationsView.as_view(), name="film_recommendations"),
    path("generate-original-embeddings/", views.GenerateOriginalEmbeddingsView.as_view(), name="generate_original_embeddings"),
]
