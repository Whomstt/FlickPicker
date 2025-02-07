from django.urls import path
from . import views
from chatbot.views import GenerateOriginalEmbeddingsView

urlpatterns = [
    path("", views.home),
    path("logout", views.logout_view, name="logout"),
    path("users/admin/", views.FetchFilmsView.as_view(), name="fetch_films"),
    path(
        "generate-original-embeddings/",
        GenerateOriginalEmbeddingsView.as_view(),
        name="generate_original_embeddings",
    ),
]
