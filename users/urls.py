from django.urls import path
from . import views
from chatbot.views.generate_embeddings import GenerateOriginalEmbeddingsView
from django.contrib.admin.views.decorators import staff_member_required
from .tmdb_api import cancel_fetch_view

urlpatterns = [
    path("", views.home),
    path("logout", views.logout_view, name="logout"),
    path(
        "users/admin/",
        staff_member_required(views.FetchFilmsView.as_view()),
        name="fetch_films",
    ),
    path(
        "generate-original-embeddings/",
        staff_member_required(GenerateOriginalEmbeddingsView.as_view()),
        name="generate_original_embeddings",
    ),
    path(
        "cancel-fetch/",
        staff_member_required(cancel_fetch_view),
        name="cancel_fetch",
    ),
]
