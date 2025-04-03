from django.urls import path
from . import views
from chatbot.views.generate_embeddings import (
    GenerateOriginalEmbeddingsView,
    cancel_generate_view,
)
from chatbot.views.regenerate_index import (
    RegenerateFaissIndexView,
    cancel_regenerate_index_view,
)
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
    path(
        "cancel-generate/",
        staff_member_required(cancel_generate_view),
        name="cancel_generate",
    ),
    path(
        "regenerate-faiss-index/",
        staff_member_required(RegenerateFaissIndexView.as_view()),
        name="regenerate_faiss_index",
    ),
    path(
        "cancel-regenerate-index/",
        staff_member_required(cancel_regenerate_index_view),
        name="cancel_regenerate_index",
    ),
]
