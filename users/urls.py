from django.urls import path
from . import views

urlpatterns = [
    path("", views.home),
    path("logout", views.logout_view, name="logout"),
    path("users/admin/", views.AdminView.as_view(), name="admin"),
]
