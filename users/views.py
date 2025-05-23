from django.shortcuts import render, redirect
from django.contrib.auth import logout
from allauth.socialaccount.models import SocialAccount
from django.views import View
from django.contrib import admin
from django.urls import path
from django.template.response import TemplateResponse
from django.conf import settings
from .tmdb_api import fetch_and_save_films
import asyncio
import aiohttp
import os
from django.conf import settings


def home(request):
    context = {
        "google_given_name": "",
        "google_picture": "",
    }
    user = request.user
    if user.is_authenticated:
        google_account = SocialAccount.objects.filter(
            user=user, provider="Google"
        ).first()
        if google_account:
            extra_data = google_account.extra_data
            context["google_given_name"] = extra_data.get("given_name", "")
            context["google_picture"] = extra_data.get("picture", "")

    return render(request, "home.html", context)


def logout_view(request):
    logout(request)
    return redirect("/")


class FetchFilmsView(View):
    def get(self, request):
        context = {
            "message": "",
        }
        return TemplateResponse(request, "admin.html", context)

    def post(self, request):
        message = ""
        try:
            asyncio.run(fetch_and_save_films())
            message = f"Successfully pulled films from TMDB and saved them."
        except Exception as e:
            message = f"Error occurred: {e}"
        context = {
            "message": message,
        }
        return TemplateResponse(request, "admin.html", context)
