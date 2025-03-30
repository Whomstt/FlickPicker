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
from django.http import HttpResponse, FileResponse
from azure.storage.blob import BlobServiceClient
from django.shortcuts import render
import os
from django.conf import settings
from django.http import HttpResponse, FileResponse
from azure.storage.blob import BlobServiceClient
from django.shortcuts import render
import logging
from azure.storage.blob import BlobServiceClient


# Set up logging
logger = logging.getLogger(__name__)


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


def download_all_files(request):
    # Retrieve connection string and container name from environment variables
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "data-files"
    download_dir = "/code"  # Root directory of the container

    # Create a BlobServiceClient to interact with Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container
    blob_list = container_client.list_blobs()

    # Download files from Azure Blob Storage
    for blob in blob_list:
        blob_client = container_client.get_blob_client(blob.name)
        file_path = os.path.join(
            download_dir, os.path.basename(blob.name)
        )  # Full path to save the file

        # Log file download attempt
        logger.info(f"Downloading file {blob.name} to {file_path}")

        try:
            # Download the file and write it to the disk
            with open(file_path, "wb") as file:
                blob_data = blob_client.download_blob()
                file.write(blob_data.readall())

            # Log success after downloading the file
            logger.info(f"Successfully downloaded {blob.name} to {file_path}")
        except Exception as e:
            # Log any error that occurs during the file download
            logger.error(f"Error downloading {blob.name}: {e}")

    return HttpResponse("Files downloaded successfully.", status=200)
