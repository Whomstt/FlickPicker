from django.shortcuts import render, redirect
from django.contrib.auth import logout
from allauth.socialaccount.models import SocialAccount

def home(request):
    user = request.user
    google_account = SocialAccount.objects.filter(user=user, provider='Google').first()
    google_given_name = ""
    google_picture = ""
    if google_account:
        extra_data = google_account.extra_data
        print("Extra Data:", extra_data)
        google_given_name = extra_data.get("given_name", "")
        google_picture = extra_data.get("picture", "")
    context = {
        "google_given_name": google_given_name,
        "google_picture": google_picture
    }
    return render(request, "home.html", context)

def logout_view(request):
    logout(request)
    return redirect("/")