from django.shortcuts import render, redirect
from django.contrib.auth import logout
from allauth.socialaccount.models import SocialAccount

def home(request):
    context = {
        "google_given_name": "",
        "google_picture": "",
    }
    user = request.user
    if user.is_authenticated:
        google_account = SocialAccount.objects.filter(user=user, provider='Google').first()
        if google_account:
            extra_data = google_account.extra_data
            context["google_given_name"] = extra_data.get("given_name", "")
            context["google_picture"] = extra_data.get("picture", "")
        
    return render(request, "home.html", context)

def logout_view(request):
    logout(request)
    return redirect("/")