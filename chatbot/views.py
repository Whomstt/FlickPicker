from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


@csrf_exempt  # Use only if you want to bypass CSRF verification for testing
def chat_view(request):
    if request.method == "POST":
        # Parse the JSON body
        data = json.loads(request.body)

        # Process the data (this is just an example)
        response_data = {
            "received_data": data,
            "message": "Data received successfully!",
        }

        return JsonResponse(response_data, status=200)

    return JsonResponse({"error": "Invalid request method."}, status=400)
