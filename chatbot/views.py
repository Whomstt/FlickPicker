import requests
from django.views import View
import json
from django.shortcuts import render


class OllamaRequestView(View):
    def get(self, request):
        return render(request, "chat.html")

    def post(self, request):
        data = {"model": "llama3.2", "prompt": "Why do you smell so bad?"}
        response = self.send_post_request_to_ollama(data)
        return render(request, "chat.html", {"response": response})

    def send_post_request_to_ollama(self, data):
        url = "http://ollama:11434/api/generate"
        full_response = ""
        try:
            response = requests.post(url, json=data, stream=True)
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    response_data = json.loads(line)
                    full_response += response_data.get("response", "")
                    if response_data.get("done", False):
                        break
            return full_response.strip()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
        except ValueError as json_error:
            return {"error": "Invalid JSON response: {}".format(json_error)}
