import json
import time
from asgiref.sync import sync_to_async
from django.http import JsonResponse
from django.views import View
import aiohttp

from chatbot.config import (
    OPENAI_API_KEY,
    OPENAI_API_URL,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)


class FilmExplanationView(View):
    """
    Asynchronous view to generate recommendation explanation
    """

    async def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body)
            prompt = data.get("prompt", "")
            films_text = data.get("films_text", "")

            SYSTEM_PROMPT = (
                f"Query: {prompt}\n\n"
                f"{films_text}\n\n"
                "You are a helpful movie recommendation assistant. Based solely on the films listed above, generate a thoughtful and detailed explanation for why each film matches the user's query.\n\n"
                "For each film:\n"
                "- Clearly explain how the film relates to the user's interests or query.\n"
                "- Highlight specific themes, genres, characters, or stylistic elements that make the film relevant.\n"
                "- Avoid generic praise and instead focus on meaningful comparisons or reasoning tied to the query.\n\n"
                "Only use the provided list of films. Do not suggest or mention any other films.\n\n"
                "Output each explanation as plain text, separated by two newline characters."
            )

            payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for recommending movies.",
                    },
                    {"role": "user", "content": SYSTEM_PROMPT},
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            }

            async with aiohttp.ClientSession() as session:
                response = await self.send_request(
                    OPENAI_API_URL, payload, session=session, headers=headers
                )

            explanation = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No explanation available.")
            )
            return JsonResponse({"explanation": explanation})
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return JsonResponse(
                {"explanation": "An error occurred while generating the explanation."},
                status=500,
            )

    async def send_request(self, url, payload, session, headers):
        async with session.post(url, json=payload, headers=headers) as resp:
            # Await and return the JSON response
            response_data = await resp.json()
            return response_data
