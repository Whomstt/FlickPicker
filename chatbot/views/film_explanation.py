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
                "Based solely on the films listed above, provide a detailed film recommendation explanation for each film. "
                "Output each explanation in plain text separated by a double newline. "
                "Do not include any films other than those provided."
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
