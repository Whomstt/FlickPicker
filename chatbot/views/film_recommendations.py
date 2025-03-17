import time
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.shortcuts import render, redirect
from django.contrib import messages

from .base_embedding import BaseEmbeddingView
from chatbot.config import NPROBE, N_TOP_MATCHES, OPENAI_API_URL, OPENAI_MODEL


class FilmRecommendationsView(BaseEmbeddingView):
    """
    View for the film recommendations chatbot.
    """

    async def get(self, request, *args, **kwargs):
        """
        Render the chatbot interface.
        """
        return await sync_to_async(render)(request, "chat.html")

    async def post(self, request, *args, **kwargs):
        """
        Handle the chatbot form submission.
        """
        start_time = time.time()

        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            messages.error(request, "Please enter a prompt.")
            return await sync_to_async(redirect)("film_recommendations")

        data, embeddings, index = self.load_cache()
        if data is None:
            messages.error(request, "Embeddings not found. Please generate them first.")
            return await sync_to_async(redirect)("film_recommendations")

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = NPROBE

        # Generate weighted query embedding
        async with aiohttp.ClientSession() as session:
            prompt_embedding = await self.fetch_embedding(
                prompt, session, service="nomic"
            )

        # Search for top matches
        distances, indices = index.search(
            prompt_embedding.reshape(1, -1), N_TOP_MATCHES
        )
        top_matches = self.prepare_top_matches(data, distances, indices)
        explanation = await self.generate_recommendation_explanation(
            prompt, top_matches
        )

        end_time = time.time()
        recommendation_time = end_time - start_time

        return await sync_to_async(render)(
            request,
            "chat.html",
            {
                "response": explanation,
                "matches": top_matches,
                "prompt": prompt,
                "recommendation_time": recommendation_time,
            },
        )

    def prepare_top_matches(self, data, distances, indices):
        """
        Prepare the top matches for display.
        """
        return [
            {**data[idx], "similarity_distance": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
        ]

    async def generate_recommendation_explanation(self, prompt, top_matches):
        """
        Generate a detailed explanation for the film recommendations.
        """
        SYSTEM_PROMPT = f"Query: {prompt}\n\n"
        SYSTEM_PROMPT += "\n\n".join(self.json_to_text(item) for item in top_matches)
        SYSTEM_PROMPT += "\n\nProvide a detailed film recommendation explanation."

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant for recommending movies.",
                },
                {"role": "user", "content": SYSTEM_PROMPT},
            ],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        try:
            async with aiohttp.ClientSession() as session:
                response = await self.send_request(OPENAI_API_URL, payload, session)

            explanation = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No explanation available.")
            )
            return explanation

        except aiohttp.ClientError as e:
            print(f"Error generating explanation with OpenAI: {e}")
            return "An error occurred while generating the explanation."
