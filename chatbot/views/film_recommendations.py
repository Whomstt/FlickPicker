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

        # Normalize the query vector using FAISS
        query_vector = prompt_embedding.reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search for top matches using the normalized query vector
        distances, indices = index.search(query_vector, N_TOP_MATCHES)
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
        Prepare the top matches for display, sorting in descending order
        by similarity score since a higher score is a closer match.
        """
        matches = []
        for sim, idx in zip(distances[0], indices[0]):
            cosine_sim = float(sim)  # This is the cosine similarity.
            # Compute the corresponding L2 distance from the cosine similarity.
            l2_distance = (2 - 2 * cosine_sim) ** 0.5
            matches.append(
                {
                    **data[idx],
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_distance,
                }
            )

        # Sort matches in descending order by cosine similarity (higher is better)
        matches.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return matches

    async def generate_recommendation_explanation(self, prompt, top_matches):
        """
        Generate a detailed explanation for the film recommendations.
        """
        # Create a text block from the top matching films
        films_text = "\n\n".join(self.json_to_text(item) for item in top_matches)

        # Construct the system prompt with explicit instructions.
        SYSTEM_PROMPT = (
            f"Query: {prompt}\n\n"
            f"{films_text}\n\n"
            "Based solely on the films listed above, provide a detailed film recommendation explanation. "
            "Do not include any films other than the ones provided."
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
