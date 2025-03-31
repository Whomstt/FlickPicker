import json
import re
from rapidfuzz import fuzz
import time
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.shortcuts import render, redirect
from django.contrib import messages

from .base_embedding import BaseEmbeddingView
from chatbot.config import (
    NPROBE,
    N_TOP_MATCHES,
    OPENAI_API_URL,
    OPENAI_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    PROMPT_WEIGHT,
    NAME_WEIGHT,
    GENRE_WEIGHT,
)

from chatbot.entity_recognition import (
    find_names_in_prompt,
    find_genres_in_prompt,
)

from chatbot.prepare_top_matches import prepare_top_matches


class FilmRecommendationsView(BaseEmbeddingView):
    """
    View for the film recommendations chatbot
    """

    async def get(self, request, *args, **kwargs):
        """
        Render the chatbot interface for GET requests
        """
        return await sync_to_async(render)(request, "chat.html")

    async def post(self, request, *args, **kwargs):
        time_breakdown = {}  # Dictionary to store timing for each event
        start_time = time.perf_counter()

        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            messages.error(request, "Please enter a prompt.")
            return await sync_to_async(redirect)("film_recommendations")

        entity_detection_start = time.perf_counter()
        # Detect names and genres
        detected_names = find_names_in_prompt(prompt)
        if detected_names:
            print(f"Detected names in prompt: " + ", ".join(detected_names))
        detected_genres = find_genres_in_prompt(prompt)
        if detected_genres:
            print(f"Detected genres in prompt: " + ", ".join(detected_genres))

        # Clean the prompt by removing detected names and genres
        clean_prompt = prompt
        for name in detected_names:
            clean_prompt = re.sub(
                rf"\b{re.escape(name)}\b", "", clean_prompt, flags=re.IGNORECASE
            ).strip()
        for genre in detected_genres:
            clean_prompt = re.sub(
                rf"\b{re.escape(genre)}\b", "", clean_prompt, flags=re.IGNORECASE
            ).strip()

        entity_detection_end = time.perf_counter()
        time_breakdown = {
            "Entity Detection & Cleaning": entity_detection_end - entity_detection_start
        }

        async with aiohttp.ClientSession() as session:
            # Get the embedding for the cleaned prompt
            embed_prompt_start = time.perf_counter()
            prompt_embedding = await self.fetch_embedding(
                clean_prompt, session, service="nomic"
            )
            embed_prompt_end = time.perf_counter()
            time_breakdown["Prompt Embedding (Nomic Embed - Atlas API)"] = (
                embed_prompt_end - embed_prompt_start
            )

            # Get the names embedding if detected
            if detected_names:
                names_str = ", ".join(detected_names)
                embed_names_start = time.perf_counter()
                names_embedding = await self.fetch_embedding(
                    names_str, session, service="nomic"
                )
                embed_names_end = time.perf_counter()
                time_breakdown["Names Embedding (Nomic Embed - Atlas API)"] = (
                    embed_names_end - embed_names_start
                )

            # Get the genre embedding if detected
            if detected_genres:
                genres_str = ", ".join(detected_genres)
                embed_genres_start = time.perf_counter()
                genres_embedding = await self.fetch_embedding(
                    genres_str, session, service="nomic"
                )
                embed_genres_end = time.perf_counter()
                time_breakdown["Genres Embedding (Nomic Embed - Atlas API)"] = (
                    embed_genres_end - embed_genres_start
                )

            # Weighted sum of embeddings
            combined_embedding = PROMPT_WEIGHT * prompt_embedding
            if detected_names:
                combined_embedding += NAME_WEIGHT * names_embedding
            if detected_genres:
                combined_embedding += GENRE_WEIGHT * genres_embedding
            query_vector = combined_embedding.reshape(1, -1)

        # Normalize the query vector using FAISS
        faiss.normalize_L2(query_vector)

        # Load FAISS Index
        load_faiss_start = time.perf_counter()

        data, embeddings, index = self.load_cache()
        if data is None:
            messages.error(request, "Embeddings not found. Please generate them first.")
            return await sync_to_async(redirect)("film_recommendations")

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = NPROBE

        load_faiss_end = time.perf_counter()
        time_breakdown["Load FAISS Index"] = load_faiss_end - load_faiss_start

        # FAISS Search
        faiss_search_start = time.perf_counter()
        distances, indices = index.search(query_vector, N_TOP_MATCHES)

        top_matches = prepare_top_matches(
            data,
            distances,
            indices,
            detected_names,
            detected_genres,
            index,
            query_vector,
        )
        faiss_search_end = time.perf_counter()
        time_breakdown["FAISS Search"] = faiss_search_end - faiss_search_start

        # GPT-4o-mini Explanation
        explanation_start = time.perf_counter()
        explanation = await self.generate_recommendation_explanation(
            prompt, top_matches
        )
        explanation_end = time.perf_counter()
        time_breakdown["Explanation Generation (GPT-4o-mini - OpenAI API)"] = (
            explanation_end - explanation_start
        )

        # Format names and genres
        detected_names = [name.title() for name in detected_names]
        detected_genres = [genre.title() for genre in detected_genres]

        time_breakdown["Total Time"] = time.perf_counter() - start_time

        return await sync_to_async(render)(
            request,
            "chat.html",
            {
                "response": explanation,
                "matches": top_matches,
                "prompt": prompt,
                "time_breakdown": time_breakdown,
                "detected_names": detected_names,
                "detected_genres": detected_genres,
            },
        )

    async def generate_recommendation_explanation(self, prompt, top_matches):
        """
        Generate a detailed explanation for the film recommendations.
        """
        films_text = "\n\n".join(self.json_to_text(item) for item in top_matches)
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
