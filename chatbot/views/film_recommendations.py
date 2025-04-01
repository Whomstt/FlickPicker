import re
import time
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.shortcuts import render, redirect
from django.contrib import messages
import asyncio

from .base_embedding import BaseEmbeddingView
from chatbot.config import (
    NPROBE,
    N_TOP_MATCHES,
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
        time_breakdown["Entity Detection & Cleaning"] = (
            entity_detection_end - entity_detection_start
        )

        # Create task results containers
        embedding_results = {"prompt": None, "names": None, "genres": None}
        faiss_results = {"data": None, "embeddings": None, "index": None}

        # Start FAISS loading task
        faiss_load_task = asyncio.create_task(self.load_faiss_async(faiss_results))
        faiss_load_start = time.perf_counter()

        # Prepare and run embedding tasks
        embeddings_start = time.perf_counter()
        embedding_tasks = []

        async with aiohttp.ClientSession() as session:
            # Add prompt embedding task
            embedding_tasks.append(
                self.fetch_embedding_task(
                    clean_prompt, session, "prompt", embedding_results
                )
            )

            # Add names embedding task if names detected
            if detected_names:
                names_str = ", ".join(detected_names)
                embedding_tasks.append(
                    self.fetch_embedding_task(
                        names_str, session, "names", embedding_results
                    )
                )

            # Add genres embedding task if genres detected
            if detected_genres:
                genres_str = ", ".join(detected_genres)
                embedding_tasks.append(
                    self.fetch_embedding_task(
                        genres_str, session, "genres", embedding_results
                    )
                )

            # Wait for all embedding tasks to complete
            await asyncio.gather(*embedding_tasks)

        embeddings_end = time.perf_counter()
        time_breakdown["Embeddings Generation (Nomic Embed - Atlas API)"] = (
            embeddings_end - embeddings_start
        )

        # Weighted sum of embeddings
        combined_embedding = PROMPT_WEIGHT * embedding_results["prompt"]
        if detected_names and embedding_results["names"] is not None:
            combined_embedding += NAME_WEIGHT * embedding_results["names"]
        if detected_genres and embedding_results["genres"] is not None:
            combined_embedding += GENRE_WEIGHT * embedding_results["genres"]

        query_vector = combined_embedding.reshape(1, -1)

        # Normalize the query vector using FAISS
        faiss.normalize_L2(query_vector)

        # Wait for FAISS loading task to complete if it hasn't already
        await faiss_load_task
        faiss_load_end = time.perf_counter()
        time_breakdown["Load FAISS Index"] = faiss_load_end - faiss_load_start

        # Extract FAISS results
        data = faiss_results["data"]
        embeddings = faiss_results["embeddings"]
        index = faiss_results["index"]

        if data is None:
            messages.error(request, "Embeddings not found. Please generate them first.")
            return await sync_to_async(redirect)("film_recommendations")

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = NPROBE

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

        # Format names and genres
        detected_names = [name.title() for name in detected_names]
        detected_genres = [genre.title() for genre in detected_genres]
        time_breakdown["Total Time (Tasks are concurrent)"] = (
            time.perf_counter() - start_time
        )

        # Render the page with matches
        return await sync_to_async(render)(
            request,
            "chat.html",
            {
                "matches": top_matches,
                "prompt": prompt,
                "time_breakdown": time_breakdown,
                "detected_names": detected_names,
                "detected_genres": detected_genres,
            },
        )

    # Create an embedding task and store the result
    async def fetch_embedding_task(self, text, session, key, results_dict):
        """
        Fetch embedding for text and store it in results_dict under the given key
        """
        embedding = await self.fetch_embedding(text, session, service="nomic")
        results_dict[key] = embedding

    # Load FAISS index asynchronously
    async def load_faiss_async(self, results_dict):
        """
        Load FAISS index asynchronously and store results in the provided dictionary
        """
        data, embeddings, index = await sync_to_async(self.load_cache)()
        results_dict["data"] = data
        results_dict["embeddings"] = embeddings
        results_dict["index"] = index
