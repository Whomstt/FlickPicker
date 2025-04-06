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
    KEYWORD_WEIGHT,
    TITLE_WEIGHT,
    RUNTIME_WEIGHT,
    RELEASE_WEIGHT,
)
from chatbot.entity_recognition import (
    find_names_in_prompt,
    find_genres_in_prompt,
    find_keywords_in_prompt,
    find_titles_in_prompt,
    clean_prompt_with_fuzzy,
    find_release_in_prompt,
    find_runtime_in_prompt,
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

        print(f"Prompt: {prompt}")

        entity_detection_start = time.perf_counter()
        # Detect names, genres, keywords, and titles in the prompt
        detected_names = find_names_in_prompt(prompt)
        if detected_names:
            print("Detected names in prompt: " + ", ".join(detected_names))
        detected_genres = find_genres_in_prompt(prompt)
        if detected_genres:
            print("Detected genres in prompt: " + ", ".join(detected_genres))
        detected_keywords = find_keywords_in_prompt(prompt)
        if detected_keywords:
            print("Detected keywords in prompt: " + ", ".join(detected_keywords))
        detected_titles = find_titles_in_prompt(prompt)
        if detected_titles:
            print("Detected titles in prompt: " + ", ".join(detected_titles))
        detected_release = find_release_in_prompt(prompt)
        if detected_release:
            print("Detected release date in prompt: " + ", ".join(detected_release))
        detected_runtime = find_runtime_in_prompt(prompt)
        if detected_runtime:
            print("Detected runtime in prompt: " + ", ".join(detected_runtime))

        if (
            detected_names
            or detected_genres
            or detected_keywords
            or detected_titles
            or detected_release
            or detected_runtime
        ):
            all_entities = (
                detected_names
                + detected_genres
                + detected_keywords
                + detected_titles
                + detected_release
                + detected_runtime
            )
            clean_prompt = clean_prompt_with_fuzzy(prompt, all_entities)
            print(f"Cleaned prompt: {clean_prompt}")
        else:
            clean_prompt = prompt

        entity_detection_end = time.perf_counter()
        time_breakdown["Entity Detection & Cleaning"] = (
            entity_detection_end - entity_detection_start
        )

        # Create task results containers
        embedding_results = {
            "prompt": None,
            "names": None,
            "genres": None,
            "keywords": None,
            "titles": None,
            "release": None,
            "runtime": None,
        }
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

            # Add keywords embedding task if keywords detected
            if detected_keywords:
                keywords_str = ", ".join(detected_keywords)
                embedding_tasks.append(
                    self.fetch_embedding_task(
                        keywords_str, session, "keywords", embedding_results
                    )
                )

            # Add titles embedding task if titles detected
            if detected_titles:
                titles_str = ", ".join(detected_titles)
                embedding_tasks.append(
                    self.fetch_embedding_task(
                        titles_str, session, "titles", embedding_results
                    )
                )

            # Add release date embedding task if detected
            if detected_release:
                release_str = ", ".join(detected_release)
                embedding_tasks.append(
                    self.fetch_embedding_task(
                        release_str, session, "release", embedding_results
                    )
                )

            # Add runtime embedding task if detected
            if detected_runtime:
                runtime_str = ", ".join(detected_runtime)
                embedding_tasks.append(
                    self.fetch_embedding_task(
                        runtime_str, session, "runtime", embedding_results
                    )
                )

            # Wait for all embedding tasks to complete
            await asyncio.gather(*embedding_tasks)

        embeddings_end = time.perf_counter()
        time_breakdown["Embeddings Generation (Nomic Embed - Atlas API)"] = (
            embeddings_end - embeddings_start
        )

        prompt_weight = PROMPT_WEIGHT
        # If any entities are detected, compute a weighted sum
        if (
            detected_names
            or detected_genres
            or detected_keywords
            or detected_titles
            or detected_release
            or detected_runtime
        ):
            if not clean_prompt:
                prompt_weight = 0
            combined_embedding = prompt_weight * embedding_results["prompt"]
            if detected_names and embedding_results["names"] is not None:
                combined_embedding += NAME_WEIGHT * embedding_results["names"]
            if detected_genres and embedding_results["genres"] is not None:
                combined_embedding += GENRE_WEIGHT * embedding_results["genres"]
            if detected_keywords and embedding_results["keywords"] is not None:
                combined_embedding += KEYWORD_WEIGHT * embedding_results["keywords"]
            if detected_titles and embedding_results["titles"] is not None:
                combined_embedding += TITLE_WEIGHT * embedding_results["titles"]
            if detected_release and embedding_results["release"] is not None:
                combined_embedding += RELEASE_WEIGHT * embedding_results["release"]
            if detected_runtime and embedding_results["runtime"] is not None:
                combined_embedding += RUNTIME_WEIGHT * embedding_results["runtime"]
        else:
            # No entities were detected, so just use the prompt embedding
            combined_embedding = embedding_results["prompt"]

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
            detected_keywords,
            detected_titles,
            detected_runtime,
            detected_release,
            index,
            query_vector,
        )
        faiss_search_end = time.perf_counter()
        time_breakdown["FAISS Search"] = faiss_search_end - faiss_search_start

        # Format names, genres, keywords, and titles for display
        detected_names = [name.title() for name in detected_names]
        detected_genres = [genre.title() for genre in detected_genres]
        detected_keywords = [keyword.title() for keyword in detected_keywords]
        detected_titles = [title.title() for title in detected_titles]
        detected_release = [release.title() for release in detected_release]
        detected_runtime = [runtime.title() for runtime in detected_runtime]
        time_breakdown["Total Time (Tasks are concurrent)"] = (
            time.perf_counter() - start_time
        )

        # Render the page with matches and detected entities
        return await sync_to_async(render)(
            request,
            "chat.html",
            {
                "matches": top_matches,
                "prompt": prompt,
                "time_breakdown": time_breakdown,
                "detected_names": detected_names,
                "detected_genres": detected_genres,
                "detected_keywords": detected_keywords,
                "detected_titles": detected_titles,
                "detected_release": detected_release,
                "detected_runtime": detected_runtime,
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
