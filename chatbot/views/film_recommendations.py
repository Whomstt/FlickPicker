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
    SEARCH_INCREMENT,
    MAX_RESULTS,
    MAX_TOKENS,
    TEMPERATURE,
    PROMPT_WEIGHT,
    NAME_WEIGHT,
    GENRE_WEIGHT,
    NAME_FUZZY_THRESHOLD,
    GENRE_FUZZY_THRESHOLD,
)

from chatbot.entity_recognition import (
    sliding_window_fuzzy,
    find_names_in_prompt,
    find_genres_in_prompt,
)


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
        start_time = time.perf_counter()
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            messages.error(request, "Please enter a prompt.")
            return await sync_to_async(redirect)("film_recommendations")

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

        data, embeddings, index = self.load_cache()
        if data is None:
            messages.error(request, "Embeddings not found. Please generate them first.")
            return await sync_to_async(redirect)("film_recommendations")

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = NPROBE

        time_breakdown = {}  # Dictionary to store timing for each event

        # Record time after entity detection and prompt cleaning
        time_breakdown["Entity Detection & Cleaning"] = time.perf_counter() - start_time

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

        # FAISS Search
        faiss_search_start = time.perf_counter()
        distances, indices = index.search(query_vector, N_TOP_MATCHES)

        top_matches = self.prepare_top_matches(
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

    def prepare_top_matches(
        self,
        data,
        distances,
        indices,
        detected_names=None,
        detected_genres=None,
        index=None,
        query_vector=None,
    ):
        """
        Prepare the top film matches for the user.
        """

        def assign_match_flags(film):
            # Name match flag
            if detected_names:
                lower_names = set(detected_names)
                directors = [director.lower() for director in film.get("directors", [])]
                actors = [actor.lower() for actor in film.get("main_actors", [])]
                film["name_match"] = any(
                    director in lower_names for director in directors
                ) or any(actor in lower_names for actor in actors)
            else:
                film["name_match"] = False

            # Genre match flag
            if detected_genres:
                lower_genres = set(detected_genres)
                film["genre_match"] = any(
                    genre.lower() in lower_genres for genre in film.get("genres", [])
                )
            else:
                film["genre_match"] = False

            return film

        # Set to track unique films and prevent duplicates
        unique_films = set()
        matches = []

        # Process initial FAISS search results
        for sim, idx in zip(distances[0], indices[0]):
            # Sanitize similarity score
            cosine_sim = max(min(float(sim), 1.0), 0.0)

            # Skip if film already processed
            if idx in unique_films:
                continue

            l2_distance = (2 - 2 * cosine_sim) ** 0.5
            film = {
                **data[idx],
                "cosine_similarity": cosine_sim,
                "l2_distance": l2_distance,
            }
            film = assign_match_flags(film)
            matches.append(film)
            unique_films.add(idx)

        # Sort matches in descending order
        matches.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        # If no detected names or genres, return top matches
        if not (detected_names or detected_genres):
            return matches[:N_TOP_MATCHES]

        # Determine filtering condition based on detected entities
        if detected_names and detected_genres:
            condition = lambda film: film["name_match"] and film["genre_match"]
        elif detected_names:
            condition = lambda film: film["name_match"]
        elif detected_genres:
            condition = lambda film: film["genre_match"]

        # Filter matches based on the condition
        filtered = [film for film in matches if condition(film)]

        current_k = 0
        unique_filtered_films = set()

        # Expand search if necessary
        while current_k < MAX_RESULTS and len(filtered) < N_TOP_MATCHES:
            current_k += SEARCH_INCREMENT
            distances, indices = index.search(query_vector, current_k)

            for sim, idx in zip(distances[0], indices[0]):
                # Sanitize similarity score
                cosine_sim = max(min(float(sim), 1.0), 0.0)

                # Skip if film already processed or not unique in filtered set
                if idx in unique_films or idx in unique_filtered_films:
                    continue

                l2_distance = (2 - 2 * cosine_sim) ** 0.5
                film = {
                    **data[idx],
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_distance,
                }
                film = assign_match_flags(film)

                if condition(film):
                    filtered.append(film)
                    unique_filtered_films.add(idx)

                unique_films.add(idx)

            # Sort and truncate to prevent excessive growth
            filtered.sort(key=lambda x: x["cosine_similarity"], reverse=True)
            filtered = filtered[:MAX_RESULTS]

        # Supplement with best unfiltered matches
        supplement = [m for m in matches if m not in filtered]
        supplement.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        # Final match selection
        if len(filtered) >= N_TOP_MATCHES:
            return filtered[:N_TOP_MATCHES]
        else:
            remaining_needed = N_TOP_MATCHES - len(filtered)
            return filtered + supplement[:remaining_needed]

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
