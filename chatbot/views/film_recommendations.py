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
)


class FilmRecommendationsView(BaseEmbeddingView):
    """
    View for the film recommendations chatbot.
    """

    def sliding_window_fuzzy(self, prompt, candidate, threshold):
        """Slide window over prompt and check fuzzy match."""
        prompt_words = prompt.split()
        candidate_words = candidate.split()
        window_size = len(candidate_words)
        best_score = 0
        for i in range(len(prompt_words) - window_size + 1):
            window = " ".join(prompt_words[i : i + window_size])
            score = fuzz.token_set_ratio(candidate, window)
            best_score = max(best_score, score)
            if best_score >= threshold:
                return True
        return False

    def find_names_in_prompt(
        self, prompt, json_path="actors_directors.json", threshold=90
    ):
        """
        Detect candidate names in user's prompt
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError):
            return []

        detected_names = set()

        def regex_match(name, text):
            pattern = r"\b" + re.escape(name) + r"\b"
            return re.search(pattern, text) is not None

        for name in data.get("unique_main_actors", []):
            if regex_match(name, prompt) or self.sliding_window_fuzzy(
                prompt, name, threshold
            ):
                detected_names.add(name.lower())

        for name in data.get("unique_directors", []):
            if regex_match(name, prompt) or self.sliding_window_fuzzy(
                prompt, name, threshold
            ):
                detected_names.add(name.lower())

        return list(detected_names)

    async def get(self, request, *args, **kwargs):
        """
        Render the chatbot interface for GET requests
        """
        return await sync_to_async(render)(request, "chat.html")

    async def post(self, request, *args, **kwargs):
        start_time = time.time()
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            messages.error(request, "Please enter a prompt.")
            return await sync_to_async(redirect)("film_recommendations")

        detected_names = self.find_names_in_prompt(prompt)
        if detected_names:
            print("Detected names in prompt: " + ", ".join(detected_names))

        data, embeddings, index = self.load_cache()
        if data is None:
            messages.error(request, "Embeddings not found. Please generate them first.")
            return await sync_to_async(redirect)("film_recommendations")

        if isinstance(index, faiss.IndexIVF):
            index.nprobe = NPROBE

        # Generate query embedding using the prompt
        async with aiohttp.ClientSession() as session:
            prompt_embedding = await self.fetch_embedding(
                prompt, session, service="nomic"
            )

        # Normalize the query vector using FAISS
        query_vector = prompt_embedding.reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # Search for top matches
        distances, indices = index.search(query_vector, N_TOP_MATCHES)
        top_matches = self.prepare_top_matches(
            data, distances, indices, detected_names, index, query_vector
        )
        explanation = await self.generate_recommendation_explanation(
            prompt, top_matches
        )
        recommendation_time = time.time() - start_time

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

    def prepare_top_matches(
        self,
        data,
        distances,
        indices,
        detected_names=None,
        index=None,
        query_vector=None,
    ):
        """
        Prepare the top film matches for the user
        """

        def filter_matches(matches, lower_names):
            filtered = []
            for film in matches:
                directors = [director.lower() for director in film.get("directors", [])]
                actors = [actor.lower() for actor in film.get("main_actors", [])]
                if any(director in lower_names for director in directors) or any(
                    actor in lower_names for actor in actors
                ):
                    filtered.append(film)
            return filtered

        # Set to track unique films and prevent duplicates
        unique_films = set()
        matches = []

        # Process initial FAISS search results
        for sim, idx in zip(distances[0], indices[0]):
            # Sanitize similarity score to prevent extreme values
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
            matches.append(film)
            unique_films.add(idx)

        # Sort matches in descending order
        matches.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        # If no detected names, return top matches
        if not detected_names:
            return matches[:N_TOP_MATCHES]

        lower_names = set(detected_names)
        filtered = filter_matches(matches, lower_names)

        current_k = 0
        unique_filtered_films = set()

        # Expand search
        while current_k < MAX_RESULTS and len(filtered) < N_TOP_MATCHES:
            current_k += SEARCH_INCREMENT
            distances, indices = index.search(query_vector, current_k)

            for sim, idx in zip(distances[0], indices[0]):
                # Sanitize similarity score
                cosine_sim = max(min(float(sim), 1.0), 0.0)

                # Skip if film already processed or not unique
                if idx in unique_films or idx in unique_filtered_films:
                    continue

                l2_distance = (2 - 2 * cosine_sim) ** 0.5
                film = {
                    **data[idx],
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_distance,
                }

                # Check name filtering
                directors = [director.lower() for director in film.get("directors", [])]
                actors = [actor.lower() for actor in film.get("main_actors", [])]

                if any(director in lower_names for director in directors) or any(
                    actor in lower_names for actor in actors
                ):
                    filtered.append(film)
                    unique_filtered_films.add(idx)

                unique_films.add(idx)

            # Sort and truncate to prevent excessive growth
            filtered.sort(key=lambda x: x["cosine_similarity"], reverse=True)
            filtered = filtered[:MAX_RESULTS]

        # Supplement with best unfiltered matches
        supplement = [
            m for m in matches if m not in filtered and m["cosine_similarity"] > 0.5
        ]
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
