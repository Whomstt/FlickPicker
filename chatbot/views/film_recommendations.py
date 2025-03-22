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
from chatbot.config import NPROBE, N_TOP_MATCHES, OPENAI_API_URL, OPENAI_MODEL


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
        Detect candidate names from the prompt by checking if any known actor or director
        appear as a full or near-full match in the prompt.
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError):
            return []

        detected_names = set()

        # Regex-based full-word match.
        def regex_match(name, text):
            pattern = r"\b" + re.escape(name) + r"\b"
            return re.search(pattern, text) is not None

        # Check each actor name from "unique_main_actors"
        for name in data.get("unique_main_actors", []):
            if regex_match(name, prompt):
                detected_names.add(name)
            else:
                if self.sliding_window_fuzzy(prompt, name, threshold):
                    detected_names.add(name)

        # Check each director name from "unique_directors"
        for name in data.get("unique_directors", []):
            if regex_match(name, prompt):
                detected_names.add(name)
            else:
                if self.sliding_window_fuzzy(prompt, name, threshold):
                    detected_names.add(name)

        return list(detected_names)

    async def get(self, request, *args, **kwargs):
        """
        Render the chatbot interface for GET requests.
        """
        return await sync_to_async(render)(request, "chat.html")

    async def post(self, request, *args, **kwargs):
        start_time = time.time()

        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            messages.error(request, "Please enter a prompt.")
            return await sync_to_async(redirect)("film_recommendations")

        # Detect names in the prompt
        detected_names = self.find_names_in_prompt(prompt)
        if detected_names:
            names_str = ", ".join(detected_names)
            print(f"Detected names in prompt: {names_str}")

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
            l2_distance = (2 - 2 * cosine_sim) ** 0.5  # Compute L2 distance.
            matches.append(
                {
                    **data[idx],
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_distance,
                }
            )

        matches.sort(key=lambda x: x["cosine_similarity"], reverse=False)
        return matches

    async def generate_recommendation_explanation(self, prompt, top_matches):
        """
        Generate a detailed explanation for the film recommendations.
        """
        films_text = "\n\n".join(self.json_to_text(item) for item in top_matches)
        SYSTEM_PROMPT = (
            f"Query: {prompt}\n\n"
            f"{films_text}\n\n"
            "Based solely on the films listed above, provide a detailed film recommendation explanation for each film. "
            "Please output your response in plain text with each film's explanation separated by a double newline. "
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
