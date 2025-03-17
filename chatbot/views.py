import os
import json
import time
import numpy as np
import faiss
import asyncio
import aiohttp
from asgiref.sync import sync_to_async
from django.views import View
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
import time
from dotenv import load_dotenv


load_dotenv()

# Constants
CACHE_DIR = os.path.join(settings.BASE_DIR, "cache")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
RAW_FILM_DATA_PATH = os.path.join(settings.BASE_DIR, "raw_film_data.json")

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"

# Nomic API settings
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Ollama settings
OLLAMA_URL = "http://ollama:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"


NPROBE = 10  # Number of clusters to be searched
NLIST = 100  # Number of clusters to be stored
N_TOP_MATCHES = 3  # Number of top matches to return
M = 16  # Number of subquantizers
NBITS = 7  # Number of bits per subquantizer

# Field weights for film attributes - higher value means more important
FIELD_WEIGHTS = {
    "genres": 1.0,
    "title": 0.8,
    "tagline": 0.6,
    "overview": 0.7,
    "keywords": 0.5,
    "director": 0.4,
    "main_actors": 0.3,
    "country_of_production": 0.1,
    "spoken_languages": 0.1,
    "runtime": 0.2,
    "release_date": 0.3,
    "budget": 0.1,
    "revenue": 0.1,
    "rating": 0.5,
}


class BaseEmbeddingView(View):
    """
    Base class for views that require embedding generation.
    Contains methods for embedding, caching, and saving.
    """

    async def send_request(self, url, payload, session):
        headers = {}
        if url == OPENAI_API_URL:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            }
        elif url == NOMIC_API_URL:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {NOMIC_API_KEY}",
            }
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error sending request to {url}: {e}")
            return {}

    async def fetch_embedding(self, text, session, service="nomic"):
        """Fetch embeddings from specified service"""
        if service == "nomic":
            payload = {
                "model": NOMIC_MODEL,
                "texts": [text],
                "task_type": "clustering",
            }
            url = NOMIC_API_URL
            headers = {"Authorization": f"Bearer {NOMIC_API_KEY}"}
        elif service == "ollama":
            payload = {"model": EMBEDDING_MODEL, "prompt": text}
            url = f"{OLLAMA_URL}/embeddings"
            headers = {}

        try:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()

                # Handle different response formats
                if service == "nomic":
                    embedding = np.array(data["embeddings"][0], dtype="float32")
                elif service == "ollama":
                    embedding = np.array(data["embedding"], dtype="float32")

                return (
                    embedding / np.linalg.norm(embedding)
                    if embedding.any()
                    else np.zeros(EMBEDDING_DIM)
                )
        except Exception as e:
            print(f"Error from {service}: {str(e)}")
            return np.zeros(EMBEDDING_DIM)

    async def generate_field_embeddings(self, item, use_ollama=False):
        """
        Generate embeddings for each field in the item separately.
        """
        field_embeddings = {}

        async with aiohttp.ClientSession() as session:
            for field, weight in FIELD_WEIGHTS.items():
                if (value := item.get(field)) is not None:
                    # Format the field value
                    if isinstance(value, list):
                        text_value = ", ".join(map(str, value))
                    else:
                        text_value = str(value)

                    # Generate embedding for the field
                    field_text = f"{field}: {text_value}"
                    service = "ollama" if use_ollama else "nomic"
                    embedding = await self.fetch_embedding(field_text, session, service)

                    # Store the embedding with its weight
                    field_embeddings[field] = (embedding, weight)

        return field_embeddings

    def combine_weighted_embeddings(self, field_embeddings):
        """
        Combine field embeddings using their weights and normalize the result.
        """
        weighted_sum = np.zeros(EMBEDDING_DIM, dtype="float32")

        for field, (embedding, weight) in field_embeddings.items():
            weighted_sum += embedding * weight

        # Normalize to ensure the final embedding is a unit vector
        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum /= norm

        return weighted_sum

    def save_cache(self, data, embeddings, index):
        """
        Save the data, embeddings, and index to the cache directory.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(
            os.path.join(CACHE_DIR, "film_embeddings.npy"), embeddings
        )  # Save embeddings
        faiss.write_index(index, FAISS_INDEX_PATH)  # Save index

    def load_cache(self):
        """
        Load the data, embeddings, and index from the cache directory.
        """
        required_files = [
            os.path.join(RAW_FILM_DATA_PATH),
            os.path.join(CACHE_DIR, "film_embeddings.npy"),
            FAISS_INDEX_PATH,
        ]
        if not all(os.path.exists(p) for p in required_files):
            return None, None, None

        with open(required_files[0], "r") as f:
            data = json.load(f)
        embeddings = np.load(required_files[1])
        index = faiss.read_index(required_files[2])
        return data, embeddings, index

    def json_to_text(self, item):
        """
        Convert film data JSON to enriched text for display or explanation.
        """
        components = []

        if title := item.get("title"):
            components.append(f"Film Title: {title}")
        if genres := item.get("genres"):
            components.append(f"Genres: {', '.join(genres)}")
        if tagline := item.get("tagline"):
            components.append(f"Tagline: {tagline}")
        if overview := item.get("overview"):
            components.append(f"Overview: {overview}")
        if director := item.get("director"):
            components.append(f"Directed by {director}")
        if main_actors := item.get("main_actors"):
            components.append(f"Featuring: {', '.join(main_actors)}")
        if runtime := item.get("runtime"):
            components.append(f"Runtime: {runtime}")
        if release_date := item.get("release_date"):
            components.append(f"Release Date: {release_date}")
        if country := item.get("country_of_production"):
            components.append(f"Country of Production: {', '.join(country)}")
        if languages := item.get("spoken_languages"):
            components.append(f"Spoken Languages: {', '.join(languages)}")
        if budget := item.get("budget"):
            components.append(f"Budget: {budget}")
        if revenue := item.get("revenue"):
            components.append(f"Revenue: {revenue}")
        if rating := item.get("rating"):
            components.append(f"Rating: {rating}")
        if keywords := item.get("keywords"):
            components.append(f"Keywords: {', '.join(keywords)}")

        # Combine all non-empty components into a single string
        return "\n".join(components)


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


class GenerateOriginalEmbeddingsView(BaseEmbeddingView):
    """
    View for generating embeddings for the original film data.
    """

    async def post(self, request, *args, **kwargs):
        """
        Generate embeddings for the original film data.
        """
        data, embeddings, index = await self.generate_original_embeddings()
        self.save_cache(data, embeddings, index)
        message = "Embeddings and index generated successfully!"
        return await sync_to_async(render)(request, "admin.html", {"message": message})

    async def generate_original_embeddings(self):
        """
        Generate embeddings for the original film data using field-specific weighting.
        """
        with open(RAW_FILM_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = []

        for item in data:
            # Generate embeddings for each field
            field_embeddings = await self.generate_field_embeddings(
                item, use_ollama=True
            )

            # Combine embeddings with weighting
            combined_embedding = self.combine_weighted_embeddings(field_embeddings)
            embeddings.append(combined_embedding)

        embeddings = np.array(embeddings, dtype="float32")

        # Create FAISS index
        quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
        index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, NLIST, M, NBITS)
        index.train(embeddings)
        index.add(embeddings)

        return data, embeddings, index
