import os
import json
import numpy as np
import faiss
import asyncio
import aiohttp
from django.views import View
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages

# Constants
CACHE_DIR = os.path.join(settings.BASE_DIR, "cache")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
ORIGINAL_FILM_DATA_PATH = os.path.join(settings.BASE_DIR, "original_film_data.json")
OLLAMA_URL = "http://ollama:11434/api"

# Parameters
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_GENERATION_MODEL = "llama3.2"
EMBEDDING_DIM = 768
NPROBE = 10  # Number of clusters to be searched
NLIST = 100  # Number of clusters to be stored
N_TOP_MATCHES = 5  # Number of top matches to return


class BaseEmbeddingView(View):
    """
    Base class for views that require embedding generation.
    Contains methods for embedding, caching, and saving.
    """

    async def send_request(self, url, payload, session):
        """
        Send a POST request to the OLLAMA API.
        """
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Error sending request to {url}: {e}")
            return {}

    async def fetch_embedding(self, text, session):
        """
        Fetch embedding for a given text and normalize it.
        """
        payload = {"model": OLLAMA_EMBEDDING_MODEL, "prompt": text, "keep_alive": -1}
        response = await self.send_request(f"{OLLAMA_URL}/embeddings", payload, session)
        embedding = np.array(response.get("embedding", []), dtype="float32")
        return (
            embedding / np.linalg.norm(embedding)
            if embedding.size > 0
            else np.zeros(EMBEDDING_DIM, dtype="float32")
        )

    async def generate_embeddings(self, data_texts):
        """
        Generate normalized embeddings for a list of texts.
        """
        async with aiohttp.ClientSession() as session:
            embeddings = await asyncio.gather(
                *[self.fetch_embedding(text, session) for text in data_texts]
            )
        return np.array(embeddings, dtype="float32")

    def save_cache(self, data, embeddings, index):
        """
        Save the data, embeddings, and index to the cache directory.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "film_data.json"), "w") as f:
            json.dump(data, f)  # Save the data
        np.save(os.path.join(CACHE_DIR, "film_embeddings.npy"), embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)

    def load_cache(self):
        """
        Load the data, embeddings, and index from the cache directory.
        """
        required_files = [
            os.path.join(CACHE_DIR, "film_data.json"),
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
        Convert Film data JSON to enriched text.
        """
        fields = [
            ("Title", "title"),
            ("Genres", "genres"),
            ("Overview", "overview"),
            ("Director", "director"),
            ("Main Actors", "main_actors"),
            ("Runtime", "runtime"),
            ("Release Date", "release_date"),
            ("Tagline", "tagline"),
            ("Country of Production", "country_of_production"),
            ("Spoken Languages", "spoken_languages"),
            ("Keywords", "keywords"),
            ("Budget", "budget"),
            ("Revenue", "revenue"),
        ]
        return "\n".join(
            f"{label}: {', '.join(map(str, item.get(key, 'N/A'))) if isinstance(item.get(key, 'N/A'), list) else item.get(key, 'N/A')}"
            for label, key in fields
        )


class FilmRecommendationsView(BaseEmbeddingView):
    """
    View for the film recommendations chatbot.
    """

    async def get(self, request, *args, **kwargs):
        """
        Render the chatbot interface.
        """
        return render(request, "chat.html")

    async def post(self, request, *args, **kwargs):
        """
        Handle the chatbot form submission.
        """
        prompt = request.POST.get("prompt", "").strip()
        if not prompt:
            messages.error(request, "Please enter a prompt.")
            return redirect("film_recommendations")

        data, embeddings, index = self.load_cache()
        if data is None:
            messages.error(request, "Embeddings not found. Please generate them first.")
            return redirect("film_recommendations")

        if isinstance(index, faiss.IndexIVFFlat):
            index.nprobe = NPROBE

        async with aiohttp.ClientSession() as session:
            prompt_embedding = await self.fetch_embedding(prompt, session)

        distances, indices = index.search(
            prompt_embedding.reshape(1, -1), N_TOP_MATCHES
        )
        top_matches = self.prepare_top_matches(data, distances, indices)
        explanation = await self.generate_recommendation_explanation(
            prompt, top_matches
        )

        return render(
            request, "chat.html", {"response": explanation, "matches": top_matches}
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

        async with aiohttp.ClientSession() as session:
            response = await self.send_request(
                f"{OLLAMA_URL}/generate",
                {
                    "model": OLLAMA_GENERATION_MODEL,
                    "prompt": SYSTEM_PROMPT,
                    "keep_alive": -1,
                    "stream": False,
                },
                session,
            )
            return response.get("response", "No explanation available.")


class GenerateOriginalEmbeddingsView(BaseEmbeddingView):
    """
    View for generating embeddings for the original film data.
    """

    async def get(self, request, *args, **kwargs):
        """
        Generate embeddings for the original film data.
        """
        data, embeddings, index = await self.generate_original_embeddings()
        self.save_cache(data, embeddings, index)
        messages.success(request, "Embeddings and index generated successfully!")
        return redirect("film_recommendations")

    async def generate_original_embeddings(self):
        """
        Generate embeddings for the original film data.
        """
        with open(ORIGINAL_FILM_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        data_texts = [self.json_to_text(item) for item in data]
        embeddings = await self.generate_embeddings(data_texts)

        quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(
            quantizer, EMBEDDING_DIM, NLIST, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)

        return data, embeddings, index
