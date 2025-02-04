import os
import json
import logging
import numpy as np
import faiss
import asyncio
import aiohttp
from django.views import View
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseServerError
from django.conf import settings

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(settings.BASE_DIR, "cache")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
EMBEDDING_DIM = 768
FILM_DATA_PATH = os.path.join(settings.BASE_DIR, "film_data.json")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_GENERATION_MODEL = "llama3.2"
OLLAMA_URL = "http://ollama:11434/api"


class BaseEmbeddingView(View):
    async def send_request(self, url, payload, session):
        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Request error to {url}: {e}")
            return None

    async def fetch_embedding(self, text, session):
        url = f"{OLLAMA_URL}/embeddings"
        payload = {"model": OLLAMA_EMBEDDING_MODEL, "prompt": text}
        response = await self.send_request(url, payload, session)
        if response and "embedding" in response:
            return np.array(response["embedding"], dtype="float32")
        return np.zeros(EMBEDDING_DIM, dtype="float32")

    async def generate_embeddings(self, data_texts):
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_embedding(text, session) for text in data_texts]
            embeddings = await asyncio.gather(*tasks)
        return np.array(embeddings, dtype="float32")

    def save_cache(self, data, embeddings, index):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "film_data.json"), "w") as f:
            json.dump(data, f)
        np.save(os.path.join(CACHE_DIR, "film_embeddings.npy"), embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)

    def load_cache(self):
        data_path = os.path.join(CACHE_DIR, "film_data.json")
        embeddings_path = os.path.join(CACHE_DIR, "film_embeddings.npy")

        if not all(
            os.path.exists(p) for p in [data_path, embeddings_path, FAISS_INDEX_PATH]
        ):
            return None, None, None

        with open(data_path, "r") as f:
            data = json.load(f)
        embeddings = np.load(embeddings_path)
        index = faiss.read_index(FAISS_INDEX_PATH)
        return data, embeddings, index

    def json_to_text(self, item):
        keys = [
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
            ("Budget", "budget"),
            ("Revenue", "revenue"),
        ]
        parts = []
        for label, key in keys:
            value = item.get(key, "N/A")
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            parts.append(f"{label}: {value}")
        return "\n".join(parts)


class FilmRecommendationsView(BaseEmbeddingView):
    async def get(self, request, *args, **kwargs):
        return render(request, "chat.html")

    async def post(self, request, *args, **kwargs):
        try:
            prompt = request.POST.get("prompt", "").strip()
            if not prompt:
                return render(request, "chat.html", {"error": "Please enter a prompt"})

            data, embeddings, index = self.load_cache()
            if data is None or embeddings is None or index is None:
                return render(
                    request,
                    "chat.html",
                    {"error": "Embeddings not found. Please generate them first."},
                )

            if isinstance(index, faiss.IndexIVFFlat):
                index.nprobe = 10

            async with aiohttp.ClientSession() as session:
                prompt_embedding = await self.fetch_embedding(prompt, session)
            prompt_embedding = prompt_embedding.reshape(1, -1)
            distances, indices = index.search(prompt_embedding, 3)
            top_matches = self.prepare_top_matches(data, distances, indices)
            explanation = await self.generate_recommendation_explanation(
                prompt, top_matches
            )

            return render(
                request, "chat.html", {"response": explanation, "matches": top_matches}
            )
        except Exception as e:
            logger.error(f"Error in POST request: {e}", exc_info=True)
            return render(request, "chat.html", {"error": str(e)})

    def prepare_top_matches(self, data, distances, indices):
        results = [(dist, idx) for dist, idx in zip(distances[0], indices[0])]
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            {**data[idx], "similarity_distance": float(dist)} for dist, idx in results
        ]

    async def generate_recommendation_explanation(self, prompt, top_matches):
        try:
            full_prompt = f"Query: {prompt}\n\n"
            full_prompt += "\n\n".join(self.json_to_text(item) for item in top_matches)
            full_prompt += "\n\nProvide a detailed film recommendation explanation."

            async with aiohttp.ClientSession() as session:
                response = await self.send_request(
                    f"{OLLAMA_URL}/generate",
                    {
                        "model": OLLAMA_GENERATION_MODEL,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                    session,
                )
                return (
                    response.get("response", "No explanation available.")
                    if response
                    else "No explanation available."
                )
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return "Error generating recommendation explanation."


class GenerateOriginalEmbeddingsView(BaseEmbeddingView):
    async def get(self, request, *args, **kwargs):
        try:
            data, embeddings, index = await self.generate_original_embeddings()
            self.save_cache(data, embeddings, index)
            return HttpResponse("Embeddings and IVF index generated successfully!")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return HttpResponseServerError(f"Error: {str(e)}")

    async def generate_original_embeddings(self):
        try:
            with open(FILM_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError("Film data file not found")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in film data file")

        data_texts = [self.json_to_text(item) for item in data]
        embeddings = await self.generate_embeddings(data_texts)

        quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
        nlist = 100
        index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)

        return data, embeddings, index
