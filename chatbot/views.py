import os
import json
import logging
import numpy as np
import faiss
import requests
from concurrent.futures import ThreadPoolExecutor
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
    def fetch_embedding(self, text):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            )
            response.raise_for_status()
            embedding = response.json().get("embedding")
            if embedding is None:
                raise ValueError("Embedding not found in response")
            return np.array(embedding, dtype="float32")
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding request error: {e}")
            return np.zeros(EMBEDDING_DIM, dtype="float32")

    def generate_embeddings(self, data_texts):
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.fetch_embedding, data_texts))
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
        fields = [f"Title: {item.get('title', 'N/A')}"]
        key_fields = [
            ("Genres", "genres"),
            ("Overview", "overview"),
            ("Director", "director"),
            ("Main Actors", "main_actors"),
            ("Runtime", "runtime"),
            ("Release Date", "release_date"),
        ]
        for label, key in key_fields:
            value = item.get(key, [])
            value = ", ".join(value) if isinstance(value, list) else value or "N/A"
            fields.append(f"{label}: {value}")
        return "\n".join(fields)


class FilmRecommendationsView(BaseEmbeddingView):
    def get(self, request, *args, **kwargs):
        return render(request, "chat.html")

    def post(self, request, *args, **kwargs):
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

            prompt_embedding = self.fetch_embedding(prompt).reshape(1, -1)
            distances, indices = index.search(prompt_embedding, 3)
            top_matches = self.prepare_top_matches(data, distances, indices)
            explanation = self.generate_recommendation_explanation(prompt, top_matches)

            return render(
                request, "chat.html", {"response": explanation, "matches": top_matches}
            )
        except Exception as e:
            logger.error(f"Error in POST request: {e}", exc_info=True)
            return render(request, "chat.html", {"error": str(e)})

    def prepare_top_matches(self, data, distances, indices):
        return [
            {**data[idx], "similarity_distance": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
        ]

    def generate_recommendation_explanation(self, prompt, top_matches):
        try:
            full_prompt = f"Query: {prompt}\n\n"
            full_prompt += "\n\n".join(self.json_to_text(item) for item in top_matches)
            full_prompt += "\n\nProvide a detailed film recommendation explanation."

            response = requests.post(
                f"{OLLAMA_URL}/generate",
                json={
                    "model": OLLAMA_GENERATION_MODEL,
                    "prompt": full_prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json().get("response", "No explanation available.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Recommendation generation error: {e}")
            return "Error generating recommendation explanation."


class GenerateOriginalEmbeddingsView(BaseEmbeddingView):
    def get(self, request, *args, **kwargs):
        try:
            data, embeddings, index = self.generate_original_embeddings()
            self.save_cache(data, embeddings, index)
            return HttpResponse("Embeddings and IVF index generated successfully!")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return HttpResponseServerError(f"Error: {str(e)}")

    def generate_original_embeddings(self):
        try:
            with open(FILM_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ValueError("Film data file not found")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in film data file")

        data_texts = [self.json_to_text(item) for item in data]
        embeddings = self.generate_embeddings(data_texts)

        quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
        nlist = 100  # Number of IVF clusters
        index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist, faiss.METRIC_L2)
        index.train(embeddings)
        index.add(embeddings)

        return data, embeddings, index
