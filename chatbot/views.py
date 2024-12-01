import os
import json
import logging
import numpy as np
import faiss
import requests
from concurrent.futures import ThreadPoolExecutor
from django.views import View
from django.shortcuts import render
from django.http import HttpResponseServerError
from django.conf import settings

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(settings.BASE_DIR, "cache")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
EMBEDDING_DIM = 768
MOVIES_DATA_PATH = os.path.join(settings.BASE_DIR, "movies_data.json")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_GENERATION_MODEL = "llama3.2"
OLLAMA_URL = "http://ollama:11434/api"


class MovieRecommendationView(View):
    def get(self, request, *args, **kwargs):
        try:
            return render(request, "chat.html")
        except Exception as e:
            logger.error(f"Error in GET request: {e}")
            return HttpResponseServerError("Internal server error")

    def post(self, request, *args, **kwargs):
        try:
            # Get the prompt from the POST request
            prompt = request.POST.get("prompt", "").strip()
            if not prompt:
                return render(request, "chat.html", {"error": "Please enter a prompt"})
            # Load or generate embeddings for films
            data, embeddings, index = self.load_or_generate_embeddings()
            # Fetch the embedding for the prompt
            prompt_embedding = self.fetch_embedding(prompt).reshape(1, -1)
            # Search for the top 3 matches
            distances, indices = index.search(prompt_embedding, 3)
            # Prepare the top matches for display
            top_matches = self.prepare_top_matches(data, distances, indices)
            # Generate recommendation explanation
            explanation = self.generate_recommendation_explanation(prompt, top_matches)

            return render(
                request, "chat.html", {"response": explanation, "matches": top_matches}
            )

        except Exception as e:
            logger.error(f"Error in POST request: {e}", exc_info=True)
            return render(
                request,
                "chat.html",
                {"error": f"An unexpected error occurred: {str(e)}"},
            )

    # Load existing embeddings for films or generate new ones
    def load_or_generate_embeddings(self):
        # Load cache if available
        data, embeddings, index = self.load_cache()
        # If cache is not available, generate new embeddings
        if data is None:
            try:
                # Load movies data from disk
                with open(MOVIES_DATA_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except FileNotFoundError:
                raise ValueError("Movies data file not found")
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in movies data file")
            # Convert JSON data to text for embedding
            data_texts = [self.json_to_text(item) for item in data]
            # Generate embeddings for the data
            embeddings = self.generate_embeddings(data_texts)
            # Create a FAISS index for the embeddings (Euclidean distance)
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            # Add the embeddings to the index
            index.add(embeddings)
            # Save the cache to disk
            self.save_cache(data, embeddings, index)
        return data, embeddings, index

    # Generate embeddings for a given text
    def fetch_embedding(self, text):
        try:
            # Request the embedding from the OLLAMA service
            response = requests.post(
                f"{OLLAMA_URL}/embeddings",
                json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text},
            )
            response.raise_for_status()
            # Extract the embedding from the response
            embedding = response.json().get("embedding")
            if embedding is None:
                raise ValueError("Embedding not found in response")
            return np.array(embedding, dtype="float32")
        except requests.exceptions.RequestException as e:
            # Dimension mismatch
            logger.error(f"Embedding request error: {e}")
            return np.zeros(EMBEDDING_DIM, dtype="float32")

    # Generate embeddings for a list of texts in parallel
    def generate_embeddings(self, data_texts):
        with ThreadPoolExecutor() as executor:
            embeddings = list(executor.map(self.fetch_embedding, data_texts))
        # Return the embeddings as a NumPy array for FAISS
        return np.array(embeddings, dtype="float32")

    # Prepare top matches for display
    def prepare_top_matches(self, data, distances, indices):
        return [
            {**data[idx], "similarity_distance": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
        ]

    # Generate recommendation explanation
    def generate_recommendation_explanation(self, prompt, top_matches):
        try:
            full_prompt = f"Query: {prompt}\n\n"
            full_prompt += "\n\n".join(self.json_to_text(item) for item in top_matches)
            full_prompt += "\n\nProvide a detailed movie recommendation explanation."

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

    # Convert JSON to text
    def json_to_text(self, item, include_all_fields=False):
        fields = [f"Title: {item.get('title', 'N/A')}"]
        key_fields = [
            ("Genres", "genres"),
            ("Overview", "overview"),
            ("Director", "director"),
            ("Main Actors", "main_actors"),
            ("Runtime", "runtime"),
            ("Release Date", "release_date"),
            ("Country", "country_of_production"),
            ("Languages", "spoken_languages"),
            ("Tagline", "tagline"),
            ("Budget", "budget"),
            ("Revenue", "revenue"),
        ]
        for label, key in key_fields:
            value = item.get(key, [])
            value = ", ".join(value) if isinstance(value, list) else value or "N/A"
            fields.append(f"{label}: {value}")

        return "\n".join(fields)

    # Save cache to disk for future use
    def save_cache(self, data, embeddings, index):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(
            os.path.join(CACHE_DIR, f"{os.path.basename(MOVIES_DATA_PATH)}_data.json"),
            "w",
        ) as f:
            json.dump(data, f)
        np.save(
            os.path.join(
                CACHE_DIR, f"{os.path.basename(MOVIES_DATA_PATH)}_embeddings.npy"
            ),
            embeddings,
        )
        faiss.write_index(index, FAISS_INDEX_PATH)

    # Load cache from disk if available
    def load_cache(self):
        data_path = os.path.join(
            CACHE_DIR, f"{os.path.basename(MOVIES_DATA_PATH)}_data.json"
        )
        embeddings_path = os.path.join(
            CACHE_DIR, f"{os.path.basename(MOVIES_DATA_PATH)}_embeddings.npy"
        )

        if not (
            os.path.exists(data_path)
            and os.path.exists(embeddings_path)
            and os.path.exists(FAISS_INDEX_PATH)
        ):
            return None, None, None

        with open(data_path, "r") as f:
            data = json.load(f)

        embeddings = np.load(embeddings_path)
        index = faiss.read_index(FAISS_INDEX_PATH)

        return data, embeddings, index
