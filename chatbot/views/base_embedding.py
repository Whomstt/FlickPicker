import os
import json
import numpy as np
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.views import View

from chatbot.config import (
    CACHE_DIR,
    FAISS_INDEX_PATH,
    RAW_FILM_DATA_PATH,
    OPENAI_API_KEY,
    OPENAI_API_URL,
    OPENAI_MODEL,
    NOMIC_API_KEY,
    NOMIC_API_URL,
    NOMIC_MODEL,
    EMBEDDING_DIM,
    OLLAMA_URL,
    EMBEDDING_MODEL,
    FIELD_WEIGHTS,
)


class BaseEmbeddingView(View):
    """
    Base class for views that require embedding generation.
    Contains methods for API requests, generating and combining embeddings,
    caching data, and converting JSON to text.
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
        """Fetch embeddings from specified service."""
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

                # Handle different service types
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
        if overview := item.get("overview"):
            components.append(f"Overview: {overview}")
        if tagline := item.get("tagline"):
            components.append(f"Tagline: {tagline}")
        if keywords := item.get("keywords"):
            components.append(f"Keywords: {', '.join(keywords)}")
        if director := item.get("director"):
            components.append(f"Directed by {director}")
        if main_actors := item.get("main_actors"):
            components.append(f"Featuring: {', '.join(main_actors)}")
        if runtime := item.get("runtime"):
            components.append(f"Runtime: {runtime}")
        if release_date := item.get("release_date"):
            components.append(f"Release Date: {release_date}")

        # Combine all non-empty components into a single string
        return "\n".join(components)
