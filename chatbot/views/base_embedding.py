import os
import json
import numpy as np
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.views import View
from datetime import datetime

from chatbot.config import (
    CACHE_DIR,
    FAISS_INDEX_PATH,
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
    TMDB_OUTPUT_FILE,
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
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    return embedding
                else:
                    return np.zeros(EMBEDDING_DIM)

        except Exception as e:
            print(f"Error from {service}: {str(e)}")
            return np.zeros(EMBEDDING_DIM)

    async def generate_field_embeddings(self, item, use_ollama=False):
        """
        Generate embeddings for each field in the film item separately,
        using enriched text for film data.
        """
        field_embeddings = {}
        async with aiohttp.ClientSession() as session:
            for field, weight in FIELD_WEIGHTS.items():
                if (value := item.get(field)) is not None:

                    # Convert value to string if it's a list
                    text_value = (
                        ", ".join(map(str, value))
                        if isinstance(value, list)
                        else str(value)
                    )
                    enriched_text = self.enrich_field_text(field, text_value)
                    service = "ollama" if use_ollama else "nomic"
                    embedding = await self.fetch_embedding(
                        enriched_text, session, service
                    )
                    field_embeddings[field] = (embedding, weight)
        return field_embeddings

    def combine_weighted_embeddings(self, field_embeddings):
        """
        Combine field embeddings using their weights
        """
        weighted_sum = np.zeros(EMBEDDING_DIM, dtype="float32")

        for field, (embedding, weight) in field_embeddings.items():
            weighted_sum += embedding * weight

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
            os.path.join(TMDB_OUTPUT_FILE),
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
            components.append(f"Title: {title}")
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

    def enrich_field_text(self, field, text_value):
        """
        Enriches the film field text with more descriptive language to match user prompts.
        """
        if field == "title":
            return f"Title: {text_value}"
        elif field == "genres":
            return f"Genres: {text_value}"
        elif field == "overview":
            return f"Overview: {text_value}"
        elif field == "tagline":
            return f"Tagline: {text_value}"
        elif field == "keywords":
            return f"Keywords: {text_value}"
        elif field == "director":
            return f"Directed by {text_value}"
        elif field == "main_actors":
            return f"Featuring: {text_value}"
        elif field == "runtime":
            runtime = int(text_value)
            if 0 <= runtime <= 90:
                label = "Short"
            elif 91 <= runtime <= 120:
                label = "Average"
            else:
                label = "Long"
            return f"Runtime: {label} ({text_value} minutes)"
        elif field == "release_date":
            year = int(text_value.split("-")[0])
            current_year = datetime.now().year
            age = current_year - year
            if 0 <= age < 2:
                label = "New"
            elif 2 <= age < 15:
                label = "Modern"
            else:
                label = "Old"
            return f"Release Date: {label} ({text_value})"
        else:
            return f"{field.capitalize()}: {text_value}"
