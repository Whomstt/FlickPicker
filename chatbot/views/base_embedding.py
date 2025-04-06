import os
import json
import numpy as np
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.views import View
from datetime import datetime
import re

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
        Load the data and index from the cache directory
        """
        data_path = os.path.join(TMDB_OUTPUT_FILE)
        index_path = FAISS_INDEX_PATH
        embeddings_path = os.path.join(CACHE_DIR, "film_embeddings.npy")

        # Check for required files (data and index)
        if not (os.path.exists(data_path) and os.path.exists(index_path)):
            return None, None, None

        with open(data_path, "r") as f:
            data = json.load(f)
        index = faiss.read_index(index_path)

        # Load embeddings if available; otherwise, set to None
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
        else:
            embeddings = None

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
        if directors := item.get("directors"):
            components.append(f"Directed by {directors}")
        if main_actors := item.get("main_actors"):
            components.append(f"Featuring: {', '.join(main_actors)}")
        if runtime := item.get("runtime"):
            components.append(f"Runtime: {runtime}")
        if release_date := item.get("release_date"):
            components.append(f"Release Date: {release_date}")

        # Combine all non-empty components into a single string
        return "\n".join(components)

    def clean_text(self, text):
        """
        Converts the text into lowercase, removes special characters, and extra spaces
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s,:\.]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def enrich_text(self, item):
        """
        Enrich film text data for embedding generation
        """
        enriched_parts = []
        # Process fields in a consistent order
        field_order = [
            "title",
            "genres",
            "overview",
            "tagline",
            "keywords",
            "directors",
            "main_actors",
            "runtime",
            "release_date",
        ]

        for field in field_order:
            value = item.get(field)
            if value is not None:
                # Join list values with commas
                text_value = (
                    ", ".join(map(str, value))
                    if isinstance(value, list)
                    else str(value)
                )
                # Clean the text while preserving useful punctuation.
                cleaned_value = self.clean_text(text_value)

                # Field-specific enrichment logic
                if field == "runtime":
                    try:
                        runtime = int(cleaned_value)
                        if runtime <= 90:
                            cleaned_value = "short"
                        elif runtime <= 120:
                            cleaned_value = "medium"
                        else:
                            cleaned_value = "long"
                    except Exception:
                        pass

                elif field == "release_date":
                    try:
                        # Extract a four-digit year using a regex.
                        match = re.search(r"(\d{4})", cleaned_value)
                        if match:
                            year = int(match.group(1))
                            current_year = datetime.now().year
                            age = current_year - year
                            if age < 2:
                                cleaned_value = "new"
                            elif age < 15:
                                cleaned_value = "modern"
                            else:
                                cleaned_value = "old"
                    except Exception:
                        pass

                # Append the enriched text with an explicit label
                enriched_parts.append(f"{field.capitalize()}: {cleaned_value}")
        return ", ".join(enriched_parts)
