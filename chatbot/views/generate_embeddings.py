import json
import logging
import aiohttp
import numpy as np
import faiss
from asgiref.sync import sync_to_async
from django.shortcuts import render

from .base_embedding import BaseEmbeddingView
from chatbot.config import TMDB_OUTPUT_FILE, EMBEDDING_DIM, NLIST, M, NBITS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
        Generate embeddings for the original film data using a single enriched text block per film.
        """
        counter = 1
        with open(TMDB_OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Fetched {len(data)} films from {TMDB_OUTPUT_FILE}")
        embeddings = []
        async with aiohttp.ClientSession() as session:
            for item in data:
                # Enrich the text
                film_text = self.enrich_text(item)
                # Embed the text
                embedding = await self.fetch_embedding(
                    film_text, session, service="ollama"
                )
                embeddings.append(embedding)
                # Log progress
                logging.info(f"Processed film {counter}/{len(data)}")
                counter += 1

        embeddings = np.array(embeddings, dtype="float32")

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        quantizer = faiss.IndexFlatIP(
            EMBEDDING_DIM
        )  # IndexFlatIP for inner product (cosine similarity)
        index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, NLIST, M, NBITS)
        index.train(embeddings)
        index.add(embeddings)

        return data, embeddings, index
