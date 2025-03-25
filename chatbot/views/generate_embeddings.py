import json
import logging
import aiohttp
import numpy as np
import faiss
from asgiref.sync import sync_to_async
from django.shortcuts import render

from .base_embedding import BaseEmbeddingView
from chatbot.config import TMDB_OUTPUT_FILE, EMBEDDING_DIM, NLIST, M, NBITS

from django.http import JsonResponse
from django.views.decorators.http import require_POST


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

cancel_generate = False


class GenerateOriginalEmbeddingsView(BaseEmbeddingView):
    """
    View for generating embeddings for the original film data.
    """

    async def post(self, request, *args, **kwargs):
        """
        Generate embeddings for the original film data.
        """
        data, embeddings, index = await self.generate_original_embeddings()
        if data is None:
            message = "Embedding generation was cancelled"
        else:
            self.save_cache(data, embeddings, index)
            message = "Embeddings and index generated successfully!"
        return await sync_to_async(render)(request, "admin.html", {"message": message})

    async def generate_original_embeddings(self):
        """
        Generate embeddings for the original film data using a single enriched text block per film.
        """
        global cancel_generate
        counter = 1
        with open(TMDB_OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"Fetched {len(data)} films from {TMDB_OUTPUT_FILE}")
        embeddings = []
        cancelled = False
        async with aiohttp.ClientSession() as session:
            for item in data:
                if cancel_generate:
                    logging.info("Embedding generation cancelled by user")
                    cancelled = True
                    break
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

        # Handle cancellation
        if cancelled:
            cancel_generate = False
            return None, None, None

        # Proceed only if not cancelled
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


@require_POST
def cancel_generate_view(request):
    """
    Cancel the embedding generation process
    """
    global cancel_generate
    cancel_generate = True
    return JsonResponse({"status": "Embedding generation cancelled"})
