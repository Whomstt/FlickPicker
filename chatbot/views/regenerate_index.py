import json
import logging
import os
import numpy as np
import faiss
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils.decorators import method_decorator
from asgiref.sync import sync_to_async
from django.shortcuts import render
from chatbot.views.base_embedding import BaseEmbeddingView
from chatbot.config import CACHE_DIR, TMDB_OUTPUT_FILE, EMBEDDING_DIM, NLIST, M, NBITS


# Global flag to track cancellation
cancel_regenerate_index = False


class RegenerateFaissIndexView(BaseEmbeddingView):
    """
    View for regenerating the FAISS index without regenerating embeddings
    """

    async def post(self, request, *args, **kwargs):
        """
        Regenerate FAISS index using existing embeddings
        """
        global cancel_regenerate_index
        cancel_regenerate_index = False
        success, message = await self.regenerate_faiss_index()
        return await sync_to_async(render)(request, "admin.html", {"message": message})

    async def regenerate_faiss_index(self):
        """
        Regenerate FAISS index from existing embeddings without recomputing them
        """
        global cancel_regenerate_index

        try:
            # Load data
            with open(TMDB_OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} films from {TMDB_OUTPUT_FILE}")

            # Load the embeddings
            embeddings_path = os.path.join(CACHE_DIR, "film_embeddings.npy")
            if not os.path.exists(embeddings_path):
                logging.error(f"Embeddings file not found at {embeddings_path}")
                return False, "Embeddings file not found, cannot regenerate index."

            embeddings = np.load(embeddings_path)
            logging.info(f"Loaded embeddings with shape {embeddings.shape}")

            # Check if data and embeddings counts match
            if len(data) != embeddings.shape[0]:
                logging.warning(
                    f"Data count ({len(data)}) doesn't match embeddings count ({embeddings.shape[0]})"
                )

            # Check for cancellation
            if cancel_regenerate_index:
                logging.info("Index regeneration cancelled by user")
                cancel_regenerate_index = False
                return False, "Index regeneration was cancelled"

            # Ensure embeddings are normalized
            faiss.normalize_L2(embeddings)

            # Create FAISS index with the same configuration
            quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
            index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, NLIST, M, NBITS)

            # Train and add embeddings to index
            logging.info("Training FAISS index...")
            index.train(embeddings)

            # Check for cancellation after training
            if cancel_regenerate_index:
                logging.info("Index regeneration cancelled by user after training")
                cancel_regenerate_index = False
                return False, "Index regeneration was cancelled"

            logging.info("Adding vectors to index...")
            index.add(embeddings)

            # Save the index
            logging.info(f"Index created with {index.ntotal} vectors. Saving...")
            self.save_cache(data, embeddings, index)

            return True, "FAISS index regenerated successfully!"

        except Exception as e:
            logging.error(f"Error regenerating FAISS index: {str(e)}")
            return False, f"Error regenerating FAISS index: {str(e)}"


@require_POST
def cancel_regenerate_index_view(request):
    """
    Cancel the index regeneration process
    """
    global cancel_regenerate_index
    cancel_regenerate_index = True
    return JsonResponse({"status": "Index regeneration cancelled"})
