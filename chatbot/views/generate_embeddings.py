import json
import numpy as np
import faiss
from asgiref.sync import sync_to_async
from django.shortcuts import render

from .base_embedding import BaseEmbeddingView
from chatbot.config import TMDB_OUTPUT_FILE, EMBEDDING_DIM, NLIST, M, NBITS


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
        Generate embeddings for the original film data using field-specific weighting.
        """
        with open(TMDB_OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = []

        for item in data:
            # Generate embeddings for each field
            field_embeddings = await self.generate_field_embeddings(
                item, use_ollama=True
            )

            # Combine embeddings with weighting
            combined_embedding = self.combine_weighted_embeddings(field_embeddings)
            embeddings.append(combined_embedding)

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
