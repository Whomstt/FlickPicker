import json
import re
from rapidfuzz import fuzz
import time
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.shortcuts import render, redirect
from django.contrib import messages

from chatbot.config import (
    N_TOP_MATCHES,
    SEARCH_INCREMENT,
    MAX_RESULTS,
    NPROBE_INCREMENT,
)


def prepare_top_matches(
    data,
    distances,
    indices,
    detected_names=None,
    detected_genres=None,
    index=None,
    query_vector=None,
):
    """
    Prepare the top film matches for the user.
    """

    def assign_match_flags(film):
        # Name match flag
        if detected_names:
            lower_names = set(detected_names)
            directors = [director.lower() for director in film.get("directors", [])]
            actors = [actor.lower() for actor in film.get("main_actors", [])]
            film["name_match"] = any(
                director in lower_names for director in directors
            ) or any(actor in lower_names for actor in actors)
        else:
            film["name_match"] = False

        # Genre match flag
        if detected_genres:
            lower_genres = set(detected_genres)
            film["genre_match"] = any(
                genre.lower() in lower_genres for genre in film.get("genres", [])
            )
        else:
            film["genre_match"] = False

        return film

    # Set to track unique films and prevent duplicates
    unique_films = set()
    matches = []

    # Process initial FAISS search results
    for sim, idx in zip(distances[0], indices[0]):
        # Sanitize similarity score
        cosine_sim = max(min(float(sim), 1.0), 0.0)

        # Skip if film already processed
        if idx in unique_films:
            continue

        l2_distance = (2 - 2 * cosine_sim) ** 0.5
        film = {
            **data[idx],
            "cosine_similarity": cosine_sim,
            "l2_distance": l2_distance,
        }
        film = assign_match_flags(film)
        matches.append(film)
        unique_films.add(idx)

    # Sort matches in descending order
    matches.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    # If no detected names or genres, return top matches
    if not (detected_names or detected_genres):
        return matches[:N_TOP_MATCHES]

    # Determine filtering condition based on detected entities
    if detected_names and detected_genres:
        condition = lambda film: film["name_match"] and film["genre_match"]
    elif detected_names:
        condition = lambda film: film["name_match"]
    elif detected_genres:
        condition = lambda film: film["genre_match"]

    # Filter matches based on the condition
    filtered = [film for film in matches if condition(film)]

    current_k = 0
    current_nprobe = 0
    unique_filtered_films = set()

    # Expand search if necessary
    while current_k < MAX_RESULTS and len(filtered) < N_TOP_MATCHES:
        current_k += SEARCH_INCREMENT
        print(f"Searching for {current_k} results...")

        # Increase nprobe
        current_nprobe += NPROBE_INCREMENT
        if hasattr(index, "nprobe"):
            index.nprobe = current_nprobe
            print(f"nprobe set to {index.nprobe}")

        distances, indices = index.search(query_vector, current_k)

        for sim, idx in zip(distances[0], indices[0]):
            # Sanitize similarity score
            cosine_sim = max(min(float(sim), 1.0), 0.0)

            # Skip if film already processed or not unique in filtered set
            if idx in unique_films or idx in unique_filtered_films:
                continue

            l2_distance = (2 - 2 * cosine_sim) ** 0.5
            film = {
                **data[idx],
                "cosine_similarity": cosine_sim,
                "l2_distance": l2_distance,
            }
            film = assign_match_flags(film)

            if condition(film):
                filtered.append(film)
                unique_filtered_films.add(idx)

            unique_films.add(idx)

        # Sort and truncate to prevent excessive growth
        filtered.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        filtered = filtered[:MAX_RESULTS]

    # Supplement with best unfiltered matches
    supplement = [m for m in matches if m not in filtered]
    supplement.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    # Final match selection
    if len(filtered) >= N_TOP_MATCHES:
        return filtered[:N_TOP_MATCHES]
    else:
        remaining_needed = N_TOP_MATCHES - len(filtered)
        return filtered + supplement[:remaining_needed]
