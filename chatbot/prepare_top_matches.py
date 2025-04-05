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
    NPROBE,
)


def prepare_top_matches(
    data,
    distances,
    indices,
    detected_names=None,
    detected_genres=None,
    detected_keywords=None,
    detected_titles=None,
    index=None,
    query_vector=None,
):
    """
    Prepare the top film matches for the user
    """

    def assign_match_flags(film):
        # Name match flag
        if detected_names:
            lower_names = set(name.lower() for name in detected_names)
            directors = [director.lower() for director in film.get("directors", [])]
            actors = [actor.lower() for actor in film.get("main_actors", [])]
            film["name_match"] = any(
                director in lower_names for director in directors
            ) or any(actor in lower_names for actor in actors)
        else:
            film["name_match"] = False

        # Genre match flag
        if detected_genres:
            lower_genres = set(genre.lower() for genre in detected_genres)
            film["genre_match"] = any(
                genre.lower() in lower_genres for genre in film.get("genres", [])
            )
        else:
            film["genre_match"] = False

        # Keyword match flag
        if detected_keywords:
            lower_keywords = set(keyword.lower() for keyword in detected_keywords)
            film["keyword_match"] = any(
                keyword.lower() in lower_keywords
                for keyword in film.get("keywords", [])
            )
        else:
            film["keyword_match"] = False

        # Title match flag
        if detected_titles:
            lower_titles = set(title.lower() for title in detected_titles)
            film["title_match"] = film.get("title", "").lower() in lower_titles
        else:
            film["title_match"] = False

        return film

    # Set to track unique films and prevent duplicates
    unique_films = set()
    matches = []

    # Process initial FAISS search results
    for sim, idx in zip(distances[0], indices[0]):
        cosine_sim = max(min(float(sim), 1.0), 0.0)
        # Skip exact match error
        if cosine_sim == 1.0:
            continue

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

    # If no entities for names or genres were detected we return the top N matches
    if not (detected_names or detected_genres):
        matches.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return matches[:N_TOP_MATCHES]

    # If both names and genres are detected we require both matches
    def condition(film):
        if detected_names and detected_genres:
            return film["name_match"] and film["genre_match"]
        elif detected_names:
            return film["name_match"]
        elif detected_genres:
            return film["genre_match"]
        return False

    filtered = [film for film in matches if condition(film)]
    filtered.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    filtered = filtered[:N_TOP_MATCHES]

    # Only expand if the filtered results (with matching names/genres) are not filled
    initial_k = len(indices[0])
    current_k = initial_k - 5
    previous_nprobe = NPROBE  # Default starting nprobe
    if hasattr(index, "nprobe"):
        previous_nprobe = index.nprobe

    while current_k < MAX_RESULTS and len(filtered) < N_TOP_MATCHES:
        next_k = current_k + SEARCH_INCREMENT

        # For the first expansion, use the same nprobe value; then increase it
        if current_k == initial_k - 5:
            next_nprobe = previous_nprobe
        else:
            next_nprobe = previous_nprobe + NPROBE_INCREMENT

        print(f"Expanding search: k={next_k}, nprobe={next_nprobe}")

        if hasattr(index, "nprobe"):
            index.nprobe = next_nprobe

        new_distances, new_indices = index.search(query_vector, next_k)

        for sim, idx in zip(new_distances[0][:next_k], new_indices[0][:next_k]):
            if idx in unique_films:
                continue

            cosine_sim = max(min(float(sim), 1.0), 0.0)
            # Skip exact match error
            if cosine_sim == 1.0:
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

            if condition(film):
                filtered.append(film)

        # Update search parameters for the next iteration
        current_k = next_k
        previous_nprobe = next_nprobe

        # Sort and restrict filtered results
        filtered.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        filtered = filtered[:N_TOP_MATCHES]

        if current_k >= MAX_RESULTS:
            break

    # Return only films that pass the matching condition
    return filtered[:N_TOP_MATCHES]
