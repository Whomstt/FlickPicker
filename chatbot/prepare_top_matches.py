import json
import re
from rapidfuzz import fuzz
import time
import faiss
import aiohttp
from asgiref.sync import sync_to_async
from django.shortcuts import render, redirect
from django.contrib import messages
from datetime import datetime

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
    detected_runtime=None,
    detected_release=None,
    index=None,
    query_vector=None,
):
    """
    Prepare the top film matches for the user
    """

    def film_priority(film):
        """
        Assign a sorting priority based on matching detected entities
        """
        base_priority = 100  # Start with a high number (low priority)
        bonus_points = 0

        # Calculate bonus points for additional matches
        if film.get("keyword_match", False):
            bonus_points += 1
        if film.get("title_match", False):
            bonus_points += 1
        if film.get("runtime_match", False):
            bonus_points += 1
        if film.get("release_match", False):
            bonus_points += 1

        # First bracket: detected names AND detected genres
        if detected_names and detected_genres:
            if film["name_match"] and film["genre_match"]:
                base_priority = 1
            elif film["name_match"]:
                base_priority = 2
            elif film["genre_match"]:
                base_priority = 3
            else:
                base_priority = 4

        # Second bracket: detected names only
        elif detected_names:
            base_priority = 1 if film["name_match"] else 2

        # Third bracket: detected genres only
        elif detected_genres:
            base_priority = 1 if film["genre_match"] else 2

        # No detected entities
        else:
            base_priority = 1

        return base_priority - (bonus_points * 0.1)

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

        # Runtime match flag
        if detected_runtime:
            runtime_category = detected_runtime[0].lower()
            runtime_raw = film.get("runtime")
            try:
                runtime_val = int(runtime_raw)
                if runtime_val < 90:
                    film["runtime_match"] = runtime_category == "short"
                elif runtime_val <= 120:
                    film["runtime_match"] = runtime_category == "medium"
                else:
                    film["runtime_match"] = runtime_category == "long"
            except (ValueError, TypeError):
                film["runtime_match"] = False
        else:
            film["runtime_match"] = False

        # Release date match flag
        if detected_release:
            release_category = detected_release[0].lower()
            release_val = film.get("release_date")
            if release_val:
                match = re.match(r"(\d{4})", release_val)
                if match:
                    year = int(match.group(1))
                    current_year = datetime.now().year
                    age = current_year - year
                    if age < 2:
                        film["release_match"] = release_category == "new"
                    elif age < 15:
                        film["release_match"] = release_category == "modern"
                    else:
                        film["release_match"] = release_category == "old"
                else:
                    film["release_match"] = False
            else:
                film["release_match"] = False
        else:
            film["release_match"] = False

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

    # # Print the original films found in similarity search before filtering
    # print("Initial films found in similarity search:\n")

    # fields = [
    #     ("title", "Title"),
    #     ("genres", "Genres"),
    #     ("overview", "Overview"),
    #     ("directors", "Directors"),
    #     ("main_actors", "Main Actors"),
    #     ("runtime", "Runtime"),
    #     ("release_date", "Release Date"),
    #     ("keywords", "Keywords"),
    # ]

    # for idx, film in enumerate(matches[:5], start=1):
    #     print(f"Film {idx}:")
    #     print("\n".join(f"{label}: {film.get(key, 'N/A')}" for key, label in fields))
    #     print()

    # If no entities detected return top matches based on cosine similarity
    if not (
        detected_names
        or detected_genres
        or detected_runtime
        or detected_release
        or detected_keywords
    ):
        matches.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return matches[:N_TOP_MATCHES]

    # Define condition for films passing all applicable checks
    def condition(film):
        checks = []
        if detected_names:
            checks.append(film["name_match"])
        if detected_genres:
            checks.append(film["genre_match"])
        if detected_runtime:
            checks.append(film["runtime_match"])
        if detected_release:
            checks.append(film["release_match"])
        if detected_keywords:
            checks.append(film["keyword_match"])
        return all(checks)

    # Select films that meet all detected entities
    filtered = [film for film in matches if condition(film)]
    filtered.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    filtered = filtered[:N_TOP_MATCHES]

    # Expand search if needed
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

    # Supplement with films that have at least a name or genre match
    if len(filtered) < N_TOP_MATCHES:
        needed = N_TOP_MATCHES - len(filtered)
        fallback_films = [film for film in matches if film not in filtered]

        fallback_step1 = []  # Both name and genre match
        fallback_step2 = []  # Only name match
        fallback_step3 = []  # The rest (either name or genre match)

        # Organize fallback films according to the desired steps
        for film in fallback_films:
            if detected_names and detected_genres:
                if film["name_match"] and film["genre_match"]:
                    fallback_step1.append(film)
                elif film["name_match"]:
                    fallback_step2.append(film)
                elif film["genre_match"]:
                    fallback_step3.append(film)
            elif detected_names:
                if film["name_match"]:
                    fallback_step1.append(film)
            elif detected_genres:
                if film["genre_match"]:
                    fallback_step1.append(film)

        # Helper to count additional matching flags (runtime, release, keywords)
        def count_rest_checks(film):
            count = 0
            if detected_runtime and film["runtime_match"]:
                count += 1
            if detected_release and film["release_match"]:
                count += 1
            if detected_keywords and film["keyword_match"]:
                count += 1
            return count

        # Sort each fallback list using the count of additional matches and cosine similarity
        fallback_step1.sort(
            key=lambda x: (count_rest_checks(x), x["cosine_similarity"]), reverse=True
        )
        filtered.extend(fallback_step1[:needed])
        needed = N_TOP_MATCHES - len(filtered)

        if needed > 0:
            fallback_step2.sort(
                key=lambda x: (count_rest_checks(x), x["cosine_similarity"]),
                reverse=True,
            )
            filtered.extend(fallback_step2[:needed])
            needed = N_TOP_MATCHES - len(filtered)

        if needed > 0:
            fallback_step3.sort(
                key=lambda x: (count_rest_checks(x), x["cosine_similarity"]),
                reverse=True,
            )
            filtered.extend(fallback_step3[:needed])

        filtered.sort(key=lambda x: (film_priority(x), -x["cosine_similarity"]))

    return filtered[:N_TOP_MATCHES]
