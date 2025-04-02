import json
import re
from rapidfuzz import fuzz
from django.shortcuts import render, redirect
from django.contrib import messages

from chatbot.config import (
    NAME_FUZZY_THRESHOLD,
    GENRE_FUZZY_THRESHOLD,
)


def load_json(json_path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def find_names_in_prompt(prompt, json_path="actors_directors.json"):
    """
    Detect candidate names in user's prompt
    """
    names_data = load_json(json_path)
    detected_names = set()
    prompt_lower = prompt.lower()

    # Add word boundary markers to the prompt for more accurate matching
    prompt_with_boundaries = f" {prompt_lower} "

    candidate_names = names_data.get("unique_main_actors", []) + names_data.get(
        "unique_directors", []
    )

    for name in candidate_names:
        name_lower = name.lower()

        # Skip very short names
        if len(name_lower) < 4:
            continue

        # Word boundary check
        if f" {name_lower} " in prompt_with_boundaries:
            detected_names.add(name)
            continue

        # For partial names that might be at the start or end of the prompt
        if prompt_lower.startswith(f"{name_lower} ") or prompt_lower.endswith(
            f" {name_lower}"
        ):
            detected_names.add(name)
            continue

        # Only if the above checks fail, use fuzzy matching
        score = fuzz.partial_ratio(name_lower, prompt_lower)

        # Higher threshold for shorter names to avoid false positives
        adjusted_threshold = NAME_FUZZY_THRESHOLD + max(0, 8 - len(name_lower)) * 5

        if score >= adjusted_threshold:
            detected_names.add(name)

    # Filter out shorter names that are substrings of longer names
    sorted_names = sorted(detected_names, key=lambda x: len(x), reverse=True)
    filtered_names = []
    for name in sorted_names:
        name_lower = name.lower()
        # Check against all already added names
        if not any(name_lower in existing.lower() for existing in filtered_names):
            filtered_names.append(name)

    return filtered_names


def find_genres_in_prompt(prompt, json_path="genres.json"):
    """
    Detect candidate genres in user's prompt
    """
    genres_data = load_json(json_path)
    genre_alternatives = load_json("genre_alternatives.json")
    detected_genres = set()
    prompt_lower = prompt.lower()

    # Add word boundary markers to the prompt
    prompt_with_boundaries = f" {prompt_lower} "

    for genre in genres_data.get("unique_genres", []):
        genre_lower = genre.lower()

        # Skip very short genres
        if len(genre_lower) < 3:
            continue

        # Word boundary check
        if f" {genre_lower} " in prompt_with_boundaries:
            detected_genres.add(genre_lower)
            continue

        # For genres that might be at the start or end of the prompt
        if prompt_lower.startswith(f"{genre_lower} ") or prompt_lower.endswith(
            f" {genre_lower}"
        ):
            detected_genres.add(genre_lower)
            continue

        # Only if the above checks fail, use fuzzy matching
        score = fuzz.partial_ratio(genre_lower, prompt_lower)

        # Higher threshold for shorter genres
        adjusted_threshold = GENRE_FUZZY_THRESHOLD + max(0, 6 - len(genre_lower)) * 5

        if score >= adjusted_threshold:
            detected_genres.add(genre_lower)

    # Check for alternatives
    for alternative, canonical in genre_alternatives.items():
        # Simple boundary check for alternatives
        if f" {alternative.lower()} " in prompt_with_boundaries:
            detected_genres.add(canonical.lower())
            continue

        # For alternatives at the start or end of the prompt
        if prompt_lower.startswith(f"{alternative.lower()} ") or prompt_lower.endswith(
            f" {alternative.lower()}"
        ):
            detected_genres.add(canonical.lower())
            continue

        # Only use fuzzy matching if necessary
        score = fuzz.partial_ratio(alternative.lower(), prompt_lower)
        if score >= GENRE_FUZZY_THRESHOLD:
            detected_genres.add(canonical.lower())

    return list(detected_genres)
