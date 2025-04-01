import json
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
    Detect candidate names in user's prompt using RapidFuzz fuzzy matching and filter out substrings.
    """
    data = load_json(json_path)
    detected_names = set()
    prompt_lower = prompt.lower()

    candidate_names = data.get("unique_main_actors", []) + data.get(
        "unique_directors", []
    )
    for name in candidate_names:
        score = fuzz.partial_ratio(name.lower(), prompt_lower)
        if score >= NAME_FUZZY_THRESHOLD:
            detected_names.add(name)  # Store original name with correct casing

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
    Detect candidate genres in user's prompt using RapidFuzz fuzzy matching
    """
    data = load_json(json_path)
    detected_genres = set()
    prompt_lower = prompt.lower()

    for genre in data.get("unique_genres", []):
        score = fuzz.partial_ratio(genre.lower(), prompt_lower)
        if score >= GENRE_FUZZY_THRESHOLD:
            detected_genres.add(genre.lower())

    # Map alternative genre names to canonical forms
    genre_alternatives = {
        "scifi": "science fiction",
        "sci-fi": "science fiction",
        "sci fi": "science fiction",
        "rom": "romantic",
        "com": "comedy",
        "doc": "documentary",
        "historic": "history",
        "historical": "history",
        "musical": "music",
    }
    for alternative, canonical in genre_alternatives.items():
        score = fuzz.partial_ratio(alternative.lower(), prompt_lower)
        if score >= GENRE_FUZZY_THRESHOLD:
            detected_genres.add(canonical.lower())
    return list(detected_genres)
