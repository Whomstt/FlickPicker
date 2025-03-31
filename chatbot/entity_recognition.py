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
    NPROBE,
    N_TOP_MATCHES,
    OPENAI_API_URL,
    OPENAI_MODEL,
    SEARCH_INCREMENT,
    MAX_RESULTS,
    MAX_TOKENS,
    TEMPERATURE,
    PROMPT_WEIGHT,
    NAME_WEIGHT,
    GENRE_WEIGHT,
    NAME_FUZZY_THRESHOLD,
    GENRE_FUZZY_THRESHOLD,
)


def sliding_window_fuzzy(prompt, candidate, threshold):
    """Slide window over prompt and search fuzzy match"""
    prompt = prompt.lower()
    candidate = candidate.lower()
    prompt_words = prompt.split()
    candidate_words = candidate.split()
    window_size = len(candidate_words)

    # Slide over prompt tokens
    for i in range(len(prompt_words) - window_size + 1):
        window = " ".join(prompt_words[i : i + window_size])
        # Require an exact match first
        if window == candidate:
            return True
        # Use a fuzzy ratio for partial matches
        score = fuzz.ratio(candidate, window)
        if score >= threshold:
            return True
    return False


def find_names_in_prompt(prompt, json_path="actors_directors.json"):
    """
    Detect candidate names in user's prompt
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError):
        return []

    detected_names = set()

    def regex_match(name, text):
        pattern = r"\b" + re.escape(name) + r"\b"
        return re.search(pattern, text, re.IGNORECASE) is not None

    for name in data.get("unique_main_actors", []):
        # Try exact match first
        if regex_match(name, prompt):
            detected_names.add(name.lower())
        # If exact match fails we use fuzzy matching
        elif sliding_window_fuzzy(prompt, name, threshold=NAME_FUZZY_THRESHOLD):
            detected_names.add(name.lower())

    for name in data.get("unique_directors", []):
        if regex_match(name, prompt):
            detected_names.add(name.lower())
        elif sliding_window_fuzzy(prompt, name, threshold=NAME_FUZZY_THRESHOLD):
            detected_names.add(name.lower())

    return list(detected_names)


def find_genres_in_prompt(prompt, json_path="genres.json"):
    """
    Detect candidate genres in user's prompt
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError):
        return []

    # Map alternative genre names to match their canonical form
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

    detected_genres = set()

    def regex_match(genre, text):
        pattern = r"\b" + re.escape(genre) + r"\b"
        return re.search(pattern, text, re.IGNORECASE) is not None

    for genre in data.get("unique_genres", []):
        # Try exact match first
        if regex_match(genre, prompt):
            detected_genres.add(genre.lower())
        # If exact match fails we use fuzzy matching
        elif sliding_window_fuzzy(prompt, genre, threshold=GENRE_FUZZY_THRESHOLD):
            detected_genres.add(genre.lower())

    for alternative, genre in genre_alternatives.items():
        if alternative.lower() in prompt.lower():
            detected_genres.add(genre.lower())
    return list(detected_genres)
