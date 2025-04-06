import json
import re
from rapidfuzz import fuzz
from django.shortcuts import render, redirect
from django.contrib import messages

from chatbot.config import (
    PROMPT_FUZZY_THRESHOLD,
    NAME_FUZZY_THRESHOLD,
    GENRE_FUZZY_THRESHOLD,
    TITLE_FUZZY_THRESHOLD,
    KEYWORD_FUZZY_THRESHOLD,
    RUNTIME_FUZZY_THRESHOLD,
    RELEASE_FUZZY_THRESHOLD,
)


def load_json(json_path):
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def clean_prompt_with_fuzzy(prompt, detected_entities):
    """
    Remove all instances of detected entities from the prompt,
    including partial matches above the fuzzy threshold
    """
    cleaned_prompt = prompt

    # Sort entities by length (descending) to handle longer entities first
    # This prevents issues where parts of longer entities remain after removal
    sorted_entities = sorted(detected_entities, key=len, reverse=True)

    for entity in sorted_entities:
        # Create a pattern that matches the entity with word boundaries
        pattern = r"\b" + re.escape(entity) + r"\b"
        cleaned_prompt = re.sub(pattern, "", cleaned_prompt, flags=re.IGNORECASE)

        # Also check for fuzzy matches
        words = cleaned_prompt.split()
        cleaned_words = []

        for word in words:
            if fuzz.ratio(word.lower(), entity.lower()) < PROMPT_FUZZY_THRESHOLD:
                cleaned_words.append(word)

        cleaned_prompt = " ".join(cleaned_words)

    # Clean up any extra spaces
    cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).strip()

    return cleaned_prompt


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
        score = fuzz.ratio(name_lower, prompt_lower)

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
        score = fuzz.ratio(genre_lower, prompt_lower)

        # Higher threshold for shorter genres
        adjusted_threshold = GENRE_FUZZY_THRESHOLD + max(0, 6 - len(genre_lower)) * 5
        if score >= adjusted_threshold:
            detected_genres.add(genre_lower)

    # Check for alternative genres (e.g. "romcom" mapping to "romantic, comedy")
    for alternative, canonical in genre_alternatives.items():

        # Split the canonical genres by comma and strip whitespace
        canonicals = [c.strip() for c in canonical.split(",")]

        if f" {alternative.lower()} " in prompt_with_boundaries:
            for c in canonicals:
                detected_genres.add(c.lower())
            continue

        if prompt_lower.startswith(f"{alternative.lower()} ") or prompt_lower.endswith(
            f" {alternative.lower()}"
        ):
            for c in canonicals:
                detected_genres.add(c.lower())
            continue

        score = fuzz.ratio(alternative.lower(), prompt_lower)
        if score >= GENRE_FUZZY_THRESHOLD:
            for c in canonicals:
                detected_genres.add(c.lower())

    return list(detected_genres)


def find_titles_in_prompt(
    prompt, json_path="titles.json", bizarre_titles_path="bizarre_titles.json"
):
    """
    Detect candidate film titles in the user's prompt
    """
    titles_data = load_json(json_path)
    bizarre_titles_data = load_json(bizarre_titles_path)
    bizarre_titles = bizarre_titles_data.get("bizarre_titles", [])

    detected_titles = set()
    prompt_lower = prompt.lower()
    prompt_with_boundaries = f" {prompt_lower} "
    candidate_titles = titles_data.get("unique_titles", [])

    for title in candidate_titles:
        title_lower = title.lower()

        # Skip titles that are in the bizarre_titles list
        if title_lower in bizarre_titles:
            continue

        # Skip very short titles
        if len(title_lower) < 3:
            continue

        # Word boundary check
        if f" {title_lower} " in prompt_with_boundaries:
            detected_titles.add(title)
            continue

        # Check for titles at the start or end of the prompt
        if prompt_lower.startswith(f"{title_lower} ") or prompt_lower.endswith(
            f" {title_lower}"
        ):
            detected_titles.add(title)
            continue

        # Use fuzzy matching if above checks don't catch a direct match
        score = fuzz.ratio(title_lower, prompt_lower)
        adjusted_threshold = TITLE_FUZZY_THRESHOLD + max(0, 8 - len(title_lower)) * 5
        if score >= adjusted_threshold:
            detected_titles.add(title)

    # Filter out shorter titles that are substrings of longer detected titles
    sorted_titles = sorted(detected_titles, key=lambda x: len(x), reverse=True)
    filtered_titles = []
    for title in sorted_titles:
        title_lower = title.lower()
        if not any(title_lower in existing.lower() for existing in filtered_titles):
            filtered_titles.append(title)

    return filtered_titles


def find_keywords_in_prompt(
    prompt, json_path="keywords.json", bizarre_keywords_path="bizarre_keywords.json"
):
    """
    Detect candidate keywords in the user's prompt
    """
    keywords_data = load_json(json_path)
    bizarre_keywords_data = load_json(bizarre_keywords_path)
    bizarre_keywords = bizarre_keywords_data.get("bizarre_keywords", [])
    detected_keywords = set()
    prompt_lower = prompt.lower()
    prompt_with_boundaries = f" {prompt_lower} "
    candidate_keywords = keywords_data.get("unique_keywords", [])

    for keyword in candidate_keywords:
        keyword_lower = keyword.lower()

        # Skip keywords that are in the bizarre_keywords list
        if keyword_lower in bizarre_keywords:
            continue

        # Skip very short keywords
        if len(keyword_lower) < 3:
            continue

        # Word boundary check
        if f" {keyword_lower} " in prompt_with_boundaries:
            detected_keywords.add(keyword)
            continue

        # Check for keywords at the start or end of the prompt
        if prompt_lower.startswith(f"{keyword_lower} ") or prompt_lower.endswith(
            f" {keyword_lower}"
        ):
            detected_keywords.add(keyword)
            continue

        # Use fuzzy matching if direct boundary checks do not match
        score = fuzz.ratio(keyword_lower, prompt_lower)
        adjusted_threshold = (
            KEYWORD_FUZZY_THRESHOLD + max(0, 6 - len(keyword_lower)) * 5
        )
        if score >= adjusted_threshold:
            detected_keywords.add(keyword)

    # Filter out shorter keywords that are substrings of longer ones
    sorted_keywords = sorted(detected_keywords, key=lambda x: len(x), reverse=True)
    filtered_keywords = []
    for keyword in sorted_keywords:
        keyword_lower = keyword.lower()
        if not any(keyword_lower in existing.lower() for existing in filtered_keywords):
            filtered_keywords.append(keyword)

    return filtered_keywords


def find_release_in_prompt(prompt, json_path="releases.json"):
    """
    Detect candidate releases in user's prompt
    """
    releases_data = load_json(json_path)
    release_alternatives = load_json("release_alternatives.json")
    detected_releases = set()
    prompt_lower = prompt.lower()

    # Add word boundary markers to the prompt
    prompt_with_boundaries = f" {prompt_lower} "

    # Check direct matches from releases_data
    for release in releases_data.get("unique_releases", []):
        release_lower = release.lower()

        # Skip very short releases
        if len(release_lower) < 3:
            continue

        # Word boundary check
        if f" {release_lower} " in prompt_with_boundaries:
            detected_releases.add(release_lower)
            continue

        # For releases that might be at the start or end of the prompt
        if prompt_lower.startswith(f"{release_lower} ") or prompt_lower.endswith(
            f" {release_lower}"
        ):
            detected_releases.add(release_lower)
            continue

        # Fuzzy matching with the defined threshold
        score = fuzz.ratio(release_lower, prompt_lower)
        if score >= RELEASE_FUZZY_THRESHOLD:
            detected_releases.add(release_lower)

    # Check for alternative releases
    for alternative, canonical in release_alternatives.items():
        # Split the canonical releases by comma and strip whitespace
        canonicals = [c.strip() for c in canonical.split(",")]
        alternative_lower = alternative.lower()

        if f" {alternative_lower} " in prompt_with_boundaries:
            for c in canonicals:
                detected_releases.add(c.lower())
            continue

        if prompt_lower.startswith(f"{alternative_lower} ") or prompt_lower.endswith(
            f" {alternative_lower}"
        ):
            for c in canonicals:
                detected_releases.add(c.lower())
            continue

        score = fuzz.ratio(alternative_lower, prompt_lower)
        if score >= RELEASE_FUZZY_THRESHOLD:
            for c in canonicals:
                detected_releases.add(c.lower())

    return list(detected_releases)


def find_runtime_in_prompt(prompt, json_path="runtimes.json"):
    """
    Detect candidate runtimes in user's prompt
    """
    runtimes_data = load_json(json_path)
    runtime_alternatives = load_json("runtime_alternatives.json")
    detected_runtimes = set()
    prompt_lower = prompt.lower()

    # Add word boundary markers to the prompt
    prompt_with_boundaries = f" {prompt_lower} "

    # Check direct matches from runtimes_data
    for runtime in runtimes_data.get("unique_runtimes", []):
        runtime_lower = runtime.lower()

        # Skip very short runtimes
        if len(runtime_lower) < 3:
            continue

        # Word boundary check
        if f" {runtime_lower} " in prompt_with_boundaries:
            detected_runtimes.add(runtime_lower)
            continue

        # For runtimes that might be at the start or end of the prompt
        if prompt_lower.startswith(f"{runtime_lower} ") or prompt_lower.endswith(
            f" {runtime_lower}"
        ):
            detected_runtimes.add(runtime_lower)
            continue

        # Fuzzy matching with the defined threshold
        score = fuzz.ratio(runtime_lower, prompt_lower)
        if score >= RUNTIME_FUZZY_THRESHOLD:
            detected_runtimes.add(runtime_lower)

    # Check for alternative runtimes
    for alternative, canonical in runtime_alternatives.items():
        # Split the canonical runtimes by comma and strip whitespace
        canonicals = [c.strip() for c in canonical.split(",")]
        alternative_lower = alternative.lower()

        if f" {alternative_lower} " in prompt_with_boundaries:
            for c in canonicals:
                detected_runtimes.add(c.lower())
            continue

        if prompt_lower.startswith(f"{alternative_lower} ") or prompt_lower.endswith(
            f" {alternative_lower}"
        ):
            for c in canonicals:
                detected_runtimes.add(c.lower())
            continue

        score = fuzz.ratio(alternative_lower, prompt_lower)
        if score >= RUNTIME_FUZZY_THRESHOLD:
            for c in canonicals:
                detected_runtimes.add(c.lower())

    return list(detected_runtimes)
