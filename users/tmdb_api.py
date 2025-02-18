import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime

BASE_URL = "https://api.themoviedb.org/3"
NUM_FILMS = 100  # Number of films to fetch
MAX_WORKERS = os.cpu_count()  # Use maximum available threads


def get_top_films(api_key, num_films):
    films = []
    page = 1
    while len(films) < num_films:
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": api_key,
            "sort_by": "popularity.desc",
            "page": page,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:  # Exit if no more results
            break

        # Add only the required number of films
        remaining_films = num_films - len(films)
        films.extend(results[:remaining_films])
        page += 1
    return films


def fetch_film_data(api_key, film_id):
    """Fetch film details, credits, and keywords concurrently."""

    def get_film_details():
        url = f"{BASE_URL}/movie/{film_id}"
        params = {"api_key": api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_film_credits():
        url = f"{BASE_URL}/movie/{film_id}/credits"
        params = {"api_key": api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_film_keywords():
        url = f"{BASE_URL}/movie/{film_id}/keywords"
        params = {"api_key": api_key}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return [kw["name"] for kw in response.json().get("keywords", [])]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "details": executor.submit(get_film_details),
            "credits": executor.submit(get_film_credits),
            "keywords": executor.submit(get_film_keywords),
        }
        results = {}
        for key, future in futures.items():
            try:
                results[key] = future.result()
            except requests.exceptions.RequestException:
                results[key] = None  # Handle failed requests gracefully

    return results


def remove_duplicates(films_data):
    seen = set()
    unique_films = []
    for film in films_data:
        film_key = (film.get("title"), film.get("release_date"))
        if film_key not in seen:
            seen.add(film_key)
            unique_films.append(film)
    return unique_films


def convert_to_meaningful_text(film_data):
    for film in film_data:
        title = film.get("title")
        if title:
            film["title"] = "Title: " + title
        genres = film.get("genres")
        if genres:
            film["genres"] = "Genres: " + ", ".join(genres)
        runtime = film.get("runtime")
        if runtime:
            if runtime <= 90:
                film["runtime"] = "Short Runtime"
            elif runtime <= 120:
                film["runtime"] = "Medium Runtime"
            else:
                film["runtime"] = "Long Runtime"
        release_date = film.get("release_date")
        if release_date:
            date_obj = datetime.strptime(release_date, "%Y-%m-%d")
            year = date_obj.year
            current_year = datetime.now().year
            age = current_year - year
            if age < 2:
                film["release_date"] = "New Release"
            elif age < 10:
                film["release_date"] = "Recent"
            elif age < 25:
                film["release_date"] = "Modern"
            else:
                film["release_date"] = "Classic"
        overview = film.get("overview")
        if overview:
            film["overview"] = "Overview: " + overview
        director = film.get("director")
        if director:
            film["director"] = "Director: " + director
        main_actors = film.get("main_actors")
        if main_actors:
            film["main_actors"] = "Main Actors: " + ", ".join(main_actors)
        country_of_production = film.get("country_of_production")
        if country_of_production:
            film["country_of_production"] = "Country of Production: " + ", ".join(
                country_of_production
            )
        spoken_languages = film.get("spoken_languages")
        if spoken_languages:
            film["spoken_languages"] = "Spoken Languages: " + ", ".join(
                spoken_languages
            )
        tagline = film.get("tagline")
        if tagline:
            film["tagline"] = "Tagline: " + tagline
        budget = film.get("budget")
        if budget:
            if budget <= 1_000_000:
                film["budget"] = "Low Budget"
            elif budget <= 10_000_000:
                film["budget"] = "Medium Budget"
            else:
                film["budget"] = "High Budget"
        revenue = film.get("revenue")
        if revenue:
            if revenue <= 1_000_000:
                film["revenue"] = "Low Revenue"
            elif revenue <= 10_000_000:
                film["revenue"] = "Medium Revenue"
            else:
                film["revenue"] = "High Revenue"
        keywords = film.get("keywords")
        if keywords:
            film["keywords"] = "Keywords: " + ", ".join(keywords)
    return film_data


def fetch_and_save_films(api_key, output_file):
    # Fetch the top films
    top_films = get_top_films(api_key, NUM_FILMS)
    films_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_film = {
            executor.submit(fetch_film_data, api_key, film["id"]): film
            for film in top_films
        }

        for future in as_completed(future_to_film):
            film = future_to_film[future]
            try:
                result = future.result()
                details = result.get("details", {})
                credits = result.get("credits", {})
                keywords = result.get("keywords", [])

                # Extract key information
                main_actors = [actor["name"] for actor in credits.get("cast", [])[:5]]
                director = next(
                    (
                        member["name"]
                        for member in credits.get("crew", [])
                        if member.get("job") == "Director"
                    ),
                    None,
                )
                film_data = {
                    "title": details.get("title"),
                    "genres": [genre["name"] for genre in details.get("genres", [])],
                    "runtime": details.get("runtime"),
                    "release_date": details.get("release_date"),
                    "overview": details.get("overview"),
                    "director": director,
                    "main_actors": main_actors,
                    "country_of_production": [
                        country["name"]
                        for country in details.get("production_countries", [])
                    ],
                    "spoken_languages": [
                        language["name"]
                        for language in details.get("spoken_languages", [])
                    ],
                    "tagline": details.get("tagline"),
                    "budget": details.get("budget"),
                    "revenue": details.get("revenue"),
                    "keywords": keywords,
                }
                # Remove empty or invalid fields and filter empty strings from lists
                cleaned_film_data = {
                    key: (
                        [item for item in value if item != ""]
                        if isinstance(value, list)
                        else value
                    )
                    for key, value in film_data.items()
                    if value not in [None, 0, [], {}, ""]
                }
                films_data.append(cleaned_film_data)
            except Exception:
                continue

    unique_films = remove_duplicates(films_data)
    meaningful_films = convert_to_meaningful_text(unique_films)

    # Write the data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(meaningful_films, f, indent=4, ensure_ascii=False)
