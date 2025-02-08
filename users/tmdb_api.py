import requests
import json

BASE_URL = "https://api.themoviedb.org/3"

NUM_FILMS = 10000


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


def get_film_details(api_key, film_id):
    url = f"{BASE_URL}/movie/{film_id}"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_film_credits(api_key, film_id):
    url = f"{BASE_URL}/movie/{film_id}/credits"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_film_keywords(api_key, film_id):
    url = f"{BASE_URL}/movie/{film_id}/keywords"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return [kw["name"] for kw in response.json().get("keywords", [])]


def fetch_and_save_films(api_key, output_file):
    # Fetch the top 100 films
    top_films = get_top_films(api_key, NUM_FILMS)
    films_data = []
    for film in top_films:
        film_id = film["id"]
        try:
            details = get_film_details(api_key, film_id)
            credits = get_film_credits(api_key, film_id)
            keywords = get_film_keywords(api_key, film_id)
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
                    language["name"] for language in details.get("spoken_languages", [])
                ],
                "tagline": details.get("tagline"),
                "budget": details.get("budget"),
                "revenue": details.get("revenue"),
                "keywords": keywords,
            }
            films_data.append(film_data)
        except requests.exceptions.RequestException:
            continue

    # Write the data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(films_data, f, indent=4, ensure_ascii=False)
