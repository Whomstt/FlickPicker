import requests
import json
from dotenv import load_dotenv
import os

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.thefilmdb.org/3"


# Function to get top films
def get_top_films(api_key, num_films=100):
    films = []
    page = 1
    while len(films) < num_films:
        url = f"{BASE_URL}/discover/film"
        params = {
            "api_key": api_key,
            "sort_by": "popularity.desc",
            "page": page,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        films.extend(data["results"])
        page += 1
    return films[:num_films]


# Get detailed film info
def get_film_details(api_key, film_id):
    url = f"{BASE_URL}/film/{film_id}"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# Get cast and crew information
def get_film_credits(api_key, film_id):
    url = f"{BASE_URL}/film/{film_id}/credits"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# Fetch and save the data
def fetch_and_save_films(api_key, output_file):
    print("Fetching top 100 films...")
    top_films = get_top_films(api_key, 100)

    films_data = []
    for idx, film in enumerate(top_films, start=1):
        film_id = film["id"]
        print(f"Fetching details for film {idx}: {film['title']}...")

        try:
            details = get_film_details(api_key, film_id)
            credits = get_film_credits(api_key, film_id)

            # Extract key information
            main_actors = [
                actor["name"] for actor in credits["cast"][:5]
            ]  # Take the top 5 actors
            director = next(
                (
                    member["name"]
                    for member in credits["crew"]
                    if member["job"] == "Director"
                ),
                None,
            )

            film_data = {
                "title": details["title"],
                "genres": [genre["name"] for genre in details["genres"]],
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
            }

            films_data.append(film_data)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for film ID {film_id}: {e}")

    # Save the data to a JSON file
    print(f"Saving data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(films_data, f, indent=4, ensure_ascii=False)

    print("Data saved successfully!")


# Execute the script
if __name__ == "__main__":
    OUTPUT_FILE = "film_data.json"
    fetch_and_save_films(API_KEY, OUTPUT_FILE)
