import requests
import json
from dotenv import load_dotenv
import os

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"


# Function to get top movies
def get_top_movies(api_key, num_movies=100):
    movies = []
    page = 1
    while len(movies) < num_movies:
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": api_key,
            "sort_by": "popularity.desc",
            "page": page,
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        movies.extend(data["results"])
        page += 1
    return movies[:num_movies]


# Get detailed movie info
def get_movie_details(api_key, movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# Get cast and crew information
def get_movie_credits(api_key, movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


# Fetch and save the data
def fetch_and_save_movies(api_key, output_file):
    print("Fetching top 100 movies...")
    top_movies = get_top_movies(api_key, 100)

    movies_data = []
    for idx, movie in enumerate(top_movies, start=1):
        movie_id = movie["id"]
        print(f"Fetching details for movie {idx}: {movie['title']}...")

        try:
            details = get_movie_details(api_key, movie_id)
            credits = get_movie_credits(api_key, movie_id)

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

            movie_data = {
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

            movies_data.append(movie_data)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for movie ID {movie_id}: {e}")

    # Save the data to a JSON file
    print(f"Saving data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(movies_data, f, indent=4, ensure_ascii=False)

    print("Data saved successfully!")


# Execute the script
if __name__ == "__main__":
    OUTPUT_FILE = "movies_data.json"
    fetch_and_save_movies(API_KEY, OUTPUT_FILE)
