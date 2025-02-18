import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://api.themoviedb.org/3"
API_KEY = os.getenv("TMDB_API_KEY")  # API key from environment variables
NUM_FILMS = 10000  # Number of films to fetch
MAX_WORKERS = os.cpu_count()  # Use maximum available threads
OUTPUT_FILE = "films_data.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_top_films(api_key, num_films):
    """Fetch the top films by popularity."""
    films = []
    page = 1
    while len(films) < num_films:
        url = f"{BASE_URL}/movie/popular"
        params = {
            "api_key": api_key,
            "page": page,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            # Add only the required number of films
            remaining_films = num_films - len(films)
            films.extend(results[:remaining_films])
            page += 1
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching top films: {e}")
            break
    return films


def fetch_film_data(api_key, film_id):
    """Fetch film details, credits, and keywords in a single API call."""
    try:
        url = f"{BASE_URL}/movie/{film_id}"
        params = {
            "api_key": api_key,
            "append_to_response": "credits,keywords",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract details, credits, and keywords
        details = data
        credits = data.get("credits", {})
        keywords = [kw["name"] for kw in data.get("keywords", {}).get("keywords", [])]

        return {
            "details": details,
            "credits": credits,
            "keywords": keywords,
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for film {film_id}: {e}")
        return None


def remove_duplicates(films_data):
    """Remove duplicate films based on title and release date."""
    seen = set()
    unique_films = []
    for film in films_data:
        film_key = (film.get("title"), film.get("release_date"))
        if film_key not in seen:
            seen.add(film_key)
            unique_films.append(film)
    return unique_films


def convert_to_meaningful_text(film_data):
    """Transform film data into a more readable and meaningful format."""
    runtime_categories = {
        (0, 90): "Short Runtime",
        (91, 120): "Medium Runtime",
        (121, float("inf")): "Long Runtime",
    }
    budget_categories = {
        (0, 1_000_000): "Low Budget",
        (1_000_001, 10_000_000): "Medium Budget",
        (10_000_001, float("inf")): "High Budget",
    }
    revenue_categories = {
        (0, 1_000_000): "Low Revenue",
        (1_000_001, 10_000_000): "Medium Revenue",
        (10_000_001, float("inf")): "High Revenue",
    }
    rating_categories = {
        (0, 4.0): "Low Rating",
        (4.0, 6.0): "Average Rating",
        (6.0, 8.0): "Good Rating",
        (8.0, float("inf")): "Excellent Rating",
    }

    for film in film_data:
        # Title
        if film.get("title"):
            film["title"] = f"Title: {film['title']}"

        # Genres
        if film.get("genres"):
            film["genres"] = f"Genres: {', '.join(film['genres'])}"

        # Runtime
        runtime = film.get("runtime")
        if runtime:
            for (min_val, max_val), category in runtime_categories.items():
                if min_val <= runtime <= max_val:
                    film["runtime"] = category
                    break

        # Release Date
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

        # Budget
        budget = film.get("budget")
        if budget:
            for (min_val, max_val), category in budget_categories.items():
                if min_val <= budget <= max_val:
                    film["budget"] = category
                    break

        # Revenue
        revenue = film.get("revenue")
        if revenue:
            for (min_val, max_val), category in revenue_categories.items():
                if min_val <= revenue <= max_val:
                    film["revenue"] = category
                    break
        # Rating
        rating = film.get("rating")
        if rating:
            for (min_val, max_val), category in rating_categories.items():
                if min_val <= rating <= max_val:
                    film["rating"] = category
                    break

    return film_data


def calculate_weighted_rating(vote_average, vote_count, C=1000, M=6.0):
    """Calculate a weighted rating using the Bayesian Average formula."""
    if vote_count == 0:
        return 0
    return (vote_average * vote_count + C * M) / (vote_count + C)


def fetch_and_save_films(api_key, output_file):
    """Fetch film data, process it, and save to a JSON file."""
    # Fetch the top films
    logging.info("Fetching top films...")
    top_films = get_top_films(api_key, NUM_FILMS)
    films_data = []

    # Fetch film details concurrently
    logging.info("Fetching film details...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_film_data, api_key, film["id"]): film
            for film in top_films
        }

        for future in as_completed(futures):
            film = futures[future]
            try:
                result = future.result()
                if result is None:
                    continue

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
                vote_average = details.get("vote_average", 0)
                vote_count = details.get("vote_count", 0)
                weighted_rating = calculate_weighted_rating(vote_average, vote_count)

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
                    "rating": weighted_rating,
                    "keywords": keywords,
                }
                # Remove empty or invalid fields
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
            except Exception as e:
                logging.error(f"Error processing film {film['id']}: {e}")
                continue

    # Remove duplicates and transform data
    logging.info("Processing film data...")
    unique_films = remove_duplicates(films_data)
    meaningful_films = convert_to_meaningful_text(unique_films)

    # Save to JSON file
    logging.info(f"Saving data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(meaningful_films, f, indent=4, ensure_ascii=False)

    logging.info("Done!")
