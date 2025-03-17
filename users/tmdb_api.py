import aiohttp
import asyncio
import json
import os
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "https://api.themoviedb.org/3"
API_KEY = os.getenv("TMDB_API_KEY")  # API key from environment variables
NUM_FILMS = 10_000  # Number of films to fetch
RATE_LIMIT = 40  # TMDB rate limit (40 requests per 10 seconds)
RATE_LIMIT_WINDOW = 10  # Rate limit window in seconds
OUTPUT_FILE = "films_data.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def fetch_top_films(api_key, num_films):
    """Fetch the top films by popularity asynchronously."""
    films = []
    page = 1
    async with aiohttp.ClientSession() as session:
        while len(films) < num_films:
            url = f"{BASE_URL}/movie/popular"
            params = {
                "api_key": api_key,
                "page": page,
            }
            try:
                async with session.get(url, params=params, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    results = data.get("results", [])
                    if not results:
                        break

                    # Add only the required number of films
                    remaining_films = num_films - len(films)
                    films.extend(results[:remaining_films])
                    page += 1
            except Exception as e:
                logging.error(f"Error fetching top films: {e}")
                break
    logging.info(f"Fetched {len(films)} top films.")
    return films


async def fetch_film_data(session, api_key, film_id):
    """Fetch film details, credits, and keywords asynchronously."""
    try:
        url = f"{BASE_URL}/movie/{film_id}"
        params = {
            "api_key": api_key,
            "append_to_response": "credits,keywords",
        }
        async with session.get(url, params=params, timeout=10) as response:
            response.raise_for_status()
            data = await response.json()

            # Extract details, credits, and keywords
            details = data
            credits = data.get("credits", {})
            keywords = [
                kw["name"] for kw in data.get("keywords", {}).get("keywords", [])
            ]

            return {
                "details": details,
                "credits": credits,
                "keywords": keywords,
            }
    except Exception as e:
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
    logging.info(f"Removed duplicates. {len(unique_films)} unique films remaining.")
    return unique_films


def convert_to_meaningful_text(films):
    """
    Transform film data by categorizing runtime and release date into descriptive labels.
    """
    runtime_categories = {
        (0, 90): "Short",
        (91, 120): "Average",
        (121, float("inf")): "Long",
    }
    release_age_categories = {
        (0, 2): "New",
        (2, 10): "Modern",
        (10, float("inf")): "Old",
    }

    current_year = datetime.now().year

    for film in films:
        # Categorize runtime
        runtime = film.get("runtime")
        if runtime:
            for (low, high), label in runtime_categories.items():
                if low <= runtime <= high:
                    film["runtime"] = label
                    break

        # Categorize release date by film age
        release_date = film.get("release_date")
        if release_date:
            try:
                year = datetime.strptime(release_date, "%Y-%m-%d").year
                age = current_year - year
                for (low, high), label in release_age_categories.items():
                    if low <= age < high:
                        film["release_date"] = label
                        break
            except ValueError:
                pass

    logging.info("Transformed film data into meaningful text.")
    return films


async def rate_limited_fetch(session, api_key, film_id, semaphore):
    """Fetch film data while respecting the rate limit."""
    async with semaphore:
        await asyncio.sleep(RATE_LIMIT_WINDOW / RATE_LIMIT)  # Spread requests evenly
        return await fetch_film_data(session, api_key, film_id)


async def fetch_and_save_films(api_key, output_file):
    """Fetch film data, process it, and save to a JSON file asynchronously."""
    # Fetch the top films
    logging.info("Fetching top films...")
    top_films = await fetch_top_films(api_key, NUM_FILMS)
    films_data = []

    # Fetch film details concurrently with rate limiting
    logging.info("Fetching film details...")
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(RATE_LIMIT)  # Limit concurrent requests

        tasks = [
            rate_limited_fetch(session, api_key, film["id"], semaphore)
            for film in top_films
        ]
        results = await asyncio.gather(*tasks)

        for result in results:
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

            film_data = {
                "title": details.get("title"),
                "genres": [genre["name"] for genre in details.get("genres", [])],
                "overview": details.get("overview"),
                "tagline": details.get("tagline"),
                "keywords": keywords,
                "director": director,
                "main_actors": main_actors,
                "runtime": details.get("runtime"),
                "release_date": details.get("release_date"),
                "poster_image": (
                    f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}"
                    if details.get("poster_path")
                    else None
                ),
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

    # Remove duplicates and transform data
    logging.info("Processing film data...")
    unique_films = remove_duplicates(films_data)
    meaningful_films = convert_to_meaningful_text(unique_films)

    # Save to JSON file
    logging.info(f"Saving data to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(meaningful_films, f, indent=4, ensure_ascii=False)
        logging.info(f"Data successfully saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving data to {output_file}: {e}")
