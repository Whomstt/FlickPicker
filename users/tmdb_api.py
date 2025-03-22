import aiohttp
import asyncio
import json
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
from chatbot.config import (
    TMDB_BASE_URL,
    TMDB_API_KEY,
    TMDB_NUM_FILMS,
    TMDB_RATE_LIMIT,
    TMDB_RATE_LIMIT_WINDOW,
    TMDB_OUTPUT_FILE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def fetch_top_films(num_films):
    """Fetch the top films by popularity asynchronously."""
    films = []
    page = 1
    async with aiohttp.ClientSession() as session:
        while len(films) < num_films:
            url = f"{TMDB_BASE_URL}/movie/popular"
            params = {
                "api_key": TMDB_API_KEY,
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


async def fetch_film_data(session, film_id):
    """Fetch film details, credits, and keywords asynchronously."""
    try:
        url = f"{TMDB_BASE_URL}/movie/{film_id}"
        params = {
            "api_key": TMDB_API_KEY,
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


async def rate_limited_fetch(session, film_id, semaphore):
    """Fetch film data while respecting the rate limit."""
    async with semaphore:
        await asyncio.sleep(
            TMDB_RATE_LIMIT_WINDOW / TMDB_RATE_LIMIT
        )  # Spread requests evenly
        return await fetch_film_data(session, film_id)


def extract_unique_names(films):
    """
    Extract and return sorted lists of unique main actors and directors
    """
    unique_actors = set()
    unique_directors = set()

    for film in films:
        director = film.get("director")
        if director:
            unique_directors.add(director)

        main_actors = film.get("main_actors", [])
        for actor in main_actors:
            unique_actors.add(actor)

    # Return sorted lists for consistency.
    return sorted(unique_actors), sorted(unique_directors)


async def fetch_and_save_films():
    """Fetch film data, process it, and save to a JSON file asynchronously."""
    # Fetch the top films
    logging.info("Fetching top films...")
    top_films = await fetch_top_films(TMDB_NUM_FILMS)
    films_data = []

    # Fetch film details concurrently with rate limiting
    logging.info("Fetching film details...")
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(TMDB_RATE_LIMIT)  # Limit concurrent requests

        tasks = [
            rate_limited_fetch(session, film["id"], semaphore) for film in top_films
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

    # Save film data to JSON file
    logging.info(f"Saving film data to {TMDB_OUTPUT_FILE}...")
    try:
        with open(TMDB_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(unique_films, f, indent=4, ensure_ascii=False)
        logging.info(f"Film data successfully saved to {TMDB_OUTPUT_FILE}.")
    except Exception as e:
        logging.error(f"Error saving film data to {TMDB_OUTPUT_FILE}: {e}")

    # Extract unique main actors and directors
    logging.info("Extracting unique main actors and directors...")
    unique_actors, unique_directors = extract_unique_names(unique_films)
    actors_directors_data = {
        "unique_main_actors": unique_actors,
        "unique_directors": unique_directors,
    }

    # Save the unique actors and directors to a JSON file.
    actors_directors_file = "actors_directors.json"
    logging.info(f"Saving actors and directors to {actors_directors_file}...")
    try:
        with open(actors_directors_file, "w", encoding="utf-8") as f:
            json.dump(actors_directors_data, f, indent=4, ensure_ascii=False)
        logging.info(
            f"Actors and directors successfully saved to {actors_directors_file}."
        )
    except Exception as e:
        logging.error(
            f"Error saving actors and directors to {actors_directors_file}: {e}"
        )


if __name__ == "__main__":
    asyncio.run(fetch_and_save_films())
