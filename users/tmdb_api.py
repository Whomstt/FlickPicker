import aiohttp
import asyncio
import json
import logging
from chatbot.config import (
    TMDB_BASE_URL,
    TMDB_API_KEY,
    TMDB_NUM_FILMS,
    TMDB_RATE_LIMIT,
    TMDB_RATE_LIMIT_WINDOW,
    TMDB_OUTPUT_FILE,
    TMDB_TOTAL_PAGES,
)

from django.http import JsonResponse
from django.views.decorators.http import require_POST

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

cancel_fetch = False


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


async def rate_limited_fetch(session, film_id, semaphore):
    """Fetch film data while respecting the rate limit."""
    async with semaphore:
        await asyncio.sleep(
            TMDB_RATE_LIMIT_WINDOW / TMDB_RATE_LIMIT
        )  # Spread requests evenly
        return await fetch_film_data(session, film_id)


def extract_unique_names(films):
    """Extract and return sorted lists of unique main actors and directors."""
    unique_actors = set()
    unique_directors = set()
    for film in films:
        directors = film.get("directors", [])
        for director in directors:
            unique_directors.add(director.lower())
        main_actors = film.get("main_actors", [])
        for actor in main_actors:
            unique_actors.add(actor.lower())
    return sorted(unique_actors), sorted(unique_directors)


def extract_unique_genres(films):
    """Extract and return sorted list of unique genres."""
    unique_genres = set()
    for film in films:
        genres = film.get("genres", [])
        for genre in genres:
            unique_genres.add(genre.lower())
    return sorted(unique_genres)


def extract_unique_titles(films):
    """Extract and return a sorted list of unique film titles."""
    unique_titles = set()
    for film in films:
        title = film.get("title")
        if title:
            unique_titles.add(title.lower().strip())
    return sorted(unique_titles)


def extract_unique_keywords(films):
    """Extract and return a sorted list of unique keywords across all films."""
    unique_keywords = set()
    for film in films:
        keywords = film.get("keywords", [])
        for keyword in keywords:
            unique_keywords.add(keyword.lower().strip())
    return sorted(unique_keywords)


async def fetch_and_save_films():
    """
    Fetch film data per year using the TMDB discover endpoint, process it,
    and save to a JSON file asynchronously.
    """
    global cancel_fetch

    all_films = []  # List to hold all films
    films_by_year = {}  # Dictionary to hold films by year
    all_film_keys = set()  # Track film keys across all years

    # List of years to search for
    years = list(range(1962, 2026))

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(TMDB_RATE_LIMIT)
        # Loop through each year
        for year in years:
            unique_films_dict = {}
            page = 1
            logging.info(f"Fetching films for the year {year}...")
            while len(unique_films_dict) < TMDB_NUM_FILMS:
                if cancel_fetch:
                    logging.info("Film fetching cancelled by user")
                    break

                logging.info(f"Year {year}: Fetching page {page}...")
                url = f"{TMDB_BASE_URL}/discover/movie"
                params = {
                    "api_key": TMDB_API_KEY,
                    "primary_release_year": year,
                    "page": page,
                }
                try:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 400:
                            logging.error(
                                f"Year {year}, page {page} returned 400. Skipping to next page."
                            )
                            page += 1
                            continue
                        response.raise_for_status()

                        data = await response.json()
                        results = data.get("results", [])
                        if not results:
                            logging.warning(
                                f"No more films available for year {year} on page {page}."
                            )
                            break
                except Exception as e:
                    logging.error(
                        f"Error fetching films for year {year}, page {page}: {e}"
                    )
                    page += 1
                    continue

                # Create tasks to fetch film details concurrently
                tasks = [
                    rate_limited_fetch(session, film["id"], semaphore)
                    for film in results
                ]
                page_results = await asyncio.gather(*tasks)

                new_films = 0
                for result in page_results:
                    if result is None:
                        continue
                    details = result.get("details", {})
                    credits = result.get("credits", {})
                    keywords = result.get("keywords", [])

                    # Extract key fields
                    main_actors = [
                        actor["name"] for actor in credits.get("cast", [])[:5]
                    ]
                    directors = [
                        member["name"]
                        for member in credits.get("crew", [])
                        if member["job"] == "Director"
                    ]

                    film_data = {
                        "title": details.get("title"),
                        "genres": [
                            genre["name"] for genre in details.get("genres", [])
                        ],
                        "overview": details.get("overview"),
                        "tagline": details.get("tagline"),
                        "keywords": keywords,
                        "directors": directors,
                        "main_actors": main_actors,
                        "runtime": details.get("runtime"),
                        "release_date": details.get("release_date"),
                        "poster_image": (
                            f"https://image.tmdb.org/t/p/w500{details.get('poster_path')}"
                            if details.get("poster_path")
                            else None
                        ),
                    }
                    # Check if runtime exists and is atleast 40 minutes
                    runtime = film_data.get("runtime")
                    if runtime is None or runtime < 40:
                        continue

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
                    # Use a tuple of title and release_date as the key for duplicate checking
                    film_key = (
                        cleaned_film_data.get("title"),
                        cleaned_film_data.get("release_date"),
                    )
                    # Only add the film if it's not already present globally
                    if (
                        film_key not in unique_films_dict
                        and film_key not in all_film_keys
                    ):
                        unique_films_dict[film_key] = cleaned_film_data
                        all_film_keys.add(film_key)
                        new_films += 1

                logging.info(
                    f"Year {year}, page {page}: Found {new_films} new unique films. Total for year: {len(unique_films_dict)}"
                )
                page += 1
                if page > TMDB_TOTAL_PAGES:
                    logging.warning(
                        f"Year {year}: Reached the last available page from TMDB"
                    )
                    break

            # Limit films per year
            logging.info(f"Year {year}: Cutting down to {TMDB_NUM_FILMS} films.")
            films_for_year = list(unique_films_dict.values())[:TMDB_NUM_FILMS]
            all_films.extend(films_for_year)

    logging.info(f"Total films across all years after deduplication: {len(all_films)}")

    logging.info(f"Saving film data to {TMDB_OUTPUT_FILE}...")
    try:
        with open(TMDB_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_films, f, indent=4, ensure_ascii=False)
        logging.info(f"Film data successfully saved to {TMDB_OUTPUT_FILE}.")
    except Exception as e:
        logging.error(f"Error saving film data to {TMDB_OUTPUT_FILE}: {e}")

    logging.info("Extracting unique main actors and directors...")
    unique_actors, unique_directors = extract_unique_names(all_films)
    actors_directors_data = {
        "unique_main_actors": unique_actors,
        "unique_directors": unique_directors,
    }
    actors_directors_file = "filmdata/original/actors_directors.json"
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
    logging.info("Extracting unique genres...")
    unique_genres = extract_unique_genres(all_films)
    genres_data = {"unique_genres": unique_genres}
    genres_file = "filmdata/original/genres.json"
    try:
        with open(genres_file, "w", encoding="utf-8") as f:
            json.dump(genres_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Genres successfully saved to {genres_file}.")
    except Exception as e:
        logging.error(f"Error saving genres to {genres_file}: {e}")

    logging.info("Extracting unique titles...")
    unique_titles = extract_unique_titles(all_films)
    titles_data = {"unique_titles": unique_titles}
    titles_file = "filmdata/original/titles.json"
    try:
        with open(titles_file, "w", encoding="utf-8") as f:
            json.dump(titles_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Titles successfully saved to {titles_file}.")
    except Exception as e:
        logging.error(f"Error saving titles to {titles_file}: {e}")

    logging.info("Extracting unique keywords...")
    unique_keywords = extract_unique_keywords(all_films)
    keywords_data = {"unique_keywords": unique_keywords}
    keywords_file = "filmdata/original/keywords.json"
    try:
        with open(keywords_file, "w", encoding="utf-8") as f:
            json.dump(keywords_data, f, indent=4, ensure_ascii=False)
        logging.info(f"Keywords successfully saved to {keywords_file}.")
    except Exception as e:
        logging.error(f"Error saving keywords to {keywords_file}: {e}")

    # Reset cancellation flag
    cancel_fetch = False


@require_POST
def cancel_fetch_view(request):
    """
    Sets the cancellation flag for the film fetching process
    """
    global cancel_fetch
    cancel_fetch = True
    return JsonResponse({"status": "Fetch cancelled"})
