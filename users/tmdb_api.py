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


async def fetch_and_save_films():
    """Fetch film data, process it, and save to a JSON file asynchronously."""
    global cancel_fetch
    unique_films_dict = {}
    page = 1
    total_pages = TMDB_TOTAL_PAGES

    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(TMDB_RATE_LIMIT)
        while len(unique_films_dict) < TMDB_NUM_FILMS:
            # Check cancellation flag on each loop iteration
            if cancel_fetch:
                logging.info("Film fetching cancelled by user")
                break

            logging.info(f"Fetching films from page {page}...")
            url = f"{TMDB_BASE_URL}/movie/popular"
            params = {"api_key": TMDB_API_KEY, "page": page}
            try:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 400:
                        logging.error(
                            f"Page {page} returned 400. Skipping to the next page."
                        )
                        page += 1
                        continue

                    response.raise_for_status()
                    data = await response.json()

                    results = data.get("results", [])
                    if not results:
                        logging.warning("No more films available to fetch.")
                        break
            except Exception as e:
                logging.error(f"Error fetching films from page {page}: {e}")
                page += 1
                continue

            # Create tasks to fetch film details concurrently
            tasks = [
                rate_limited_fetch(session, film["id"], semaphore) for film in results
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
                main_actors = [actor["name"] for actor in credits.get("cast", [])[:5]]
                directors = [
                    member["name"]
                    for member in credits.get("crew", [])
                    if member["job"] == "Director"
                ]

                film_data = {
                    "title": details.get("title"),
                    "genres": [genre["name"] for genre in details.get("genres", [])],
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
                if film_key not in unique_films_dict:
                    unique_films_dict[film_key] = cleaned_film_data
                    new_films += 1

            logging.info(
                f"Page {page}: Added {new_films} new unique films. Total unique films: {len(unique_films_dict)}"
            )
            page += 1

            if page > total_pages:
                logging.warning("Reached the last available page from TMDB.")
                break

    unique_films = list(unique_films_dict.values())
    if len(unique_films) > TMDB_NUM_FILMS:
        unique_films = unique_films[:TMDB_NUM_FILMS]

    logging.info(f"Total unique films after deduplication: {len(unique_films)}")

    logging.info(f"Saving film data to {TMDB_OUTPUT_FILE}...")
    try:
        with open(TMDB_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(unique_films, f, indent=4, ensure_ascii=False)
        logging.info(f"Film data successfully saved to {TMDB_OUTPUT_FILE}.")
    except Exception as e:
        logging.error(f"Error saving film data to {TMDB_OUTPUT_FILE}: {e}")

    logging.info("Extracting unique main actors and directors...")
    unique_actors, unique_directors = extract_unique_names(unique_films)
    actors_directors_data = {
        "unique_main_actors": unique_actors,
        "unique_directors": unique_directors,
    }
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
