import os
from django.conf import settings
from dotenv import load_dotenv

load_dotenv()

# Cache and data paths
CACHE_DIR = os.path.join(settings.BASE_DIR, "cache")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")

# OpenAI API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"
MAX_TOKENS = 1500
TEMPERATURE = 0.7

# Nomic API settings
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Ollama settings
OLLAMA_URL = "http://ollama:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"

# FAISS and embedding parameters
NPROBE = 64  # Number of clusters to be searched
NLIST = 512  # Number of clusters to be stored
M = 16  # Number of subquantizers
NBITS = 8  # Number of bits per subquantizer

N_TOP_MATCHES = 5  # Number of top matches to return
SEARCH_INCREMENT = 12480  # Increment for searching more results
MAX_RESULTS = 99840  # Maximum number of search results
NPROBE_INCREMENT = 64  # Increment for nprobe

# Weights for different embeddings
PROMPT_WEIGHT = 0.2  # Clean prompt (after entities removed)
NAME_WEIGHT = 1.0  # Actors and directors
GENRE_WEIGHT = 1.2  # Film genres
KEYWORD_WEIGHT = 0.8  # Thematic keywords
TITLE_WEIGHT = 1.5  # Film titles
RUNTIME_WEIGHT = 0.1  # Film runtime
RELEASE_WEIGHT = 0.1  # Film release date

# Fuzzy matching settings
PROMPT_FUZZY_THRESHOLD = 80  # Threshold for prompt matching
NAME_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for names in %
GENRE_FUZZY_THRESHOLD = 80  # Fuzzy matching threshold for genres in %
TITLE_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for titles in %
KEYWORD_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for keywords in %
RUNTIME_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for runtime in %
RELEASE_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for release date in %

# Settings for TMDB API
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # API key from environment variables
TMDB_NUM_FILMS = 3072  # Number of films to fetch per year (Total Films = TMDB_NUM_FILMS * ((2025 - 1962 + 2) / 2) or TMDB_NUM_FILMS * 32.5)
TMDB_RATE_LIMIT = 50  # TMDB rate limit 50 requests per second
TMDB_RATE_LIMIT_WINDOW = 1  # Rate limit window in seconds
TMDB_TOTAL_PAGES = 500  # Total number of pages available
TMDB_OUTPUT_FILE = "filmdata/original/films_data.json"
