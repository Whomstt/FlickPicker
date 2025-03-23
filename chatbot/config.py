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

# Nomic API settings
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")
NOMIC_API_URL = "https://api-atlas.nomic.ai/v1/embedding/text"
NOMIC_MODEL = "nomic-embed-text-v1.5"
EMBEDDING_DIM = 768

# Ollama settings
OLLAMA_URL = "http://ollama:11434/api"
EMBEDDING_MODEL = "nomic-embed-text"

# FAISS and embedding parameters
NPROBE = 5  # Number of clusters to be searched
NLIST = 50  # Number of clusters to be stored
N_TOP_MATCHES = 5  # Number of top matches to return
SEARCH_INCREMENT = 10  # Increment for searching more results
MAX_RESULTS = 100  # Maximum number of search results
M = 16  # Number of subquantizers
NBITS = 7  # Number of bits per subquantizer

# Field weights for film attributes - higher value means more important
FIELD_WEIGHTS = {
    "genres": 1.0,
    "title": 0.8,
    "overview": 0.7,
    "tagline": 0.5,
    "keywords": 0.5,
    "director": 0.8,
    "main_actors": 0.8,
    "runtime": 0.5,
    "release_date": 0.5,
}

# Settings for TMDB API
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # API key from environment variables
TMDB_NUM_FILMS = 10000  # Number of films to fetch
TMDB_RATE_LIMIT = 40  # TMDB rate limit (40 requests per 10 seconds)
TMDB_RATE_LIMIT_WINDOW = 10  # Rate limit window in seconds
TMDB_OUTPUT_FILE = "films_data.json"
