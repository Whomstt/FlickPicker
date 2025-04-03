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
NPROBE = 1  # Number of clusters to be searched
NLIST = 1  # Number of clusters to be stored
M = 16  # Number of subquantizers
NBITS = 4  # Number of bits per subquantizer

N_TOP_MATCHES = 5  # Number of top matches to return
SEARCH_INCREMENT = 8192  # Increment for searching more results
MAX_RESULTS = 65536  # Maximum number of search results
NPROBE_INCREMENT = 128  # Increment for nprobe

PROMPT_WEIGHT = 1  # Weight for prompt embedding
NAME_WEIGHT = 0.6  # Weight for actor / director names embedding
GENRE_WEIGHT = 0.4  # Weight for genre embedding
TITLE_WEIGHT = 0.4  # Weight for title embedding
KEYWORD_WEIGHT = 0.4  # Weight for keyword embedding

# Fuzzy matching settings
NAME_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for names in %
GENRE_FUZZY_THRESHOLD = 80  # Fuzzy matching threshold for genres in %
TITLE_FUZZY_THRESHOLD = 90  # Fuzzy matching threshold for titles in %
KEYWORD_FUZZY_THRESHOLD = 80  # Fuzzy matching threshold for keywords in %

# Settings for TMDB API
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # API key from environment variables
TMDB_NUM_FILMS = 10  # Number of films to fetch per year
TMDB_RATE_LIMIT = 50  # TMDB rate limit 50 requests per second
TMDB_RATE_LIMIT_WINDOW = 1  # Rate limit window in seconds
TMDB_TOTAL_PAGES = 500  # Total number of pages available
TMDB_OUTPUT_FILE = "films_data.json"
