import os
from django.conf import settings
from dotenv import load_dotenv

load_dotenv()

# Cache and data paths
CACHE_DIR = os.path.join(settings.BASE_DIR, "cache")
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
RAW_FILM_DATA_PATH = os.path.join(settings.BASE_DIR, "raw_film_data.json")

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
NPROBE = 10  # Number of clusters to be searched
NLIST = 100  # Number of clusters to be stored
N_TOP_MATCHES = 3  # Number of top matches to return
M = 16  # Number of subquantizers
NBITS = 7  # Number of bits per subquantizer

# Field weights for film attributes - higher value means more important
FIELD_WEIGHTS = {
    "genres": 1.0,
    "title": 0.6,
    "overview": 1.0,
    "tagline": 0.8,
    "keywords": 0.7,
    "director": 0.5,
    "main_actors": 0.4,
    "runtime": 0.2,
    "release_date": 0.3,
}
