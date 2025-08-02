# FlickPicker
## Description

FlickPicker is an AI-powered film recommender system that delivers personalized, context-aware suggestions based on user prompts. By using Retrieval-Augmented Generation (RAG), it overcomes the limitations of static LLM knowledge by retrieving rich, real-time film data to generate more accurate and relevant recommendations.

The system combines Nomic Embed for textual embeddings, FAISS for efficient similarity search, and GPT-4o-mini for a natural language final output. Unlike traditional recommender systems, it supports natural language queries to better capture user intent, making it especially effective for users with niche or evolving tastes.

## Setup Guide:

### Docker
* docker-compose build
* docker-compose up

### Env file (Create a .env file at root and add these keys with your own values)
* DJANGO_ADMIN_PASSWORD
* GOOGLE_CLIENT_ID
* GOOGLE_CLIENT_SECRET
* SECRET_KEY
* TMDB_API_KEY
* OPENAI_API_KEY
* NOMIC_API_KEY
* PGDATABASE
* PGUSER
* PGPASSWORD
* PGHOST
* PGPORT
