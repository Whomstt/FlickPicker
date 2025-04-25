# FlickPicker
## A film recommender system built using retrieval-augmented generation (RAG)
Deployed at https://flickpicker.site

### Project Overview

#### Building a Film Recommender System using Retrieval-Augmented Generation

This project is a film recommender system developed for film enthusiasts who desire personalized and meaningful recommendations. The system takes a user prompt and recommends films that are a close match. RAG addresses the limitations of traditional recommendation systems by tackling the topics of semantic understanding and inclusivity, providing users with a deeper and more personalized experience. The target audience is individuals with diverse tastes and interests, so that both mainstream and niche interests are provided for.

The project was envisioned through a combination of research into state-of-the-art machine learning techniques and experimentation through trial and error. The system makes use of Nomic Embed for textual embeddings, FAISS for similarity search, and ChatGPT for explanations. Inspiration was drawn from existing limitations in traditional recommender systems and the potential provided by AI to better align user interests with meaningful, contextually relevant recommendations.

### Setup Guide:

#### Docker
* docker-compose build
* docker-compose up

#### Env file (Create a .env file at root and add these keys with your own values)
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
