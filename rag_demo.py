import faiss
import ollama
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = "cache"
FAISS_INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index")
EMBEDDING_DIM = 768  # Our embedding model's output dimension


def json_to_text(item, include_all_fields=False):
    # Convert JSON data to text
    fields = [f"Title: {item.get('title', 'N/A')}"]
    key_fields = [
        ("Genres", "genres"),
        ("Overview", "overview"),
        ("Director", "director"),
        ("Main Actors", "main_actors"),
    ]
    for label, key in key_fields:
        value = item.get(key, [])
        value = ", ".join(value) if isinstance(value, list) else value or "N/A"
        fields.append(f"{label}: {value}")

    if include_all_fields:
        optional_fields = [
            ("Runtime", "runtime"),
            ("Release Date", "release_date"),
            ("Country", "country_of_production"),
            ("Languages", "spoken_languages"),
            ("Tagline", "tagline"),
            ("Budget", "budget"),
            ("Revenue", "revenue"),
        ]
        for label, key in optional_fields:
            value = item.get(key, "N/A")
            value = ", ".join(value) if isinstance(value, list) else value
            fields.append(f"{label}: {value}")

    return "\n".join(fields)


def save_cache(filename, data, embeddings, index):
    # Save data, embeddings, and FAISS index to cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(
        os.path.join(CACHE_DIR, f"{filename}_data.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(data, f)
    with open(
        os.path.join(CACHE_DIR, f"{filename}_embeddings.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(embeddings.tolist(), f)
    faiss.write_index(index, FAISS_INDEX_PATH)


def load_cache(filename):
    # Load data, embeddings, and FAISS index from cache
    data_path = os.path.join(CACHE_DIR, f"{filename}_data.json")
    embeddings_path = os.path.join(CACHE_DIR, f"{filename}_embeddings.json")
    if not (
        os.path.exists(data_path)
        and os.path.exists(embeddings_path)
        and os.path.exists(FAISS_INDEX_PATH)
    ):
        return None, None, None
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(embeddings_path, "r", encoding="utf-8") as f:
        embeddings = np.array(json.load(f), dtype="float32")
    index = faiss.read_index(FAISS_INDEX_PATH)
    return data, embeddings, index


def generate_embeddings(data_texts):
    # Generate embeddings for the given data texts in parallel

    def fetch_embedding(text):
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

    print("Generating embeddings...")
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(fetch_embedding, data_texts))
    return np.array(embeddings, dtype="float32")


def main():
    filename = input("JSON Filename? -> ").strip()
    # Load cache if available
    data, embeddings, index = load_cache(filename)
    if data is None:
        print("No cache found. Generating new embeddings and FAISS index...")
        # Load data and preprocess
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_texts = [json_to_text(item) for item in data]

        # Generate embeddings
        embeddings = generate_embeddings(data_texts)

        # Create FAISS index
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index.add(embeddings)

        # Save to cache
        save_cache(filename, data, embeddings, index)
    else:
        print("Using cached data and index.")

    # Query processing
    prompt = input("What movie are you looking for? -> ").strip()
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)[
        "embedding"
    ]
    prompt_embedding = np.array(prompt_embedding, dtype="float32").reshape(1, -1)

    # Search for top matches
    distances, indices = index.search(prompt_embedding, 3)
    print("\nBest Matching Movies:\n")
    for distance, idx in zip(distances[0], indices[0]):
        print(f"Match (Similarity Distance: {distance:.4f}):")
        print(json_to_text(data[idx], include_all_fields=True))
        print()

    # Generate explanation for matches
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a movie recommendation expert."},
            {
                "role": "user",
                "content": f"Query: {prompt}\n\n"
                + "\n\n".join(json_to_text(data[idx]) for idx in indices[0]),
            },
        ],
    )
    print("\nRecommendation Explanation:")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
