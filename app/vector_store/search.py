import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("data/processed/faiss_index.index")

print("Loading metadata...")
with open("data/processed/faiss_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)


def search(query, top_k=5):

    print("Embedding query...")

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results


if __name__ == "__main__":

    query = input("Enter your query: ")

    results = search(query)

    print("\nTop Results:\n")

    for r in results:

        print("Company:", r["company_name"])
        print("Section:", r["section"])
        print("Page:", r["page"])
        print("Text:", r["text"][:300])
        print("-" * 50)