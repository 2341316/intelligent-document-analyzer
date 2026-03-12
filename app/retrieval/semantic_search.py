import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load dataset
with open("data/processed/combined_chunks.json", "r") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

# Attach Embeddings to Chunks
for i, chunk in enumerate(chunks):
    chunk["embedding"] = embeddings[i]

# to Create Semantic Search Function
def semantic_search(query, chunks, model, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode([query])

    # Collect all chunk embeddings
    chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Get indices of top results
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:
        results.append({
            "score": float(similarities[idx]),
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "section": chunks[idx]["section"],
            "company": chunks[idx]["company_name"]
        })

    return results

# for Testing
query = "risk management framework"

results = semantic_search(query, chunks, model)

# print results
for r in results:
    print("\nSimilarity Score:", r["score"])
    print("Company:", r["company"])
    print("Section:", r["section"])
    print("Page:", r["page"])
    print("Text:", r["text"][:300])