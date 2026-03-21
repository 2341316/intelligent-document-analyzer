import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load dataset
with open("data/processed/combined_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Total chunks:", len(chunks))

# Extract text
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

# Inspect first embedding
print("\nFirst embedding vector (first 10 numbers):")
print(list(embeddings[0][:10]))


# Cosine Similarity Test

similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print("Similarity between chunk 1 and 2:", similarity[0][0])


# Query Similarity Example

query = "financial performance and revenue growth"

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)

best_match = similarities.argmax()

print("\nBest matching chunk:")
print(chunks[best_match]["text"])


# Save embeddings

for i, chunk in enumerate(chunks):
    chunk["embedding"] = embeddings[i].tolist()

with open("data/processed/chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("\nEmbeddings saved to chunks_with_embeddings.json")
