import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading chunks...")

with open("data/processed/combined_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract text from chunks
texts = [chunk["text"] for chunk in chunks]

print("Generating embeddings...")

embeddings = model.encode(texts, show_progress_bar=True)

# Convert to numpy float32
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

print("Embedding dimension:", dimension)

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

print("Adding embeddings to FAISS index...")

index.add(embeddings)

print("Total vectors stored:", index.ntotal)

# Save index
faiss.write_index(index, "data/processed/faiss_index.index")

# Save metadata
with open("data/processed/faiss_metadata.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index saved successfully")