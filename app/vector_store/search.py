import faiss
import pickle
import numpy as np

## LAZY LOAD VARIABLES
embedding_model = None
index = None
chunks = None


## LOADERS 

def get_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


def get_index():
    global index
    if index is None:
        print("Loading FAISS index...")
        index = faiss.read_index("data/processed/faiss_index.index")
    return index


def get_chunks():
    global chunks
    if chunks is None:
        print("Loading metadata...")
        with open("data/processed/faiss_metadata.pkl", "rb") as f:
            chunks = pickle.load(f)
    return chunks


## SEARCH FUNCTION

def search(query, top_k=5):

    print("Embedding query...")

    model = get_model()
    idx = get_index()
    data = get_chunks()

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = idx.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        results.append(data[i])

    return results


## TEST 

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