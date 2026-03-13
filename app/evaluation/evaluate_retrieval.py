from sentence_transformers import SentenceTransformer
import faiss
import json
import pickle
import numpy as np

#Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

#Load the FAISS Index
index = faiss.read_index("data/processed/faiss_index.index")

# Load Metadata
with open("data/processed/faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

test_queries = [
    {
        "query": "chairman message",
        "expected_keywords": ["chairman", "message"]
    },
    {
        "query": "risk management framework",
        "expected_keywords": ["risk"]
    },
    {
        "query": "revenue growth",
        "expected_keywords": ["revenue"]
    },
    {
        "query": "corporate governance",
        "expected_keywords": ["governance"]
    },
    {
        "query": "financial performance",
        "expected_keywords": ["financial"]
    },
    {
        "query": "sustainability initiatives",
        "expected_keywords": ["sustainability"]
    },
    {
        "query": "board of directors",
        "expected_keywords": ["board"]
    },
    {
        "query": "company strategy",
        "expected_keywords": ["strategy"]
    },
    {
        "query": "operating segments",
        "expected_keywords": ["segment"]
    },
    {
        "query": "future outlook",
        "expected_keywords": ["outlook"]
    }
]

#Creating Retrieval Function
def retrieve_chunks(query, k=5):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k)

    results = []

    for idx in indices[0]:
        results.append(metadata[idx]["text"])

    return results

# Evaluation Function
def evaluate_query(query, expected_keywords, k=5):

    results = retrieve_chunks(query, k)

    relevant = 0

    for chunk in results:

        if any(keyword.lower() in chunk.lower() for keyword in expected_keywords):
            relevant += 1

    precision = relevant / k

    return precision, results

# Run Evaluation
precisions = []

for test in test_queries:

    query = test["query"]
    expected = test["expected_keywords"]

    precision, results = evaluate_query(query, expected)

    print("\n===========================")
    print("Query:", query)
    print("Precision@5:", precision)

    print("\nTop Retrieved Chunks:")

    for r in results:
        print("-", r[:200], "...")

    precisions.append(precision)

avg_precision = sum(precisions) / len(precisions)

print("\n===========================")
print("Average Precision@5:", avg_precision)