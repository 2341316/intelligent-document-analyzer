import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

## load FAISS Index
index = faiss.read_index("data/processed/faiss_index.index")

# Load Metadata
with open("data/processed/faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Retrieval Function
def retrieve_chunks(query, k=2):

    query_embedding = embedding_model.encode([query]).astype("float32")

    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(metadata[idx])

    return retrieved_chunks

# Load the LLM
generator = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    max_new_tokens=150,
    do_sample=False
)

#LLM needs document text as context
def build_context(chunks):

    context = ""

    for chunk in chunks:
        context += chunk["text"][:500] + "\n"

    return context

# prompt
def build_prompt(context, question):

    prompt = f"""
You are an AI assistant analyzing an annual report.

From the context below, identify the key risks mentioned in the report.

Context:
{context}

Question:
{question}

Provide a short and clear answer listing the risks.
Answer:
"""

    return prompt

# for generating answer
def generate_answer(prompt):

    response = generator(prompt)

    answer = response[0]["generated_text"]

    # Remove the prompt from the generated output
    answer = answer.replace(prompt, "").strip()

    return answer

# RAG Pipeline
def ask_question(question):

    retrieved = retrieve_chunks(question)

    print("\nRetrieved Chunks:\n")
    for chunk in retrieved:
        print(chunk["text"][:200])
        print("-----")

    context = build_context(retrieved)

    prompt = build_prompt(context, question)

    answer = generate_answer(prompt)

    return answer

# Testing system
if __name__ == "__main__":

    question = "What risks does the company mention?"

    answer = ask_question(question)

    print("\nAnswer:\n")
    print(answer)
