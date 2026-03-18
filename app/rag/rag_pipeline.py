import faiss
import pickle
import numpy as np

##  LAZY LOADING VARIABLES 
embedding_model = None
generator = None
index = None
metadata = None
generator = None


##  LOADERS 

def get_index():
    global index
    if index is None:
        print("Loading FAISS index...")
        index = faiss.read_index("data/processed/faiss_index.index")
    return index


def get_metadata():
    global metadata
    if metadata is None:
        print("Loading metadata...")
        with open("data/processed/faiss_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
    return metadata


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


def get_generator():
    global generator
    if generator is None:
        print("Loading LLM...")
        from transformers import pipeline
        generator = pipeline(
            "text-generation",
            model="microsoft/phi-2",
            max_new_tokens=150,
            do_sample=False
        )
    return generator


## RETRIEVAL

def retrieve_chunks(query, k=2):

    model = get_embedding_model()
    idx = get_index()
    meta = get_metadata()

    query_embedding = model.encode([query]).astype("float32")

    distances, indices = idx.search(query_embedding, k)

    retrieved_chunks = []

    for i in indices[0]:
        retrieved_chunks.append(meta[i])

    return retrieved_chunks


## CONTEXT BUILDING

def build_context(chunks):

    context = ""

    for chunk in chunks:
        context += chunk["text"][:500] + "\n"

    return context


## PROMPT

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


## GENERATION

def generate_answer(prompt):
    global generator

    if generator is None:
        print("Loading LLM...")
        from transformers import pipeline

        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=150
        )

    response = generator(prompt)

    answer = response[0]["generated_text"]
    answer = answer.replace(prompt, "").strip()

    return answer


## RAG PIPELINE

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


##  TEST 

if __name__ == "__main__":

    question = "What risks does the company mention?"

    answer = ask_question(question)

    print("\nAnswer:\n")
    print(answer)