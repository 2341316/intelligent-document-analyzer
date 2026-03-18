from fastapi import FastAPI, UploadFile, File
import os

from app.ingestion.pdf_reader import parse_pdf
from app.cleaning.text_cleaner import clean_document
from app.vector_store.search import search
from app.rag.rag_pipeline import ask_question

import joblib

# ✅ Correct order
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))

# Load models
classifier_model = joblib.load(os.path.join(ROOT_DIR, "baseline_model.pkl"))
vectorizer = joblib.load(os.path.join(ROOT_DIR, "tfidf_vectorizer.pkl"))

app = FastAPI(
    title="Intelligent Document Analysis API",
    description="API for document ingestion, semantic search, and RAG QA",
    version="1.0"
)

# Root Endpoint
@app.get("/")
def home():
    return {"message": "Document Intelligence API is running"}


# Upload + Process PDF
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    data_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)

    save_path = os.path.join(data_dir, file.filename)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    parsed = parse_pdf(save_path)
    cleaned = clean_document(parsed)

    return {
        "filename": file.filename,
        "total_chunks": len(cleaned["chunks"])
    }


# Semantic Search
@app.post("/search")
def semantic_search(query: str):

    results = search(query)

    return {
        "query": query,
        "results": results
    }


# RAG Question Answering
@app.post("/rag")
def rag_query(question: str):

    answer = ask_question(question)

    return {
        "question": question,
        "answer": answer
    }


# Classification
@app.post("/classify")
def classify(text: str):

    vector = vectorizer.transform([text])
    prediction = classifier_model.predict(vector)[0]

    return {
        "text": text,
        "prediction": prediction
    }