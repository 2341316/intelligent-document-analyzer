from fastapi import FastAPI, UploadFile, File
import os


from app.ingestion.pdf_reader import parse_pdf
from app.cleaning.text_cleaner import clean_document
from app.vector_store.search import search
from app.rag.rag_pipeline import ask_question

import joblib

classifier_model = joblib.load("baseline_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


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

    save_path = f"data/raw/{file.filename}"

    # Save file
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Parse PDF
    parsed = parse_pdf(save_path)

    # Clean + chunk
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


# Classification Placeholder

@app.post("/classify")
def classify(text: str):

    # Convert text → TFIDF features
    vector = vectorizer.transform([text])

    # Predict label
    prediction = classifier_model.predict(vector)[0]

    return {
        "text": text,
        "prediction": prediction
    }