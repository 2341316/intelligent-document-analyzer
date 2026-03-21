# intelligent-document-analyzer
PDF to Insights Engine – Project

# Intelligent Document Analyzer (NLP + ML + RAG) – Stable Version

This project is an end-to-end system for analyzing corporate PDF documents and extracting meaningful insights using Natural Language Processing (NLP), Machine Learning (ML), and Retrieval-Augmented Generation (RAG).

This branch (`stable-flan`) is an optimized and deployment-ready version of the system. It uses a lightweight language model to ensure faster response times, lower memory usage, and stable API performance.

---

## Overview

Corporate annual reports are large and complex documents. This system converts unstructured PDF data into a searchable and interactive format.

It supports:

* Document ingestion and processing
* Semantic search using embeddings
* Document classification
* Question answering using retrieved context

This version is designed for **practical usage and deployment**.

---

## Features

* PDF ingestion and text extraction using pdfplumber
* Text cleaning and preprocessing
* Sentence-aware chunking using spaCy
* Document classification using TF-IDF + Logistic Regression
* Semantic embeddings using Sentence Transformers
* Vector similarity search using FAISS
* Named Entity Recognition (NER) using spaCy
* Retrieval-Augmented Generation (RAG) for question answering
* FastAPI-based backend for real-time interaction
* Dockerized deployment for reproducibility

---

## System Architecture

```id="b98wkp"
PDF → Ingestion → Cleaning → Chunking → Classification → Embeddings → FAISS → Retrieval → NER → RAG → FastAPI → Docker
```

---

## Project Structure

```id="rxgtmr"
intelligent-document-analyzer/

app/
   ingestion/
   cleaning/
   classification/
   embeddings/
   vector_store/
   retrieval/
   rag/
   api/

data/
   raw/
   processed/

notebooks/

run_pipeline.py
check_distribution.py
test_chunking.py

requirements.txt
Dockerfile
README.md
```

---

## Models Used

### Classification

* TF-IDF + Logistic Regression
* DistilBERT (experimental)

### Embeddings

* sentence-transformers/all-MiniLM-L6-v2

### Vector Database

* FAISS

### RAG Model (This Branch)

* google/flan-t5-base

This model is lightweight and optimized for faster inference, making it suitable for API deployment and Docker environments.

---

## Results

* Classification Accuracy: ~74%
* Macro F1 Score: ~0.58
* Semantic Retrieval: Precision@5 = 0.88

The system provides reliable retrieval and stable answer generation in real-time scenarios.

---

## Installation

1. Clone the repository

```id="7qk9ne"
git clone https://github.com/your-username/intelligent-document-analyzer
cd intelligent-document-analyzer
git checkout stable-flan
```

2. Install dependencies

```id="zfsrte"
pip install -r requirements.txt
```

3. Run the API

```id="rsyczd"
uvicorn app.api.main:app --reload
```

---

## Docker Usage

Build the Docker image:

```id="q9n8o6"
docker build -t doc-intelligence .
```

Run the container:

```id="5ok2hz"
docker run -p 8000:8000 doc-intelligence
```

---

## API Endpoints

* **POST /upload** → Upload and process PDF documents
* **POST /search** → Perform semantic search
* **POST /rag** → Ask questions based on document content
* **POST /classify** → Classify input text

Swagger UI:

```id="4o2r9h"
http://127.0.0.1:8000/docs
```

---

## Example Workflow

1. Upload a PDF document
2. System extracts and processes text into chunks
3. User submits a query
4. Relevant chunks are retrieved using FAISS
5. RAG model generates an answer based on context

---

## Notes on This Branch (stable-flan)

* Uses **google/flan-t5-base** for answer generation
* Faster and more stable compared to larger models
* Lower memory usage (suitable for CPU environments)
* Recommended for deployment and real-time applications

---

## Limitations

* Answers may be shorter or less detailed compared to larger models
* Performance depends on retrieved context quality
* Limited reasoning compared to larger LLMs

---

## Future Improvements

* Add reranking for better retrieval quality
* Integrate larger models for hybrid setups
* Build frontend UI for user interaction
* Improve dataset balance and coverage

---

## Author

Maria P A
