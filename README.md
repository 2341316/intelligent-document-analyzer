# intelligent-document-analyzer
PDF to Insights Engine – Project

# Intelligent Document Analyzer (NLP + ML + RAG)

This project is an end-to-end system for analyzing corporate PDF documents and extracting meaningful insights using Natural Language Processing (NLP), Machine Learning (ML), and Retrieval-Augmented Generation (RAG).

The system can process large annual reports, perform semantic search, classify document sections, and answer questions based on document content. This branch uses a higher-capability language model for better answer quality.

---

## Overview

Corporate reports are long and complex, making manual analysis difficult. This system converts unstructured PDF documents into an interactive knowledge system.

It supports:

* Document ingestion and processing
* Semantic search based on meaning (not keywords)
* Document classification
* Question answering using retrieved context

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

---

## System Architecture

```
PDF → Ingestion → Cleaning → Chunking → Classification → Embeddings → FAISS → Retrieval → NER → RAG → FastAPI
```

---

## Project Structure

```
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

* microsoft/phi-2

This model provides better reasoning and more detailed answers but requires higher memory and longer loading time.

---

## Results

* Classification Accuracy: ~74%
* Macro F1 Score: ~0.58
* Semantic Retrieval: Precision@5 = 0.88

The system performs well in retrieving relevant document sections and generating context-based answers.

---

## Installation

1. Clone the repository

```
git clone https://github.com/your-username/intelligent-document-analyzer
cd intelligent-document-analyzer
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the API

```
uvicorn app.api.main:app --reload
```

---

## API Endpoints

* **POST /upload** → Upload and process PDF documents
* **POST /search** → Perform semantic search
* **POST /rag** → Ask questions based on document content
* **POST /classify** → Classify input text

Swagger UI:

```
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

## Notes on This Branch (main)

* Uses **microsoft/phi-2** for answer generation
* Produces higher-quality and more detailed responses
* Slower and more memory-intensive
* May take time to load model initially

---

## Limitations

* Requires significant memory for model loading
* Slower response time compared to lightweight models
* Performance depends on quality of extracted text

---

## Future Improvements

* Use larger datasets for better model performance
* Add reranking for improved retrieval accuracy
* Build a frontend interface
* Optimize model performance further

---

## Author

Maria P A
