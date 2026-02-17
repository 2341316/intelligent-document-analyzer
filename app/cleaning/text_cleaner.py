import re
from app.utils.logger import logger
from app.cleaning.chunker import create_chunks


def clean_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_line_breaks(text: str) -> str:
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')
    return text


def remove_headers_footers(text: str) -> str:
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    return text


def clean_page_text(text: str) -> str:
    text = normalize_line_breaks(text)
    text = remove_headers_footers(text)
    text = clean_whitespace(text)
    return text


def clean_document(document: dict) -> dict:
    cleaned_pages = []

    # Step 1: Clean each page
    for i, page in enumerate(document["pages"], start=1):
        if page and page.strip():
            logger.info(f"Cleaning page {i}...")
            cleaned_page = clean_page_text(page)
            cleaned_pages.append(cleaned_page)
        else:
            logger.warning(f"Skipping empty page {i}")

    # Step 2: Chunk each cleaned page
    all_chunks = []

    for page_number, page_text in enumerate(cleaned_pages, start=1):
        logger.info(f"Chunking page {page_number}...")
        page_chunks = create_chunks(page_text, page_number)
        all_chunks.extend(page_chunks)

    # Step 3: Return chunks instead of pages
    return {
        "filename": document["filename"],
        "chunks": all_chunks
    }