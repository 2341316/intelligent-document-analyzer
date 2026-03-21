import re
from app.utils.logger import logger
from app.cleaning.chunker import create_chunks


# SECTION LABEL MAP

SECTION_LABELS = {
    "corporate": "Corporate_Overview",
    "management": "Management_Discussion",
    "risk": "Risk_Management",
    "governance": "Governance",
    "sustainability": "Sustainability",
    "financial": "Financial_Statements",
    "unknown": "Other"
}


# BASIC CLEANING 

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


#MAIN FUNCTION

def clean_document(document: dict) -> dict:

    cleaned_pages = []

    # Step 1: Clean each page
    for page in document["pages"]:
        if page and page.strip():
            cleaned_page = clean_page_text(page)
            cleaned_pages.append(cleaned_page)
        else:
            cleaned_pages.append("")

    all_chunks = []
    current_section = "unknown"

    for page_number, page_text in enumerate(cleaned_pages, start=1):

        text_lower = page_text.lower()

        # Default early pages to corporate
        if current_section == "unknown" and page_number < 40:
            current_section = "corporate"

        #  STRICT SECTION DETECTION

        # Financial (very strict)
        if (
            "financial statements" in text_lower
            or "consolidated balance sheet" in text_lower
            or "statement of profit and loss" in text_lower
            or "independent auditor" in text_lower
            or "notes forming part" in text_lower
        ):
            current_section = "financial"

        elif "sustainability" in text_lower:
            current_section = "sustainability"

        elif (
            "corporate governance" in text_lower
            or "board of directors" in text_lower
            or "statutory report" in text_lower
            or "statutory section" in text_lower
        ):
            current_section = "governance"

        elif "risk management" in text_lower:
            current_section = "risk"

        elif (
            "management discussion" in text_lower
            or "performance review" in text_lower
            or "performance overview" in text_lower
            or "performance and outlook" in text_lower
            or "chairman" in text_lower
            or "ceo" in text_lower
        ):
            current_section = "management"

        elif (
            ("corporate" in text_lower and "overview" in text_lower)
            or "about infosys" in text_lower
            or "about tcs" in text_lower
            or "about wipro" in text_lower
        ):
            current_section = "corporate"

        # Chunk page 

        page_chunks = create_chunks(page_text, page_number)

        for chunk in page_chunks:
            chunk["section"] = current_section
            chunk["label"] = SECTION_LABELS[current_section]

        all_chunks.extend(page_chunks)

    return {
        "filename": document["filename"],
        "chunks": all_chunks
    }
