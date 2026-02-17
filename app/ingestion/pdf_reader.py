import pdfplumber
import os


def parse_pdf(file_path):
    """
    Reads a PDF file and extracts text from all pages.
    Returns structured dictionary output.
    """

    result = {
        "filename": os.path.basename(file_path),
        "pages": []
    }

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()

                # Handle empty pages
                if text:
                    result["pages"].append(text)
                else:
                    result["pages"].append("")

        return result

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
