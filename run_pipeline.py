from app.ingestion.pdf_reader import parse_pdf
from app.cleaning.text_cleaner import clean_document
import json
import os

SECTION_LABELS = {
    "corporate": "Corporate_Overview",
    "management": "Management_Discussion",
    "risk": "Risk_Management",
    "governance": "Governance",
    "sustainability": "Sustainability",
    "financial": "Financial_Statements",
    "unknown": "Other"
}

#  Helper: Detect Company 
def get_company_name(filename):
    filename = filename.lower()

    if "infosys" in filename:
        return "Infosys"
    elif "tcs" in filename:
        return "TCS"
    elif "wipro" in filename:
        return "Wipro"
    else:
        return "Unknown"

#  Folders
raw_folder = "data/raw"
output_path = "data/processed/combined_chunks.json"

all_chunks = []
chunk_counter = 1

# Process All PDFs 
for filename in os.listdir(raw_folder):

    if filename.endswith(".pdf"):

        pdf_path = os.path.join(raw_folder, filename)
        company_name = get_company_name(filename)

        print(f"\nProcessing {company_name} report...")

        # Step 1: Ingestion
        document = parse_pdf(pdf_path)

        if document is None:
            print(f"Skipping {filename} (parse failed)")
            continue

        # Step 2: Cleaning + Chunking
        processed_document = clean_document(document)

        print("Total chunks:", len(processed_document["chunks"]))

        for chunk in processed_document["chunks"]:
            chunk["chunk_id"] = chunk_counter
            chunk["company_name"] = company_name

            all_chunks.append(chunk)
            chunk_counter += 1

#  Save Combined File 
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=4)

print("\nAll chunks saved successfully to combined_chunks.json")
print("Total combined chunks:", len(all_chunks))