from app.ingestion.pdf_reader import parse_pdf
from app.cleaning.text_cleaner import clean_document

file_path = "sample.pdf"

document = parse_pdf(file_path)

print("\n================ RAW DOCUMENT ================\n")
print(document["pages"][0][:500])  # show first 500 characters only

cleaned_document = clean_document(document)

print("\n================ CLEANED DOCUMENT ================\n")

print("Total Pages Extracted:", len(cleaned_document["pages"]))

print("\nPage 1 Preview:\n")
print(cleaned_document["pages"][0][:300])

print("\nPage 5 Preview:\n")
print(cleaned_document["pages"][4][:300])