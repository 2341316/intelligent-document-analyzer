from app.ingestion.pdf_reader import parse_pdf
from app.cleaning.text_cleaner import clean_document
import json

pdf_path = "sample.pdf"

# Step 1: Ingestion
document = parse_pdf(pdf_path)

if document is None:
    print("PDF parsing failed.")
    exit()

# Step 2: Cleaning + Chunking
processed_document = clean_document(document)

# Step 3: Inspect output
print("Total chunks:", len(processed_document["chunks"]))

if processed_document["chunks"]:
    print("\nFirst chunk preview:\n")
    print(processed_document["chunks"][0]["text"][:500])
else:
    print("No chunks were created.")

sizes = [len(chunk["text"].split()) for chunk in processed_document["chunks"]]
print("Average chunk size:", sum(sizes) / len(sizes))

# Assign proper unique chunk IDs
all_chunks = []
chunk_counter = 1

for chunk in processed_document["chunks"]:
    chunk["chunk_id"] = chunk_counter
    all_chunks.append(chunk)
    chunk_counter += 1

# Assign proper unique chunk IDs
all_chunks = []
chunk_counter = 1

for chunk in processed_document["chunks"]:
    chunk["chunk_id"] = chunk_counter
    all_chunks.append(chunk)
    chunk_counter += 1

# Save chunks to file
with open("chunks.json", "w") as f:
    json.dump(all_chunks, f, indent=4)

print("\nChunks saved successfully to chunks.json")