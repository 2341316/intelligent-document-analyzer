import json
from app.extraction.entity_extractor import extract_entities

INPUT_FILE = "data/processed/chunks_with_embeddings.json"
OUTPUT_FILE = "data/processed/chunks_with_entities.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

for chunk in chunks:

    entities = extract_entities(chunk["text"])

    chunk["entities"] = entities

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("Entity extraction complete.")