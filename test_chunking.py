from app.cleaning.chunker import create_chunks

sample_text = """
Paste some long cleaned text here from your PDF.
"""

chunks = create_chunks(sample_text, page_number=1)

print("Number of chunks:", len(chunks))

sizes = [len(chunk["text"].split()) for chunk in chunks]
print("Average chunk size:", sum(sizes) / len(sizes))

print("\nPreview:\n")
print(chunks[0]["text"][:500])