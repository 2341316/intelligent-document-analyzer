import spacy

# Load model once
nlp = spacy.load("en_core_web_sm")


def split_into_sentences(text):
    """
    Split text into sentences using spaCy
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def create_chunks(text, page_number, min_words=400, max_words=600):
    """
    Create sentence-aware chunks
    """
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_word_count = 0
    chunk_id = 1

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words:
            chunks.append({
                "chunk_id": chunk_id,
                "text": " ".join(current_chunk),
                "page": page_number
            })

            chunk_id += 1
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(current_chunk),
            "page": page_number
        })

    return chunks