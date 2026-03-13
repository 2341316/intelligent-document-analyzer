import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Only keep useful entity types
IMPORTANT_LABELS = {"ORG", "MONEY", "DATE", "PERCENT", "GPE"}


def extract_entities(text):
    """
    Extract named entities from a chunk of text.
    """

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ in IMPORTANT_LABELS:

            entities.append({
                "text": ent.text,
                "label": ent.label_
            })

    return entities