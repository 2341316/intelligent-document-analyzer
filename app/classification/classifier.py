import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


# STEP 1: Load chunk data
with open("chunks.json", "r") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
pages = [chunk["page"] for chunk in chunks]

print("Total chunks loaded:", len(texts))


# STEP 2: Assign labels
def assign_label(page):
    if 18 <= page <= 27:
        return 0  # Corporate overview
    elif 28 <= page <= 36:
        return 1  # Performance overview
    elif 37 <= page <= 41:
        return 2  # Approaching value creation
    elif 42 <= page <= 55:
        return 3  # Delivering value
    elif 56 <= page <= 203:
        return 4  # Statutory reports
    elif 204 <= page <= 352:
        return 5  # Financial statements
    else:
        return -1


labels = [assign_label(p) for p in pages]

# Remove unlabeled chunks
filtered = [(t, l) for t, l in zip(texts, labels) if l != -1]
texts, labels = zip(*filtered)

print("After filtering:", len(texts))
print("Class distribution:", Counter(labels))


# STEP 3: TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(texts)


# STEP 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)


# STEP 5: Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)


# STEP 6: Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# STEP 7: Save model
joblib.dump(model, "baseline_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully.")