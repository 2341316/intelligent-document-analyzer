import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import RandomOverSampler


# STEP 1: Load combined chunk data
with open("data/processed/combined_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# STEP 2: Extract text and labels (already assigned in JSON)
texts = [chunk["text"] for chunk in chunks]

labels = []
for chunk in chunks:
    label = chunk["label"]

    if label in ["Corporate_Overview", "Risk_Management", "Management_Discussion"]:
        labels.append("Other")
    else:
        labels.append(label)

print("Total chunks loaded:", len(texts))
print("Class distribution:", Counter(labels))


# STEP 3: TF-IDF (Optimized -  Best Config)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=1,
    stop_words="english"
)
X = vectorizer.fit_transform(texts)


# STEP 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels   # IMPORTANT improvement
)

# OVERSAMPLING

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("After Oversampling:", Counter(y_train_resampled))


# STEP 5: Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)


# STEP 6: Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# STEP 7: Save model
joblib.dump(model, "baseline_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully.")