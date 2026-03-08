import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


# LOAD DATA 

with open("data/processed/combined_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
labels = [chunk["label"] for chunk in chunks]

print("Total chunks:", len(texts))
print("Class distribution:", Counter(labels))


# STRATIFIED K-FOLD 

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
    print(f"\n========== Fold {fold+1} ==========")

    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred))

    acc = model.score(X_test_tfidf, y_test)
    print("Fold Accuracy:", acc)

    accuracies.append(acc)


#  FINAL RESULT 

print("\nAverage Accuracy:", sum(accuracies) / len(accuracies))


#  CONFUSION MATRIX (LAST FOLD) 

# Get sorted unique labels
unique_labels = sorted(list(set(labels)))

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, unique_labels, rotation=45, ha="right")
plt.yticks(tick_marks, unique_labels)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Add numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")

plt.tight_layout()
plt.show()