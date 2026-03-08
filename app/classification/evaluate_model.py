import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


# LOAD DATA 

with open("data/processed/combined_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

labels = []
for chunk in chunks:
    label = chunk["label"]

    if label in ["Corporate_Overview", "Risk_Management", "Management_Discussion"]:
        labels.append("Other")
    else:
        labels.append(label)

print("Total chunks:", len(texts))
print("Class distribution:", Counter(labels))


#  EXPERIMENT CONFIGS


#experiments = [
#    {
#        "name": "LogReg_unigram",
#        "model": "logreg",
#        "ngram_range": (1,1),
#        "max_features": 5000,
#        "min_df": 1
#    },
#    {
#        "name": "LogReg_bigram",
#        "model": "logreg",
#        "ngram_range": (1,2),
#        "max_features": 5000,
#        "min_df": 1
#    },
#    {
#        "name": "LogReg_bigram_min_df2",
#        "model": "logreg",
#        "ngram_range": (1,2),
#        "max_features": 5000,
#        "min_df": 2
#    },
#    {
#        "name": "LinearSVC_bigram",
#        "model": "svm",
#        "ngram_range": (1,2),
#        "max_features": 5000,
#        "min_df": 2
#    }
#]

experiments = [
    {
        "name": "Final_LogReg_Bigram_Merged",
        "model": "logreg",
        "ngram_range": (1,2),
        "max_features": 5000,
        "min_df": 1
    }
]


for exp in experiments:
    print("\n==============================")
    print("Running Experiment:", exp["name"])
    print("==============================")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    macro_f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):

        X_train = [texts[i] for i in train_idx]
        X_test = [texts[i] for i in test_idx]
        y_train = [labels[i] for i in train_idx]
        y_test = [labels[i] for i in test_idx]

        vectorizer = TfidfVectorizer(
            ngram_range=exp["ngram_range"],
            max_features=exp["max_features"],
            min_df=exp["min_df"],
            stop_words="english"
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        if exp["model"] == "logreg":
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
        else:
            model = LinearSVC()

        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        fold_f1 = f1_score(y_test, y_pred, average="macro")
        macro_f1_scores.append(fold_f1)

        print(f"Fold {fold+1} Macro F1:", fold_f1)

    print("Average Macro F1:", sum(macro_f1_scores)/len(macro_f1_scores))



# FINAL TRAIN-TEST CONFUSION MATRIX

print("\nGenerating final confusion matrix...")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    min_df=1,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("\nClassification Report (Final Split):\n")
print(classification_report(y_test, y_pred))

unique_labels = sorted(list(set(labels)))
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (Final Model)")
plt.colorbar()

tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, unique_labels, rotation=45, ha="right")
plt.yticks(tick_marks, unique_labels)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center")

plt.tight_layout()
plt.show()