import json
import pandas as pd
from datasets import Dataset

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load combined dataset
with open("data/processed/combined_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# To convert labels to numeric
df["label"] = df["label"].astype("category")
df["label_id"] = df["label"].cat.codes

print("Label Mapping:")
print(df[["label", "label_id"]].drop_duplicates())

# Keeping only required columns
df = df[["text", "label_id"]]
df = df.rename(columns={"label_id": "label"})

print("Dataset shape:", df.shape)
print("Number of classes:", df["label"].nunique())

# Now to convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

print(dataset)



# ---------------- TOKENIZATION (MOVE THIS UP) ----------------

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

print(tokenized_dataset)

# ---------------- MODEL LOADING ----------------

num_labels = df["label"].nunique()

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

print("Model loaded successfully.")

# ---------------- TRAINING ----------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro")
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print("Final Results:", results)