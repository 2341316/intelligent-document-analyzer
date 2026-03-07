import pandas as pd

df = pd.read_json("data/processed/combined_chunks.json")

print("\nLabel Distribution:\n")
print(df["label"].value_counts())

print("\nCompany Distribution:\n")
print(df["company_name"].value_counts())