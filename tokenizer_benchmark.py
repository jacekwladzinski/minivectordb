from transformers import AutoTokenizer
import time
import pandas as pd

models = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "DistilBERT": "distilbert-base-uncased",
    "ELECTRA-small": "google/electra-small-discriminator",
    "RoBERTa-base": "roberta-base"
}

texts = ["Attention Is All You Need"] * 100_000

results = []
for name, model_name in models.items():
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    start = time.time()
    tokenizer(texts, padding=True, truncation=True)
    duration = time.time() - start
    results.append({"Model": name, "Time (s)": duration})

df = pd.DataFrame(results).sort_values("Time (s)")
print(df)
