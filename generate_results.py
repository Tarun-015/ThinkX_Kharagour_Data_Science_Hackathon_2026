import pandas as pd
import torch
import os

from embedding.encoder import TextEncoder
from model.compatibility_classifier import CompatibilityClassifier

TEST_PATH = "input/test.csv"
MODEL_PATH = "outputs/classifier.pth"
OUTPUT_PATH = "results.csv"

def hard_contradiction_rule(story, snippet):
    rules = [
        ("born blind", "watched"),
        ("never", "always"),
        ("died", "ten years later"),
        ("never left", "traveled"),
    ]
    s = story.lower()
    c = snippet.lower()
    return any(a in s and b in c for a, b in rules)

# Load model
encoder = TextEncoder()
model = CompatibilityClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

df = pd.read_csv(TEST_PATH)

results = []

with torch.no_grad():
    for _, row in df.iterrows():
        story = str(row["content"])
        snippet = f"{row.get('char','')} {row.get('caption','')} {row['content']}"

        if hard_contradiction_rule(story, snippet):
            pred = 0
        else:
            e1 = encoder.encode(story)
            e2 = encoder.encode(snippet)
            logits = model(e1, e2)
            pred = torch.argmax(logits, dim=1).item()

        label = "consistent" if pred == 1 else "contradict"

        results.append({
            "id": row["id"],
            "label": label
        })

pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print("✅ results.csv generated successfully")
