import os
import json
import torch
import torch.nn as nn
import pandas as pd
from embedding.encoder import TextEncoder
from model.compatibility_classifier import CompatibilityClassifier

TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "classifier.pth")
MEMORY_PATH = os.path.join(OUTPUT_DIR, "memory.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")

def train_classifier():
    print("\n Training classifier...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")

    df = pd.read_csv(TRAIN_PATH)

    if "label" not in df.columns:
        raise ValueError("'label' column required in train.csv")

    encoder = TextEncoder()
    model = CompatibilityClassifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Balance labels
    label_counts = df["label"].value_counts().to_dict()
    weight = torch.tensor(
        [
            1.0 / label_counts.get(0, 1),
            1.0 / label_counts.get(1, 1),
        ],
        dtype=torch.float,
    )
    criterion = nn.CrossEntropyLoss(weight=weight)

    X1, X2, y = [], [], []
    label_map = {
    "contradict": 0,
    "consistent": 1
    }

    for _, row in df.iterrows():
        story = str(row["content"])
        snippet = f"{row.get('char','')} {row.get('caption','')} {row['content']}"

        X1.append(encoder.encode(story))
        X2.append(encoder.encode(snippet))

        label = str(row["label"]).strip().lower()
        y.append(label_map[label])


    X1 = torch.cat(X1).to(device)
    X2 = torch.cat(X2).to(device)
    y = torch.tensor(y).to(device)

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        logits = model(X1, X2)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# INFERENCE
def run_inference():
    print("\n🔍 Running inference...")

    if not os.path.exists(TEST_PATH):
        print(" No test.csv found — skipping inference")
        return

    encoder = TextEncoder()
    model = CompatibilityClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    df = pd.read_csv(TEST_PATH)
    results = []

    with torch.no_grad():
        for _, row in df.iterrows():
            story = str(row["content"])
            snippet = f"{row.get('char','')} {row.get('caption','')} {row['content']}"

            e1 = encoder.encode(story)
            e2 = encoder.encode(snippet)

            logits = model(e1, e2)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs))
            conf = float(probs[0, pred])

            results.append({
                "story": story[:100],
                "prediction": pred,
                "confidence": round(conf, 3)
            })

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Inference saved to {MEMORY_PATH}")

#main
if __name__ == "__main__":
    print("\n Track B Pipeline Starting")
    train_classifier()
    run_inference()
    print(" Done")
