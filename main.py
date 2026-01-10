import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from embedding.encoder import TextEncoder
from model.compatibility_classifier import CompatibilityClassifier
from bdh.update import update_memory

# ---------------- PATHS ----------------
DATA_DIR = "data"
INPUT_DIR = "input"
OUTPUT_PATH = "outputs/memory.json"

# ---------------- HELPERS ----------------
def combine_text(row):
    parts = [str(row['book_name']), str(row['char']), str(row['caption']), str(row['content'])]
    return " ".join([p for p in parts if p.strip()])

def chunk_text(text, max_chars=2000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# ---------------- TRAINING ----------------
def train_classifier(train_path="data/train.csv"):
    print("📘 Training classifier on train.csv ...")
    df = pd.read_csv(train_path)
    df['text'] = df.apply(combine_text, axis=1)
    label_map = {'contradict': 0, 'consistent': 1}
    y = torch.tensor(df['label'].map(label_map).values)

    encoder = TextEncoder()
    classifier = CompatibilityClassifier()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    embeddings = []
    for text in df['text']:
        emb = encoder.encode(text)
        embeddings.append(emb)
    X = torch.vstack(embeddings)

    # --- training loop ---
    classifier.train()
    for epoch in range(5):  # small hackathon run
        optimizer.zero_grad()
        # simple self-pairing (X,X) for demonstration
        outputs = classifier(X, X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    torch.save(classifier.state_dict(), "outputs/classifier.pth")
    print("✅ Training complete — model saved to outputs/classifier.pth")
    return classifier

# ---------------- INFERENCE ----------------
def run_inference():
    print("🔍 Running inference on input stories ...")

    # 1️⃣ Load stories
    stories = {}
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(".txt"):
            with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
                stories[file.replace(".txt", "").strip()] = f.read()

    # 2️⃣ Load test dataset
    test_path = os.path.join(INPUT_DIR, "test.csv")
    test_df = pd.read_csv(test_path)

    # 3️⃣ Load encoder + trained classifier
    encoder = TextEncoder()
    classifier = CompatibilityClassifier()
    ckpt = "outputs/classifier.pth"
    if os.path.exists(ckpt):
        classifier.load_state_dict(torch.load(ckpt))
    classifier.eval()

    # 4️⃣ Process each story
    for story_name, story_text in stories.items():
        related = test_df[test_df["book_name"].str.lower().str.contains(story_name.lower())]
        if related.empty:
            print(f"⚠️ No test rows for {story_name}")
            continue

        # Encode long story in chunks → mean-pool
        chunks = chunk_text(story_text)
        story_embs = [encoder.encode(c) for c in chunks]
        story_emb = torch.mean(torch.vstack(story_embs), dim=0, keepdim=True)

        for _, row in related.iterrows():
            snippet = str(row["content"])
            character = str(row["char"])
            snippet_emb = encoder.encode(snippet)

            with torch.no_grad():
                out = classifier(story_emb, snippet_emb)
                pred = torch.argmax(out, dim=1).item()
                conf = out[0, pred].item()

            entry = {
                "story_name": story_name,
                "character": character,
                "snippet": snippet[:200] + "...",
                "prediction": int(pred),
                "confidence": round(conf, 3)
            }
            update_memory(entry)

    print("✅ Inference complete. Results saved in outputs/memory.json")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists("outputs/classifier.pth"):
        train_classifier()  # train once
    run_inference()
