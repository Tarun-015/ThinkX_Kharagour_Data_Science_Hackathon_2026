import streamlit as st
import torch
import os
import json
import pandas as pd

from embedding.encoder import TextEncoder
from model.compatibility_classifier import CompatibilityClassifier
from bdh.update import update_memory


# ================== CONFIG ==================
st.set_page_config(
    page_title="Kharagpur Hackathon – Track B",
    layout="wide"
)

MODEL_PATH = "outputs/classifier.pth"
MEMORY_PATH = "outputs/memory.json"


# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    encoder = TextEncoder()
    classifier = CompatibilityClassifier()

    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model not found. Run `python main.py` first.")
        st.stop()

    classifier.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )
    classifier.eval()
    return encoder, classifier


encoder, classifier = load_model()


# ================== UI ==================
st.markdown(
    "<h1 style='text-align:center;'>📘 Story Consistency Checker</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    story = st.text_area(
        "Full Story / Novel Text",
        height=300,
        placeholder="Paste story text here..."
    )

with col2:
    character = st.text_input("Character Name")
    caption = st.text_input("Caption / Context")
    snippet = st.text_area(
        "Character Snippet / Event",
        height=300,
        placeholder="Describe the character backstory..."
    )


# ================== HARD CONTRADICTION RULE ==================
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


# ================== ANALYSIS ==================
if st.button("🔍 Analyze Consistency"):
    if not story.strip() or not snippet.strip():
        st.warning("Please provide both story and snippet.")
    else:
        with st.spinner("Analyzing..."):

            # 🔴 HARD RULE OVERRIDE
            if hard_contradiction_rule(story, snippet):
                pred = 0
                confidence = 0.99

            # 🧠 MODEL INFERENCE
            else:
                story_emb = encoder.encode(story)      # (1, 384)
                snippet_emb = encoder.encode(snippet)  # (1, 384)

                with torch.no_grad():
                    logits = classifier(story_emb, snippet_emb)
                    probs = torch.softmax(logits, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred].item()

            # 🎯 DISPLAY RESULT
            label = "✅ CONSISTENT" if pred == 1 else "❌ CONTRADICT"
            color = "#2ECC71" if pred == 1 else "#E74C3C"

            st.markdown(
                f"<h2 style='text-align:center; color:{color};'>{label}</h2>",
                unsafe_allow_html=True
            )

            st.caption(f"Confidence: {confidence * 100:.2f}%")

            # 💾 SAVE MEMORY
            entry = {
                "character": character or "Unknown",
                "caption": caption,
                "prediction": int(pred),
                "confidence": round(confidence, 3),
                "snippet": snippet[:200]
            }
            update_memory(entry)

            st.success("Result saved to memory.")


# ================== MEMORY VIEW ==================
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📚 Memory Log")

if os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        memory = json.load(f)

    if memory:
        df = pd.DataFrame(memory).T
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Memory is empty.")
else:
    st.info("No memory file yet.")
