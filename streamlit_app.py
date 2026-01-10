import streamlit as st
import torch
from embedding.encoder import TextEncoder
from model.compatibility_classifier import CompatibilityClassifier

st.title("📖 Story Consistency Checker (Track B)")

story = st.text_area("Paste story (up to 100k words)")
question = st.text_area("Paste character snippet / question")

if st.button("Analyze"):
    if story.strip() and question.strip():
        encoder = TextEncoder()
        classifier = CompatibilityClassifier()
        classifier.load_state_dict(torch.load("outputs/classifier.pth"))
        classifier.eval()
        s_emb = encoder.encode(story)
        q_emb = encoder.encode(question)
        with torch.no_grad():
            out = classifier(s_emb, q_emb)
            pred = torch.argmax(out, dim=1).item()
            conf = out[0, pred].item()
        label = "✅ Consistent" if pred == 1 else "❌ Contradict"
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {conf:.3f}")
    else:
        st.warning("Please paste both story and question.")
