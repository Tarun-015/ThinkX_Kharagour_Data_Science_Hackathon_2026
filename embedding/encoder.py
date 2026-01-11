from sentence_transformers import SentenceTransformer
import torch

class TextEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, text: str):
        emb = self.model.encode([text], convert_to_tensor=True)
        return emb 
