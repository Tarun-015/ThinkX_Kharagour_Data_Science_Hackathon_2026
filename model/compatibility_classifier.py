# model/compatibility_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompatibilityClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super().__init__()
        # combining story + snippet embeddings (concatenate)
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 classes: contradict / consistent

    def forward(self, story_emb, snippet_emb):
        # story_emb, snippet_emb → [1, embedding_dim]
        x = torch.cat((story_emb, snippet_emb), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
