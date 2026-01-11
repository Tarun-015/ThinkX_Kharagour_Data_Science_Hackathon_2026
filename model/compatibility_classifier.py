import torch
import torch.nn as nn
import torch.nn.functional as F

class CompatibilityClassifier(nn.Module):
    def __init__(self, emb_dim=384):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim * 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, e1, e2):
        diff = torch.abs(e1 - e2)
        prod = e1 * e2
        x = torch.cat([e1, e2, diff, prod], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
