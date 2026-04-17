import torch
import torch.nn as nn
from torch.utils.data import Dataset

class JetDataset(Dataset):
    def __init__(self, events, labels):
        self.events = torch.tensor(events, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.events[idx], self.labels[idx]

class TinyTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2):
        super(TinyTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1) 
        )

    # CRITICAL FIX: explicit mask parameter routed to the TransformerEncoder
    def forward(self, x, mask=None):
        x = self.embedding(x)
        latent = self.transformer(x, mask=mask)
        pooled = latent.mean(dim=1) 
        out = self.classifier(pooled)
        return out.squeeze()