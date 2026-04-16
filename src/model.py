import torch
import torch.nn as nn
from torch.utils.data import Dataset

class JetDataset(Dataset):
    def __init__(self, events, labels):
        # Convert numpy arrays to PyTorch tensors
        self.events = torch.tensor(events, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.events[idx], self.labels[idx]

class TinyTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2):
        super(TinyTransformer, self).__init__()
        
        # 1. Project the 4-vector into a higher dimensional latent space
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Transformer Encoder (batch_first=True is crucial for [Batch, Seq, Features])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1) # Outputs a single logit (Signal vs Background)
        )

    def forward(self, x):
        # x shape: [Batch, 2 particles, 4 features]
        x = self.embedding(x)
        
        # Pass through transformer
        # We don't use positional encoding to maintain permutation invariance
        latent = self.transformer(x)
        
        # Mean Pooling: Combine the information from both particle tokens
        pooled = latent.mean(dim=1) 
        
        # Classify
        out = self.classifier(pooled)
        return out.squeeze() # Remove extra dimension for the loss function