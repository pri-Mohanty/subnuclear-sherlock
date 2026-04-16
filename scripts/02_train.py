import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import JetDataset, TinyTransformer

def main():
    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Load Data
    events = np.load('data/raw/events.npy')
    labels = np.load('data/raw/labels.npy')
    
    full_dataset = JetDataset(events, labels)
    
    # Train/Test Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. Initialize Model
    model = TinyTransformer().to(device)
    
    # Using BCEWithLogitsLoss because our model outputs raw logits (no sigmoid at the end)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_events, batch_labels in train_loader:
            batch_events, batch_labels = batch_events.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_events)
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
            
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")

    # Add this at the very end of main() in 02_train.py
    os.makedirs('data/models', exist_ok=True)
    torch.save(model.state_dict(), 'data/models/tiny_transformer.pth')
    print("Model saved to data/models/tiny_transformer.pth")

if __name__ == "__main__":
    main()