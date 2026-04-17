import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import TinyTransformer

def main():
    device = torch.device("cpu") 
    
    # 1. Load Data and Model
    events = np.load('data/raw/events.npy')
    model = TinyTransformer().to(device)
    model.load_state_dict(torch.load('data/models/tiny_transformer.pth', map_location=device, weights_only=True))
    model.eval()

    # 2. Calculate the "Ground Truth" Physics
    print("Calculating Ground Truth Physics...")
    E_tot = events[:, 0, 0] + events[:, 1, 0]
    px_tot = events[:, 0, 1] + events[:, 1, 1]
    py_tot = events[:, 0, 2] + events[:, 1, 2]
    pz_tot = events[:, 0, 3] + events[:, 1, 3]

    true_total_energy = E_tot
    true_inv_mass = np.sqrt(np.maximum(0, E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)))

    # 3. Setup PyTorch Hooks
    hidden_states = {}
    
    def get_activation(name):
        def hook(model, input, output):
            hidden_states[name] = output.mean(dim=1).detach().numpy()
        return hook

    model.transformer.register_forward_hook(get_activation('encoder_out'))

    # 4. Capture States
    print("Pushing data through the model to capture internal states...")
    with torch.no_grad():
        tensor_events = torch.tensor(events, dtype=torch.float32)
        _ = model(tensor_events)
        
    captured_embeddings = hidden_states['encoder_out']

    # 5. Train/Test Split for the Probes (Crucial for honest evaluation)
    print("\n--- Training Linear Probes (with Held-Out Test Set) ---")
    
    X_train, X_test, energy_train, energy_test, mass_train, mass_test = train_test_split(
        captured_embeddings, true_total_energy, true_inv_mass, test_size=0.2, random_state=42
    )
    
    # Probe 1: Total Energy
    probe_energy = LinearRegression().fit(X_train, energy_train)
    energy_preds = probe_energy.predict(X_test)
    r2_energy = r2_score(energy_test, energy_preds)
    
    # Probe 2: Invariant Mass
    probe_mass = LinearRegression().fit(X_train, mass_train)
    mass_preds = probe_mass.predict(X_test)
    r2_mass = r2_score(mass_test, mass_preds)

    print(f"Held-out R^2 for predicting Total Energy:   {r2_energy:.4f}")
    print(f"Held-out R^2 for predicting Invariant Mass: {r2_mass:.4f}")
    
    # 6. Honest Evaluation Logic
    print("\n--- Pipeline Evaluation ---")
    if r2_mass > 0.90:
        print("Success: Strong evidence of invariant mass encoding in this run.")
    elif r2_mass > r2_energy:
        print("Note: Model prefers invariant mass over raw energy, but encoding is weak/distributed.")
    else:
        print("Warning: Model failed to strongly encode relativistic invariants in this specific layer/run.")

if __name__ == "__main__":
    main()