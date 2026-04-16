import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import TinyTransformer

def main():
    device = torch.device("cpu") # Probing is easier on CPU RAM
    
    # 1. Load Data and Model
    events = np.load('data/raw/events.npy')
    model = TinyTransformer().to(device)
    model.load_state_dict(torch.load('data/models/tiny_transformer.pth', map_location=device, weights_only=True))
    model.eval()

    # 2. Calculate the "Ground Truth" Physics for all events
    print("Calculating Ground Truth Physics...")
    E_tot = events[:, 0, 0] + events[:, 1, 0]
    px_tot = events[:, 0, 1] + events[:, 1, 1]
    py_tot = events[:, 0, 2] + events[:, 1, 2]
    pz_tot = events[:, 0, 3] + events[:, 1, 3]

    true_total_energy = E_tot
    # M^2 = E^2 - p^2 (Using max(0, x) to avoid floating point negative roots)
    true_inv_mass = np.sqrt(np.maximum(0, E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)))

    # 3. Setup PyTorch Hooks to steal the hidden states
    hidden_states = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # output shape is [Batch, Seq, d_model]. We mean-pool to get [Batch, d_model]
            hidden_states[name] = output.mean(dim=1).detach().numpy()
        return hook

    # Attach the hook to the output of the Transformer Encoder
    model.transformer.register_forward_hook(get_activation('encoder_out'))

    # 4. Run the data through the model to capture the states
    print("Pushing data through the model to capture internal states...")
    with torch.no_grad():
        tensor_events = torch.tensor(events, dtype=torch.float32)
        _ = model(tensor_events)
        
    captured_embeddings = hidden_states['encoder_out']

    # 5. The Linear Probes
    print("\n--- Training Linear Probes ---")
    
    # Probe 1: Did it learn Total Energy?
    probe_energy = LinearRegression()
    probe_energy.fit(captured_embeddings, true_total_energy)
    energy_preds = probe_energy.predict(captured_embeddings)
    r2_energy = r2_score(true_total_energy, energy_preds)
    
    # Probe 2: Did it learn Invariant Mass?
    probe_mass = LinearRegression()
    probe_mass.fit(captured_embeddings, true_inv_mass)
    mass_preds = probe_mass.predict(captured_embeddings)
    r2_mass = r2_score(true_inv_mass, mass_preds)

    print(f"Probe R^2 for predicting Total Energy:   {r2_energy:.4f}")
    print(f"Probe R^2 for predicting Invariant Mass: {r2_mass:.4f}")
    
    if r2_energy > r2_mass:
        print("\nConclusion: The model cheated! It heavily encodes raw Energy rather than the true Invariant Mass.")
    else:
        print("\nConclusion: The model learned true physics! It encodes the invariant mass formula.")

if __name__ == "__main__":
    main()