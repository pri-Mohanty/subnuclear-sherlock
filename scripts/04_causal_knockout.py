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

    # 2. Ground Truth Physics
    print("Calculating Ground Truth Physics...")
    E_tot = events[:, 0, 0] + events[:, 1, 0]
    px_tot = events[:, 0, 1] + events[:, 1, 1]
    py_tot = events[:, 0, 2] + events[:, 1, 2]
    pz_tot = events[:, 0, 3] + events[:, 1, 3]
    true_inv_mass = np.sqrt(np.maximum(0, E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)))

    # 3. Hook setup
    hidden_states = {}
    def get_activation(name):
        def hook(model, input, output):
            hidden_states[name] = output.mean(dim=1).detach().numpy()
        return hook
    model.transformer.register_forward_hook(get_activation('encoder_out'))

    tensor_events = torch.tensor(events, dtype=torch.float32)

    # 4. --- RUN 1: Baseline (Normal Attention) ---
    print("Running baseline forward pass...")
    with torch.no_grad():
        _ = model(tensor_events)
    baseline_embeddings = hidden_states['encoder_out'].copy()

    # 5. --- RUN 2: Knockout (Severed Attention) ---
    print("Running causal knockout forward pass...")
    # We create a mask that prevents Particle 1 from looking at Particle 2, and vice versa.
    # In PyTorch, True means "do not attend". Mask shape: [2, 2]
    isolation_mask = torch.tensor([[False, True], 
                                   [True, False]], dtype=torch.bool)
    
    with torch.no_grad():
        _ = model(tensor_events, mask=isolation_mask)
    knockout_embeddings = hidden_states['encoder_out'].copy()

    # 6. --- Train/Test Split (The Methodological Fix) ---
    print("\n--- Training Linear Probes (with Held-Out Test Set) ---")
    
    # We split baseline embeddings, knockout embeddings, and the target mass synchronously
    X_train_base, X_test_base, X_train_knock, X_test_knock, mass_train, mass_test = train_test_split(
        baseline_embeddings, knockout_embeddings, true_inv_mass, test_size=0.2, random_state=42
    )

    # 7. --- The Probes ---
    
    # Baseline Probe
    probe_baseline = LinearRegression().fit(X_train_base, mass_train)
    preds_baseline = probe_baseline.predict(X_test_base)
    r2_baseline = r2_score(mass_test, preds_baseline)

    # Knockout Probe
    probe_knockout = LinearRegression().fit(X_train_knock, mass_train)
    preds_knockout = probe_knockout.predict(X_test_knock)
    r2_knockout = r2_score(mass_test, preds_knockout)

    # 8. --- Results & Evaluation ---
    print("\n--- Causal Knockout Results ---")
    print(f"Held-out Probe R^2 (Normal Attention):  {r2_baseline:.4f}")
    print(f"Held-out Probe R^2 (Severed Attention): {r2_knockout:.4f}")
    
    drop = r2_baseline - r2_knockout
    print(f"\nAbsolute Information Loss: {drop:.4f} R^2 points")
    
    if r2_baseline < 0.8:
         print("Warning: Baseline encoding is too weak to run a valid causal knockout test.")
    elif drop > 0.5:
        print("Conclusion: Causal link established. The Self-Attention mechanism is strictly responsible for computing the physical invariants.")
    elif drop > 0.1:
        print("Conclusion: Partial causal link. Attention contributes to the calculation, but the model has backup routing pathways.")
    else:
        print("Conclusion: No causal link. Severing attention did not degrade the representation of invariant mass.")

if __name__ == "__main__":
    main()