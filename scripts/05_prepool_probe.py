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
    
    events = np.load('data/raw/events.npy')
    model = TinyTransformer().to(device)
    model.load_state_dict(torch.load('data/models/tiny_transformer.pth', map_location=device, weights_only=True))
    model.eval()

    print("Calculating Ground Truth Physics...")
    E_tot = events[:, 0, 0] + events[:, 1, 0]
    px_tot = events[:, 0, 1] + events[:, 1, 1]
    py_tot = events[:, 0, 2] + events[:, 1, 2]
    pz_tot = events[:, 0, 3] + events[:, 1, 3]
    true_inv_mass = np.sqrt(np.maximum(0, E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)))

    hidden_states = {}
    def get_activation(name):
        def hook(model, input, output):
            # Capture the FULL sequence: [Batch, 2, d_model]
            hidden_states[name] = output.detach().numpy()
        return hook
    model.transformer.register_forward_hook(get_activation('encoder_out'))

    tensor_events = torch.tensor(events, dtype=torch.float32)

    print("Running forward pass...")
    with torch.no_grad():
        _ = model(tensor_events)
        
    full_embeddings = hidden_states['encoder_out']
    
    # Isolate the pre-pooling representations
    token_1_embeddings = full_embeddings[:, 0, :]
    token_2_embeddings = full_embeddings[:, 1, :]
    pooled_embeddings = full_embeddings.mean(axis=1)

    print("\n--- Training Linear Probes (Held-Out Test Set) ---")
    
    # Split all sets synchronously
    X1_train, X1_test, X2_train, X2_test, Xp_train, Xp_test, mass_train, mass_test = train_test_split(
        token_1_embeddings, token_2_embeddings, pooled_embeddings, true_inv_mass, test_size=0.2, random_state=42
    )

    # Probes
    probe_t1 = LinearRegression().fit(X1_train, mass_train)
    r2_t1 = r2_score(mass_test, probe_t1.predict(X1_test))

    probe_t2 = LinearRegression().fit(X2_train, mass_train)
    r2_t2 = r2_score(mass_test, probe_t2.predict(X2_test))
    
    probe_pooled = LinearRegression().fit(Xp_train, mass_train)
    r2_pooled = r2_score(mass_test, probe_pooled.predict(Xp_test))

    print(f"Held-out R^2 (Token 1 only): {r2_t1:.4f}")
    print(f"Held-out R^2 (Token 2 only): {r2_t2:.4f}")
    print(f"Held-out R^2 (Mean Pooled):  {r2_pooled:.4f}")
    
    print("\n--- Conclusion ---")
    if r2_pooled > 0.9 and r2_t1 < 0.5 and r2_t2 < 0.5:
        print("Hypothesis Confirmed: Individual tokens do not encode global invariant mass. The representation is fundamentally constructed during the pooling step.")
    elif r2_t1 > 0.8 or r2_t2 > 0.8:
        print("Anomaly: A single isolated token encodes the global invariant mass. This indicates unexpected local aggregation prior to pooling.")

if __name__ == "__main__":
    main()