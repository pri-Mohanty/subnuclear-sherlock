import numpy as np
import os
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.generator import JetSimulator

def main():
    print("Initializing Jet Simulator...")
    sim = JetSimulator(num_events=10000)
    
    print("Generating Signal (Top Quarks)...")
    sig_events, sig_labels = sim.generate_signal()
    
    print("Generating Background (QCD Noise)...")
    bkg_events, bkg_labels = sim.generate_background()
    
    # Combine and shuffle
    all_events = np.vstack((sig_events, bkg_events))
    all_labels = np.concatenate((sig_labels, bkg_labels))
    
    # Shuffle the dataset
    indices = np.random.permutation(len(all_labels))
    all_events = all_events[indices]
    all_labels = all_labels[indices]
    
    # Save to disk
    os.makedirs('data/raw', exist_ok=True)
    np.save('data/raw/events.npy', all_events)
    np.save('data/raw/labels.npy', all_labels)
    
    print(f"Dataset generated and saved! Total events: {len(all_labels)}")

if __name__ == "__main__":
    main()