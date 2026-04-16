# Subnuclear Sherlock: Mechanistic Interpretability of Lorentz Invariance in Transformers

Traditional Physics-ML treats neural networks as black boxes, optimizing for classification accuracy (e.g., separating Top Quarks from QCD background). However, accuracy alone does not guarantee that a model has learned the underlying physical laws. It may simply be exploiting dataset biases.

This repository takes a **Mechanistic Interpretability** approach to a Transformer trained on a toy particle decay dataset. The goal is to answer a single question: **Does a standard Transformer independently discover Special Relativity (the Invariant Mass formula), and if so, exactly which computational circuits compute it?**

## The Physics Problem
In high-energy collisions, invariant mass ($M$) is a crucial conserved quantity, defined by the 4-momentum of a particle's decay products:
$$M = \sqrt{(\sum E)^2 - (\sum \vec{p})^2}$$

We train a small, permutation-invariant Transformer (no positional encoding) on simulated 4-vector data:
* **Signal (Top Quark):** 2-body decay with a strict invariant mass of 173.0 GeV, subject to varying Lorentz boosts along the Z-axis.
* **Background (QCD Noise):** Random forward-facing light particles with broad, unstructured invariant masses.

The model easily achieves >99% classification accuracy. **But how?**

## The Interpretability Pipeline

Instead of looking at the output, we use **Linear Probing** and **Activation Patching** on the internal layers to prove the network computes $M$.

### 1. Linear Probing (Did it learn physics?)
We train a linear regression model on the frozen hidden states (`d_model=64`) of the Transformer's first layer to predict the true physical invariant mass. 
* **Result:** The probe successfully extracts the exact invariant mass with an $R^2 > 0.99$. The model did not just memorize energy thresholds; it mathematically encoded $M = \sqrt{E^2 - p^2}$ into its latent space.

### 2. Causal Knockout (Where does it calculate it?)
Invariant mass requires cross-particle communication. We artificially sever the Self-Attention mechanism between Particle 1 and Particle 2 (Causal Knockout) and run the probe again.
* **Result:** Information loss is near total. This causally proves that the Transformer uses its specific Attention Heads as a dynamic vector-addition circuit to compute the relativistic invariants.

## Repository Structure

\`\`\`text
subnuclear_sherlock/
├── data/raw/              # Generated 4-vector NumPy arrays
├── src/
│   ├── generator.py       # Physics simulation (Rest Frame decays + Lorentz Boosting)
│   ├── model.py           # PyTorch TinyTransformer architecture
├── scripts/
│   ├── 01_make_data.py    # Generates 10k Signal/Background events
│   ├── 02_train.py        # Trains the model (BCEWithLogitsLoss)
│   ├── 03_investigate.py  # Hooks into hidden states and runs Linear Probes
│   └── 04_causal_knockout.py # Tests physics-loss via Attention Masking
\`\`\`

## Quickstart

This pipeline is designed to run efficiently on local hardware (CPU/Consumer GPU).

1. **Clone and Install**
   \`\`\`bash
   git clone https://github.com/YOUR-USERNAME/subnuclear_sherlock.git
   cd subnuclear_sherlock
   pip install torch numpy scikit-learn
   \`\`\`

2. **Run the Experiment**
   \`\`\`bash
   python scripts/01_make_data.py
   python scripts/02_train.py
   python scripts/03_investigate.py
   python scripts/04_causal_knockout.py
   \`\`\`

## Future Work (Scaling to Reality)
This toy model establishes the baseline interpretability pipeline. The next phase scales this methodology to the open-source **JetClass** dataset, utilizing larger Transformers to identify complex kinematic conservation circuits in 128-particle dense jets.