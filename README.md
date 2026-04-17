# Subnuclear Sherlock: Mechanistic Interpretability of Lorentz Invariance in Transformers

Traditional Physics-ML treats neural networks as black boxes, optimising for classification accuracy — for example, separating Top Quarks from QCD background. High accuracy, however, does not guarantee that a model has learned the underlying physical laws. It may simply be exploiting dataset biases, such as energy thresholds or angular correlations that happen to be discriminative without being physically fundamental.

This repository takes a **Mechanistic Interpretability** approach to a Transformer trained on a toy particle decay dataset. The central question is: **does a standard Transformer independently discover Special Relativity — specifically the Lorentz-invariant mass formula — and if so, through what computational pathway?**

## The Physics Problem

In high-energy collisions, invariant mass ($M$) is a Lorentz-invariant conserved quantity defined by the 4-momentum of a particle's decay products:

$$M = \sqrt{\left(\sum E\right)^2 - \left(\sum \vec{p}\right)^2}$$

We train a small, permutation-invariant Transformer (no positional encoding) on simulated 4-vector data:

- **Signal (Top Quark):** 2-body decay products with a strict invariant mass of 173.0 GeV, subject to varying Lorentz boosts along the Z-axis. The boost randomises individual particle energies and momenta, but leaves $M$ constant.
- **Background (QCD Noise):** Two independent forward-facing particles with broad, unstructured invariant masses.

The model achieves >99% classification accuracy by epoch 2. **But how?**

## The Interpretability Pipeline

Rather than inspecting outputs, we use **Linear Probing** and **Causal Knockout** on internal representations to characterise what the model has learned and where.

### 1. Linear Probing — Did it learn physics?

We train a linear regression probe on the frozen, mean-pooled hidden states (`d_model=64`) of the Transformer encoder to predict two candidate physical quantities: raw total energy $E$ and invariant mass $M$.

**Results (held-out test set, 20% split):**

| Target Quantity | Held-out R² |
|---|---|
| Total Energy $E$ | 0.6837 |
| Invariant Mass $M$ | **0.9925** |

The strong preference for $M$ over $E$ is significant. Because the signal is generated with random Lorentz boosts, raw energy is highly variable across events — a model exploiting energy thresholds would generalise poorly. The probe result shows the model has instead encoded the Lorentz-invariant quantity that remains constant across all boosts: $M = \sqrt{E^2 - p^2}$.

### 2. Causal Knockout — Where does it aggregate cross-particle information?

To identify the computational mechanism behind this encoding, we ran a causal knockout experiment: we severed all cross-particle attention connections using PyTorch's `src_mask`, forcing each particle token to process only its own 4-vector through the Transformer layers, and re-ran the linear probe.

**Results:**

| Condition | Held-out R² | Drop |
|---|---|---|
| Normal attention | 0.9925 | — |
| Severed attention | 0.9871 | 0.0054 |

The negligible degradation (ΔR² = 0.005) is the key finding: **the model does not use cross-particle attention to compute invariant mass.** The architecture's mean pooling layer — which aggregates across particle tokens *after* the Transformer — is sufficient to combine both 4-vectors. The model learned to encode each particle's momentum information in its individual token representation, and delegates cross-particle aggregation to pooling rather than attention heads.

This is a meaningful mechanistic result. The attention mechanism is not doing what one might naively expect; the architecture's inductive bias toward pooling-based aggregation shapes which pathway the model learns to use.

### 3. Open Question — Pre-pooling token probing

The natural next experiment is to probe *individual particle tokens* (before mean pooling) rather than the pooled representation. If individual tokens encode $M$, this would require single-particle local computation of a global quantity — a surprising and theoretically interesting result. More likely, neither token alone encodes $M$ strongly, but their combination under pooling does. This experiment is in progress.

## Repository Structure

```
subnuclear_sherlock/
├── data/raw/              # Generated 4-vector NumPy arrays
├── src/
│   ├── generator.py       # Physics simulation (rest frame decays + Lorentz boosting)
│   ├── model.py           # PyTorch TinyTransformer (mask-aware)
├── scripts/
│   ├── 01_make_data.py    # Generates 20k Signal/Background events
│   ├── 02_train.py        # Trains the model (BCEWithLogitsLoss)
│   ├── 03_investigate.py  # Linear probes on encoder hidden states (held-out eval)
│   └── 04_causal_knockout.py  # Attention masking + comparative probing
```

## Quickstart

Designed to run on local hardware (CPU or consumer GPU).

```bash
git clone https://github.com/pri-Mohanty/subnuclear-sherlock.git
cd subnuclear-sherlock
pip install torch numpy scikit-learn

python scripts/01_make_data.py
python scripts/02_train.py
python scripts/03_investigate.py
python scripts/04_causal_knockout.py
```

## Summary of Findings

| Claim | Status | Evidence |
|---|---|---|
| Model encodes invariant mass, not raw energy | ✅ Confirmed | R² = 0.99 vs 0.68 on held-out set |
| Encoding is Lorentz-invariant | ✅ Confirmed | Boost randomisation rules out energy shortcuts |
| Cross-particle attention computes $M$ | ❌ Falsified | ΔR² = 0.005 after full attention knockout |
| Mean pooling is the aggregation mechanism | 🔍 Hypothesis | Supported by knockout null result |
| Pre-pooling token probing | 🔍 In progress | Next experiment |

## Future Work

This pipeline establishes the baseline interpretability methodology. The immediate next step is pre-pooling token-level probing to confirm the pooling hypothesis. The longer-term goal is to scale this methodology to the open-source **JetClass** dataset, applying it to larger Transformers processing 128-particle dense jets — where the question of which circuits compute conserved quantities becomes substantially harder and more physically interesting.