# Subnuclear Sherlock: Mechanistic Interpretability of Lorentz Invariance in Transformers

Traditional Physics-ML treats neural networks as black boxes, optimising for classification accuracy  - for example, separating Top Quarks from QCD background. High accuracy, however, does not guarantee that a model has learned the underlying physical laws. It may simply be exploiting dataset biases, such as energy thresholds or angular correlations that are discriminative without being physically fundamental.

This repository takes a **Mechanistic Interpretability** approach to a Transformer trained on a toy particle decay dataset. The central question is: **does a standard Transformer independently discover Special Relativity  -specifically the Lorentz-invariant mass formula  -and if so, through what computational pathway?**

The experiments produce a complete mechanistic story, including one result that contradicts the naive expectation and reveals something physically meaningful about the dataset structure.

---

## The Physics Problem

In high-energy collisions, invariant mass ($M$) is a Lorentz-invariant conserved quantity defined by the 4-momentum of a particle's decay products:

$$M = \sqrt{\left(\sum E\right)^2 - \left(\sum \vec{p}\right)^2}$$

We train a small, permutation-invariant Transformer (no positional encoding) on simulated 4-vector data:

- **Signal (Top Quark):** 2-body decay products with a strict invariant mass of 173.0 GeV, subject to varying Lorentz boosts along the Z-axis. The boost randomises individual particle energies and momenta, but leaves $M$ invariant by construction.
- **Background (QCD Noise):** Two independent forward-facing particles with broad, unstructured invariant masses.

The model achieves >99% classification accuracy by epoch 2. **But how?**

---

## The Interpretability Pipeline

Four experiments progressively narrow down the computational mechanism.

### Experiment 1  -Linear Probing: Did it learn physics?

We train a linear regression probe on the frozen, mean-pooled encoder hidden states (`d_model=64`) to predict two candidate physical quantities from the internal representation: raw total energy $E$ and invariant mass $M$.

**Results (held-out test set, 20% split):**

| Target Quantity | Held-out R² |
|---|---|
| Total Energy $E$ | 0.6837 |
| Invariant Mass $M$ | **0.9925** |

The strong preference for $M$ over $E$ is significant. Because signal events are generated under random Lorentz boosts, raw energy is highly variable  -a model exploiting energy thresholds would generalise poorly across boosts. The probe result shows the model has instead encoded the Lorentz-invariant quantity that remains constant: $M = \sqrt{E^2 - p^2}$. The model learned relativistic kinematics, not a shortcut.

---

### Experiment 2  -Causal Knockout: Is cross-particle attention responsible?

To identify the computational mechanism, we severed all cross-particle attention connections using PyTorch's `src_mask`, forcing each particle token to process only its own 4-vector through the Transformer layers. We then re-ran the linear probe on the resulting representations.

**Results:**

| Condition | Held-out R² | Drop |
|---|---|---|
| Normal attention | 0.9925 |  -|
| Severed attention | 0.9871 | **0.0054** |

The negligible degradation establishes a clear negative result: **the model does not use cross-particle attention to compute invariant mass.** Severing the attention mechanism entirely causes no meaningful loss of physical information. The hypothesis that attention heads implement a vector-addition circuit for relativistic invariants is falsified.

---

### Experiment 3  -Pre-Pooling Token Probing: Where is the information?

Since attention is not responsible, we probe the computational pathway more precisely by hooking into the encoder output *before* mean pooling, examining individual particle token representations separately.

**Results:**

| Representation | Held-out R² |
|---|---|
| Token 1 only (Particle 1's representation) | **0.9860** |
| Token 2 only (Particle 2's representation) | **0.9861** |
| Mean pooled (both tokens combined) | 0.9925 |

Each individual particle token, having only ever seen its own 4-vector as input, encodes the invariant mass of the full two-particle system with R² > 0.98.

---

### Experiment 4  -Physical Interpretation: Why can a single token encode a global quantity?

This result is not anomalous  -it is physically meaningful. The explanation lies in the kinematics of 2-body decay.

In the signal class, both particles are decay products of the same parent at rest:

$$E_{1,\text{rest}} = E_{2,\text{rest}} = \frac{M}{2}$$

After a Lorentz boost $\beta$ along the Z-axis:

$$E_1 = \gamma\left(\frac{M}{2} + \beta p_{z,\text{rest}}\right)$$

The parent mass $M$ is therefore recoverable from particle 1's lab-frame 4-vector alone, up to the shared boost parameters $(\beta, \gamma)$  -which are also encoded in particle 1's momentum components. The information is locally available in each token because the 2-body decay kinematics make each particle's 4-vector individually sufficient to reconstruct the parent mass.

This is why cross-particle attention is unnecessary: the model learned to extract $M$ from each particle independently, because the physics of this specific dataset permits it. The marginal improvement from mean pooling (0.9925 vs 0.9860) reflects the slight additional constraint from combining both tokens.

---

## Summary of Findings

| Claim | Status | Evidence |
|---|---|---|
| Model encodes invariant mass, not raw energy | ✅ Confirmed | R² = 0.99 vs 0.68 on held-out set |
| Encoding is Lorentz-invariant | ✅ Confirmed | Boost randomisation rules out energy shortcuts |
| Cross-particle attention computes $M$ | ❌ Falsified | ΔR² = 0.005 after full attention knockout |
| Individual tokens encode $M$ strongly | ✅ Confirmed | R² ≈ 0.986 per token on held-out set |
| Physical explanation: 2-body kinematics permit local recovery of $M$ | ✅ Confirmed | Analytic derivation from decay + boost equations |

---

## Repository Structure

```
subnuclear_sherlock/
├── data/raw/              # Generated 4-vector NumPy arrays
├── src/
│   ├── generator.py       # Physics simulation (rest frame decays + Lorentz boosting)
│   ├── model.py           # PyTorch TinyTransformer (mask-aware forward pass)
├── scripts/
│   ├── 01_make_data.py    # Generates 20k Signal/Background events
│   ├── 02_train.py        # Trains the model (BCEWithLogitsLoss)
│   ├── 03_investigate.py  # Linear probes on pooled encoder states (held-out eval)
│   ├── 04_causal_knockout.py  # Attention masking + comparative probing
│   └── 05_prepool_probe.py    # Per-token probing before mean pooling
```

---

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
python scripts/05_prepool_probe.py
```

---

## Limitations and Future Work

**The key limitation this analysis reveals** is that the toy dataset is *too clean* to test cross-particle attention as an aggregation mechanism. In 2-body decay with a fixed parent mass, the physics guarantees that each particle individually carries sufficient information  -attention is never forced to do cross-particle work.

**The motivated next step** is scaling to the open-source **JetClass** dataset, where the model processes 128-particle dense jets. In that setting, no single particle's 4-vector is sufficient to reconstruct jet-level quantities like jet mass, $p_T$, or substructure variables. Cross-particle communication through attention will be *necessary* in a way it is not here. This is the experiment that will determine whether attention heads actually form the cross-particle aggregation circuits originally hypothesised  -and whether mechanistic interpretability tools transfer meaningfully to particle physics settings.

The pipeline established here  -linear probing, causal knockout, pre-pooling token analysis  -provides the full methodological scaffold for that investigation.