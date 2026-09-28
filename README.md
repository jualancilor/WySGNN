# WySGNN — Wyckoff-Position-Aware Space Group Neural Network

An E(3)-equivariant message-passing network for crystal **tensor** property
prediction (piezoelectric, rank-3) that imposes crystallographic **site/space
group** symmetry as an architectural constraint:

1. **Wyckoff site projection** — after every message layer, each atom's features
   are projected onto its site-symmetry-group (`G_i`) invariant subspace,
   `x_i ← P_i x_i` with `P_i = (1/|G_i|) Σ_{g∈G_i} D(g)`.
2. **Neumann-exact readout** — the predicted tensor is projected onto the
   subspace invariant under the crystal point group, so it satisfies Neumann's
   principle exactly (e.g. exactly zero for centrosymmetric crystals).

## A note on what the symmetry projections actually buy you

A key, test-verified finding from this build: for an **exactly E(3)-equivariant**
network on a perfectly symmetric crystal, the Wyckoff and Neumann projections are
**no-ops up to numerical error** — equivariance already forces node features to
be site-symmetric and pooled tensors to be Neumann-compliant (because every
`g ∈ G_i` is a global symmetry of the crystal that fixes site `i`). See
[`tests/test_model_smoke.py`](tests/test_model_smoke.py)
(`test_projections_are_noop_on_equivariant_network`).

The projections therefore matter as (a) an **exactness/robustness guarantee**
under numerical equivariance error, and (b) **missing symmetry added to
non-equivariant models** — the realistic invariant-feature baselines
(CGCNN/MEGNet-style) that dominate materials ML and that genuinely violate
Neumann's principle. The experiments contrast against those baselines.

## Layout

```
wysgnn/symmetry/wyckoff.py     U1  site symmetry extraction (spglib)
wysgnn/symmetry/projectors.py  U2  G-invariant projectors (real Wigner-D)
wysgnn/data/graph.py           U3  periodic graph + per-site projectors
wysgnn/nn/irreps.py            U4  rank-3 tensor irrep decomposition (2x1o+2o+3o)
wysgnn/nn/message.py           U5  equivariant message passing layer
wysgnn/nn/wyckoff_layer.py     U6  site-symmetry projection layer (contribution)
wysgnn/nn/readout.py           U7  Neumann-exact tensor readout
wysgnn/nn/model.py             U8  WySGNN model
wysgnn/nn/baseline.py              invariant-feature baseline (CGCNN/MEGNet-style)
wysgnn/data/datamodule.py      U9  data module (JARVIS or synthetic teacher)
wysgnn/train.py                U10 training (Frobenius-MSE, AdamW, EMA, early stop)
wysgnn/eval.py                 U11 metrics incl. symmetry-violation rate
```

## Install

```bash
pip install torch torch_geometric e3nn pymatgen spglib ase numpy scipy pyyaml pytest
# optional, for the real dataset:
pip install jarvis-tools
```

## Reproduce

```bash
bash scripts/run_train.sh          # tests + full ablation + sample efficiency
# or, quick pilot:
python scripts/run_ablation.py --quick
```

Outputs land in `results/`: `ablation.json`, `sample_efficiency.json`,
`metrics.csv`, `run_meta.json` (git SHA, config hash, env freeze), `splits.json`.

## Data

By default the pipeline uses a **synthetic teacher/student pilot**: a frozen,
randomly-initialised equivariant WySGNN labels a symmetry-preserving structure
pool, giving exactly Neumann-compliant, learnable targets without any download.
Set `data.source: jarvis` in `config.yaml` (and run `scripts/fetch_data.py`) to
use the real JARVIS-DFT piezoelectric tensors.

## Tests / execution gates

```bash
python -m pytest tests/ -q
```

All units ship with assertion tests; `U1–U2` (symmetry), message + end-to-end
equivariance, and the overfit-16 sanity are the gates guarding training.
