"""U9 — Piezoelectric tensor data module.

Provides crystal structures with rank-3 piezoelectric target tensors, an
80/10/10 split *by structure* (persisted), and an equivariance-preserving
standardizer.

Two data sources are supported:

* **JARVIS** (``source="jarvis"``): the JARVIS-DFT ``dft_3d`` ``piezoelectric_tensor``
  field, loaded via ``jarvis-tools`` if installed and the dataset is cached.
* **synthetic teacher** (``source="synthetic"``, default): a frozen randomly
  initialised :class:`WySGNN` (equivariant + Neumann) labels a symmetry-preserving
  structure pool.  Targets are exactly Neumann-compliant and a smooth learnable
  function of geometry — a clean pilot when JARVIS is unavailable.

Normalisation uses a single global scale with zero mean, because per-component
standardisation in the Cartesian/irrep basis would break the equivariance of the
targets (the mean of an equivariant rank-3 tensor over orientations is zero).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
from e3nn import o3
from torch_geometric.data import Batch

from ..nn.irreps import node_irreps
from ..nn.model import WySGNN, WySGNNConfig
from ..symmetry.wyckoff import crystal_point_group_ops_cart
from .graph import structure_to_graph
from .synthetic import build_structure_pool

_DTYPE = torch.float64


@dataclass
class Standardizer:
    scale: float

    def transform(self, t: torch.Tensor) -> torch.Tensor:
        return t / self.scale

    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.scale


@dataclass
class PiezoSample:
    sid: str
    space_group: int
    centrosymmetric: bool
    data: object  # torch_geometric Data with y (normalized) and y_raw
    ops: torch.Tensor


@dataclass
class PiezoDataModule:
    n_structures: int = 240
    cutoff: float = 5.0
    lmax: int = 2
    mul: int = 8
    symprec: float = 1e-3
    seed: int = 0
    source: str = "synthetic"
    splits_path: str | None = None

    samples: list[PiezoSample] = field(default_factory=list)
    split_ids: dict[str, list[str]] = field(default_factory=dict)
    standardizer: Standardizer | None = None
    irreps_node: o3.Irreps = field(init=False)

    def __post_init__(self):
        self.irreps_node = node_irreps(self.lmax, self.mul)

    # ------------------------------------------------------------------ build
    def setup(self):
        torch.manual_seed(self.seed)
        if self.source == "jarvis":
            structures, targets, sgs, centro = self._load_jarvis()
        else:
            structures, targets, sgs, centro, sids = self._build_synthetic()

        if self.source == "jarvis":
            sids = [f"jvasp-{i}" for i in range(len(structures))]

        # Build graphs + ops + attach targets.
        samples = []
        for sid, s, t, sg, c in zip(sids, structures, targets, sgs, centro):
            data = structure_to_graph(s, self.cutoff, self.irreps_node, self.symprec)
            data.y_raw = t.to(_DTYPE).unsqueeze(0)  # (1,3,3,3)
            ops = crystal_point_group_ops_cart(s, self.symprec)
            samples.append(PiezoSample(sid, int(sg), bool(c), data, ops))
        self.samples = samples

        self._make_splits()
        self._fit_standardizer()
        for smp in self.samples:
            smp.data.y = self.standardizer.transform(smp.data.y_raw)

    def _build_synthetic(self):
        pool = build_structure_pool(self.n_structures, seed=self.seed, symprec=self.symprec)
        sids = [p[0] for p in pool]
        structures = [p[1] for p in pool]
        sgs = [p[2] for p in pool]
        centro = [p[3] for p in pool]
        # Frozen teacher generates Neumann-compliant targets.  A shallow (1-layer)
        # teacher keeps the target function smooth/low-complexity so it is
        # learnable from a modest pilot dataset.
        teacher = WySGNN(
            WySGNNConfig(
                lmax=self.lmax, mul=self.mul, num_layers=1, cutoff=self.cutoff,
                wyckoff_enabled=True, neumann_enabled=True,
            )
        ).to(_DTYPE)
        teacher.eval()
        targets = []
        with torch.no_grad():
            for s in structures:
                batch, ops = _one(s, self.cutoff, self.irreps_node, self.symprec)
                out = teacher(batch, ops)[0]
                targets.append(out * 50.0)  # scale into a pC/N-like magnitude
        return structures, targets, sgs, centro, sids

    def _load_jarvis(self):
        try:
            from jarvis.db.figshare import data as jdata
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "source='jarvis' requires jarvis-tools and a cached dft_3d dataset"
            ) from e
        from pymatgen.core import Structure

        records = jdata("dft_3d")
        structures, targets, sgs, centro = [], [], [], []
        for r in records:
            pt = r.get("piezoelectric_tensor")
            if pt in (None, "na", "") or not isinstance(pt, (list, tuple)):
                continue
            voigt = torch.tensor(pt, dtype=_DTYPE)  # (3,6) expected
            if voigt.shape != (3, 6):
                continue
            try:
                s = Structure.from_dict(r["atoms"]) if isinstance(r["atoms"], dict) else None
            except Exception:
                continue
            if s is None:
                continue
            from ..nn.irreps import voigt_to_full
            structures.append(s)
            targets.append(voigt_to_full(voigt.unsqueeze(0))[0])
            sgs.append(int(r.get("spg_number", 1)))
            centro.append(False)
        if not structures:
            raise RuntimeError("no valid JARVIS piezoelectric records found")
        return structures, targets, sgs, centro

    # ------------------------------------------------------------- splits/norm
    def _make_splits(self):
        rng = torch.Generator().manual_seed(self.seed)
        ids = [s.sid for s in self.samples]
        perm = torch.randperm(len(ids), generator=rng).tolist()
        n = len(ids)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        train = [ids[i] for i in perm[:n_train]]
        val = [ids[i] for i in perm[n_train:n_train + n_val]]
        test = [ids[i] for i in perm[n_train + n_val:]]
        self.split_ids = {"train": train, "val": val, "test": test}
        if self.splits_path:
            os.makedirs(os.path.dirname(self.splits_path), exist_ok=True)
            with open(self.splits_path, "w") as f:
                json.dump(self.split_ids, f, indent=2)

    def _fit_standardizer(self):
        train_set = set(self.split_ids["train"])
        ts = torch.stack([s.data.y_raw[0] for s in self.samples if s.sid in train_set])
        scale = ts.pow(2).mean().sqrt().item()
        scale = scale if scale > 1e-12 else 1.0
        self.standardizer = Standardizer(scale=scale)

    # ------------------------------------------------------------- iteration
    def _split_samples(self, split: str) -> list[PiezoSample]:
        wanted = set(self.split_ids[split])
        return [s for s in self.samples if s.sid in wanted]

    def iter_batches(self, split: str, batch_size: int, shuffle: bool = False):
        samples = self._split_samples(split)
        order = list(range(len(samples)))
        if shuffle:
            g = torch.Generator().manual_seed(self.seed + len(samples))
            order = torch.randperm(len(samples), generator=g).tolist()
        for i in range(0, len(samples), batch_size):
            chunk = [samples[j] for j in order[i:i + batch_size]]
            batch = Batch.from_data_list([c.data for c in chunk])
            ops = [c.ops for c in chunk]
            y = torch.cat([c.data.y for c in chunk], dim=0)  # (B,3,3,3) normalized
            y_raw = torch.cat([c.data.y_raw for c in chunk], dim=0)
            meta = [(c.sid, c.space_group, c.centrosymmetric) for c in chunk]
            yield batch, ops, y, y_raw, meta


def _one(structure, cutoff, irreps, symprec):
    data = structure_to_graph(structure, cutoff, irreps, symprec)
    ops = [crystal_point_group_ops_cart(structure, symprec)]
    return Batch.from_data_list([data]), ops
