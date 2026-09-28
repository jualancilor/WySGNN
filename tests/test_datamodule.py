"""U9 tests — data module (splits, normalization, Neumann sanity)."""

from __future__ import annotations

import torch

from wysgnn.data.datamodule import PiezoDataModule
from wysgnn.nn.readout import neumann_project

torch.set_default_dtype(torch.float64)


def _dm(tmp_path):
    dm = PiezoDataModule(
        n_structures=24, cutoff=4.0, lmax=2, mul=4, seed=1,
        splits_path=str(tmp_path / "splits.json"),
    )
    dm.setup()
    return dm


def test_no_structure_leakage(tmp_path):
    dm = _dm(tmp_path)
    tr = set(dm.split_ids["train"])
    va = set(dm.split_ids["val"])
    te = set(dm.split_ids["test"])
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    assert len(tr) + len(va) + len(te) == len(dm.samples)


def test_standardizer_roundtrip(tmp_path):
    dm = _dm(tmp_path)
    x = torch.randn(5, 3, 3, 3, dtype=torch.float64)
    rec = dm.standardizer.inverse(dm.standardizer.transform(x))
    assert torch.linalg.norm(rec - x) < 1e-12


def test_targets_satisfy_neumann(tmp_path):
    dm = _dm(tmp_path)
    violators = 0
    for s in dm.samples:
        T = s.data.y_raw[0]
        proj = neumann_project(T, s.ops)
        if torch.linalg.norm(T - proj) > 1e-6:
            violators += 1
        # centrosymmetric crystals must have (near) zero piezoelectric tensor
        if s.centrosymmetric:
            assert torch.linalg.norm(T) < 1e-6
    assert violators == 0


def test_batches_iterate(tmp_path):
    dm = _dm(tmp_path)
    seen = 0
    for batch, ops, y, y_raw, meta in dm.iter_batches("train", batch_size=8):
        assert y.shape[1:] == (3, 3, 3)
        assert len(ops) == y.shape[0]
        assert len(meta) == y.shape[0]
        seen += y.shape[0]
    assert seen == len(dm.split_ids["train"])
