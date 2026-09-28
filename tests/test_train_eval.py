"""U10/U11 tests — overfit sanity and symmetry-violation metric."""

from __future__ import annotations

import torch

from wysgnn.data.datamodule import PiezoDataModule
from wysgnn.eval import evaluate
from wysgnn.nn.baseline import InvariantBaseline, InvariantBaselineConfig
from wysgnn.nn.model import WySGNN, WySGNNConfig
from wysgnn.train import overfit_subset

torch.set_default_dtype(torch.float64)


def _dm():
    dm = PiezoDataModule(n_structures=20, cutoff=4.0, lmax=2, mul=4, seed=2)
    dm.setup()
    return dm


def test_overfit_16_subset():
    # U10 sanity: the model can drive train loss near zero on a small subset.
    dm = _dm()
    torch.manual_seed(0)
    model = WySGNN(WySGNNConfig(lmax=2, mul=4, num_layers=2, cutoff=4.0)).double()
    final = overfit_subset(model, dm, k=10, steps=300, lr=5e-3)
    assert final < 1e-2, f"overfit train loss too high: {final}"


def test_symmetry_violation_contrast():
    # U11 headline: an invariant baseline violates Neumann; WySGNN does not.
    dm = _dm()
    torch.manual_seed(0)
    wys = WySGNN(WySGNNConfig(lmax=2, mul=4, num_layers=2, cutoff=4.0)).double()
    m_wys = evaluate(wys, dm, "test")
    assert m_wys["symmetry_violation_rate"] < 1e-9

    base = InvariantBaseline(InvariantBaselineConfig(cutoff=4.0)).double()
    # make the head produce clearly non-zero output
    for p in base.head.parameters():
        torch.nn.init.normal_(p, std=1.0)
    m_base = evaluate(base, dm, "test")
    assert m_base["n_forced_zero_components"] > 0
    assert m_base["symmetry_violation_rate"] > 0.0
