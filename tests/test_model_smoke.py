"""U8 tests — full WySGNN model smoke tests."""

from __future__ import annotations

import torch

from tests._structures import batio3_tetragonal, nacl, quartz
from wysgnn.data.graph import structures_to_batch
from wysgnn.nn.irreps import node_irreps
from wysgnn.nn.model import WySGNN, WySGNNConfig

torch.set_default_dtype(torch.float64)

CFG = WySGNNConfig(lmax=2, mul=4, num_layers=2, cutoff=4.0)


def _batch():
    irreps = node_irreps(CFG.lmax, CFG.mul)
    return structures_to_batch(
        [batio3_tetragonal(), quartz(), nacl()], CFG.cutoff, irreps
    )


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    model = WySGNN(CFG).double()
    batch, ops = _batch()
    out = model(batch, ops)
    assert out.shape == (3, 3, 3, 3)
    assert torch.isfinite(out).all()


def test_backward_finite_grads():
    torch.manual_seed(0)
    model = WySGNN(CFG).double()
    batch, ops = _batch()
    out = model(batch, ops)
    out.pow(2).sum().backward()
    n = 0
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "missing grad"
            assert torch.isfinite(p.grad).all(), "non-finite grad"
            n += 1
    assert n > 0


def test_centrosymmetric_prediction_is_zero():
    # With Neumann on, NaCl (centrosymmetric) must predict exactly zero.
    torch.manual_seed(0)
    model = WySGNN(CFG).double()
    batch, ops = _batch()
    out = model(batch, ops)
    assert out[2].norm() < 1e-10  # NaCl is index 2


def test_baseline_reduction_runs():
    # wyckoff + neumann OFF => plain equivariant baseline; runs without ops.
    torch.manual_seed(0)
    cfg = WySGNNConfig(
        lmax=2, mul=4, num_layers=2, cutoff=4.0,
        wyckoff_enabled=False, neumann_enabled=False,
    )
    model = WySGNN(cfg).double()
    batch, _ops = _batch()
    out = model(batch)  # no point group ops needed
    assert out.shape == (3, 3, 3, 3)
    assert torch.isfinite(out).all()


def test_projections_are_noop_on_equivariant_network():
    # Key theoretical property (verified): for an exactly-equivariant network on
    # a symmetric crystal, the Wyckoff and Neumann projections are no-ops --
    # node features are already site-symmetric and pooled outputs already
    # Neumann-compliant.  So toggling the projections leaves outputs unchanged
    # to machine precision.  (Their value is robustness + enforcing symmetry on
    # NON-equivariant baselines; see InvariantBaseline.)
    batch, ops = _batch()

    torch.manual_seed(0)
    full = WySGNN(WySGNNConfig(lmax=2, mul=4, num_layers=2, cutoff=4.0)).double()
    out_full = full(batch, ops)

    torch.manual_seed(0)
    base = WySGNN(
        WySGNNConfig(
            lmax=2, mul=4, num_layers=2, cutoff=4.0,
            wyckoff_enabled=False, neumann_enabled=False,
        )
    ).double()
    out_base = base(batch)

    # Equal up to the network's numerical equivariance error (~1e-10).
    assert (out_full - out_base).norm() < 1e-8
