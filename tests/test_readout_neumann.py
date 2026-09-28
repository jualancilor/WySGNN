"""U7 tests — Neumann-exact tensor readout."""

from __future__ import annotations

import torch

from tests._structures import batio3_tetragonal, nacl
from wysgnn.nn.irreps import full_to_voigt, piezo_irreps
from wysgnn.nn.readout import neumann_project, tensor_readout
from wysgnn.symmetry.wyckoff import crystal_point_group_ops_cart

torch.set_default_dtype(torch.float64)

IRREPS_OUT = piezo_irreps()


def _single_graph_features(n_nodes=5):
    torch.manual_seed(0)
    x = torch.randn(n_nodes, IRREPS_OUT.dim, dtype=torch.float64)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return x, batch


def test_neumann_invariance_exact():
    s = batio3_tetragonal()
    ops = crystal_point_group_ops_cart(s)
    x, batch = _single_graph_features()
    out = tensor_readout(x, batch, IRREPS_OUT, [ops], neumann_enabled=True)
    T = out[0]
    for g in ops:
        gT = torch.einsum("ia,jb,kc,abc->ijk", g, g, g, T)
        assert torch.linalg.norm(gT - T) < 1e-5


def test_centrosymmetric_gives_zero():
    # NaCl point group m-3m is centrosymmetric -> piezoelectric tensor is zero.
    s = nacl()
    ops = crystal_point_group_ops_cart(s)
    x, batch = _single_graph_features()
    out = tensor_readout(x, batch, IRREPS_OUT, [ops], neumann_enabled=True)
    assert torch.linalg.norm(out[0]) < 1e-10


def test_p4mm_allowed_components_only():
    # BaTiO3 P4mm, point group 4mm: only d15=d24, d31=d32, d33 are nonzero.
    s = batio3_tetragonal()
    ops = crystal_point_group_ops_cart(s)
    x, batch = _single_graph_features()
    out = tensor_readout(x, batch, IRREPS_OUT, [ops], neumann_enabled=True)
    voigt = full_to_voigt(out[0])  # (3,6)

    allowed = {(0, 4), (1, 3), (2, 0), (2, 1), (2, 2)}  # d15,d24,d31,d32,d33
    for i in range(3):
        for v in range(6):
            if (i, v) not in allowed:
                assert abs(voigt[i, v].item()) < 1e-6, f"d[{i},{v}] should be 0"
    # symmetry relations and non-triviality
    assert abs(voigt[0, 4].item() - voigt[1, 3].item()) < 1e-6  # d15 == d24
    assert abs(voigt[2, 0].item() - voigt[2, 1].item()) < 1e-6  # d31 == d32
    assert voigt.abs().max().item() > 1e-3  # not trivially zero


def test_neumann_disabled_passthrough():
    s = batio3_tetragonal()
    ops = crystal_point_group_ops_cart(s)
    x, batch = _single_graph_features()
    out_on = tensor_readout(x, batch, IRREPS_OUT, [ops], neumann_enabled=True)
    out_off = tensor_readout(x, batch, IRREPS_OUT, [ops], neumann_enabled=False)
    # disabled output generally violates the zero pattern
    assert torch.linalg.norm(out_on - out_off) > 1e-3


def test_neumann_project_idempotent():
    s = batio3_tetragonal()
    ops = crystal_point_group_ops_cart(s)
    T = torch.randn(3, 3, 3, dtype=torch.float64)
    once = neumann_project(T, ops)
    twice = neumann_project(once, ops)
    assert torch.linalg.norm(twice - once) < 1e-8
