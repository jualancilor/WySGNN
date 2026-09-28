"""U4 tests — rank-3 tensor irrep decomposition and change of basis."""

from __future__ import annotations

import torch
from e3nn import o3

from wysgnn.nn.irreps import (
    cart_to_irrep,
    full_to_voigt,
    irrep_to_cart,
    node_irreps,
    piezo_change_of_basis,
    piezo_irreps,
    voigt_to_full,
)

torch.set_default_dtype(torch.float64)


def _rand_sym_tensor(*batch):
    t = torch.randn(*batch, 3, 3, 3, dtype=torch.float64)
    return 0.5 * (t + t.transpose(-1, -2))


def test_decomposition_is_18_dim():
    ir = piezo_irreps()
    assert ir == o3.Irreps("2x1o + 1x2o + 1x3o")
    assert ir.dim == 18


def test_change_of_basis_orthogonal():
    Q = piezo_change_of_basis().reshape(18, 27)
    assert torch.linalg.norm(Q @ Q.T - torch.eye(18, dtype=torch.float64)) < 1e-10


def test_roundtrip_identity():
    t = _rand_sym_tensor(7)
    rec = irrep_to_cart(cart_to_irrep(t))
    assert torch.linalg.norm(rec - t) < 1e-10


def test_equivariance_of_change_of_basis():
    t = _rand_sym_tensor()
    c = cart_to_irrep(t)
    R = o3.rand_matrix().to(torch.float64)
    t_rot = torch.einsum("ia,jb,kc,abc->ijk", R, R, R, t)
    c_rot = cart_to_irrep(t_rot)
    D = piezo_irreps().D_from_matrix(R)
    assert torch.linalg.norm(c_rot - D @ c) < 1e-8


def test_voigt_roundtrip():
    t = _rand_sym_tensor(4)
    v = full_to_voigt(t)
    assert v.shape[-2:] == (3, 6)
    rec = voigt_to_full(v)
    assert torch.linalg.norm(rec - t) < 1e-10


def test_node_irreps_shape():
    ir = node_irreps(lmax=2, mul=4)
    # mul copies of (0e,0o,1e,1o,2e,2o)
    assert ir.dim == 4 * (1 + 1 + 3 + 3 + 5 + 5)
    assert o3.Irrep("3o") in [irr for _m, irr in node_irreps(3, 1)]
