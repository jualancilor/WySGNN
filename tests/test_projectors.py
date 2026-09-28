"""U2 tests — site-symmetry projectors."""

from __future__ import annotations

import torch
from e3nn import o3

from tests._structures import quartz
from wysgnn.symmetry.projectors import build_site_projectors, wigner_D_irreps
from wysgnn.symmetry.wyckoff import extract_site_symmetry

IRREPS = o3.Irreps("2x0e + 2x1o + 1x1e + 1x2e")


def _c2_z() -> torch.Tensor:
    """Group {I, C2(z)} as Cartesian matrices."""
    c2 = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    return torch.stack([torch.eye(3, dtype=torch.float64), c2])


def _inversion() -> torch.Tensor:
    return torch.stack([torch.eye(3, dtype=torch.float64), -torch.eye(3, dtype=torch.float64)])


def _all_test_groups():
    yield "trivial", torch.eye(3, dtype=torch.float64).unsqueeze(0)
    yield "C2z", _c2_z()
    yield "inversion", _inversion()
    ss = extract_site_symmetry(quartz(), 1e-4)
    yield "quartz-Si", ss.site_rotations_cart[0]


def test_idempotent():
    for _name, g in _all_test_groups():
        P = build_site_projectors(g, IRREPS)
        assert torch.linalg.norm(P @ P - P) < 1e-5


def test_symmetric():
    for _name, g in _all_test_groups():
        P = build_site_projectors(g, IRREPS)
        assert torch.linalg.norm(P - P.T) < 1e-5


def test_eigenvalues_zero_one():
    for _name, g in _all_test_groups():
        P = build_site_projectors(g, IRREPS)
        evals = torch.linalg.eigvalsh(P)
        # each eigenvalue close to 0 or 1
        dist = torch.minimum(evals.abs(), (evals - 1).abs())
        assert dist.max() < 1e-5


def test_trivial_group_is_identity():
    g = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    P = build_site_projectors(g, IRREPS)
    assert torch.linalg.norm(P - torch.eye(IRREPS.dim, dtype=torch.float64)) < 1e-6


def test_inversion_vector_and_scalar():
    g = _inversion()
    # vector irrep l=1, p=-1 -> projector is zero
    P_vec = build_site_projectors(g, o3.Irreps("1o"))
    assert torch.linalg.norm(P_vec) < 1e-6
    # scalar irrep l=0, p=+1 -> projector is identity
    P_scal = build_site_projectors(g, o3.Irreps("0e"))
    assert torch.linalg.norm(P_scal - torch.eye(1, dtype=torch.float64)) < 1e-6


def test_conjugation_equivariance_lemma():
    # Assertion 6: P_{R G R^T} = D(R) P_G D(R)^T for random rotation R.
    torch.manual_seed(0)
    for _name, g in _all_test_groups():
        P = build_site_projectors(g, IRREPS)
        for _ in range(3):
            R = o3.rand_matrix().to(torch.float64)  # proper rotation, det +1
            g_conj = torch.stack([R @ op @ R.T for op in g])
            P_conj = build_site_projectors(g_conj, IRREPS)
            DR = wigner_D_irreps(IRREPS, R)
            rhs = DR @ P @ DR.T
            assert torch.linalg.norm(P_conj - rhs) < 1e-4, _name
