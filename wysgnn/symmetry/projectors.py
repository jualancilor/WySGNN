"""U2 — Site-symmetry projectors.

For a site whose symmetry group ``G`` is given as Cartesian orthogonal matrices,
build the orthogonal projector onto the ``G``-invariant subspace of an e3nn
feature space::

    P = (1 / |G|) * sum_{g in G} D(g)

where ``D(g)`` is the (block-diagonal over irreps) real representation matrix of
``g`` acting on the feature space.  For an irrep ``(l, p)`` and an operation with
``det g = -1`` (improper), the rotation part is ``-g`` and the parity factor
``p`` multiplies the Wigner-D block.
"""

from __future__ import annotations

import torch
from e3nn import o3


def wigner_D_irreps(irreps: o3.Irreps, R_cart: torch.Tensor) -> torch.Tensor:
    """Block-diagonal real representation matrix ``D(g)`` of one operation.

    Parameters
    ----------
    irreps:
        Feature-space irreps.
    R_cart:
        A single Cartesian orthogonal ``(3, 3)`` matrix (det +/-1).

    Returns
    -------
    Tensor ``(irreps.dim, irreps.dim)``.
    """
    if R_cart.shape != (3, 3):
        raise ValueError(f"R_cart must be (3,3), got {tuple(R_cart.shape)}")
    R = R_cart.to(torch.float64)
    det = torch.det(R)
    if abs(abs(det.item()) - 1.0) > 1e-4:
        raise ValueError(f"R_cart must be orthogonal with |det|=1, got det={det.item()}")
    sign = torch.sign(det)
    proper = sign * R                      # proper rotation part (det +1)
    improper = det.item() < 0
    alpha, beta, gamma = o3.matrix_to_angles(proper)

    blocks: list[torch.Tensor] = []
    for mul, (ell, p) in irreps:
        Dl = o3.wigner_D(ell, alpha, beta, gamma).to(torch.float64)
        if improper:
            Dl = Dl * float(p)             # parity factor p^[det<0]
        for _ in range(mul):
            blocks.append(Dl)
    if not blocks:
        return torch.zeros((0, 0), dtype=torch.float64)
    return torch.block_diag(*blocks)


def build_site_projectors(rotations_cart: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    """Projector onto the ``G``-invariant subspace for one site.

    Parameters
    ----------
    rotations_cart:
        ``(|G|, 3, 3)`` Cartesian orthogonal matrices of the site symmetry group.
    irreps:
        Feature-space irreps.

    Returns
    -------
    Tensor ``(irreps.dim, irreps.dim)`` — the orthogonal projector ``P``.
    """
    if rotations_cart.ndim != 3 or rotations_cart.shape[1:] != (3, 3):
        raise ValueError(
            f"rotations_cart must be (|G|,3,3), got {tuple(rotations_cart.shape)}"
        )
    if rotations_cart.shape[0] == 0:
        raise ValueError("rotations_cart is empty; a group must contain the identity")

    dim = irreps.dim
    P = torch.zeros((dim, dim), dtype=torch.float64)
    for g in rotations_cart:
        P = P + wigner_D_irreps(irreps, g)
    P = P / rotations_cart.shape[0]
    return P
