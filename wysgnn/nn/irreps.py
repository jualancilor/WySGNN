"""U4 — Irreps schedule and rank-3 (piezoelectric) tensor decomposition.

The piezoelectric strain tensor ``d_{ijk}`` is a rank-3 Cartesian tensor that is
symmetric in its last two (strain) indices ``j, k``.  As a real ``O(3)``
representation it reduces to::

    V (x) Sym^2(V) = 1o (x) (0e + 2e) = 2x1o + 1x2o + 1x3o     (dim 18)

All irreps are parity-odd, which is exactly why the tensor must vanish in
centrosymmetric crystals.  This module provides the orthogonal change of basis
between the 18-dim symmetric Cartesian space and the irrep coefficients, plus
the feature-irreps schedule used by the network.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from e3nn import o3
from e3nn.o3 import ReducedTensorProducts

# Voigt index -> (j, k) pair for the symmetric strain indices.
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def node_irreps(lmax: int, mul: int) -> o3.Irreps:
    """Hidden node-feature irreps: ``mul`` copies of every ``(l, p)`` up to ``lmax``.

    Both parities are included at each ``l`` so the network can represent the
    parity-odd output irreps (e.g. ``3o``) required for piezoelectric tensors.
    """
    terms = []
    for ell in range(lmax + 1):
        terms.append((mul, (ell, 1)))
        terms.append((mul, (ell, -1)))
    return o3.Irreps(terms)


def sh_irreps(lmax: int) -> o3.Irreps:
    """Spherical-harmonic irreps used to embed edge directions."""
    return o3.Irreps.spherical_harmonics(lmax)


@lru_cache(maxsize=None)
def _reduced() -> ReducedTensorProducts:
    return ReducedTensorProducts("ijk=ikj", i="1o", j="1o", k="1o")


def piezo_irreps() -> o3.Irreps:
    """Output irreps for the rank-3 piezoelectric tensor (``2x1o+1x2o+1x3o``)."""
    return _reduced().irreps_out


@lru_cache(maxsize=None)
def piezo_change_of_basis() -> torch.Tensor:
    """Orthogonal change of basis ``Q`` with shape ``(18, 3, 3, 3)``.

    ``c_z = sum_{ijk} Q_{z,ijk} T_{ijk}`` and ``T_{ijk} = sum_z Q_{z,ijk} c_z``
    for any ``jk``-symmetric Cartesian tensor ``T``.
    """
    return _reduced().change_of_basis.to(torch.float64).contiguous()


def cart_to_irrep(t_cart: torch.Tensor) -> torch.Tensor:
    """Map ``(...,3,3,3)`` symmetric Cartesian tensors to ``(...,18)`` coeffs."""
    Q = piezo_change_of_basis().to(t_cart.dtype)
    return torch.einsum("zijk,...ijk->...z", Q, t_cart)


def irrep_to_cart(coeffs: torch.Tensor) -> torch.Tensor:
    """Map ``(...,18)`` irrep coeffs back to ``(...,3,3,3)`` Cartesian tensors."""
    Q = piezo_change_of_basis().to(coeffs.dtype)
    return torch.einsum("zijk,...z->...ijk", Q, coeffs)


def full_to_voigt(t_cart: torch.Tensor) -> torch.Tensor:
    """Convert ``(...,3,3,3)`` to ``(...,3,6)`` strain-Voigt ``d`` (shear x2)."""
    out = torch.zeros(t_cart.shape[:-3] + (3, 6), dtype=t_cart.dtype, device=t_cart.device)
    for v, (j, k) in enumerate(_VOIGT_PAIRS):
        factor = 1.0 if v < 3 else 2.0
        out[..., :, v] = factor * t_cart[..., :, j, k]
    return out


def voigt_to_full(voigt: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`full_to_voigt` producing a ``jk``-symmetric tensor."""
    out = torch.zeros(voigt.shape[:-2] + (3, 3, 3), dtype=voigt.dtype, device=voigt.device)
    for v, (j, k) in enumerate(_VOIGT_PAIRS):
        factor = 1.0 if v < 3 else 0.5
        val = factor * voigt[..., :, v]
        out[..., :, j, k] = val
        out[..., :, k, j] = val
    return out
