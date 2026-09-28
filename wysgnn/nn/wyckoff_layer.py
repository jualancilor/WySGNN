"""U6 — Site-symmetry projection layer (the WySGNN contribution).

After each message-passing layer, every node's feature vector is projected onto
its site-symmetry-invariant subspace via the precomputed per-node projector
``P_i`` (U2).  Because ``P_i`` is an orthogonal projector that commutes with the
group representation, this both (a) forces each node feature to be exactly
``G_i``-invariant and (b) preserves global ``E(3)`` equivariance (the projector
of a rotated crystal is the conjugate ``D(R) P_i D(R)^T``).
"""

from __future__ import annotations

import torch
from torch import nn


def apply_site_projection(
    x: torch.Tensor,
    site_proj: torch.Tensor,
    enabled: bool = True,
) -> torch.Tensor:
    """Apply per-node projectors ``x_out[i] = P_i @ x[i]``.

    Parameters
    ----------
    x:
        Node features ``(N, dim)``.
    site_proj:
        Per-node projectors ``(N, dim, dim)``.
    enabled:
        If ``False`` the projection is the identity (ablation / no-Wyckoff
        baseline).

    Returns
    -------
    Tensor ``(N, dim)``.
    """
    if not enabled:
        return x
    if x.ndim != 2:
        raise ValueError(f"x must be (N,dim), got {tuple(x.shape)}")
    if site_proj.shape != (x.shape[0], x.shape[1], x.shape[1]):
        raise ValueError(
            f"site_proj must be (N,dim,dim)={(x.shape[0], x.shape[1], x.shape[1])}, "
            f"got {tuple(site_proj.shape)}"
        )
    P = site_proj.to(x.dtype)
    return torch.bmm(P, x.unsqueeze(-1)).squeeze(-1)


class WyckoffProjection(nn.Module):
    """Thin module wrapper around :func:`apply_site_projection` with a flag."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def forward(self, x: torch.Tensor, site_proj: torch.Tensor) -> torch.Tensor:
        return apply_site_projection(x, site_proj, enabled=self.enabled)
