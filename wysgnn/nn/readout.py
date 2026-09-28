"""U7 — Neumann-exact tensor readout.

Pools node features into a graph-level irrep vector, converts it to a Cartesian
rank-3 tensor (U4 change of basis), and projects it onto the subspace invariant
under the crystal point group (Neumann's principle)::

    T  ->  (1 / |G|) * sum_{g in G} g . T,   (g . T)_{ijk} = g_ia g_jb g_kc T_abc

For a centrosymmetric point group this maps any rank-3 (parity-odd) tensor to
exactly zero.
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch_geometric.utils import scatter

from .irreps import irrep_to_cart


def neumann_project(tensor: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
    """Project a single ``(3,3,3)`` Cartesian tensor onto the ``G``-invariant subspace.

    ``ops`` is ``(|G|, 3, 3)`` Cartesian point-group matrices.
    """
    g = ops.to(tensor.dtype)
    sym = torch.einsum("gia,gjb,gkc,abc->ijk", g, g, g, tensor)
    return sym / ops.shape[0]


def tensor_readout(
    node_features: torch.Tensor,
    batch: torch.Tensor,
    irreps_out: o3.Irreps,
    crystal_point_group_ops: list[torch.Tensor],
    neumann_enabled: bool = True,
    reduce: str = "sum",
) -> torch.Tensor:
    """Pool per-node irrep features into Neumann-projected Cartesian tensors.

    Parameters
    ----------
    node_features:
        ``(N, irreps_out.dim)`` per-node features already in ``irreps_out``.
    batch:
        ``(N,)`` graph index per node.
    irreps_out:
        Output irreps (must equal the rank-3 piezo irreps, dim 18).
    crystal_point_group_ops:
        Length-``B`` list of ``(|G_b|, 3, 3)`` Cartesian point-group matrices.
    neumann_enabled:
        If ``False`` the Neumann projection is skipped (ablation baseline).
    reduce:
        Node pooling reduction, ``"sum"`` or ``"mean"``.

    Returns
    -------
    Tensor ``(B, 3, 3, 3)``.
    """
    if node_features.shape[1] != irreps_out.dim:
        raise ValueError(
            f"node_features dim {node_features.shape[1]} != irreps_out.dim {irreps_out.dim}"
        )
    n_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if len(crystal_point_group_ops) != n_graphs:
        raise ValueError(
            f"expected {n_graphs} point-group op sets, got {len(crystal_point_group_ops)}"
        )

    pooled = scatter(node_features, batch, dim=0, dim_size=n_graphs, reduce=reduce)
    cart = irrep_to_cart(pooled)  # (B,3,3,3)

    if not neumann_enabled:
        return cart

    out = torch.empty_like(cart)
    for b in range(n_graphs):
        out[b] = neumann_project(cart[b], crystal_point_group_ops[b].to(cart.dtype))
    return out
