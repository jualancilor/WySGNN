"""Invariant-feature baseline (CGCNN/MEGNet-style).

This model uses only rotation-*invariant* scalar features and regresses the 18
tensor components with an MLP head in the crystal's stored Cartesian frame.  It
is therefore **not** equivariant and does not satisfy Neumann's principle: it
predicts nonzero values for symmetry-forbidden components and for
centrosymmetric crystals.  It is the realistic non-symmetry-aware comparison
point for WySGNN, and the model on which the optional Neumann projection has a
visible effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from e3nn.math import soft_one_hot_linspace
from torch import nn
from torch_geometric.utils import scatter

from .irreps import voigt_to_full
from .readout import neumann_project


@dataclass
class InvariantBaselineConfig:
    hidden: int = 64
    num_layers: int = 3
    cutoff: float = 5.0
    num_basis: int = 8
    num_elements: int = 100
    neumann_enabled: bool = False


class _InvariantConv(nn.Module):
    def __init__(self, hidden: int, num_basis: int):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + num_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())

    def forward(self, h, edge_index, edge_embed):
        dst, src = edge_index[0], edge_index[1]
        m = self.msg(torch.cat([h[src], h[dst], edge_embed], dim=-1))
        agg = scatter(m, dst, dim=0, dim_size=h.shape[0], reduce="sum")
        return h + self.upd(agg)


class InvariantBaseline(nn.Module):
    def __init__(self, config: InvariantBaselineConfig | None = None, **overrides):
        super().__init__()
        cfg = config or InvariantBaselineConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.num_elements, cfg.hidden)
        self.convs = nn.ModuleList(
            _InvariantConv(cfg.hidden, cfg.num_basis) for _ in range(cfg.num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, 18),
        )

    def forward(self, data, point_group_ops: list[torch.Tensor] | None = None) -> torch.Tensor:
        edge_index, edge_vec = data.edge_index, data.edge_vec
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.z.shape[0], dtype=torch.long, device=data.z.device)
        edge_len = torch.linalg.norm(edge_vec, dim=1)
        edge_embed = soft_one_hot_linspace(
            edge_len, 0.0, self.cfg.cutoff, self.cfg.num_basis, basis="gaussian", cutoff=True
        )
        h = self.embedding(data.z)
        for conv in self.convs:
            h = conv(h, edge_index, edge_embed)
        n_graphs = int(batch.max().item()) + 1
        pooled = scatter(h, batch, dim=0, dim_size=n_graphs, reduce="mean")
        voigt = self.head(pooled).view(n_graphs, 3, 6)
        cart = voigt_to_full(voigt)
        if self.cfg.neumann_enabled:
            if point_group_ops is None:
                raise ValueError("point_group_ops required when neumann_enabled=True")
            cart = torch.stack(
                [neumann_project(cart[b], point_group_ops[b].to(cart.dtype)) for b in range(n_graphs)]
            )
        return cart
