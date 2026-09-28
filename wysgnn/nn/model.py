"""U8 — The WySGNN model.

Assembles: atomic embedding -> ``[message -> site projection] x L`` ->
equivariant tensor readout conv -> Neumann-projected pooling.  Every symmetry
component is config-toggleable so the model degrades cleanly to a plain
equivariant baseline (``wyckoff_enabled=False, neumann_enabled=False``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from e3nn import o3
from torch import nn

from .irreps import node_irreps, piezo_irreps, sh_irreps
from .message import EquivariantMessageLayer
from .readout import tensor_readout
from .wyckoff_layer import apply_site_projection


@dataclass
class WySGNNConfig:
    lmax: int = 2
    mul: int = 8
    num_layers: int = 3
    cutoff: float = 5.0
    sh_lmax: int = 2
    out_sh_lmax: int = 3
    num_basis: int = 8
    radial_hidden: int = 64
    avg_num_neighbors: float = 12.0
    num_elements: int = 100
    wyckoff_enabled: bool = True
    neumann_enabled: bool = True


class WySGNN(nn.Module):
    def __init__(self, config: WySGNNConfig | None = None, **overrides):
        super().__init__()
        cfg = config or WySGNNConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        self.cfg = cfg

        self.hidden_irreps = node_irreps(cfg.lmax, cfg.mul)
        self.irreps_out = piezo_irreps()
        sh = sh_irreps(cfg.sh_lmax)
        out_sh = sh_irreps(cfg.out_sh_lmax)

        # Embedding: atomic number -> mul scalars -> hidden irreps.
        self.embedding = nn.Embedding(cfg.num_elements, cfg.mul)
        self.embed_lin = o3.Linear(o3.Irreps(f"{cfg.mul}x0e"), self.hidden_irreps)

        self.layers = nn.ModuleList(
            EquivariantMessageLayer(
                self.hidden_irreps,
                sh,
                self.hidden_irreps,
                cutoff=cfg.cutoff,
                num_basis=cfg.num_basis,
                radial_hidden=cfg.radial_hidden,
                avg_num_neighbors=cfg.avg_num_neighbors,
                use_gate=True,
            )
            for _ in range(cfg.num_layers)
        )

        # Final equivariant readout conv: hidden -> piezo irreps (no gate).
        self.readout_conv = EquivariantMessageLayer(
            self.hidden_irreps,
            out_sh,
            self.irreps_out,
            cutoff=cfg.cutoff,
            num_basis=cfg.num_basis,
            radial_hidden=cfg.radial_hidden,
            avg_num_neighbors=cfg.avg_num_neighbors,
            use_gate=False,
        )

    def forward(self, data, point_group_ops: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Predict per-graph Cartesian rank-3 tensors ``(B, 3, 3, 3)``.

        ``point_group_ops`` is a length-``B`` list of ``(|G|,3,3)`` Cartesian
        point-group matrices; required when ``neumann_enabled``.
        """
        edge_index, edge_vec = data.edge_index, data.edge_vec
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.z.shape[0], dtype=torch.long, device=data.z.device)

        if self.cfg.neumann_enabled and point_group_ops is None:
            raise ValueError("point_group_ops required when neumann_enabled=True")

        x = self.embed_lin(self.embedding(data.z))
        for layer in self.layers:
            x = layer(x, edge_index, edge_vec)
            x = apply_site_projection(x, data.site_proj, enabled=self.cfg.wyckoff_enabled)

        node_out = self.readout_conv(x, edge_index, edge_vec)
        n_graphs = int(batch.max().item()) + 1
        ops = point_group_ops if point_group_ops is not None else [None] * n_graphs
        return tensor_readout(
            node_out,
            batch,
            self.irreps_out,
            ops,
            neumann_enabled=self.cfg.neumann_enabled,
            reduce="sum",
        )
