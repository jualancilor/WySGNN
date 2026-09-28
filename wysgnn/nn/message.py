"""U5 — Equivariant message-passing layer (e3nn-style convolution).

A radial MLP on edge lengths produces the weights of a tensor product between
neighbour node features and the spherical harmonics of the edge direction; the
aggregated messages pass through a gated equivariant nonlinearity.
"""

from __future__ import annotations

import torch
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Gate
from torch import nn
from torch_geometric.utils import scatter


def _gate_activations(irreps_scalars: o3.Irreps):
    acts = []
    for _mul, (ell, p) in irreps_scalars:
        assert ell == 0
        acts.append(torch.tanh if p == -1 else torch.nn.functional.silu)
    return acts


class EquivariantMessageLayer(nn.Module):
    """One equivariant convolution + gated nonlinearity.

    Parameters
    ----------
    irreps_in, irreps_out:
        Node feature irreps before/after the layer.
    irreps_sh:
        Spherical-harmonic irreps used to embed edge directions.
    cutoff:
        Radial cutoff (Angstrom) used for the basis embedding.
    num_basis, radial_hidden, radial_layers:
        Radial MLP configuration.
    avg_num_neighbors:
        Normalisation constant applied to aggregated messages.
    """

    def __init__(
        self,
        irreps_in,
        irreps_sh,
        irreps_out,
        cutoff: float,
        num_basis: int = 8,
        radial_hidden: int = 64,
        radial_layers: int = 2,
        avg_num_neighbors: float = 1.0,
        use_gate: bool = True,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)
        self.cutoff = float(cutoff)
        self.num_basis = int(num_basis)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.use_gate = use_gate

        if use_gate:
            # Split the requested output into the scalar and gated (l>0) parts
            # and add one even gate scalar per gated multiplicity.
            irreps_scalars = o3.Irreps([(m, ir) for m, ir in self.irreps_out if ir.l == 0])
            irreps_gated = o3.Irreps([(m, ir) for m, ir in self.irreps_out if ir.l > 0])
            irreps_gates = o3.Irreps([(m, "0e") for m, _ir in irreps_gated])
            self.gate = Gate(
                irreps_scalars,
                _gate_activations(irreps_scalars),
                irreps_gates,
                [torch.sigmoid] * len(irreps_gates),
                irreps_gated,
            )
            conv_out = self.gate.irreps_in  # scalars + gates + gated
        else:
            self.gate = None
            conv_out = self.irreps_out

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, conv_out, shared_weights=False
        )
        self.radial = FullyConnectedNet(
            [self.num_basis] + [radial_hidden] * radial_layers + [self.tp.weight_numel],
            torch.nn.functional.silu,
        )
        self.self_connection = o3.Linear(self.irreps_in, conv_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
    ) -> torch.Tensor:
        dst, src = edge_index[0], edge_index[1]
        edge_len = torch.linalg.norm(edge_vec, dim=1)
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=True, normalization="component"
        )
        radial_embed = soft_one_hot_linspace(
            edge_len,
            start=0.0,
            end=self.cutoff,
            number=self.num_basis,
            basis="gaussian",
            cutoff=True,
        )
        weights = self.radial(radial_embed)
        messages = self.tp(x[src], edge_sh, weights)
        agg = scatter(messages, dst, dim=0, dim_size=x.shape[0], reduce="sum")
        agg = agg / self.avg_num_neighbors
        h = agg + self.self_connection(x)
        return self.gate(h) if self.use_gate else h
