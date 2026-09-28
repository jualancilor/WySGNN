"""U5 tests — equivariant message-passing layer."""

from __future__ import annotations

import torch
from e3nn import o3

from tests._structures import quartz
from wysgnn.data.graph import structure_to_graph
from wysgnn.nn.irreps import node_irreps, sh_irreps
from wysgnn.nn.message import EquivariantMessageLayer

torch.set_default_dtype(torch.float64)

IRREPS = node_irreps(2, 4)
SH = sh_irreps(3)
CUTOFF = 4.0


def _layer():
    torch.manual_seed(0)
    return EquivariantMessageLayer(
        IRREPS, SH, IRREPS, cutoff=CUTOFF, avg_num_neighbors=8.0
    ).double()


def _graph_and_x():
    data = structure_to_graph(quartz(), CUTOFF, IRREPS)
    x = IRREPS.randn(data.num_nodes, -1)
    return data, x


def test_output_irreps_match():
    layer = _layer()
    data, x = _graph_and_x()
    out = layer(x, data.edge_index, data.edge_vec)
    assert out.shape == (data.num_nodes, IRREPS.dim)


def test_rotation_equivariance():
    layer = _layer()
    data, x = _graph_and_x()
    out = layer(x, data.edge_index, data.edge_vec)
    for seed in range(3):
        torch.manual_seed(seed)
        R = o3.rand_matrix().double()
        D = IRREPS.D_from_matrix(R)
        out_rot = layer(x @ D.T, data.edge_index, data.edge_vec @ R.T)
        assert torch.linalg.norm(out_rot - out @ D.T) < 1e-4


def test_permutation_equivariance():
    layer = _layer()
    data, x = _graph_and_x()
    out = layer(x, data.edge_index, data.edge_vec)
    N = data.num_nodes
    perm = torch.randperm(N)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N)
    x_perm = x[perm]
    edge_index_perm = inv[data.edge_index]
    out_perm = layer(x_perm, edge_index_perm, data.edge_vec)
    assert torch.linalg.norm(out_perm - out[perm]) < 1e-8
