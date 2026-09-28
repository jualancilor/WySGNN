"""U6 tests — site-symmetry projection layer."""

from __future__ import annotations

import torch
from e3nn import o3
from pymatgen.core import Lattice, Structure

from tests._structures import quartz
from wysgnn.data.graph import structure_to_graph
from wysgnn.nn.irreps import node_irreps, sh_irreps
from wysgnn.nn.message import EquivariantMessageLayer
from wysgnn.nn.wyckoff_layer import apply_site_projection
from wysgnn.symmetry.projectors import wigner_D_irreps
from wysgnn.symmetry.wyckoff import extract_site_symmetry

torch.set_default_dtype(torch.float64)

IRREPS = node_irreps(2, 4)
SH = sh_irreps(3)
CUTOFF = 4.0


def test_site_invariance():
    s = quartz()
    data = structure_to_graph(s, CUTOFF, IRREPS)
    sym = extract_site_symmetry(s, 1e-4)
    x = IRREPS.randn(data.num_nodes, -1)
    x_out = apply_site_projection(x, data.site_proj)
    for i, ops in enumerate(sym.site_rotations_cart):
        for g in ops:
            D = wigner_D_irreps(IRREPS, g).to(x_out.dtype)
            assert torch.linalg.norm(D @ x_out[i] - x_out[i]) < 1e-8


def test_idempotence():
    data = structure_to_graph(quartz(), CUTOFF, IRREPS)
    x = IRREPS.randn(data.num_nodes, -1)
    once = apply_site_projection(x, data.site_proj)
    twice = apply_site_projection(once, data.site_proj)
    assert torch.linalg.norm(twice - once) < 1e-10


def test_ablation_identity():
    data = structure_to_graph(quartz(), CUTOFF, IRREPS)
    x = IRREPS.randn(data.num_nodes, -1)
    assert torch.equal(apply_site_projection(x, data.site_proj, enabled=False), x)


def test_global_equivariance_preserved():
    # rotate whole crystal -> recompute graph -> [message -> projection]
    # equals D(R) on the un-rotated output.
    torch.manual_seed(0)
    s = quartz()
    data = structure_to_graph(s, CUTOFF, IRREPS)
    layer = EquivariantMessageLayer(
        IRREPS, SH, IRREPS, cutoff=CUTOFF, avg_num_neighbors=8.0
    ).double()
    x = IRREPS.randn(data.num_nodes, -1)

    def forward(d, xin):
        h = layer(xin, d.edge_index, d.edge_vec)
        return apply_site_projection(h, d.site_proj)

    out = forward(data, x)

    R = o3.rand_matrix().double()
    D = IRREPS.D_from_matrix(R)
    new_matrix = s.lattice.matrix @ R.T.numpy()
    s_rot = Structure(Lattice(new_matrix), [site.specie for site in s], s.frac_coords)
    data_rot = structure_to_graph(s_rot, CUTOFF, IRREPS)
    out_rot = forward(data_rot, x @ D.T)

    assert torch.linalg.norm(out_rot - out @ D.T) < 1e-4
