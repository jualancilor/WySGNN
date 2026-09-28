"""U3 tests — periodic graph construction."""

from __future__ import annotations

import numpy as np
import torch
from e3nn import o3
from pymatgen.core import Lattice, Structure

from tests._structures import nacl, quartz
from wysgnn.symmetry.projectors import build_site_projectors
from wysgnn.symmetry.wyckoff import extract_site_symmetry
from wysgnn.data.graph import structure_to_graph

IRREPS = o3.Irreps("2x0e + 1x1o + 1x2e")
CUTOFF = 4.0


def test_edge_index_bounds_and_no_zero_selfloops():
    s = quartz()
    data = structure_to_graph(s, CUTOFF, IRREPS)
    N = data.num_nodes
    assert int(data.edge_index.min()) >= 0
    assert int(data.edge_index.max()) < N
    # No true zero-length self-loops (periodic-image self neighbours are allowed).
    self_mask = data.edge_index[0] == data.edge_index[1]
    assert torch.all(data.edge_len[self_mask] > 1e-6)


def test_periodicity_rotation_translation_invariance():
    s = quartz()
    base = structure_to_graph(s, CUTOFF, IRREPS)
    base_lens = torch.sort(base.edge_len).values

    # Rotate the whole crystal: new lattice rows = R @ a_i  ==>  M' = M @ R^T.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    new_matrix = s.lattice.matrix @ Q.T
    # Also translate fractional coords and re-image.
    shift = rng.random(3)
    new_frac = (s.frac_coords + shift) % 1.0
    s2 = Structure(Lattice(new_matrix), [site.specie for site in s], new_frac)

    rot = structure_to_graph(s2, CUTOFF, IRREPS)
    rot_lens = torch.sort(rot.edge_len).values
    assert rot_lens.shape == base_lens.shape
    assert torch.allclose(base_lens, rot_lens, atol=1e-6)


def test_site_proj_matches_unit2():
    s = quartz()
    data = structure_to_graph(s, CUTOFF, IRREPS)
    sym = extract_site_symmetry(s, 1e-4)
    for i in range(data.num_nodes):
        P = build_site_projectors(sym.site_rotations_cart[i], IRREPS)
        assert torch.linalg.norm(data.site_proj[i] - P) < 1e-8


def test_known_neighbor_count_nacl():
    # Rock salt: with a cutoff between 1st (a/2=2.82) and 2nd (a/sqrt2=3.99)
    # neighbour shells, every atom has exactly 6 neighbours.
    s = nacl()
    data = structure_to_graph(s, 3.0, IRREPS)
    N = data.num_nodes
    counts = torch.bincount(data.edge_index[0], minlength=N)
    assert torch.all(counts == 6)
    assert data.edge_index.shape[1] == N * 6
