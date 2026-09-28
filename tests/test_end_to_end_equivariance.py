"""End-to-end equivariance of the full WySGNN model (Execution Gate 2).

Rotating the whole crystal by ``R`` must rotate the predicted rank-3 tensor by
the same ``R`` (Cartesian index rotation)::

    model(R . structure)_{ijk} = R_ia R_jb R_kc model(structure)_{abc}
"""

from __future__ import annotations

import torch
from e3nn import o3
from pymatgen.core import Lattice, Structure

from tests._structures import batio3_tetragonal, quartz
from wysgnn.data.graph import structures_to_batch
from wysgnn.nn.irreps import node_irreps
from wysgnn.nn.model import WySGNN, WySGNNConfig

torch.set_default_dtype(torch.float64)

CFG = WySGNNConfig(lmax=2, mul=4, num_layers=2, cutoff=4.0)


def _rotate_structure(s: Structure, R: torch.Tensor) -> Structure:
    new_matrix = s.lattice.matrix @ R.T.numpy()
    return Structure(Lattice(new_matrix), [site.specie for site in s], s.frac_coords)


def test_end_to_end_equivariance():
    torch.manual_seed(0)
    model = WySGNN(CFG).double()
    irreps = node_irreps(CFG.lmax, CFG.mul)

    for builder in (quartz, batio3_tetragonal):
        s = builder()
        for seed in range(3):
            torch.manual_seed(seed)
            R = o3.rand_matrix().double()
            s_rot = _rotate_structure(s, R)

            batch, ops = structures_to_batch([s], CFG.cutoff, irreps)
            batch_r, ops_r = structures_to_batch([s_rot], CFG.cutoff, irreps)

            out = model(batch, ops)[0]
            out_rot = model(batch_r, ops_r)[0]
            expected = torch.einsum("ia,jb,kc,abc->ijk", R, R, R, out)
            assert torch.linalg.norm(out_rot - expected) < 1e-4
            assert out.norm() > 1e-6  # non-trivial (non-centrosymmetric) output
