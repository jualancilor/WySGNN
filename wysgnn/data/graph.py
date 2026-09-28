"""U3 — Crystal structure to periodic graph.

Converts a :class:`pymatgen.core.Structure` into a
:class:`torch_geometric.data.Data` object carrying geometry, atomic numbers,
periodic edges (minimum-image convention via ASE), and the per-site symmetry
projectors from U2.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.neighborlist import neighbor_list
from e3nn import o3
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Batch, Data

from ..symmetry.projectors import build_site_projectors
from ..symmetry.wyckoff import crystal_point_group_ops_cart, extract_site_symmetry

_DTYPE = torch.float64


def structure_to_graph(
    structure: Structure,
    cutoff: float,
    irreps_node: o3.Irreps,
    symprec: float = 1e-4,
) -> Data:
    """Build a periodic graph for ``structure``.

    Edges use the minimum-image convention; periodic images of an atom can be
    its own neighbour, so ``i == j`` edges with a non-zero displacement are
    legitimate and intentional (true zero-length self-loops are never emitted).

    Returns
    -------
    torch_geometric.data.Data with fields:
        ``pos (N,3)``, ``z (N,)``, ``edge_index (2,E)``, ``edge_vec (E,3)``,
        ``edge_len (E,)``, ``site_proj (N,dim,dim)``, ``lattice (1,3,3)``,
        ``space_group (1,)``, ``y`` (target slot, filled by U9).
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    atoms = AseAtomsAdaptor.get_atoms(structure)
    i, j, D = neighbor_list("ijD", atoms, float(cutoff))
    if len(i) == 0:
        raise ValueError("no neighbours found within cutoff; increase cutoff")

    edge_index = torch.from_numpy(np.vstack([i, j])).to(torch.long)
    edge_vec = torch.from_numpy(np.ascontiguousarray(D)).to(_DTYPE)
    edge_len = torch.linalg.norm(edge_vec, dim=1)

    pos = torch.from_numpy(np.ascontiguousarray(structure.cart_coords)).to(_DTYPE)
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # Per-site projectors (U1 + U2).
    sym = extract_site_symmetry(structure, symprec)
    dim = irreps_node.dim
    site_proj = torch.empty((len(structure), dim, dim), dtype=_DTYPE)
    for idx, ops in enumerate(sym.site_rotations_cart):
        site_proj[idx] = build_site_projectors(ops, irreps_node)

    lattice = torch.from_numpy(
        np.array(structure.lattice.matrix, dtype=np.float64)
    ).to(_DTYPE).unsqueeze(0)

    data = Data(
        pos=pos,
        z=z,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_len=edge_len,
        site_proj=site_proj,
        lattice=lattice,
        space_group=torch.tensor([sym.space_group_number], dtype=torch.long),
    )
    data.num_nodes = len(structure)
    return data


def structures_to_batch(
    structures: list[Structure],
    cutoff: float,
    irreps_node: o3.Irreps,
    symprec: float = 1e-4,
) -> tuple[Batch, list[torch.Tensor]]:
    """Build a PyG ``Batch`` and the matching list of point-group op tensors.

    Point-group operations are ragged across crystals (different ``|G|``), so
    they are returned as a separate length-``B`` list aligned with the batch
    order rather than collated into the ``Batch``.
    """
    datas = [structure_to_graph(s, cutoff, irreps_node, symprec) for s in structures]
    ops = [crystal_point_group_ops_cart(s, symprec) for s in structures]
    return Batch.from_data_list(datas), ops
