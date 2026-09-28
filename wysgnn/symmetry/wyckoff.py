"""U1 — Site symmetry extraction from a crystal structure.

Given a :class:`pymatgen.core.Structure`, this module determines, for every
atomic site, the *site symmetry group* ``G_i`` (the subset of space-group
operations that fix the site modulo a lattice translation) expressed as
**Cartesian** orthogonal 3x3 matrices.  These per-site groups are the objects
the rest of WySGNN projects node features onto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spglib
import torch
from pymatgen.core import Structure


@dataclass
class SiteSymmetry:
    """Container for per-site symmetry information of a crystal.

    All per-site lists have length ``n_sites`` and are aligned with the site
    order of the input :class:`pymatgen.core.Structure`.
    """

    space_group_number: int
    wyckoff_letters: list[str]
    site_symmetry_symbols: list[str]
    site_rotations_cart: list[torch.Tensor]  # each (|G_i|, 3, 3)
    equivalent_atoms: list[int]


def _cell_from_structure(structure: Structure):
    """Build the ``(lattice, frac_coords, numbers)`` tuple spglib expects.

    spglib expects the lattice as **row vectors**, which is exactly the
    convention of :attr:`pymatgen.core.Lattice.matrix`.
    """
    lattice = np.ascontiguousarray(structure.lattice.matrix, dtype=np.float64)
    frac = np.ascontiguousarray(structure.frac_coords, dtype=np.float64)
    numbers = [site.specie.Z for site in structure]
    return (lattice, frac, numbers)


def extract_site_symmetry(structure: Structure, symprec: float = 1e-4) -> SiteSymmetry:
    """Extract space-group and per-site symmetry data for ``structure``.

    Parameters
    ----------
    structure:
        Input crystal structure.
    symprec:
        Cartesian distance tolerance (Angstrom) passed to spglib and used to
        decide which operations fix a site.

    Returns
    -------
    SiteSymmetry

    Raises
    ------
    ValueError
        If spglib fails to determine the symmetry (no silent fallback).
    """
    cell = _cell_from_structure(structure)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None:
        raise ValueError("spglib.get_symmetry_dataset returned None (malformed cell?)")

    lattice = cell[0]  # rows = lattice vectors
    frac = cell[1]
    n_sites = len(frac)

    # Cartesian basis matrix M maps fractional -> Cartesian column vectors:
    #   r_cart = M @ x_frac,  with columns of M = lattice vectors = lattice.T
    M = lattice.T
    Minv = np.linalg.inv(M)

    rotations = np.asarray(ds.rotations)          # (n_ops, 3, 3) integer, fractional
    translations = np.asarray(ds.translations)    # (n_ops, 3)

    site_rotations_cart: list[torch.Tensor] = []
    for i in range(n_sites):
        xi = frac[i]
        # Operations (R|t) with R xi + t == xi (mod 1) within Cartesian symprec.
        ops_frac: list[np.ndarray] = []
        seen: set[tuple] = set()
        for R, t in zip(rotations, translations):
            xp = R @ xi + t
            diff = xp - xi
            diff -= np.round(diff)               # reduce modulo lattice
            cart_disp = M @ diff                 # displacement in Angstrom
            if np.linalg.norm(cart_disp) < symprec:
                key = tuple(np.round(R).astype(int).flatten().tolist())
                if key not in seen:
                    seen.add(key)
                    ops_frac.append(R.astype(np.float64))
        if not ops_frac:
            raise ValueError(f"site {i}: identity operation not found (symprec too tight?)")
        # Convert each fractional point operation to a Cartesian orthogonal matrix.
        ops_cart = np.stack([M @ R @ Minv for R in ops_frac], axis=0)
        site_rotations_cart.append(torch.from_numpy(ops_cart).to(torch.float64))

    return SiteSymmetry(
        space_group_number=int(ds.number),
        wyckoff_letters=list(ds.wyckoffs),
        site_symmetry_symbols=list(ds.site_symmetry_symbols),
        site_rotations_cart=site_rotations_cart,
        equivalent_atoms=[int(x) for x in ds.equivalent_atoms],
    )


def crystal_point_group_ops_cart(structure: Structure, symprec: float = 1e-4) -> torch.Tensor:
    """Return the crystal **point group** as Cartesian 3x3 matrices ``(|G|,3,3)``.

    This is the set of distinct rotational parts of all space-group operations,
    used for the Neumann projection in the readout (U7).
    """
    cell = _cell_from_structure(structure)
    sym = spglib.get_symmetry(cell, symprec=symprec)
    if sym is None:
        raise ValueError("spglib.get_symmetry returned None (malformed cell?)")
    lattice = cell[0]
    M = lattice.T
    Minv = np.linalg.inv(M)
    rotations = np.asarray(sym["rotations"])
    seen: set[tuple] = set()
    ops: list[np.ndarray] = []
    for R in rotations:
        key = tuple(np.round(R).astype(int).flatten().tolist())
        if key not in seen:
            seen.add(key)
            ops.append(M @ R.astype(np.float64) @ Minv)
    return torch.from_numpy(np.stack(ops, axis=0)).to(torch.float64)
