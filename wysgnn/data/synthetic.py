"""Symmetry-preserving structure pool for the pilot dataset.

Generates crystals across several prototypes by isotropic/axial lattice scaling
and jitter of genuinely-free Wyckoff coordinates, rebuilding through
:meth:`Structure.from_spacegroup` so the intended space group is exact.  Each
generated structure is validated with spglib and rejected if the space group
drifts.  This pool feeds a teacher/student pilot (see :mod:`wysgnn.data.datamodule`)
when the real JARVIS piezoelectric dataset is unavailable.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Each prototype: name, space group, and a sampler(rng) -> (lattice, species, coords).
# Non-centrosymmetric prototypes carry real (nonzero) piezoelectric tensors;
# centrosymmetric ones must map to exactly zero (symmetry sanity samples).


def _batio3(rng):  # P4mm (99), non-centrosymmetric
    a = rng.uniform(3.90, 4.05)
    c = rng.uniform(4.00, 4.15)
    ti_z = rng.uniform(0.500, 0.525)
    o1_z = rng.uniform(0.000, 0.040)
    o2_z = rng.uniform(0.455, 0.495)
    coords = [[0, 0, 0], [0.5, 0.5, ti_z], [0.5, 0.5, o1_z], [0.5, 0, o2_z]]
    return Lattice.tetragonal(a, c), ["Ba", "Ti", "O", "O"], coords


def _quartz(rng):  # P3_121 (152), non-centrosymmetric
    a = rng.uniform(4.85, 4.98)
    c = rng.uniform(5.32, 5.48)
    si_x = rng.uniform(0.45, 0.49)
    ox = rng.uniform(0.40, 0.43)
    oy = rng.uniform(0.25, 0.28)
    oz = rng.uniform(0.10, 0.13)
    return Lattice.hexagonal(a, c), ["Si", "O"], [[si_x, 0.0, 1 / 3], [ox, oy, oz]]


def _wurtzite(rng):  # P6_3mc (186), non-centrosymmetric
    a = rng.uniform(3.74, 3.90)
    c = rng.uniform(6.10, 6.40)
    s_z = rng.uniform(0.36, 0.39)
    return Lattice.hexagonal(a, c), ["Zn", "S"], [[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, s_z]]


def _gaas(rng):  # F-43m (216), non-centrosymmetric
    a = rng.uniform(5.55, 5.80)
    return Lattice.cubic(a), ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]


def _nacl(rng):  # Fm-3m (225), centrosymmetric -> zero piezo
    a = rng.uniform(5.45, 5.85)
    return Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]


def _srtio3(rng):  # Pm-3m (221), centrosymmetric -> zero piezo
    a = rng.uniform(3.85, 4.00)
    return Lattice.cubic(a), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]


def _rutile(rng):  # P4_2/mnm (136), centrosymmetric -> zero piezo
    a = rng.uniform(4.55, 4.65)
    c = rng.uniform(2.93, 3.00)
    ox = rng.uniform(0.30, 0.31)
    return Lattice.tetragonal(a, c), ["Ti", "O"], [[0, 0, 0], [ox, ox, 0]]


PROTOTYPES = [
    ("BaTiO3", 99, _batio3, False),
    ("quartz", 152, _quartz, False),
    ("wurtzite", 186, _wurtzite, False),
    ("GaAs", 216, _gaas, False),
    ("NaCl", 225, _nacl, True),
    ("SrTiO3", 221, _srtio3, True),
    ("rutile", 136, _rutile, True),
]

# Sampling weights (emphasise non-centrosymmetric prototypes).
_WEIGHTS = np.array([3.0, 3.0, 3.0, 2.0, 1.0, 1.0, 1.0])
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def build_structure_pool(n: int, seed: int = 0, symprec: float = 1e-3):
    """Return a list of ``(structure_id, Structure, space_group, centrosymmetric)``.

    ``structure_id`` is a stable string unique per generated crystal.
    """
    rng = np.random.default_rng(seed)
    pool = []
    proto_names = [p[0] for p in PROTOTYPES]
    attempts = 0
    while len(pool) < n and attempts < n * 20:
        attempts += 1
        idx = rng.choice(len(PROTOTYPES), p=_WEIGHTS)
        name, sg, sampler, centro = PROTOTYPES[idx]
        lattice, species, coords = sampler(rng)
        try:
            s = Structure.from_spacegroup(sg, lattice, species, coords)
        except Exception:
            continue
        # Validate the space group is exactly as intended.
        try:
            got = SpacegroupAnalyzer(s, symprec=symprec).get_space_group_number()
        except Exception:
            continue
        if got != sg:
            continue
        sid = f"{name}-{len(pool):04d}"
        pool.append((sid, s, sg, centro))
    if len(pool) < n:
        raise RuntimeError(f"only generated {len(pool)}/{n} valid structures")
    return pool
