"""Canonical reference crystal structures for tests.

Each builder returns a :class:`pymatgen.core.Structure` generated from its
space group and asymmetric unit via :meth:`Structure.from_spacegroup`, so the
intended space group is encoded by construction and recoverable by spglib.
"""

from __future__ import annotations

from pymatgen.core import Lattice, Structure


def nacl() -> Structure:  # Fm-3m, 225
    lat = Lattice.cubic(5.6402)
    return Structure.from_spacegroup(225, lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def cscl() -> Structure:  # Pm-3m, 221
    lat = Lattice.cubic(4.123)
    return Structure.from_spacegroup(221, lat, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def diamond_si() -> Structure:  # Fd-3m, 227
    lat = Lattice.cubic(5.4307)
    return Structure.from_spacegroup(227, lat, ["Si"], [[0, 0, 0]])


def gaas() -> Structure:  # F-43m, 216
    lat = Lattice.cubic(5.653)
    return Structure.from_spacegroup(216, lat, ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def fluorite_caf2() -> Structure:  # Fm-3m, 225
    lat = Lattice.cubic(5.4626)
    return Structure.from_spacegroup(225, lat, ["Ca", "F"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def srtio3() -> Structure:  # Pm-3m, 221 (cubic perovskite)
    lat = Lattice.cubic(3.905)
    return Structure.from_spacegroup(
        221, lat, ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
    )


def rutile_tio2() -> Structure:  # P4_2/mnm, 136
    lat = Lattice.tetragonal(4.5937, 2.9587)
    return Structure.from_spacegroup(
        136, lat, ["Ti", "O"], [[0, 0, 0], [0.3053, 0.3053, 0]]
    )


def wurtzite_zns() -> Structure:  # P6_3mc, 186
    lat = Lattice.hexagonal(3.8227, 6.2607)
    return Structure.from_spacegroup(
        186, lat, ["Zn", "S"], [[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, 0.375]]
    )


def quartz() -> Structure:  # P3_121, 152 (alpha-quartz)
    lat = Lattice.hexagonal(4.9134, 5.4052)
    return Structure.from_spacegroup(
        152, lat, ["Si", "O"], [[0.4697, 0.0, 1 / 3], [0.4135, 0.2669, 0.1191]]
    )


def batio3_tetragonal() -> Structure:  # P4mm, 99 (ferroelectric)
    lat = Lattice.tetragonal(3.9998, 4.0180)
    return Structure.from_spacegroup(
        99,
        lat,
        ["Ba", "Ti", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.512], [0.5, 0.5, 0.023], [0.5, 0, 0.486]],
    )


def corundum_al2o3() -> Structure:  # R-3c, 167
    lat = Lattice.hexagonal(4.7607, 12.9947)
    return Structure.from_spacegroup(
        167, lat, ["Al", "O"], [[0, 0, 0.3522], [0.3064, 0, 0.25]]
    )


def anatase_tio2() -> Structure:  # I4_1/amd, 141
    lat = Lattice.tetragonal(3.7845, 9.5143)
    return Structure.from_spacegroup(
        141, lat, ["Ti", "O"], [[0, 0, 0], [0, 0, 0.2081]]
    )


# (builder, expected space-group number)
REFERENCE_STRUCTURES: list[tuple[str, "callable", int]] = [
    ("NaCl", nacl, 225),
    ("CsCl", cscl, 221),
    ("diamond-Si", diamond_si, 227),
    ("GaAs", gaas, 216),
    ("CaF2", fluorite_caf2, 225),
    ("SrTiO3", srtio3, 221),
    ("rutile-TiO2", rutile_tio2, 136),
    ("wurtzite-ZnS", wurtzite_zns, 186),
    ("quartz", quartz, 152),
    ("BaTiO3", batio3_tetragonal, 99),
    ("corundum-Al2O3", corundum_al2o3, 167),
    ("anatase-TiO2", anatase_tio2, 141),
]
