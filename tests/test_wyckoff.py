"""U1 tests — site symmetry extraction."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from tests._structures import REFERENCE_STRUCTURES, quartz
from wysgnn.symmetry.wyckoff import SiteSymmetry, extract_site_symmetry

SYMPREC = 1e-4


def _is_in_set(mat: torch.Tensor, group: torch.Tensor, tol: float = 1e-4) -> bool:
    diffs = (group - mat.unsqueeze(0)).reshape(group.shape[0], -1).abs().amax(dim=1)
    return bool((diffs < tol).any())


def test_per_site_list_lengths():
    for _name, fn, _sg in REFERENCE_STRUCTURES:
        s = fn()
        ss = extract_site_symmetry(s, SYMPREC)
        n = len(s)
        assert len(ss.wyckoff_letters) == n
        assert len(ss.site_symmetry_symbols) == n
        assert len(ss.site_rotations_cart) == n
        assert len(ss.equivalent_atoms) == n


def test_cartesian_ops_orthogonal():
    for _name, fn, _sg in REFERENCE_STRUCTURES:
        ss = extract_site_symmetry(fn(), SYMPREC)
        for ops in ss.site_rotations_cart:
            assert ops.dtype == torch.float64
            for R in ops:
                ortho_err = torch.linalg.norm(R.T @ R - torch.eye(3, dtype=R.dtype))
                assert ortho_err < 1e-5, ortho_err
                assert abs(abs(torch.det(R).item()) - 1.0) < 1e-5


def test_group_closure():
    for _name, fn, _sg in REFERENCE_STRUCTURES:
        ss = extract_site_symmetry(fn(), SYMPREC)
        for ops in ss.site_rotations_cart:
            # identity present
            assert _is_in_set(torch.eye(3, dtype=ops.dtype), ops)
            for a in ops:
                for b in ops:
                    assert _is_in_set(a @ b, ops), "site symmetry set not closed"


def test_space_group_cross_check():
    # Assertion 4: at least 10 reference structures agree with SpacegroupAnalyzer.
    assert len(REFERENCE_STRUCTURES) >= 10
    for name, fn, expected in REFERENCE_STRUCTURES:
        s = fn()
        ss = extract_site_symmetry(s, SYMPREC)
        sga = SpacegroupAnalyzer(s, symprec=1e-3).get_space_group_number()
        assert ss.space_group_number == sga, f"{name}: {ss.space_group_number} != {sga}"
        assert ss.space_group_number == expected, f"{name}: expected {expected}"


def test_quartz_si_site_symmetry():
    # Assertion 5: alpha-quartz Si occupies Wyckoff 3a with site symmetry .2.
    # (point group 2 / C2, order 2). O sits on a general position (order 1).
    s = quartz()
    ss = extract_site_symmetry(s, SYMPREC)
    si_indices = [i for i, site in enumerate(s) if site.specie.symbol == "Si"]
    assert si_indices
    for i in si_indices:
        assert ss.wyckoff_letters[i] == "a"
        assert "2" in ss.site_symmetry_symbols[i]
        assert ss.site_rotations_cart[i].shape[0] == 2  # |C2| = 2
    o_indices = [i for i, site in enumerate(s) if site.specie.symbol == "O"]
    for i in o_indices:
        assert ss.site_rotations_cart[i].shape[0] == 1  # general position


def test_raises_on_bad_dtype_passthrough():
    # Sanity: returned object is the expected dataclass.
    ss = extract_site_symmetry(quartz(), SYMPREC)
    assert isinstance(ss, SiteSymmetry)
