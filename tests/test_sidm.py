"""Checks of the isothermal-Jeans SIDM model (sidm/jeans.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sidm"))
import jeans as J  # noqa: E402

Z = 0.6


def test_small_x_recovers_cdm():
    # X -> 0: r1 -> 0 and the profile is the CDM halo + stars at all lensing radii
    m200, ms, re = J.galaxy(12.3, Z, 1.5)
    m_cdm, _, m_b = J.halo(m200, Z, ms, re)
    b = J.branch(m200, Z, ms, re, np.geomspace(0.02, 50, 60))
    m3 = J.m3d_at_x(m200, Z, ms, re, 0.02, b)
    r = np.array([0.5, 1.0, 3.0, 10.0])
    assert np.allclose(m3(r), m_cdm(r) + m_b(r), rtol=2e-3)


def test_matching_and_rate_condition():
    # the solution at X matches density and mass at r1 and satisfies rho_CDM(r1) (sigma/m) <v> t = 1
    m200, ms, re = J.galaxy(12.3, Z, 1.5)
    m_cdm, rho_cdm, m_b = J.halo(m200, Z, ms, re)
    b = J.branch(m200, Z, ms, re, np.geomspace(0.02, 50, 60))
    for X in (0.3, 2.0):
        r1, rho0, s0 = J.solve_at_x(m200, Z, ms, re, X, b)
        rho, mm = J.isothermal(rho0, s0, m_b, r1)
        assert abs(rho[0] / rho_cdm(r1) - 1) < 1e-5 and abs(mm[0] / m_cdm(r1) - 1) < 1e-5
        assert abs(J.x_of(s0, rho_cdm(r1)) / X - 1) < 1e-6
        m3 = J.m3d_at_x(m200, Z, ms, re, X, b)
        assert abs(m3(r1 * 1.0001) / m3(r1 * 0.9999) - 1) < 1e-3          # continuous at r1


def test_no_stars_core():
    # without baryons SIDM makes a core: inner dark-matter density below the NFW cusp at small r
    m200, ms, re = J.galaxy(12.3, Z, 1.0)
    m_cdm, rho_cdm, _ = J.halo(m200, Z, 1.0, re)
    b = J.branch(m200, Z, 1.0, re, np.geomspace(0.02, 80, 60))
    r1, rho0, s0 = J.solve_at_x(m200, Z, 1.0, re, 3.0, b)
    assert rho0 < rho_cdm(0.1 * r1) and r1 > 5


def test_beyond_fold_is_flagged():
    m200, ms, re = J.galaxy(11.8, Z, 1.5)
    b = J.branch(m200, Z, ms, re, np.geomspace(0.02, 150, 100))
    assert J.m3d_at_x(m200, Z, ms, re, 1.5 * b[-1, 3], b) is None
