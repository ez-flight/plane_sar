"""
Базовые тесты модулей комплекса SAR (ТЗ п. 5 — проверка преобразований и формул).
Запуск: python -m unittest tests.test_sar_model
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sar_model import geometry, synthesis


class TestGeometry(unittest.TestCase):
    def test_squint_broadside(self):
        """Угол скоса 90° когда вектор визирования перпендикулярен скорости."""
        V_s = np.array([1.0, 0.0, 0.0])
        R = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(geometry.squint_angle_deg(R, V_s), 90.0, places=5)

    def test_squint_along_velocity(self):
        """Угол скоса 0° когда R совпадает с направлением скорости."""
        V_s = np.array([1.0, 0.0, 0.0])
        R = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(geometry.squint_angle_deg(R, V_s), 0.0, places=5)

    def test_slant_range(self):
        """Наклонная дальность = норма вектора."""
        R = np.array([3.0, 4.0, 0.0])
        self.assertAlmostEqual(geometry.slant_range_km(R), 5.0, places=9)


class TestSynthesis(unittest.TestCase):
    def test_synthesis_consistency(self):
        """Связь T_c, L_c и Δf_d: L_c = V_s*T_c."""
        R0_km = 700.0
        V_s_km_s = 7.0
        wavelength_m = 0.1
        rho_x_m = 0.75
        beamwidth_deg = 1.0
        B_hz = 300e6
        out = synthesis.synthesis_parameters(
            R0_km, V_s_km_s, wavelength_m, rho_x_m, beamwidth_deg, B_hz
        )
        T_c = out["T_c_s"]
        L_c_m = out["L_c_km"] * 1000.0
        V_s_m_s = V_s_km_s * 1000.0
        self.assertAlmostEqual(L_c_m, V_s_m_s * T_c, places=0)
        R0_m = R0_km * 1000.0
        delta_f_d_expected = (2.0 * V_s_m_s ** 2 * T_c) / (wavelength_m * R0_m)
        self.assertAlmostEqual(out["delta_f_d_hz"], delta_f_d_expected, places=0)


if __name__ == "__main__":
    unittest.main()
