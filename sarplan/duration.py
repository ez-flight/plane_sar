"""
Модель длительности наблюдения одного кадра (статья, формулы (1)–(5), (9))
и оценка числа кадров за пролёт (19).
"""

from typing import Dict, Optional

import numpy as np
from scipy.integrate import quad

from .frames import MU, R_MEAN


def orbit_p_e(h_apo_km: float, h_peri_km: float, r_earth: float = R_MEAN):
    """Формула (2): фокальный параметр p и эксцентриситет e."""
    e = (h_apo_km - h_peri_km) / (h_apo_km + h_peri_km + 2 * r_earth)
    p = (r_earth + h_apo_km) * (1 - e)
    return p, e


def _antiderivative(theta: float, e: float) -> float:
    """Первообразная (3) без множителя p^{3/2} μ^{-1/2}."""
    return (e * np.sin(theta) / ((e ** 2 - 1) * (1 + e * np.cos(theta)))
            + 2 * (1 - e ** 2) ** -1.5 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(theta / 2)))


def duration_closed_form(theta0: float, theta1: float, p: float, e: float,
                         mu: float = MU, correct_branch: bool = True) -> float:
    """
    Формула (3). tg(ϑ/2) разрывен при ϑ = π(2m+1): на каждом переходе к
    первообразной прибавляется 2π(1-e²)^{-3/2} (для ϑ₁ ∈ (π, 3π) — один раз,
    как в статье). correct_branch=False воспроизводит ошибку без поправки.
    """
    k = p ** 1.5 * mu ** -0.5

    def f(th):
        val = _antiderivative(th, e)
        if correct_branch:
            n = np.floor((th + np.pi) / (2 * np.pi))
            val += n * 2 * np.pi * (1 - e ** 2) ** -1.5
        return val

    return k * (f(theta1) - f(theta0))


def duration_numeric(theta0: float, theta1: float, p: float, e: float, mu: float = MU) -> float:
    """Формула (1) численным интегрированием."""
    val, _ = quad(lambda th: (1 + e * np.cos(th)) ** -2, theta0, theta1, limit=200)
    return p ** 1.5 * mu ** -0.5 * val


def duration_kepler(theta0: float, theta1: float, p: float, e: float, mu: float = MU) -> float:
    """Через уравнение Кеплера: t = (E - e sin E)/n, с раскруткой витков."""
    a = p / (1 - e ** 2)
    n = np.sqrt(mu / a ** 3)

    def t_of(th):
        rev = np.floor((th + np.pi) / (2 * np.pi))
        thw = th - 2 * np.pi * rev
        ecc_an = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(thw / 2))
        return (ecc_an - e * np.sin(ecc_an) + 2 * np.pi * rev) / n

    return t_of(theta1) - t_of(theta0)


def theta0_from_height(p: float, e: float, h0_km: float, descending: bool = False,
                       r_earth: float = R_MEAN) -> float:
    """Формула (4): истинная аномалия начала съёмки по высоте H₀."""
    r = r_earth + h0_km
    th = float(np.arccos(np.clip((p - r) / (e * r), -1, 1)))
    return 2 * np.pi - th if descending else th


def epsilon_corrected(slant_km: float, roll_deg: float, az_deg: float,
                      r_earth: float = R_MEAN) -> float:
    """
    Формула (5): центральный угол вдоль трассы между КА и проекцией цели
    на плоскость орбиты (исправленный вид), рад.
    """
    lam, gam = np.deg2rad(az_deg), np.deg2rad(roll_deg)
    sin_zeta = slant_km * np.sin(gam) * np.cos(lam) / r_earth
    cos_zeta = np.sqrt(max(1 - sin_zeta ** 2, 1e-12))
    return float(np.arcsin(np.clip(slant_km * np.sin(abs(lam)) / (r_earth * cos_zeta), -1, 1)))


def epsilon_ref4(slant_km: float, roll_deg: float, az_deg: float, r_earth: float = R_MEAN) -> float:
    """ε в виде [4]: arcsin(R sinγ cosλ / R_З) — на деле поперечный угол ζ."""
    lam, gam = np.deg2rad(az_deg), np.deg2rad(roll_deg)
    return float(np.arcsin(np.clip(slant_km * np.sin(gam) * np.cos(lam) / r_earth, -1, 1)))


def duration_circular(eps0: float, eps1: float, length_km: float, alt_km: float,
                      r_earth: float = R_MEAN, mu: float = MU) -> float:
    """(1) для круговой орбиты: t_s = (ϑ₁ - ϑ₀)/ω, ϑ₁ - ϑ₀ = ε₀ + ε₁ + L_p/R_З (4)."""
    omega = np.sqrt(mu / (r_earth + alt_km) ** 3)
    return (eps0 + eps1 + length_km / r_earth) / omega


def n_frames_mechanical(t_s: float, t_turn: float, t_stab: float, t_on: float) -> int:
    """Формула (9): число кадров при механическом перенацеливании."""
    return int(t_s // (t_turn + t_stab + t_on))


def n_frames_max(t_on_max: float, t_frame_mean: float, t_union: float, gap: float) -> int:
    """Формула (19): оценка числа кадров за пролёт при электронном переключении."""
    return int(min(np.floor(t_on_max / t_frame_mean), np.floor((t_union + gap) / (t_frame_mean + gap))))


def verify_closed_form(h_apo_km: float = 1500.0, h_peri_km: float = 500.0) -> Dict[str, float]:
    """
    Проверка (3) против численного интеграла (1) и уравнения Кеплера
    (статья: орбита 500×1500 км, расхождение не более 1e-13).
    """
    p, e = orbit_p_e(h_apo_km, h_peri_km)
    cases = [(0.3, 2.0), (-1.0, 1.0), (2.0, 4.0), (0.5, 3.5)]
    worst = 0.0
    for th0, th1 in cases:
        tn = duration_numeric(th0, th1, p, e)
        tc = duration_closed_form(th0, th1, p, e)
        tk = duration_kepler(th0, th1, p, e)
        worst = max(worst, abs(tc - tn) / tn, abs(tc - tk) / tk)
    th0, th1 = 2.0, 4.0
    return {
        "p_km": p, "e": e,
        "max_rel_error": worst,
        "theta0": th0, "theta1": th1,
        "t_correct_s": duration_closed_form(th0, th1, p, e),
        "t_no_correction_s": duration_closed_form(th0, th1, p, e, correct_branch=False),
        "t_kepler_s": duration_kepler(th0, th1, p, e),
    }
