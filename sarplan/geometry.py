"""
Геометрия гибридного электромеханического режима (статья, формулы (10)–(15)).

Все векторы в ECEF; V — скорость КА относительно вращающейся Земли (V_отн).
Функции векторизованы: допускается любое согласованное по последней оси
вещание форм (..., 3).
"""

from typing import Tuple

import numpy as np

from .frames import R_EQ, R_POL


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def body_axes(r_s: np.ndarray, v_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Формула (10): оси крена (вдоль скорости), рыскания (к центру Земли)
    и тангажа (дополняет тройку до правой).
    """
    e_kr = _unit(v_s)
    e_rys = -_unit(r_s)
    e_tan = _unit(np.cross(e_kr, e_rys))
    return e_kr, e_rys, e_tan


def ray_ellipsoid(r_s: np.ndarray, r_t: np.ndarray, h_km: float = 0.0) -> np.ndarray:
    """
    Формула (11): пересечение луча R(s) = R_S + s·D, D = R_T - R_S, с эллипсоидом
    (радиусы R_e + h, R_p + h). Возвращает центр пятна ДН R_пят (наименьший
    положительный корень) или NaN, если луч не пересекает эллипсоид.
    """
    re2, rp2 = (R_EQ + h_km) ** 2, (R_POL + h_km) ** 2
    d = np.asarray(r_t) - np.asarray(r_s)
    xs, ys, zs = r_s[..., 0], r_s[..., 1], r_s[..., 2]
    a = (d[..., 0] ** 2 + d[..., 1] ** 2) / re2 + d[..., 2] ** 2 / rp2
    b = 2 * ((xs * d[..., 0] + ys * d[..., 1]) / re2 + zs * d[..., 2] / rp2)
    c = (xs ** 2 + ys ** 2) / re2 + zs ** 2 / rp2 - 1.0
    disc = b ** 2 - 4 * a * c
    sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
    s1, s2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    s = np.where(s1 > 0, s1, np.where(s2 > 0, s2, np.nan))
    return r_s + s[..., None] * d


def squint_look(r_s: np.ndarray, v_s: np.ndarray, r_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формула (15): угол скоса β_ск (от плоскости нулевого доплера, + вперёд)
    и угол визирования β_ум (от надира), градусы.
    """
    d = r_t - r_s
    dn = np.linalg.norm(d, axis=-1)
    sin_sk = _dot(d, v_s) / (dn * np.linalg.norm(v_s, axis=-1))
    cos_um = -_dot(r_s, d) / (np.linalg.norm(r_s, axis=-1) * dn)
    return (np.rad2deg(np.arcsin(np.clip(sin_sk, -1, 1))),
            np.rad2deg(np.arccos(np.clip(cos_um, -1, 1))))


def above_horizon(r_s: np.ndarray, r_t: np.ndarray) -> np.ndarray:
    """КА над горизонтом цели: (R_S - R_T)·R_T > 0 (сферическое приближение нормали)."""
    return _dot(r_s - r_t, r_t) > 0


def doppler_hz(squint_deg: np.ndarray, v_rel_km_s: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Формула (15): f_d = 2·|V_отн|·sin β_ск / λ_в."""
    return 2.0 * np.asarray(v_rel_km_s) * 1e3 * np.sin(np.deg2rad(squint_deg)) / wavelength_m


def _oriented_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Угол от a к b (°) со знаком относительно оси axis; |угол| = arccos(â·b̂)."""
    return np.rad2deg(np.arctan2(_dot(np.cross(a, b), axis), _dot(a, b)))


def beam_deflection(r_s: np.ndarray, v_s: np.ndarray, r_point: np.ndarray,
                    r_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формулы (12)–(14): углы электронного отклонения луча от оси антенны,
    направленной на точку наведения R_Tп, к центру кадра R_Tц.

    Δβ_ск — угол между проекциями D_п и D_ц на азимутальную плоскость
    (ортогональную оси тангажа), Δβ_ум — на угломестную (ортогональную оси
    крена). Модуль совпадает с arccos(...) из (13), (14); знак взят как
    ориентированный угол от D_п к D_ц (положительный — вперёд/дальше).
    """
    e_kr, _, e_tan = body_axes(r_s, v_s)
    d_p = r_point - r_s
    d_c = r_target - r_s
    d_p_az = d_p - _dot(d_p, e_tan)[..., None] * e_tan
    d_c_az = d_c - _dot(d_c, e_tan)[..., None] * e_tan
    d_p_um = d_p - _dot(d_p, e_kr)[..., None] * e_kr
    d_c_um = d_c - _dot(d_c, e_kr)[..., None] * e_kr
    d_sk = -_oriented_angle(d_p_az, d_c_az, e_tan)
    d_um = _oriented_angle(d_p_um, d_c_um, e_kr)
    return d_sk, d_um


def roll_azimuth(r_s: np.ndarray, v_s: np.ndarray, r_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Углы в определениях [4] для формулы (5): наклонная дальность R, крен γ
    (в плоскости, ортогональной оси крена, от надира) и отклонение по
    азимуту λ (угол линии визирования с этой плоскостью). Градусы, км.
    """
    e_kr, e_rys, e_tan = body_axes(r_s, v_s)
    d = r_t - r_s
    rng = np.linalg.norm(d, axis=-1)
    lam = np.rad2deg(np.arcsin(np.clip(_dot(d, e_kr) / rng, -1, 1)))
    gam = np.rad2deg(np.arctan2(np.abs(_dot(d, e_tan)), _dot(d, e_rys)))
    return rng, gam, lam
