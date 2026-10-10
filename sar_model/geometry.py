"""
Модуль расчёта базовых геометрических и кинематических параметров (ТЗ п. 2.3).

Для каждого момента t и цели: R(t), R0(t), угол скоса α(t), V_t, доплеровская частота f_d(t)
с учётом вращения Земли (статья v8).
"""

from datetime import datetime
from typing import Tuple

import numpy as np

from calc_cord import get_xyzv_from_latlon, MFACTOR


# Угловая скорость Земли, рад/с (для V_t = ω_e × R_t)
OMEGA_EARTH_RAD_S = MFACTOR  # 7.292115e-5


def target_velocity_eci(R_t_km: np.ndarray) -> np.ndarray:
    """
    Вектор линейной скорости точки на поверхности Земли из-за вращения: V_t = ω_e × R_t.
    R_t в км, возврат в км/с.
    """
    omega = np.array([0, 0, OMEGA_EARTH_RAD_S])
    return np.cross(omega, R_t_km)


def vector_sat_to_target(R_s: np.ndarray, R_t: np.ndarray) -> np.ndarray:
    """Вектор от КА к цели: R = R_t - R_s (в км)."""
    return np.asarray(R_t) - np.asarray(R_s)


def slant_range_km(R: np.ndarray) -> float:
    """Наклонная дальность R0 = ||R||, км."""
    return float(np.linalg.norm(R))


def squint_angle_rad(R: np.ndarray, V_s: np.ndarray) -> float:
    """
    Угол скоса α (рад): угол между вектором визирования R (от КА к цели) и вектором скорости V_s.
    cos α = (R · V_s) / (R0 * ||V_s||). 90° — траверзная плоскость.
    """
    R0 = np.linalg.norm(R)
    Vs = np.linalg.norm(V_s)
    if R0 < 1e-9 or Vs < 1e-9:
        return np.pi / 2.0
    cos_a = np.clip(np.dot(R, V_s) / (R0 * Vs), -1.0, 1.0)
    return float(np.arccos(cos_a))


def squint_angle_deg(R: np.ndarray, V_s: np.ndarray) -> float:
    """Угол скоса α в градусах."""
    return np.degrees(squint_angle_rad(R, V_s))


def doppler_frequency_hz(
    R_s: np.ndarray,
    V_s: np.ndarray,
    R_t: np.ndarray,
    V_t: np.ndarray,
    wavelength_m: float,
) -> float:
    """
    Доплеровская частота (Гц) с учётом вращения Земли (статья v8):

    f_d = -(2/λ) * (R · (V_s - V_t)) / R0,

    где R = R_t - R_s (вектор от КА к цели), R0 = ||R||.
    R_s, R_t в км; V_s, V_t в км/с; λ в м.
    """
    R_vec = vector_sat_to_target(R_s, R_t)
    R0_km = slant_range_km(R_vec)
    if R0_km < 1e-9:
        return 0.0
    # R0 в м для согласования с λ
    R0_m = R0_km * 1000.0
    V_s_ms = np.asarray(V_s) * 1000.0  # км/с -> м/с
    V_t_ms = np.asarray(V_t) * 1000.0
    R_m = R_vec * 1000.0
    rel_vel = V_s_ms - V_t_ms
    f_d = -2.0 / wavelength_m * np.dot(R_m, rel_vel) / R0_m
    return float(f_d)


def incidence_angle_deg(R_s: np.ndarray, R_t: np.ndarray) -> float:
    """
    Угол падения (визирования): угол между направлением от КА к цели и нормалью к поверхности
    (направление от центра Земли к цели). Градусы.
    """
    R = vector_sat_to_target(R_s, R_t)
    R0 = np.linalg.norm(R)
    Rt_norm = np.linalg.norm(R_t)
    if R0 < 1e-9 or Rt_norm < 1e-9:
        return 0.0
    n_surface = R_t / Rt_norm
    look = R / R0
    cos_inc = np.clip(np.dot(look, n_surface), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_inc)))


def compute_geometry_at_time(
    utc_time: datetime,
    R_s: np.ndarray,
    V_s: np.ndarray,
    target_lat_deg: float,
    target_lon_deg: float,
    target_height_km: float,
    wavelength_m: float = 0.1,
) -> dict:
    """
    Полный набор геометрических и кинематических параметров в один момент времени.

    Returns
    -------
    dict с ключами: R_s, V_s, R_t, V_t, R, R0_km, squint_deg, doppler_hz, incidence_deg.
    """
    pos_t, vel_t = get_xyzv_from_latlon(utc_time, target_lon_deg, target_lat_deg, target_height_km)
    R_t = np.array(pos_t)
    V_t = np.array(vel_t)
    R_s = np.asarray(R_s)
    V_s = np.asarray(V_s)
    R = vector_sat_to_target(R_s, R_t)
    R0_km = slant_range_km(R)
    return {
        "R_s": R_s.copy(),
        "V_s": V_s.copy(),
        "R_t": R_t.copy(),
        "V_t": V_t.copy(),
        "R": R.copy(),
        "R0_km": R0_km,
        "squint_deg": squint_angle_deg(R, V_s),
        "doppler_hz": doppler_frequency_hz(R_s, V_s, R_t, V_t, wavelength_m),
        "incidence_deg": incidence_angle_deg(R_s, R_t),
    }
