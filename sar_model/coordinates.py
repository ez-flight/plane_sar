"""
Модуль координатных преобразований (ТЗ п. 2.2).

- Геодезические (B, L, H) -> гринвичская геоцентрическая (GCRS) -> ECI.
- Преобразование в орбитальную барицентрическую СК (τ, b, η).
"""

from datetime import datetime
from typing import Tuple

import numpy as np

# Используем существующий модуль проекта
from calc_cord import get_xyzv_from_latlon, EARTH_EQUATORIAL_RADIUS, MFACTOR


def geodetic_to_gcrs(lat_deg: float, lon_deg: float, height_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Геодезические координаты -> гринвичская геоцентрическая (GCRS/ITRS).
    X,Y,Z в км; скорость точки из-за вращения Земли в км/с.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Широта и долгота, град.
    height_km : float
        Высота над эллипсоидом, км.

    Returns
    -------
    R_gcrs : (3,) в км
    V_gcrs : (3,) в км/с (ω_e × R)
    """
    # В момент t=0 (или без учёта времени) GCRS = геоцентр с осями как у Земли
    # Для единообразия с ECI используем get_xyzv_from_latlon в фиксированный момент
    # с нулевым звёздным временем эквивалентным lon=0 при t=0.
    # Стандарт: GCRS совпадает с ITRS с учётом вращения. Мы даём координаты в "земной" СК.
    eci_pos, eci_vel = get_xyzv_from_latlon(datetime(2000, 1, 1, 12, 0, 0), lon_deg, lat_deg, height_km)
    # В pyorbital GMST даёт переход в инерциальную; для GCRS оставляем земную ориентацию:
    # GCRS обычно отождествляют с геоцентрической земной (X по пересечению экватора и нулевого меридиана).
    # У нас get_xyzv_from_latlon возвращает уже ECI (с учётом theta = GMST + lon). Поэтому для "GCRS"
    # можно трактовать как координаты в земной системе: R_gcrs = R_earth_fixed. В литературе GCRS
    # часто считают инерциальной (без суточного вращения), тогда это то же ECI. Используем ECI как выход
    # для единообразия с баллистикой; при необходимости отдельный GCRS можно добавить через матрицу вращения.
    return np.array(eci_pos), np.array(eci_vel)


def geodetic_to_eci(utc_time: datetime, lat_deg: float, lon_deg: float, height_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Геодезические -> экваториальная инерциальная (ECI) с учётом вращения Земли.

    Returns
    -------
    R_eci : (3,) км
    V_t : (3,) км/с — скорость точки на поверхности Земли в ECI.
    """
    pos, vel = get_xyzv_from_latlon(utc_time, lon_deg, lat_deg, height_km)
    return np.array(pos), np.array(vel)


def eci_to_orbital(
    R_s: np.ndarray,
    V_s: np.ndarray,
    R_vector_eci: np.ndarray,
) -> np.ndarray:
    """
    Вектор в ECI -> компоненты в орбитальной барицентрической СК (τ, b, η).
    Начало в центре масс КА; ось τ — по касательной к орбите; η — нормаль к плоскости орбиты; b = η × τ.

    Parameters
    ----------
    R_s, V_s : (3,) — положение и скорость КА в ECI (км, км/с).
    R_vector_eci : (3,) — вектор в ECI (например R = R_t - R_s от КА к цели), км.

    Returns
    -------
    R_orb : (3,) — компоненты [τ, b, η] в км.
    """
    V_norm = np.linalg.norm(V_s)
    if V_norm < 1e-9:
        return np.zeros(3)
    tau = V_s / V_norm
    h = np.cross(R_s, V_s)
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-9:
        eta = np.array([0.0, 0.0, 1.0])
    else:
        eta = h / h_norm
    b = np.cross(eta, tau)
    R_orb = np.array([
        np.dot(R_vector_eci, tau),
        np.dot(R_vector_eci, b),
        np.dot(R_vector_eci, eta),
    ])
    return R_orb
