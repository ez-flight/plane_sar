"""
Модуль баллистического прогнозирования (ТЗ п. 2.1).

Расчёт эфемерид КА по модели SGP4 на основе TLE.
Прогноз положения R_s(t) и скорости V_s(t) в инерциальной системе (ECI) с заданным шагом.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from pyorbital.orbital import Orbital


def get_position_velocity(orb: Orbital, utc_time: datetime) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Положение и скорость КА в заданный момент времени в ECI (геоцентрическая инерциальная).

    Returns
    -------
    R_s : (X, Y, Z) в км
    V_s : (Vx, Vy, Vz) в км/с
    """
    R_s, V_s = orb.get_position(utc_time, False)
    return tuple(R_s), tuple(V_s)


def ephemeris(
    orb: Orbital,
    t_start: datetime,
    t_end: datetime,
    step_s: float = 10.0,
) -> List[Tuple[datetime, np.ndarray, np.ndarray]]:
    """
    Эфемериды КА: список (t, R_s, V_s) с заданным временным шагом.

    Parameters
    ----------
    orb : Orbital
        Объект орбиты (pyorbital) на основе TLE.
    t_start, t_end : datetime
        Начало и конец интервала (UTC).
    step_s : float
        Шаг по времени, с.

    Returns
    -------
    list of (datetime, R_s_km, V_s_km_s)
        R_s_km, V_s_km_s — numpy arrays shape (3,) в км и км/с.
    """
    out = []
    t = t_start
    while t <= t_end:
        R_s, V_s = get_position_velocity(orb, t)
        out.append((t, np.array(R_s), np.array(V_s)))
        t += timedelta(seconds=step_s)
    return out
