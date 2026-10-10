"""
Модуль анализа временных окон наблюдения (ТЗ п. 2.4).

Сканирование временного интервала, определение окон, в которых выполняются условия:
- угол визирования (угол падения) в рабочем диапазоне (например 24°–55°);
- угол скоса α в пределах сектора электронного сканирования (±4.1° от траверза);
- наклонная дальность R0 в допустимых пределах.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
from pyorbital.orbital import Orbital

from calc_cord import get_xyzv_from_latlon, EARTH_EQUATORIAL_RADIUS
from . import ballistic
from . import geometry


def is_visible(R_s: np.ndarray, R_t: np.ndarray) -> bool:
    """Проверка видимости цели с КА: прямая КА–цель не пересекает Землю."""
    vec_sat = np.asarray(R_s)
    vec_tgt = np.asarray(R_t)
    vec_dir = vec_tgt - vec_sat
    dist = np.linalg.norm(vec_dir)
    if dist < 1e-6:
        return True
    vec_dir_unit = vec_dir / dist
    t = -np.dot(vec_sat, vec_dir_unit)
    if t < 0 or t > dist:
        return True
    closest = vec_sat + t * vec_dir_unit
    return np.linalg.norm(closest) >= EARTH_EQUATORIAL_RADIUS


def find_observation_windows(
    orb: Orbital,
    target_lat_deg: float,
    target_lon_deg: float,
    target_height_km: float,
    t_start: datetime,
    t_end: datetime,
    *,
    time_step_s: float = 10.0,
    R0_min_km: float = 561.0,
    R0_max_km: float = 964.0,
    incidence_min_deg: float = 24.0,
    incidence_max_deg: float = 55.0,
    squint_half_angle_deg: float = 4.1,
    min_window_duration_s: float = 30.0,
    wavelength_m: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Найти временные окна наблюдения для одной цели.

    Условия окна (ТЗ 2.4):
    - угол визирования (угол падения) в [incidence_min_deg, incidence_max_deg];
    - угол скоса α в пределах ±squint_half_angle_deg от 90° (т.е. 90 ± 4.1°);
    - R0 в [R0_min_km, R0_max_km];
    - цель видима с КА.

    Returns
    -------
    Список словарей, по одному на окно:
    - start_time, end_time : datetime
    - duration_s : float
    - R0_mean_km, R0_min_km, R0_max_km
    - squint_mean_deg, squint_min_deg, squint_max_deg
    - incidence_mean_deg
    - samples : список моментов времени в окне (опционально, для детального отчёта)
    """
    angle_min = 90.0 - squint_half_angle_deg
    angle_max = 90.0 + squint_half_angle_deg
    windows_out = []
    t = t_start
    window_start = None
    window_samples = []

    while t <= t_end:
        R_s, V_s = ballistic.get_position_velocity(orb, t)
        R_s = np.array(R_s)
        V_s = np.array(V_s)
        pos_t, vel_t = get_xyzv_from_latlon(t, target_lon_deg, target_lat_deg, target_height_km)
        R_t = np.array(pos_t)
        if not is_visible(R_s, R_t):
            if window_start is not None:
                _append_window(windows_out, window_start, window_samples, orb,
                    target_lat_deg, target_lon_deg, target_height_km, wavelength_m,
                    min_window_duration_s)
                window_start = None
                window_samples = []
            t += timedelta(seconds=time_step_s)
            continue
        R = geometry.vector_sat_to_target(R_s, R_t)
        R0 = geometry.slant_range_km(R)
        squint = geometry.squint_angle_deg(R, V_s)
        inc = geometry.incidence_angle_deg(R_s, R_t)
        ok = (
            R0_min_km <= R0 <= R0_max_km
            and angle_min <= squint <= angle_max
            and incidence_min_deg <= inc <= incidence_max_deg
        )
        if ok:
            if window_start is None:
                window_start = t
            window_samples.append({
                "time": t,
                "R0_km": R0,
                "squint_deg": squint,
                "incidence_deg": inc,
            })
        else:
            if window_start is not None:
                _append_window(windows_out, window_start, window_samples, orb,
                    target_lat_deg, target_lon_deg, target_height_km, wavelength_m,
                    min_window_duration_s)
                window_start = None
                window_samples = []
        t += timedelta(seconds=time_step_s)

    if window_start is not None:
        _append_window(windows_out, window_start, window_samples, orb,
            target_lat_deg, target_lon_deg, target_height_km, wavelength_m,
            min_window_duration_s)
    return windows_out


def _append_window(
    windows_out: List[Dict],
    window_start: datetime,
    samples: List[Dict],
    orb: Orbital,
    target_lat_deg: float,
    target_lon_deg: float,
    target_height_km: float,
    wavelength_m: float,
    min_duration_s: float,
) -> None:
    if not samples:
        return
    t_end = samples[-1]["time"]
    duration_s = (t_end - window_start).total_seconds()
    if duration_s < min_duration_s:
        return
    R0_list = [s["R0_km"] for s in samples]
    squint_list = [s["squint_deg"] for s in samples]
    inc_list = [s["incidence_deg"] for s in samples]
    windows_out.append({
        "start_time": window_start,
        "end_time": t_end,
        "duration_s": duration_s,
        "R0_mean_km": float(np.mean(R0_list)),
        "R0_min_km": float(np.min(R0_list)),
        "R0_max_km": float(np.max(R0_list)),
        "squint_mean_deg": float(np.mean(squint_list)),
        "squint_min_deg": float(np.min(squint_list)),
        "squint_max_deg": float(np.max(squint_list)),
        "incidence_mean_deg": float(np.mean(inc_list)),
        "n_samples": len(samples),
    })
