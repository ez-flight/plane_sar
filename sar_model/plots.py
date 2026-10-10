"""
Модуль визуализации (ТЗ п. 2.6, рис. 5–7 статьи).

Графики зависимости ключевых параметров от времени: R0(t), α(t), f_d(t) для выбранного окна/прохода.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyorbital.orbital import Orbital

from . import ballistic
from . import geometry


def time_series_for_window(
    orb: Orbital,
    target_lat_deg: float,
    target_lon_deg: float,
    target_height_km: float,
    t_start: datetime,
    t_end: datetime,
    step_s: float = 1.0,
    wavelength_m: float = 0.1,
) -> List[Dict]:
    """Временные ряды R0, squint, doppler, incidence в пределах окна."""
    out = []
    t = t_start
    while t <= t_end:
        R_s, V_s = ballistic.get_position_velocity(orb, t)
        R_s = np.array(R_s)
        V_s = np.array(V_s)
        g = geometry.compute_geometry_at_time(
            t, R_s, V_s, target_lat_deg, target_lon_deg, target_height_km, wavelength_m
        )
        out.append({
            "time": t,
            "R0_km": g["R0_km"],
            "squint_deg": g["squint_deg"],
            "incidence_deg": g["incidence_deg"],
            "doppler_hz": g["doppler_hz"],
        })
        t += timedelta(seconds=step_s)
    return out


def plot_window_parameters(
    series: List[Dict],
    out_path: str,
    title_prefix: str = "",
) -> None:
    """
    Построить графики R0(t), α(t), f_d(t) и сохранить в файл.
    """
    if not series:
        return
    times = [s["time"] for s in series]
    R0 = [s["R0_km"] for s in series]
    squint = [s["squint_deg"] for s in series]
    doppler = [s["doppler_hz"] for s in series]
    # Относительное время в секундах от начала
    t0 = times[0]
    x = [(t - t0).total_seconds() for t in times]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title_prefix + " Параметры съёмки в окне наблюдения", fontsize=12)
    axes[0].plot(x, R0, "b-")
    axes[0].set_ylabel("R₀, км")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, squint, "g-")
    axes[1].set_ylabel("Угол скоса α, °")
    axes[1].axhline(90, color="gray", ls="--", alpha=0.7)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(x, doppler, "r-")
    axes[2].set_ylabel("Доплеровская частота, Гц")
    axes[2].set_xlabel("Время от начала окна, с")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_first_window(
    orb: Orbital,
    target_lat_deg: float,
    target_lon_deg: float,
    target_height_km: float,
    window: Dict,
    wavelength_m: float,
    out_path: str,
) -> None:
    """Построить графики для первого (или заданного) окна и сохранить."""
    series = time_series_for_window(
        orb, target_lat_deg, target_lon_deg, target_height_km,
        window["start_time"], window["end_time"],
        step_s=1.0, wavelength_m=wavelength_m,
    )
    plot_window_parameters(series, out_path, title_prefix="")
