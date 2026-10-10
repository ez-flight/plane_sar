"""
Модуль расчёта параметров синтеза апертуры (ТЗ п. 2.5).

Для выбранного окна и целевого разрешения ρ_x: T_c, L_c, Δf_d, PRF_min, ΔR_max (RCM), L_az.
"""

from typing import Dict, Any

import numpy as np

# Скорость света, м/с
C_M_S = 2.99792458e8


def synthesis_parameters(
    R0_km: float,
    V_s_km_s: float,
    wavelength_m: float,
    rho_x_m: float,
    beamwidth_deg: float,
    bandwidth_hz: float,
) -> Dict[str, Any]:
    """
    Параметры синтеза апертуры для одного набора (R0, V_s) в середине окна.

    Parameters
    ----------
    R0_km : float
        Наклонная дальность, км.
    V_s_km_s : float
        Модуль скорости КА, км/с.
    wavelength_m : float
        Длина волны, м.
    rho_x_m : float
        Целевое азимутальное разрешение, м.
    beamwidth_deg : float
        Ширина ДН по азимуту, град (для L_az).
    bandwidth_hz : float
        Ширина полосы сигнала B, Гц.

    Returns
    -------
    dict : T_c_s, L_c_km, delta_f_d_hz, PRF_min_hz, delta_R_max_m, range_cell_m, rcm_critical, L_az_km.
    """
    R0_m = R0_km * 1000.0
    V_s_m_s = V_s_km_s * 1000.0
    # Время синтеза: T_c ≈ R0*λ/(2*V_s*ρ_x)
    T_c_s = (R0_m * wavelength_m) / (2.0 * V_s_m_s * rho_x_m)
    # Длина синтезированной апертуры: L_c = V_s*T_c
    L_c_m = V_s_m_s * T_c_s
    L_c_km = L_c_m / 1000.0
    # Доплеровская полоса: Δf_d = 2*V_s²*T_c/(λ*R0) или Δf_d ≈ V_s/ρ_x
    delta_f_d_hz = (2.0 * V_s_m_s ** 2 * T_c_s) / (wavelength_m * R0_m)
    # Минимальная PRF по Найквисту
    PRF_min_hz = delta_f_d_hz  # строго PRF > Δf_d для однозначности
    # Миграция по дальности: ΔR_max ≈ λ²*R0*Δf_d²/(8*V_s²)
    delta_R_max_m = (wavelength_m ** 2 * R0_m * delta_f_d_hz ** 2) / (8.0 * V_s_m_s ** 2)
    # Ширина дальностного канала
    range_cell_m = C_M_S / (2.0 * bandwidth_hz)
    rcm_critical = delta_R_max_m > range_cell_m
    # Размер кадра по азимуту: L_az = R0*Θ_az (Θ_az в радианах)
    theta_az_rad = np.deg2rad(beamwidth_deg)
    L_az_km = R0_km * theta_az_rad
    return {
        "T_c_s": T_c_s,
        "L_c_km": L_c_km,
        "delta_f_d_hz": delta_f_d_hz,
        "PRF_min_hz": PRF_min_hz,
        "delta_R_max_m": delta_R_max_m,
        "range_cell_m": range_cell_m,
        "rcm_critical": rcm_critical,
        "L_az_km": L_az_km,
    }
