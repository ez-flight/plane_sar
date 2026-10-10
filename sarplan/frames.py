"""
Системы координат.

Внутри sarplan вся геометрия считается в гринвичской (земной) СК ECEF:
цели и точки наведения в ней неподвижны, а скорость КА берётся относительно
вращающейся Земли, V_отн = V_S - [ω_e × R_S] (статья, формула (15)).
Эфемериды pyorbital выдаются в ECI и переводятся в ECEF поворотом на
звёздное время, как в calc_cord.get_xyzv_from_latlon.
"""

from datetime import datetime
from typing import Tuple

import numpy as np
from pyorbital import astronomy

from calc_cord import EARTH_ECCENTRICITY_SQ, EARTH_EQUATORIAL_RADIUS, MFACTOR

R_EQ = EARTH_EQUATORIAL_RADIUS                 # R_e, км
E2 = EARTH_ECCENTRICITY_SQ
R_POL = R_EQ * np.sqrt(1.0 - E2)               # R_p, км
OMEGA_E = MFACTOR                              # ω_e, рад/с
MU = 398600.4418                               # μ, км³/с²
R_MEAN = 6371.0                                # R_З (сферическая Земля), км


def to_dt64(t) -> np.ndarray:
    """datetime / строка / массив -> numpy datetime64[us]."""
    if isinstance(t, datetime):
        t = t.replace(tzinfo=None)
    return np.asarray(t, dtype="datetime64[us]")


def dt64_to_datetime(t: np.datetime64) -> datetime:
    return t.astype("datetime64[us]").astype(datetime)


def gmst(times) -> np.ndarray:
    """Гринвичское среднее звёздное время, рад."""
    return np.asarray(astronomy.gmst(to_dt64(times)))


def eci_to_ecef(r_eci: np.ndarray, v_eci: np.ndarray, times) -> Tuple[np.ndarray, np.ndarray]:
    """
    ECI -> ECEF. r, v: (N, 3) в км и км/с.
    Возвращает R_S и V_отн = V_S - ω_e × R_S, оба в осях ECEF.
    """
    r = np.atleast_2d(r_eci)
    v = np.atleast_2d(v_eci)
    th = np.atleast_1d(gmst(times))
    c, s = np.cos(th), np.sin(th)
    v_rel = np.column_stack([v[:, 0] + OMEGA_E * r[:, 1], v[:, 1] - OMEGA_E * r[:, 0], v[:, 2]])
    r_e = np.column_stack([c * r[:, 0] + s * r[:, 1], -s * r[:, 0] + c * r[:, 1], r[:, 2]])
    v_e = np.column_stack([c * v_rel[:, 0] + s * v_rel[:, 1], -s * v_rel[:, 0] + c * v_rel[:, 1], v_rel[:, 2]])
    return r_e, v_e


def ecef_to_eci(r_ecef: np.ndarray, times) -> np.ndarray:
    """ECEF -> ECI (положение), для сверки с calc_cord."""
    r = np.atleast_2d(r_ecef)
    th = np.atleast_1d(gmst(times))
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1], r[:, 2]])


def geodetic_to_ecef(lat_deg, lon_deg, h_km=0.0) -> np.ndarray:
    """Геодезические (WGS-84) -> ECEF, км. Векторизовано, форма (..., 3)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    h = np.asarray(h_km, dtype=float)
    n = R_EQ / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - E2) + h) * np.sin(lat)
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ecef_to_geodetic(r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF -> (широта°, долгота°, высота км), итерационно (как calc_cord.get_lonlatalt)."""
    r = np.asarray(r, dtype=float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - E2))
    for _ in range(10):
        n = R_EQ / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - E2 * n / (n + h)))
    n = R_EQ / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.rad2deg(lat), np.rad2deg(lon), h


def enu_axes(lat_deg, lon_deg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Орты восток, север, вверх (нормаль к эллипсоиду) в ECEF."""
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    e = np.array([-np.sin(lon), np.cos(lon), 0.0])
    n = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    u = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return e, n, u


def offset_point(lat_deg: float, lon_deg: float, azimuth_deg: float,
                 along_km, across_km=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Точка, смещённая на along_km по азимуту azimuth_deg (от севера по часовой)
    и на across_km вправо от этого направления. Смещение в касательной
    плоскости с проекцией обратно на эллипсоид (для расстояний до ~200 км).
    """
    e, n, _ = enu_axes(lat_deg, lon_deg)
    az = np.deg2rad(azimuth_deg)
    along = np.asarray(along_km, dtype=float)[..., None]
    across = np.asarray(across_km, dtype=float)[..., None]
    fwd = np.cos(az) * n + np.sin(az) * e
    right = -np.sin(az) * n + np.cos(az) * e
    r = geodetic_to_ecef(lat_deg, lon_deg, 0.0) + along * fwd + across * right
    lat, lon, _ = ecef_to_geodetic(r)
    return lat, lon


def azimuth_between(lat1, lon1, lat2, lon2) -> float:
    """Начальный азимут (°) с точки 1 на точку 2 (сфера)."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Расстояние по дуге большого круга, км."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_MEAN * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
