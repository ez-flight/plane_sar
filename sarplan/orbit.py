"""
Орбиты и эфемериды.

Эфемериды считаются моделью SGP4 через pyorbital, как в sar_model.ballistic.
Для гипотетических КА (класс ICEYE, группировки) TLE синтезируются из
элементов орбиты: солнечно-синхронная круговая орбита заданной высоты,
местное время восходящего узла (МВВУ) и фаза.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

import numpy as np
from pyorbital import astronomy
from pyorbital.orbital import Orbital
from sgp4.api import Satrec

from . import geometry
from .frames import MU, R_EQ, dt64_to_datetime, eci_to_ecef, to_dt64

J2_SSO_CONST = 12352.0  # км: cos i = -(a / 12352)^{7/2} для ССО


def tle_checksum(line: str) -> int:
    s = 0
    for ch in line[:68]:
        if ch.isdigit():
            s += int(ch)
        elif ch == "-":
            s += 1
    return s % 10


def mean_motion_rev_day(alt_km: float) -> float:
    a = R_EQ + alt_km
    return np.sqrt(MU / a ** 3) * 86400.0 / (2 * np.pi)


def sso_inclination_deg(alt_km: float) -> float:
    return float(np.rad2deg(np.arccos(-((R_EQ + alt_km) / J2_SSO_CONST) ** 3.5)))


def raan_for_ltan(epoch: datetime, ltan_h: float) -> float:
    """ДВУ (°) для заданного местного времени восходящего узла."""
    ra_sun, _ = astronomy.sun_ra_dec(to_dt64(epoch))
    return float((np.rad2deg(ra_sun) + (ltan_h - 12.0) * 15.0) % 360.0)


def make_tle(satnum: int, epoch: datetime, incl_deg: float, raan_deg: float,
             ecc: float, argp_deg: float, mean_anom_deg: float,
             mean_motion: float) -> tuple:
    """Сформировать две строки TLE (формат NORAD) с контрольными суммами."""
    yy = epoch.year % 100
    doy = (epoch - datetime(epoch.year, 1, 1)).total_seconds() / 86400.0 + 1.0
    epoch_str = f"{yy:02d}{doy:012.8f}"
    l1 = f"1 {satnum:05d}U 26001A   {epoch_str}  .00000000  00000-0  00000-0 0  999"
    ecc_str = f"{int(round(ecc * 1e7)):07d}"
    l2 = (f"2 {satnum:05d} {incl_deg % 180:8.4f} {raan_deg % 360:8.4f} {ecc_str} "
          f"{argp_deg % 360:8.4f} {mean_anom_deg % 360:8.4f} {mean_motion:11.8f}{1:5d}")
    assert len(l1) == 68 and len(l2) == 68, (l1, l2)
    l1 += str(tle_checksum(l1))
    l2 += str(tle_checksum(l2))
    Satrec.twoline2rv(l1, l2)  # проверка формата
    return l1, l2


class Satellite:
    """КА: TLE + распространение SGP4 (pyorbital)."""

    def __init__(self, name: str, line1: str, line2: str):
        self.name = name
        self.line1, self.line2 = line1, line2
        self.orb = Orbital(name, line1=line1, line2=line2)

    def rev_number(self, t) -> int:
        """
        Номер витка (счёт от восходящего узла, как в TLE): номер на эпоху плюс
        число прохождений узла с эпохи (по среднему движению из TLE).
        """
        l1, l2 = self.line1, self.line2
        yy, doy = int(l1[18:20]), float(l1[20:32])
        epoch = np.datetime64(f"{2000 + yy if yy < 57 else 1900 + yy}-01-01T00:00:00", "us") + \
            np.timedelta64(int(round((doy - 1) * 86400e6)), "us")
        argp, m0, n = float(l2[34:42]), float(l2[43:51]), float(l2[52:63])
        rev0 = int(l2[63:68] or 0)
        days = float((to_dt64(t) - epoch) / np.timedelta64(1, "s")) / 86400.0
        return rev0 + int(np.floor((argp + m0) / 360.0 + n * days))

    def states_eci(self, times):
        t = np.atleast_1d(to_dt64(times))
        pos, vel = self.orb.get_position(t, normalize=False)
        return np.asarray(pos).reshape(3, -1).T, np.asarray(vel).reshape(3, -1).T

    def states_ecef(self, times):
        """R_S, V_отн в ECEF (км, км/с), форма (N, 3)."""
        t = np.atleast_1d(to_dt64(times))
        r, v = self.states_eci(t)
        return eci_to_ecef(r, v, t)

    def __repr__(self):
        return f"Satellite({self.name!r})"


def make_sso_satellite(name: str, satnum: int, epoch: datetime, alt_km: float = 560.0,
                       ltan_h: float = 10.5, mean_anom_deg: float = 0.0,
                       incl_deg: Optional[float] = None) -> Satellite:
    incl = sso_inclination_deg(alt_km) if incl_deg is None else incl_deg
    l1, l2 = make_tle(satnum, epoch, incl, raan_for_ltan(epoch, ltan_h), 1e-4, 0.0,
                      mean_anom_deg, mean_motion_rev_day(alt_km))
    return Satellite(name, l1, l2)


def load_tle_file(path: str) -> List[Satellite]:
    """Прочитать файл TLE (трёхстрочный формат, как backup.tle)."""
    lines = [ln.rstrip() for ln in open(path, encoding="utf-8") if ln.strip()]
    sats = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name = lines[i - 1].strip() if i > 0 and not lines[i - 1].startswith(("1 ", "2 ")) else lines[i][2:7]
            sats.append(Satellite(name, lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return sats


@dataclass
class Pass:
    """Пролёт КА над районом: сетка времени с шагом dt и состояние КА в ECEF."""
    sat_name: str
    t_ref: np.datetime64          # момент t = 0
    t: np.ndarray                 # с от t_ref
    r: np.ndarray                 # (T, 3) R_S, км
    v: np.ndarray                 # (T, 3) V_отн, км/с
    index: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else 0.5

    @property
    def start(self) -> np.datetime64:
        return self.t_ref + np.timedelta64(int(self.t[0] * 1e6), "us")

    def time_at(self, i: int) -> np.datetime64:
        return self.t_ref + np.timedelta64(int(round(self.t[i] * 1e6)), "us")

    def hours_since(self, t0) -> float:
        return float((self.start - to_dt64(t0)) / np.timedelta64(1, "s")) / 3600.0

    @property
    def ascending(self) -> bool:
        return bool(self.v[len(self.v) // 2, 2] > 0)


def find_passes(sat: Satellite, center_ecef: np.ndarray, t_start, t_end, *,
                look_min_deg: float, look_max_deg: float, squint_max_deg: float,
                look_margin_deg: float = 10.0, squint_margin_deg: float = 8.0,
                coarse_step_s: float = 20.0, dt_s: float = 0.5) -> List[Pass]:
    """
    Пролёты, в которых центр района попадает в полосу углов визирования
    [look_min - margin, look_max + margin] при |β_ск| <= squint_max + margin.
    Запасы учитывают размер района (кадры вокруг центра).
    """
    t0 = to_dt64(t_start)
    total = float((to_dt64(t_end) - t0) / np.timedelta64(1, "s"))
    tc = np.arange(0.0, total + coarse_step_s, coarse_step_s)
    times = t0 + (tc * 1e6).astype("timedelta64[us]")
    out: List[Pass] = []
    chunk = 20000
    ok_all = np.zeros(len(tc), bool)
    for a in range(0, len(tc), chunk):
        r, v = sat.states_ecef(times[a:a + chunk])
        sq, look = geometry.squint_look(r, v, center_ecef[None, :])
        ok_all[a:a + chunk] = (geometry.above_horizon(r, center_ecef[None, :])
                               & (np.abs(sq) <= squint_max_deg + squint_margin_deg)
                               & (look >= look_min_deg - look_margin_deg)
                               & (look <= look_max_deg + look_margin_deg))
    idx = np.flatnonzero(ok_all)
    if len(idx) == 0:
        return out
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for g in groups:
        a = tc[g[0]] - coarse_step_s
        b = tc[g[-1]] + coarse_step_s
        tf = np.arange(a, b + dt_s / 2, dt_s)
        tt = t0 + (tf * 1e6).astype("timedelta64[us]")
        r, v = sat.states_ecef(tt)
        sq, look = geometry.squint_look(r, v, center_ecef[None, :])
        okf = (geometry.above_horizon(r, center_ecef[None, :]) & (np.abs(sq) <= squint_max_deg + squint_margin_deg)
               & (look >= look_min_deg - look_margin_deg) & (look <= look_max_deg + look_margin_deg))
        sel = np.flatnonzero(okf)
        if len(sel) < 4:
            continue
        i0, i1 = sel[0], sel[-1] + 1
        t_ref = tt[i0]
        out.append(Pass(sat.name, t_ref, tf[i0:i1] - tf[i0], r[i0:i1], v[i0:i1], index=len(out),
                        meta={"min_look_deg": float(look[i0:i1].min())}))
    return out
