"""
Группировки малых КА в ССО и распределение кадров по пролётам.

Кадры распределяются по пролётам всех КА в порядке времени: на каждом
пролёте методика выбирает подмножество оставшихся кадров.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import SensorParams
from .criterion import CoverageModel, Element, Frame
from .frames import geodetic_to_ecef
from .orbit import Pass, Satellite, find_passes, make_sso_satellite
from .planner import PlanParams, plan_pass
from .windows import PassGeometry

PLANE_LTANS = (10.5, 14.5, 6.0, 18.0)   # МВВУ плоскостей: 10:30, 14:30, 6:00, 18:00


def build_constellation(n_planes: int, per_plane: int, epoch: datetime, sensor: SensorParams,
                        phase_deg: float = 0.0, ltans=PLANE_LTANS, satnum0: int = 99001) -> List[Satellite]:
    sats = []
    for i in range(n_planes):
        for j in range(per_plane):
            ma = phase_deg + j * 360.0 / per_plane + i * 360.0 / (per_plane * max(n_planes, 1)) / 2
            sats.append(make_sso_satellite(f"P{i + 1}S{j + 1}", satnum0 + len(sats), epoch,
                                           sensor.alt_km, ltans[i], ma, sensor.incl_deg))
    return sats


def constellation_passes(sats: Sequence[Satellite], center, t_start, t_end,
                         sensor: SensorParams) -> List[Pass]:
    c = geodetic_to_ecef(center[0], center[1], 0.0)
    out = []
    for s in sats:
        out += find_passes(s, c, t_start, t_end, look_min_deg=sensor.beta_um_min,
                           look_max_deg=sensor.beta_um_max, squint_max_deg=sensor.beta_sk_max,
                           dt_s=sensor.dt_s)
    out.sort(key=lambda p: p.start)
    for i, p in enumerate(out):
        p.index = i
    return out


@dataclass
class CompletionResult:
    first_h: float          # от заявки до первой съёмки
    all_h: float            # до съёмки всех кадров (nan — не закрыто за горизонт)
    frac_12h: float         # доля кадров, снятых за 12 ч
    max_per_pass: int       # наибольшее число кадров за пролёт
    per_pass: List[int]


class PassCache:
    """Кэш геометрии пролётов для неизменного набора кадров."""

    def __init__(self, frames: Sequence[Frame], sensor: SensorParams, center):
        self.frames, self.sensor = list(frames), sensor
        self.center = geodetic_to_ecef(center[0], center[1], 0.0)
        self._cache: Dict[tuple, PassGeometry] = {}

    def get(self, p: Pass) -> PassGeometry:
        key = (p.sat_name, str(p.start))
        if key not in self._cache:
            self._cache[key] = PassGeometry(p, self.frames, self.sensor, center_ecef=self.center)
        return self._cache[key]


def time_to_complete(passes: Sequence[Pass], model: CoverageModel, cache: PassCache,
                     t_request: np.datetime64, horizon_h: float,
                     method: str = "proposed_cluster") -> CompletionResult:
    """Последовательное распределение кадров по пролётам после поступления заявки."""
    remaining = set(range(len(model.frames)))
    acc = model.new_acc()
    first, last = np.nan, np.nan
    shot_times = []
    per_pass = []
    for p in passes:
        h = float((p.start - t_request) / np.timedelta64(1, "s")) / 3600.0
        if h < 0:
            continue
        if h > horizon_h:
            break
        g = cache.get(p)
        plan = plan_pass(g, model, acc, h, sorted(remaining), method)
        if not plan.items:
            continue
        for k in plan.frames:
            model.add(acc, k, h)
            remaining.discard(k)
            shot_times.append(h)
        per_pass.append(len(plan.items))
        if np.isnan(first):
            first = h
        if not remaining:
            last = h
            break
    n = len(model.frames)
    return CompletionResult(first, last, float(np.sum(np.array(shot_times) <= 12.0)) / n,
                            max(per_pass) if per_pass else 0, per_pass)
