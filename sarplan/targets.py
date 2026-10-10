"""
Модель сложнораспределённого объекта: элементы и кадры-кандидаты,
а также сценарии моделирования из статьи.

Кадры: для протяжённых элементов — вдоль оси с перекрытием, превышающим
погрешность прицеливания; для площадных — регулярная раскладка;
для точечных — один кадр.
"""

from typing import List, Sequence, Tuple

import numpy as np

from .criterion import Element, Frame
from .frames import azimuth_between, distance_km, offset_point

# Ось нижнего Днестра от устья (Днестровский лиман) до Тирасполя, оцифрована приближённо
DNESTR_AXIS = [
    (46.07, 30.46), (46.20, 30.33), (46.33, 30.25), (46.42, 30.27), (46.45, 30.12),
    (46.55, 30.02), (46.62, 29.93), (46.70, 29.80), (46.77, 29.70), (46.84, 29.63),
]

Scenario = Tuple[List[Element], List[Frame]]


def point_element(eid: str, lat: float, lon: float, alpha: float = 1.0,
                  deadline_h: float = float("inf"), n_looks: int = 1) -> Scenario:
    e = Element(eid, "point", alpha, deadline_h)
    return [e], [Frame(f"{eid}", lat, lon, {eid: 1.0}, n_looks)]


def resample_polyline(vertices: Sequence[Tuple[float, float]], step_km: float,
                      max_points: int = None) -> List[Tuple[float, float]]:
    """Равномерные точки вдоль ломаной (lat, lon) с шагом step_km."""
    pts = [vertices[0]]
    carry = 0.0
    for (la1, lo1), (la2, lo2) in zip(vertices[:-1], vertices[1:]):
        seg = float(distance_km(la1, lo1, la2, lo2))
        az = azimuth_between(la1, lo1, la2, lo2)
        s = step_km - carry
        while s <= seg + 1e-9:
            la, lo = offset_point(la1, lo1, az, s)
            pts.append((float(la), float(lo)))
            s += step_km
        carry = seg - (s - step_km)
    if max_points:
        pts = pts[:max_points]
    return pts


def line_element(eid: str, vertices, frame_km: float = 5.0, overlap_km: float = 0.1,
                 alpha: float = 1.0, deadline_h: float = float("inf"),
                 n_frames: int = None) -> Scenario:
    """Линейный элемент: цепочка кадров вдоль оси с шагом frame_km - overlap_km."""
    step = frame_km - overlap_km
    pts = resample_polyline(vertices, step, n_frames)
    pts = [(float(a), float(b)) for a, b in pts]
    e = Element(eid, "line", alpha, deadline_h, meta={"n_frames": len(pts)})
    c = 1.0 / len(pts)
    return [e], [Frame(f"{eid}.{j}", la, lo, {eid: c}) for j, (la, lo) in enumerate(pts)]


def area_element(eid: str, lat: float, lon: float, n_along: int, n_across: int,
                 azimuth_deg: float = 0.0, frame_km: float = 5.0, alpha: float = 1.0,
                 deadline_h: float = float("inf")) -> Scenario:
    """Площадной элемент: регулярная раскладка n_along × n_across кадров."""
    e = Element(eid, "area", alpha, deadline_h)
    a = (np.arange(n_along) - (n_along - 1) / 2) * frame_km
    b = (np.arange(n_across) - (n_across - 1) / 2) * frame_km
    A, B = np.meshgrid(a, b, indexing="ij")
    la, lo = offset_point(lat, lon, azimuth_deg, A.ravel(), B.ravel())
    c = 1.0 / A.size
    frames = [Frame(f"{eid}.{j}", float(x), float(y), {eid: c}) for j, (x, y) in enumerate(zip(la, lo))]
    return [e], frames


def merge(*parts: Scenario) -> Scenario:
    els, frs = [], []
    for e, f in parts:
        els += e
        frs += f
    return els, frs


def _uniform_in_region(rng, center, azimuth_deg, size_az_km, size_rg_km, n):
    along = rng.uniform(-size_az_km / 2, size_az_km / 2, n)
    across = rng.uniform(-size_rg_km / 2, size_rg_km / 2, n)
    la, lo = offset_point(center[0], center[1], azimuth_deg, along, across)
    return np.atleast_1d(la), np.atleast_1d(lo)


def random_targets(rng, center, azimuth_deg, n: int = 20, size_az_km: float = 100.0,
                   size_rg_km: float = 150.0, p_two_look: float = 0.3) -> Scenario:
    """Случайные точечные цели в районе 100×150 км (азимут × дальность), приоритеты 1–5."""
    la, lo = _uniform_in_region(rng, center, azimuth_deg, size_az_km, size_rg_km, n)
    parts = []
    for j in range(n):
        parts.append(point_element(f"T{j:02d}", la[j], lo[j], float(rng.integers(1, 6)),
                                   n_looks=2 if rng.random() < p_two_look else 1))
    return merge(*parts)


def duplicate_targets(rng, center, azimuth_deg, n_dup: int = 14, n_single: int = 10,
                      dup_shift_km: float = 2.0, size_az_km: float = 100.0,
                      size_rg_km: float = 150.0) -> Scenario:
    """
    Сценарий с альтернативными кадрами: n_dup объектов, каждый покрывается любым из
    двух кадров-дублей (смещены на dup_shift_km), и n_single одиночных объектов.
    """
    n = n_dup + n_single
    la, lo = _uniform_in_region(rng, center, azimuth_deg, size_az_km, size_rg_km, n)
    els, frs = [], []
    for j in range(n):
        eid = f"O{j:02d}"
        alpha = float(rng.integers(1, 6))
        els.append(Element(eid, "point", alpha))
        if j < n_dup:
            for s, sign in enumerate((-1, 1)):
                a, b = offset_point(la[j], lo[j], azimuth_deg, sign * dup_shift_km / 2, 0.0)
                frs.append(Frame(f"{eid}.{s}", float(a), float(b), {eid: 1.0}))
        else:
            frs.append(Frame(eid, float(la[j]), float(lo[j]), {eid: 1.0}))
    return els, frs


def rolling_targets(rng, center, azimuth_deg, size_az_km: float = 100.0,
                    size_rg_km: float = 150.0) -> Scenario:
    """
    Сценарий текущей коррекции: 12 площадных объектов по 3 кадра, 10 точечных,
    линия из 14 кадров; срок 30 или 48 ч, после срока вклад умножается на 0,3.
    """
    parts = []
    la, lo = _uniform_in_region(rng, center, azimuth_deg, size_az_km - 20, size_rg_km - 20, 22)
    for j in range(12):
        parts.append(area_element(f"A{j:02d}", la[j], lo[j], 3, 1, azimuth_deg,
                                  alpha=float(rng.integers(1, 6)),
                                  deadline_h=float(rng.choice([30.0, 48.0]))))
    for j in range(10):
        parts.append(point_element(f"P{j:02d}", la[12 + j], lo[12 + j], float(rng.integers(1, 6)),
                                   float(rng.choice([30.0, 48.0]))))
    a0 = offset_point(center[0], center[1], azimuth_deg, -35.0, rng.uniform(-40, 40))
    a1 = offset_point(center[0], center[1], azimuth_deg, 35.0, rng.uniform(-40, 40))
    parts.append(line_element("L00", [(float(a0[0]), float(a0[1])), (float(a1[0]), float(a1[1]))],
                              alpha=float(rng.integers(1, 6)),
                              deadline_h=float(rng.choice([30.0, 48.0])), n_frames=14))
    return merge(*parts)


def urgent_request(rng, center, azimuth_deg, alpha: float = 8.0, deadline_h: float = 30.0) -> Scenario:
    """Срочная заявка: площадной объект из 3 кадров, приоритет 8, срок 30 ч."""
    la, lo = _uniform_in_region(rng, center, azimuth_deg, 60.0, 100.0, 1)
    return area_element("URGENT", la[0], lo[0], 3, 1, azimuth_deg, alpha=alpha, deadline_h=deadline_h)


def dnestr_targets(n_chain: int = 21, frame_km: float = 5.0, n_points: int = 19,
                   seed: int = 54, chain_id: str = "DNESTR") -> Scenario:
    """
    Участок Днестра от Чёрного моря к Тирасполю: цепочка из n_chain рамок
    frame_km × frame_km вдоль оси реки и n_points точечных объектов района.
    """
    chain = line_element(chain_id, DNESTR_AXIS, frame_km=frame_km, overlap_km=frame_km * 0.02,
                         n_frames=n_chain)
    if n_points == 0:
        return chain
    rng = np.random.default_rng(seed)
    axis = resample_polyline(DNESTR_AXIS, 5.0)
    parts = [chain]
    for j in range(n_points):
        la0, lo0 = axis[rng.integers(0, len(axis))]
        la, lo = offset_point(la0, lo0, float(rng.uniform(0, 360)), float(rng.uniform(3, 25)))
        parts.append(point_element(f"D{j:02d}", float(la), float(lo), float(rng.integers(1, 6))))
    return merge(*parts)


def region_center_of(frames: Sequence[Frame]) -> Tuple[float, float]:
    return float(np.mean([f.lat for f in frames])), float(np.mean([f.lon for f in frames]))
