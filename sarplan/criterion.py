"""
Модель СРО и критерий взвешенного покрытия с насыщением (статья, формула (16)):

    Э = Σ_i α_i · min{1, Σ_{k,ν} c_{i,k} · χ_i(t_{k,ν}) · x_{k,ν}} → max,

χ_i(t) = 1 при t ≤ T_i^дир и равна decay после срока.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .frames import geodetic_to_ecef


@dataclass
class Element:
    """Элемент СРО: точечный, линейный или площадной."""
    eid: str
    kind: str = "point"                 # point | line | area
    alpha: float = 1.0                  # приоритет α_i > 0
    deadline_h: float = float("inf")    # директивный срок T_i^дир, ч от начала горизонта
    decay: float = 0.3                  # χ после срока
    res_m: float = 0.5                  # требуемое разрешение
    meta: dict = field(default_factory=dict)


@dataclass
class Frame:
    """Кадр-кандидат: центр, покрываемые элементы c_{i,k}, число видов, объём данных."""
    fid: str
    lat: float
    lon: float
    cover: Dict[str, float]
    n_looks: int = 1
    q: float = 1.0
    ecef: np.ndarray = None

    def __post_init__(self):
        if self.ecef is None:
            self.ecef = geodetic_to_ecef(self.lat, self.lon, 0.0)

    def duration(self, t_frame_s: float) -> float:
        """t_вкл,k = n_k · t_вкл,1."""
        return self.n_looks * t_frame_s


class CoverageModel:
    """Вычисление Э и приращений для набора элементов и кадров."""

    def __init__(self, elements: Sequence[Element], frames: Sequence[Frame]):
        self.elements = list(elements)
        self.frames = list(frames)
        self.eidx = {e.eid: i for i, e in enumerate(self.elements)}
        self.alpha = np.array([e.alpha for e in self.elements], float)
        self.deadline = np.array([e.deadline_h for e in self.elements], float)
        self.decay = np.array([e.decay for e in self.elements], float)
        self.fcov = []
        for f in self.frames:
            idx = np.array([self.eidx[e] for e in f.cover], int)
            c = np.array(list(f.cover.values()), float)
            self.fcov.append((idx, c))
        # приоритет кадра: максимум приоритетов его элементов (α_k в (20))
        self.frame_alpha = np.array([self.alpha[i].max() if len(i) else 0.0 for i, _ in self.fcov])

    @property
    def total_alpha(self) -> float:
        return float(self.alpha.sum())

    def new_acc(self) -> np.ndarray:
        return np.zeros(len(self.elements))

    def chi(self, idx: np.ndarray, t_h: float, strict: bool = False) -> np.ndarray:
        late = 0.0 if strict else self.decay[idx]
        return np.where(t_h <= self.deadline[idx], 1.0, late)

    def value(self, acc: np.ndarray) -> float:
        return float(np.sum(self.alpha * np.minimum(1.0, acc)))

    def coverage(self, acc: np.ndarray) -> float:
        """Нормированное покрытие Э / Σα ∈ [0, 1]."""
        return self.value(acc) / self.total_alpha if self.total_alpha > 0 else 0.0

    def gain(self, acc: np.ndarray, k: int, t_h: float) -> float:
        idx, c = self.fcov[k]
        old = acc[idx]
        new = old + c * self.chi(idx, t_h)
        return float(np.sum(self.alpha[idx] * (np.minimum(1.0, new) - np.minimum(1.0, old))))

    def additive_value(self, k: int, t_h: float) -> float:
        """Аддитивный вклад кадра без насыщения (критерий [5, 6])."""
        idx, c = self.fcov[k]
        return float(np.sum(self.alpha[idx] * c * self.chi(idx, t_h)))

    def add(self, acc: np.ndarray, k: int, t_h: float, strict: bool = False) -> None:
        idx, c = self.fcov[k]
        acc[idx] += c * self.chi(idx, t_h, strict)

    def set_value(self, acc: np.ndarray, ks: Sequence[int], t_h: float) -> float:
        a = acc.copy()
        for k in ks:
            self.add(a, k, t_h)
        return self.value(a)

    def element_coverage(self, acc: np.ndarray, eid: str) -> float:
        return float(min(1.0, acc[self.eidx[eid]]))
