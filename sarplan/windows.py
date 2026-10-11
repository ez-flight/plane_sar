"""
Окна видимости и допустимости кадров (статья, формула (17)), сетка точек
наведения Ξ_ν, показатель Φ_k и кластеризация кадров по профилю допустимости.

Для пролёта ν, кадра k и точки наведения R_Tп окно допустимого начала съёмки

    Ω_{k,ν}(R_Tп) = { t ∈ [t_н, t_к - t_вкл,k] :
        |Δβ_ск| ≤ Δβ_ск max, |Δβ_ум| ≤ Δβ_ум max,
        |β_ск| ≤ β_ск max, β_ум min ≤ β_ум ≤ β_ум max  на всём [t, t + t_вкл,k] }.
"""

from typing import List, Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from . import geometry
from .config import SensorParams
from .criterion import Frame
from .frames import ecef_to_geodetic
from .orbit import Pass

# Сетка Ξ: смещение l_с вдоль трассы (км) и множители глубины h_п относительно номинальной
ALONG_GRID_KM = (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0)
DEPTH_MULTS = (0.0, 0.5, 1.0, 2.0, 4.0, 1000.0)


def _runs_ok(mask: np.ndarray, n: int) -> np.ndarray:
    """ok[..., s] = mask[..., s:s+n] целиком True (s ≤ T-n), иначе False."""
    T = mask.shape[-1]
    out = np.zeros(mask.shape, bool)
    if n > T:
        return out
    cs = np.concatenate([np.zeros(mask.shape[:-1] + (1,), int), np.cumsum(mask, axis=-1)], axis=-1)
    out[..., :T - n + 1] = (cs[..., n:] - cs[..., :T - n + 1]) == n
    return out


def _longest_run(mask: np.ndarray) -> np.ndarray:
    """Оставить только самый длинный непрерывный участок True (окно видимости [t_н, t_к])."""
    out = np.zeros_like(mask)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    g = max(groups, key=len)
    out[g[0]:g[-1] + 1] = True
    return out


class PassGeometry:
    """
    Геометрия пролёта для набора кадров: окна видимости, допустимости
    по каждой точке наведения, Φ_k, длительности в отсчётах сетки.
    """

    def __init__(self, pas: Pass, frames: Sequence[Frame], sensor: SensorParams,
                 center_ecef: Optional[np.ndarray] = None,
                 along_grid=ALONG_GRID_KM, depth_mults=DEPTH_MULTS):
        self.pas = pas
        self.frames = list(frames)
        self.sensor = sensor
        self.dt = pas.dt
        self.T = len(pas.t)
        K = len(self.frames)
        F = np.array([f.ecef for f in self.frames]).reshape(K, 3)
        r, v = pas.r, pas.v
        self.n_dur = np.array([int(np.ceil(f.duration(sensor.t_frame_s) / self.dt - 1e-9)) for f in self.frames])
        self.n_gap = int(np.ceil(sensor.gap_s / self.dt - 1e-9))

        sk, um = geometry.squint_look(r[None], v[None], F[:, None, :])     # (K, T)
        vis = (geometry.above_horizon(r[None], F[:, None, :]) & (np.abs(sk) <= sensor.beta_sk_max)
               & (um >= sensor.beta_um_min) & (um <= sensor.beta_um_max))
        self.vis = np.array([_longest_run(m) for m in vis]).reshape(K, self.T)
        self.squint, self.look = sk, um

        if center_ecef is None:
            center_ecef = F.mean(axis=0)
            from .frames import geodetic_to_ecef
            la, lo, _ = ecef_to_geodetic(center_ecef)
            center_ecef = geodetic_to_ecef(la, lo, 0.0)
        self.center = center_ecef

        if sensor.mode == "hybrid":
            self.points, self.point_params = self._pointing_grid(along_grid, depth_mults)
            self.nominal = self.point_params.index((0.0, 1.0))
        elif sensor.mode == "stripmap":
            self.points = None
            self.point_params = self._swath_grid()
            self.nominal = len(self.point_params) // 2
        else:
            self.points, self.point_params, self.nominal = None, [("broadside", None)], 0
        P = len(self.point_params)

        self.adm = np.zeros((P, K, self.T), bool)
        self.phi = np.full((P, K), np.nan)
        self._dsk_cache = {}
        for p in range(P):
            dsk, dum = self.deflection(p)
            a = (self.vis & (np.abs(dsk) <= sensor.d_beta_sk_max) & (np.abs(dum) <= sensor.d_beta_um_max))
            self.adm[p] = a
            dev = np.where(a, np.abs(dsk) + np.abs(dum), np.nan)
            cnt = a.sum(axis=1)
            self.phi[p] = np.where(cnt > 0, np.nansum(dev, axis=1) / np.maximum(cnt, 1), np.nan)
        self.start_ok = np.zeros((P, K, self.T), bool)
        self.start_vis = np.zeros((K, self.T), bool)
        for n in np.unique(self.n_dur):
            ks = np.flatnonzero(self.n_dur == n)
            self.start_ok[:, ks] = _runs_ok(self.adm[:, ks], n)
            self.start_vis[ks] = _runs_ok(self.vis[ks], n)

    # --- точки наведения -------------------------------------------------
    def _pointing_grid(self, along_grid, depth_mults):
        """
        Ξ_ν: точки R_Tп = C + l_с·a + h_п·u, где C — центр района, a — направление
        вдоль трассы в C, u — линия визирования C в момент траверза. Номинальная
        глубина выбирается так, чтобы пятно ДН проходило протяжённость района по
        азимуту за бюджет T_вкл max: h₀ = R₀·v_f / (V_g - v_f), v_f = L_az / T_вкл max.
        """
        r, v = self.pas.r, self.pas.v
        sk_c, _ = geometry.squint_look(r, v, self.center[None, :])
        ib = int(np.argmin(np.abs(sk_c)))
        rb, vb = r[ib], v[ib]
        up = self.center / np.linalg.norm(self.center)
        a = vb - np.dot(vb, up) * up
        a /= np.linalg.norm(a)
        u = self.center - rb
        R0 = np.linalg.norm(u)
        u /= R0
        F = np.array([f.ecef for f in self.frames])
        along = (F - self.center) @ a
        L_az = float(along.max() - along.min()) + 5.0
        v_f = L_az / self.sensor.budget_s
        V_g = np.linalg.norm(vb) * np.linalg.norm(self.center) / np.linalg.norm(rb)
        h0 = R0 * v_f / max(V_g - v_f, 1e-3)
        self.h0_km, self.broadside_index = h0, ib
        pts, params = [], []
        for l in along_grid:
            for m in depth_mults:
                pts.append(self.center + l * a + m * h0 * u)
                params.append((l, m))
        return np.array(pts), params

    def _swath_grid(self):
        """
        Маршрутный режим: варианты положения полосы (крен на пролёт). x_k — удаление
        центра кадра от подспутниковой трассы по поверхности на траверзе; полоса с
        центром x_c снимает кадр, если он целиком в ней: |x_k - x_c| ≤ (W - f)/2.
        """
        r, v = self.pas.r, self.pas.v
        F = np.array([f.ecef for f in self.frames])
        sk_c, _ = geometry.squint_look(r, v, self.center[None, :])
        ib = int(np.argmin(np.abs(sk_c)))
        up = r[ib] / np.linalg.norm(r[ib])
        right = np.cross(v[ib], up)
        right /= np.linalg.norm(right)
        nadir = r[ib] * (np.linalg.norm(self.center) / np.linalg.norm(r[ib]))
        self.cross_km = (F - nadir) @ right
        self.broadside_index = ib
        W, f = self.sensor.swath_km, self.sensor.frame_km
        half = max((W - f) / 2.0, 0.0)
        lo, hi = self.cross_km.min() - half, self.cross_km.max() + half
        sign = np.sign(np.median(self.cross_km)) or 1.0
        grid = np.arange(np.floor(lo / 2.5) * 2.5, hi + 2.5, 2.5)
        return [("swath", float(x)) for x in grid if abs(x) >= 1.0 and np.sign(x) == sign] or [("swath", 0.0)]

    def deflection(self, p: int):
        """Δβ_ск, Δβ_ум (K, T) для точки наведения p."""
        r, v = self.pas.r, self.pas.v
        F = np.array([f.ecef for f in self.frames])
        if self.sensor.mode == "stripmap":
            half = max((self.sensor.swath_km - self.sensor.frame_km) / 2.0, 0.0)
            off = np.abs(self.cross_km - self.point_params[p][1])
            # «отклонение» в км поперёк трассы относительно допустимой полуширины (≤ 0 — кадр в полосе)
            dum = np.where(off <= half + 1e-9, 0.0, 999.0)[:, None] * np.ones((1, self.T))
            return self.squint, dum
        if self.points is None:
            return self.squint, np.zeros_like(self.squint)
        return geometry.beam_deflection(r[None], v[None], self.points[p][None, None, :], F[:, None, :])

    def point_description(self, p: int) -> dict:
        if self.sensor.mode == "stripmap":
            return {"mode": "stripmap", "swath_center_km": self.point_params[p][1],
                    "swath_km": self.sensor.swath_km}
        if self.points is None:
            return {"mode": "electronic"}
        la, lo, h = ecef_to_geodetic(self.points[p])
        l, m = self.point_params[p]
        return {"lat": float(la), "lon": float(lo), "depth_km": float(-h),
                "along_km": l, "depth_mult": m, "h0_km": float(self.h0_km)}

    # --- производные показатели -----------------------------------------
    def window_bounds(self, k: int):
        idx = np.flatnonzero(self.vis[k])
        return (self.pas.t[idx[0]], self.pas.t[idx[-1]]) if len(idx) else (np.nan, np.nan)

    def admissible_frames(self, p: int) -> np.ndarray:
        return self.start_ok[p].any(axis=1)

    def union_length_s(self, p: int, ks=None) -> float:
        """t_объед — длина объединения окон допустимости (для оценки (19))."""
        ks = np.arange(len(self.frames)) if ks is None else ks
        return float(self.adm[p][ks].any(axis=0).sum() * self.dt)

    def candidate_points(self, weights: np.ndarray, threshold: float = 0.5) -> List[int]:
        """
        Короткий список точек наведения: кадры кластеризуются по профилю
        допустимости (бинарный вектор по точкам Ξ, расстояние Жаккара,
        иерархический метод); для кластера берётся точка с наибольшей суммой
        «приоритет × длина окна».
        """
        if self.points is None:
            return [0]
        prof = self.start_ok.any(axis=2).T            # (K, P)
        keep = np.flatnonzero(prof.any(axis=1))
        out = {self.nominal}
        if len(keep) >= 2:
            d = pdist(prof[keep], "jaccard")
            labels = fcluster(linkage(d, "average"), threshold, "distance")
        else:
            labels = np.ones(len(keep), int)
        wlen = self.start_ok.sum(axis=2) * self.dt    # (P, K)
        for lab in np.unique(labels):
            ks = keep[labels == lab]
            score = (wlen[:, ks] * weights[ks]).sum(axis=1)
            out.add(int(np.argmax(score)))
        return sorted(out)

    def min_phi_point(self) -> int:
        """Точка с минимальным средним Φ по кадрам (критерий оптимизатора [2])."""
        if self.points is None:
            return 0
        n_adm = self.start_ok.any(axis=2).sum(axis=1)
        with np.errstate(all="ignore"):
            mphi = np.nanmean(self.phi, axis=1)
        mphi = np.where(np.isnan(mphi), np.inf, mphi)
        best = n_adm.max()
        cand = np.flatnonzero(n_adm == best)
        return int(cand[np.argmin(mphi[cand])])
