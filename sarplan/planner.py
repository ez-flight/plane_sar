"""
Выбор и упорядочение кадров за пролёт (статья, формулы (18), (19)) и базовые
методы для сравнения.

Ограничения пролёта ν: окно допустимости t_k ∈ Ω_k(R_Tп); отсутствие наложений
t_k + t_вкл,k + Δt ≤ t_l; бюджет Σ t_вкл,k ≤ T_вкл max; ресурсы Σ Q_k ≤ Q_max.

Методы (номера — как в таблице 3 статьи):
    greedy_priority   (1) жадный по приоритету, номинальная точка
    seq_analog2       (2) аналог [2]: последовательная съёмка без выбора подмножества
    proposed          (3) предлагаемый: жадная вставка по ρ_k + локальное улучшение
    proposed_cluster  (4) то же, точка из кластерного списка
    proposed_full     (5) то же, полный перебор точек наведения
    dp_additive       (6) ДП с аддитивным критерием при заданном порядке кадров [5, 6]
    exact             точный перебор подмножеств (эталон для десятков кандидатов)
    greedy_priority_sat  жадный по приоритету с учётом насыщения
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .criterion import CoverageModel
from .windows import PassGeometry

EPS = 1e-9


@dataclass
class PlanParams:
    budget_s: Optional[float] = None    # по умолчанию из SensorParams
    q_max: Optional[float] = None
    zeta: float = 0.01                  # ζ — вес Φ во внешнем цикле по точкам наведения
    local_search: bool = True
    max_ls_rounds: int = 20
    exact_max_frames: int = 14


@dataclass
class PassPlan:
    method: str
    point: int
    items: List[Tuple[int, int]]        # (кадр k, индекс начала на сетке пролёта)
    gain: float = 0.0                   # прирост Э за пролёт
    objective: float = 0.0              # с учётом штрафа/бонуса и ζΦ
    phi: float = 0.0
    runtime_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def frames(self) -> List[int]:
        return [k for k, _ in self.items]


class _Ctx:
    """Данные одной задачи «пролёт × точка наведения»."""

    def __init__(self, geom: PassGeometry, p: int, model: CoverageModel, acc: np.ndarray,
                 t_h: float, cand: Sequence[int], params: PlanParams,
                 bonus: Optional[np.ndarray] = None, start_ok: Optional[np.ndarray] = None):
        s = geom.sensor
        self.geom, self.p, self.model, self.acc, self.t_h = geom, p, model, acc, t_h
        self.start_ok = geom.start_ok[p] if start_ok is None else start_ok
        self.n_dur, self.n_gap, self.T = geom.n_dur, geom.n_gap, geom.T
        self.dur_s = np.array([f.duration(s.t_frame_s) for f in geom.frames])
        self.q = np.array([f.q for f in geom.frames])
        self.budget = params.budget_s if params.budget_s is not None else s.budget_s
        self.q_max = params.q_max if params.q_max is not None else s.q_max
        self.bonus = np.zeros(len(geom.frames)) if bonus is None else bonus
        self.cand = [k for k in cand if self.start_ok[k].any()]
        self.pad = int(self.n_dur.max() + self.n_gap + 2) if len(self.n_dur) else 2

    def value(self, ks) -> float:
        return self.model.set_value(self.acc, ks, self.t_h) - self.model.value(self.acc) + float(self.bonus[list(ks)].sum())

    def new_occ(self):
        return np.zeros(self.T + self.pad, bool)

    def earliest(self, k: int, occ: np.ndarray, after: int = 0) -> int:
        """Самое раннее допустимое начало кадра k без пересечения с занятыми интервалами."""
        L = self.n_dur[k] + self.n_gap
        cs = np.concatenate([[0], np.cumsum(occ)])
        free = (cs[L:L + self.T] - cs[:self.T]) == 0
        ok = self.start_ok[k] & free
        if after:
            ok[:after] = False
        idx = np.flatnonzero(ok)
        return int(idx[0]) if len(idx) else -1

    def occupy(self, occ: np.ndarray, k: int, s: int):
        occ[s:s + self.n_dur[k] + self.n_gap] = True

    def resources_ok(self, ks) -> bool:
        ks = list(ks)
        return self.dur_s[ks].sum() <= self.budget + EPS and self.q[ks].sum() <= self.q_max + EPS

    def pack(self, ks) -> Optional[List[Tuple[int, int]]]:
        """Разместить множество кадров (несколько порядков вставки); None — не удалось."""
        ks = list(ks)
        if not ks:
            return []
        first = {k: np.flatnonzero(self.start_ok[k])[0] for k in ks}
        last = {k: np.flatnonzero(self.start_ok[k])[-1] for k in ks}
        orders = [sorted(ks, key=lambda k: (last[k], first[k])),
                  sorted(ks, key=lambda k: (first[k], last[k])),
                  sorted(ks, key=lambda k: (first[k] + last[k]))]
        for order in orders:
            occ = self.new_occ()
            items = []
            for k in order:
                s = self.earliest(k, occ)
                if s < 0:
                    break
                self.occupy(occ, k, s)
                items.append((k, s))
            else:
                return sorted(items, key=lambda x: x[1])
        return None


# --- предлагаемый метод -------------------------------------------------------
def _greedy_rho(ctx: _Ctx) -> List[Tuple[int, int]]:
    """Жадная вставка в порядке убывания ρ_k(S) = (Э(S∪{k}) - Э(S)) / (t_вкл,k + Δt), формула (18)."""
    model, gap = ctx.model, ctx.geom.sensor.gap_s
    acc = ctx.acc.copy()
    occ = ctx.new_occ()
    items, used, q_used = [], 0.0, 0.0
    alive = set(ctx.cand)
    while alive:
        scored = []
        for k in alive:
            if used + ctx.dur_s[k] > ctx.budget + EPS or q_used + ctx.q[k] > ctx.q_max + EPS:
                continue
            g = model.gain(acc, k, ctx.t_h) + ctx.bonus[k]
            if g > EPS:
                scored.append((g / (ctx.dur_s[k] + gap), k))
        if not scored:
            break
        scored.sort(reverse=True)
        placed = False
        for _, k in scored:
            s = ctx.earliest(k, occ)
            if s < 0:
                alive.discard(k)      # занятость только растёт — кадр больше не поместится
                continue
            ctx.occupy(occ, k, s)
            model.add(acc, k, ctx.t_h)
            items.append((k, s))
            used += ctx.dur_s[k]
            q_used += ctx.q[k]
            alive.discard(k)
            placed = True
            break
        if not placed:
            break
    return sorted(items, key=lambda x: x[1])


def _local_search(ctx: _Ctx, items: List[Tuple[int, int]], max_rounds: int) -> List[Tuple[int, int]]:
    """Локальное улучшение: вставка с переупаковкой, замена и обмен кадров."""
    sel = [k for k, _ in items]
    best_val = ctx.value(sel)
    for _ in range(max_rounds):
        improved = False
        out = [k for k in ctx.cand if k not in sel]
        solo = {k: ctx.model.gain(ctx.acc, k, ctx.t_h) + ctx.bonus[k] for k in out}
        out = [k for k in sorted(out, key=lambda k: -solo[k]) if solo[k] > EPS][:15]
        # вставка
        for u in out:
            new = sel + [u]
            if ctx.resources_ok(new):
                v = ctx.value(new)
                if v > best_val + EPS:
                    packed = ctx.pack(new)
                    if packed is not None:
                        sel, best_val, items, improved = new, v, packed, True
                        break
        if improved:
            continue
        # замена одного кадра (обмен выбранного на невыбранный)
        best_move = None
        for u in out:
            for s in sel:
                new = [k for k in sel if k != s] + [u]
                if not ctx.resources_ok(new):
                    continue
                v = ctx.value(new)
                if v > best_val + EPS and (best_move is None or v > best_move[0]):
                    packed = ctx.pack(new)
                    if packed is not None:
                        best_move = (v, new, packed)
        if best_move:
            best_val, sel, items = best_move
            continue
        break
    return items


def _exact(ctx: _Ctx) -> List[Tuple[int, int]]:
    """
    Точное решение перебором подмножеств: для каждого подмножества — минимальное
    время окончания последовательной съёмки (ДП по подмножествам), затем
    максимум Э по достижимым подмножествам при ограничениях бюджета.
    """
    ks = ctx.cand
    n = len(ks)
    if n == 0:
        return []
    T = ctx.T
    nxt = np.full((n, T + ctx.pad + 1), -1, int)
    for j, k in enumerate(ks):
        ok = ctx.start_ok[k]
        cur = -1
        for s in range(T - 1, -1, -1):
            if ok[s]:
                cur = s
            nxt[j, s] = cur
    full = 1 << n
    INF = 1 << 30
    end = np.full(full, INF, np.int64)
    used = np.zeros(full)
    parent = np.full(full, -1, np.int64)
    pstart = np.full(full, -1, np.int64)
    end[0] = 0
    dur = [ctx.dur_s[k] for k in ks]
    L = [ctx.n_dur[k] + ctx.n_gap for k in ks]
    for m in range(full):
        e = end[m]
        if e >= INF:
            continue
        for j in range(n):
            b = 1 << j
            if m & b:
                continue
            if used[m] + dur[j] > ctx.budget + EPS:
                continue
            s = nxt[j, e] if e < nxt.shape[1] else -1
            if s < 0:
                continue
            ne = s + L[j]
            m2 = m | b
            if ne < end[m2]:
                end[m2], used[m2], parent[m2], pstart[m2] = ne, used[m] + dur[j], j, s
    best_m, best_v = 0, 0.0
    for m in np.flatnonzero(end < INF):
        sel = [ks[j] for j in range(n) if m >> j & 1]
        if ctx.q[sel].sum() > ctx.q_max + EPS:
            continue
        v = ctx.value(sel)
        if v > best_v + EPS:
            best_m, best_v = int(m), v
    items, m = [], best_m
    while m:
        j = parent[m]
        items.append((ks[j], int(pstart[m])))
        m &= ~(1 << j)
    return sorted(items, key=lambda x: x[1])


# --- базовые методы ------------------------------------------------------------
def _greedy_priority(ctx: _Ctx, saturation: bool = False) -> List[Tuple[int, int]]:
    """Жадный по приоритету: кадры по убыванию α, вставка на самое раннее время."""
    acc = ctx.acc.copy()
    first = {k: np.flatnonzero(ctx.start_ok[k])[0] for k in ctx.cand}
    order = sorted(ctx.cand, key=lambda k: (-(ctx.model.frame_alpha[k] + ctx.bonus[k]), first[k]))
    occ = ctx.new_occ()
    items, used, q_used = [], 0.0, 0.0
    for k in order:
        if used + ctx.dur_s[k] > ctx.budget + EPS or q_used + ctx.q[k] > ctx.q_max + EPS:
            continue
        if saturation and ctx.model.gain(acc, k, ctx.t_h) + ctx.bonus[k] <= EPS:
            continue
        s = ctx.earliest(k, occ)
        if s < 0:
            continue
        ctx.occupy(occ, k, s)
        ctx.model.add(acc, k, ctx.t_h)
        items.append((k, s))
        used += ctx.dur_s[k]
        q_used += ctx.q[k]
    return sorted(items, key=lambda x: x[1])


def _sequential(ctx: _Ctx) -> List[Tuple[int, int]]:
    """
    Аналог [2]: заданный набор целей снимается последовательно вдоль трассы
    (по середине окна), каждая — на самое раннее время после предыдущей;
    выбор подмножества не выполняется, съёмка прекращается по исчерпании бюджета.
    """
    mid = {k: np.flatnonzero(ctx.start_ok[k]).mean() for k in ctx.cand}
    order = sorted(ctx.cand, key=lambda k: mid[k])
    items, e, used = [], 0, 0.0
    occ = ctx.new_occ()
    for k in order:
        if used + ctx.dur_s[k] > ctx.budget + EPS:
            break
        s = ctx.earliest(k, occ, after=e)
        if s < 0:
            continue
        ctx.occupy(occ, k, s)
        items.append((k, s))
        e = s + ctx.n_dur[k] + ctx.n_gap
        used += ctx.dur_s[k]
    return items


def _dp_additive(ctx: _Ctx) -> List[Tuple[int, int]]:
    """
    ДП по времени для аддитивного критерия (сумма приоритетов) при заданном
    порядке кадров (по середине окна), как в [5, 6]. Состояние — (время окончания,
    использованный бюджет); доминируемые состояния отбрасываются.
    """
    mid = {k: np.flatnonzero(ctx.start_ok[k]).mean() for k in ctx.cand}
    order = sorted(ctx.cand, key=lambda k: mid[k])
    w = {k: ctx.model.additive_value(k, ctx.t_h) + ctx.bonus[k] for k in order}
    states = [(0, 0.0, 0.0, ())]       # end, used, value, items
    for k in order:
        okk = np.flatnonzero(ctx.start_ok[k])
        new = list(states)
        for e, u, v, it in states:
            if u + ctx.dur_s[k] > ctx.budget + EPS:
                continue
            j = np.searchsorted(okk, e)
            if j >= len(okk):
                continue
            s = int(okk[j])
            new.append((s + ctx.n_dur[k] + ctx.n_gap, u + ctx.dur_s[k], v + w[k], it + ((k, s),)))
        new.sort(key=lambda x: (x[0], x[1], -x[2]))
        kept = []
        for st in new:
            if not any(o[0] <= st[0] and o[1] <= st[1] + EPS and o[2] >= st[2] - EPS for o in kept):
                kept.append(st)
        states = kept
    best = max(states, key=lambda x: x[2])
    return list(best[3])


CORES: Dict[str, Callable] = {
    "proposed": _greedy_rho,
    "greedy_priority": _greedy_priority,
    "greedy_priority_sat": lambda ctx: _greedy_priority(ctx, saturation=True),
    "seq_analog2": _sequential,
    "dp_additive": _dp_additive,
    "exact": _exact,
}


def _points_for(method: str, geom: PassGeometry, model: CoverageModel) -> List[int]:
    if method == "proposed_full":
        return list(range(len(geom.point_params)))
    if method == "proposed_cluster":
        return geom.candidate_points(model.frame_alpha)
    if method == "seq_analog2":
        return [geom.min_phi_point()]
    return [geom.nominal]


def plan_pass(geom: PassGeometry, model: CoverageModel, acc: np.ndarray, t_h: float,
              cand: Sequence[int], method: str = "proposed", params: Optional[PlanParams] = None,
              bonus: Optional[np.ndarray] = None, points: Optional[List[int]] = None,
              ignore_deflection: bool = False) -> PassPlan:
    """
    План одного пролёта. Внешний цикл выбирает точку наведения по максимуму
    Э_ν(R_Tп) - ζ·Φ(R_Tп). ignore_deflection=True — «планировщик без модели»:
    учитываются только окно видимости и скос, пределы Δβ игнорируются.
    """
    params = params or PlanParams()
    t0 = time.perf_counter()
    core_name = method.replace("_full", "").replace("_cluster", "")
    core = CORES[core_name]
    pts = points if points is not None else _points_for(method, geom, model)
    best = None
    for p in pts:
        ctx = _Ctx(geom, p, model, acc, t_h, cand, params, bonus,
                   start_ok=geom.start_vis if ignore_deflection else None)
        items = core(ctx)
        if core_name == "proposed" and params.local_search:
            items = _local_search(ctx, items, params.max_ls_rounds)
        ks = [k for k, _ in items]
        gain = model.set_value(acc, ks, t_h) - model.value(acc)
        phi = float(np.nanmean(geom.phi[p, ks])) if ks and not ignore_deflection else 0.0
        obj = gain + float(ctx.bonus[ks].sum()) - params.zeta * phi
        if best is None or obj > best.objective + EPS:
            best = PassPlan(method, p, items, gain, obj, phi)
    best.runtime_s = time.perf_counter() - t0
    return best


def admissible_items(plan: PassPlan, geom: PassGeometry) -> List[Tuple[int, int]]:
    """Оставить кадры, реально снимаемые по полной модели допустимости в точке плана."""
    return [(k, s) for k, s in plan.items if geom.start_ok[plan.point, k, s]]


def plan_n_max(geom: PassGeometry, p: int, ks=None) -> int:
    """Оценка (19) для пролёта и точки наведения."""
    from .duration import n_frames_max
    s = geom.sensor
    ks = np.arange(len(geom.frames)) if ks is None else np.asarray(ks)
    t_mean = float(np.mean([geom.frames[k].duration(s.t_frame_s) for k in ks]))
    return n_frames_max(s.budget_s, t_mean, geom.union_length_s(p, ks), s.gap_s)


def plan_to_rows(plan: PassPlan, geom: PassGeometry) -> List[dict]:
    """Строки плана: время, кадр, углы — для отчёта."""
    rows = []
    dsk, dum = geom.deflection(plan.point)
    pd = geom.point_description(plan.point)
    for k, s in plan.items:
        f = geom.frames[k]
        n = geom.n_dur[k]
        sl = slice(s, s + n)
        rows.append({
            "sat": geom.pas.sat_name,
            "start_utc": str(geom.pas.time_at(s)),
            "frame": f.fid,
            "elements": ";".join(f.cover),
            "lat": round(f.lat, 5), "lon": round(f.lon, 5),
            "duration_s": round(f.duration(geom.sensor.t_frame_s), 2),
            "squint_deg": round(float(geom.squint[k, sl].mean()), 3),
            "look_deg": round(float(geom.look[k, sl].mean()), 3),
            "d_beta_sk_deg": round(float(np.abs(dsk[k, sl]).max()), 3),
            "d_beta_um_deg": round(float(np.abs(dum[k, sl]).max()), 3),
            **{f"point_{a}": (round(b, 4) if isinstance(b, float) else b) for a, b in pd.items()},
        })
    return rows
