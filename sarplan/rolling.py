"""
Текущая коррекция плана на скользящем горизонте (статья, формула (20)).

План строится на горизонт T_г, ближайший интервал T_фикс (до сеанса загрузки)
фиксируется. При событии (новая заявка, отказ ресурса, отклонение съёмки)
остаток плана перепланируется со штрафом за изменение решений:

    Э' = Э - κ Σ_{k,ν} |x_{k,ν} - x_{k,ν}^пред| · α_k.

Показатель устойчивости — доля изменённых решений Д = Σ|x - x^пред| / Σ x^пред.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .criterion import CoverageModel, Element, Frame
from .planner import PlanParams, plan_pass
from .windows import PassGeometry


@dataclass
class RollingEvent:
    t_h: float
    kind: str                       # urgent | budget | (служебные: pass)
    data: dict = field(default_factory=dict)


@dataclass
class RollingResult:
    strategy: str
    coverage: float                 # Э / Σα с учётом χ
    on_time: float                  # покрытие к сроку (χ = 0 после срока)
    urgent: float                   # выполнение срочной заявки к сроку
    changes_share: float            # Д
    cycle_time_s: float
    executed: Dict[int, List[int]] = field(default_factory=dict)


class RollingPlanner:
    """
    Моделирование исполнения плана на горизонте с событиями.

    geoms   — PassGeometry пролётов (в порядке времени) для полного набора кадров;
    pass_h  — время пролётов, ч от начала горизонта;
    hidden  — элементы, появляющиеся по событию urgent (их кадры недоступны до события);
    loss    — матрица (K, ν): кадр k, снятый на пролёте ν, потерян.
    """

    def __init__(self, model: CoverageModel, geoms: Sequence[PassGeometry], pass_h: Sequence[float],
                 events: Sequence[RollingEvent], loss: np.ndarray, hidden: Sequence[str] = (),
                 method: str = "proposed_cluster", t_fix_h: float = 1.0,
                 params: Optional[PlanParams] = None):
        self.model, self.geoms, self.pass_h = model, list(geoms), list(pass_h)
        self.events = sorted(events, key=lambda e: e.t_h)
        self.loss = loss
        self.hidden = set(hidden)
        self.method, self.t_fix_h = method, t_fix_h
        self.params = params or PlanParams()
        self.K = len(model.frames)
        self.hidden_frames = {k for k, f in enumerate(model.frames) if set(f.cover) & self.hidden}

    # ---------------------------------------------------------------------------
    def _budget(self, nu: int, t_known: float) -> float:
        b = self.geoms[nu].sensor.budget_s
        for e in self.events:
            if e.kind == "budget" and e.t_h <= t_known and nu in e.data["passes"]:
                b = e.data["budget_s"]
        return b

    def _truncate(self, items, nu, budget):
        out, used = [], 0.0
        tf = self.geoms[nu].sensor.t_frame_s
        for k, s in sorted(items, key=lambda x: x[1]):
            d = self.model.frames[k].duration(tf)
            if used + d <= budget + 1e-9:
                out.append((k, s))
                used += d
        return out

    def _plan_from(self, t_now: float, acc0: np.ndarray, done: set, plans: Dict[int, list],
                   executed: set, kappa: Optional[float]):
        """Перепланировать незафиксированные пролёты; вернуть новые планы и число изменений."""
        acc = acc0.copy()
        avail = [k for k in range(self.K) if k not in done
                 and (t_now >= self._urgent_time() or k not in self.hidden_frames)]
        new_plans = dict(plans)
        changes, prev_total = 0, 0
        taken = set()
        for nu, g in enumerate(self.geoms):
            if nu in executed:
                continue
            fixed = self.pass_h[nu] < t_now + self.t_fix_h and nu in plans
            if fixed:
                items = self._truncate(plans[nu], nu, self._budget(nu, t_now))
                new_plans[nu] = items
                for k, _ in items:
                    self.model.add(acc, k, self.pass_h[nu])
                    taken.add(k)
                continue
        for nu, g in enumerate(self.geoms):
            if nu in executed or (self.pass_h[nu] < t_now + self.t_fix_h and nu in plans):
                continue
            cand = [k for k in avail if k not in taken]
            prev = {k for k, _ in plans.get(nu, [])}
            bonus = None
            if kappa is not None and nu in plans:
                fa = self.model.frame_alpha
                bonus = np.where(np.isin(np.arange(self.K), list(prev)), kappa * fa, -kappa * fa)
            pp = PlanParams(budget_s=self._budget(nu, t_now), zeta=self.params.zeta,
                            local_search=self.params.local_search)
            plan = plan_pass(g, self.model, acc, self.pass_h[nu], cand, self.method, pp, bonus=bonus)
            new_plans[nu] = plan.items
            cur = set(plan.frames)
            if nu in plans:
                changes += len(cur ^ prev)
                prev_total += len(prev)
            for k in cur:
                self.model.add(acc, k, self.pass_h[nu])
                taken.add(k)
        return new_plans, changes, prev_total

    def _urgent_time(self) -> float:
        ts = [e.t_h for e in self.events if e.kind == "urgent"]
        return min(ts) if ts else 0.0

    # ---------------------------------------------------------------------------
    def run(self, kappa: Optional[float], replan: bool = True) -> RollingResult:
        """replan=False — план строится один раз и не меняется (стратегия «без перепланирования»)."""
        model = self.model
        acc, acc_strict = model.new_acc(), model.new_acc()
        done, executed = set(), set()
        plans: Dict[int, list] = {}
        log: Dict[int, List[int]] = {}
        changes = prev_total = 0
        cycle_times = []

        cycles = [0.0]
        if replan:
            cycles += [e.t_h for e in self.events]
            cycles += [h + 0.01 for h in self.pass_h]
        cycles = sorted(set(cycles))

        def execute_until(t):
            for nu in range(len(self.geoms)):
                if nu in executed or self.pass_h[nu] > t:
                    continue
                items = self._truncate(plans.get(nu, []), nu, self._budget(nu, self.pass_h[nu]))
                shot = []
                for k, _ in items:
                    if k in done or self.loss[k, nu]:
                        continue
                    model.add(acc, k, self.pass_h[nu])
                    model.add(acc_strict, k, self.pass_h[nu], strict=True)
                    done.add(k)
                    shot.append(k)
                log[nu] = shot
                executed.add(nu)

        for t in cycles:
            execute_until(t)
            t0 = time.perf_counter()
            plans, ch, pt = self._plan_from(t, acc, done, plans, executed, kappa if t > 0 else None)
            cycle_times.append(time.perf_counter() - t0)
            if t > 0:
                changes += ch
                prev_total += pt
        execute_until(float("inf"))

        urgent = [e for e in model.elements if e.eid in self.hidden]
        urg = float(np.mean([min(1.0, acc_strict[model.eidx[e.eid]]) for e in urgent])) if urgent else float("nan")
        name = "static" if not replan else f"kappa={kappa}"
        return RollingResult(name, model.coverage(acc), model.coverage(acc_strict), urg,
                             changes / prev_total if prev_total else float("nan"),
                             float(np.mean(cycle_times)), log)
