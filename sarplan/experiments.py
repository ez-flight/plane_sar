"""
Численная проверка и моделирование (статья, раздел «Численная проверка и
моделирование»): таблицы 3–7 и рисунки 1–2.

Сценарии синтетические, параметры малого КА приняты по аналогам (таблица 2),
поэтому числа близки к статье по характеру, но не обязаны совпадать до знака.
"""

import csv
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import duration, geometry, targets
from .config import ICEYE_CLASS, KONDOR_FKA, OBZOR_R, REGION_CENTER, SensorParams
from .constellation import PassCache, build_constellation, constellation_passes, time_to_complete
from .criterion import CoverageModel
from .frames import azimuth_between, ecef_to_geodetic, geodetic_to_ecef, offset_point, to_dt64
from .orbit import find_passes, make_sso_satellite
from .planner import PlanParams, admissible_items, plan_n_max, plan_pass
from .rolling import RollingEvent, RollingPlanner
from .windows import PassGeometry

EPOCH = datetime(2026, 6, 1)
METHOD_LABELS = {
    "greedy_priority": "(1) Жадный по приоритету, номинальная точка",
    "seq_analog2": "(2) Аналог [2] без выбора подмножества",
    "proposed": "(3) Предлагаемый, номинальная точка",
    "proposed_cluster": "(4) Предлагаемый, кластерный список точек",
    "proposed_full": "(5) Предлагаемый, полный перебор точек",
    "dp_additive": "(6) ДП, номинальная точка",
    "greedy_priority_sat": "Жадный с учётом насыщения",
}


# --- общие ----------------------------------------------------------------------
def _write(out_dir: Optional[str], name: str, text: str, rows: Optional[List[dict]] = None):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(text)
    if rows:
        with open(os.path.join(out_dir, name + ".csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def _md_table(header: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _wilcoxon(a, b) -> str:
    from scipy.stats import wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    same = int(np.sum(np.isclose(a, b)))
    try:
        p = wilcoxon(a, b).pvalue if same < len(a) else 1.0
    except ValueError:
        p = 1.0
    return f"совпадение в {same} из {len(a)}, p = {p:.2g}"


def _ground_azimuth(pas, i: int) -> float:
    la1, lo1, _ = ecef_to_geodetic(pas.r[max(i - 4, 0)])
    la2, lo2, _ = ecef_to_geodetic(pas.r[min(i + 4, len(pas.r) - 1)])
    return azimuth_between(la1, lo1, la2, lo2)


def reference_pass(sensor: SensorParams = ICEYE_CLASS, center=REGION_CENTER, epoch=EPOCH,
                   look_deg: float = 37.5, days: int = 3):
    """Опорный пролёт для сценариев одного пролёта: центр района под углом визирования ≈ look_deg."""
    sat = make_sso_satellite("REF", 99001, epoch, sensor.alt_km, 10.5, 0.0, sensor.incl_deg)
    c = geodetic_to_ecef(center[0], center[1], 0.0)
    passes = find_passes(sat, c, epoch, epoch + timedelta(days=days), look_min_deg=sensor.beta_um_min,
                         look_max_deg=sensor.beta_um_max, squint_max_deg=sensor.beta_sk_max, dt_s=sensor.dt_s)

    def look_at_broadside(p):
        sk, lk = geometry.squint_look(p.r, p.v, c[None])
        return lk[np.argmin(np.abs(sk))]

    p = min(passes, key=lambda q: abs(look_at_broadside(q) - look_deg))
    sk, _ = geometry.squint_look(p.r, p.v, c[None])
    return sat, p, _ground_azimuth(p, int(np.argmin(np.abs(sk))))


# --- проверка формулы длительности и рис. 1 ---------------------------------------
def exp_duration(out_dir: Optional[str] = None, length_km: float = 103.0, cross_km: float = 353.0,
                 betas=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20)) -> str:
    v = duration.verify_closed_form()
    text = "## Проверка формулы длительности наблюдения (3)\n\n"
    text += (f"Орбита 500×1500 км (p = {v['p_km']:.1f} км, e = {v['e']:.4f}). Максимальное "
             f"относительное расхождение (3) с численным интегралом (1) и уравнением Кеплера: "
             f"{v['max_rel_error']:.1e}.\n\n"
             f"При ϑ₀ = {v['theta0']}, ϑ₁ = {v['theta1']} рад (ϑ₁ > π): с поправкой "
             f"{v['t_correct_s']:.1f} с, без поправки {v['t_no_correction_s']:.1f} с, "
             f"по Кеплеру {v['t_kepler_s']:.1f} с.\n\n")

    sens = ICEYE_CLASS
    sat, pas, az = reference_pass()
    c = geodetic_to_ecef(*REGION_CENTER, 0.0)
    sk_c, _ = geometry.squint_look(pas.r, pas.v, c[None])
    ib = int(np.argmin(np.abs(sk_c)))
    la_s, lo_s, _ = ecef_to_geodetic(pas.r[ib])
    side = 1.0 if np.dot(np.cross(pas.v[ib], pas.r[ib]), c - pas.r[ib]) > 0 else -1.0
    la_c, lo_c = offset_point(float(la_s), float(lo_s), az, 0.0, side * cross_km)
    ends = [offset_point(float(la_c), float(lo_c), az, s * length_km / 2, 0.0) for s in (-1, 1)]
    E = np.array([geodetic_to_ecef(float(a), float(b), 0.0) for a, b in ends])   # [задний, передний]
    tt = pas.time_at(ib) + (np.arange(-400.0, 400.0, 0.05) * 1e6).astype("timedelta64[us]")
    r, vv = sat.states_ecef(tt)
    ts = (tt - tt[0]) / np.timedelta64(1, "s")
    sk_rear, _ = geometry.squint_look(r, vv, E[0][None])
    sk_front, _ = geometry.squint_look(r, vv, E[1][None])
    rows, num, cor, ref = [], [], [], []
    alt = float(np.mean(np.linalg.norm(r, axis=1))) - 6371.0
    for b in betas:
        i0 = int(np.flatnonzero(sk_rear <= b)[0])
        i1 = int(np.flatnonzero(sk_front >= -b)[-1])
        t_num = ts[i1] - ts[i0]
        R0, g0, l0 = geometry.roll_azimuth(r[i0], vv[i0], E[0])
        R1, g1, l1 = geometry.roll_azimuth(r[i1], vv[i1], E[1])
        t_cor = duration.duration_circular(duration.epsilon_corrected(R0, g0, l0),
                                           duration.epsilon_corrected(R1, g1, l1), length_km, alt)
        t_ref = duration.duration_circular(duration.epsilon_ref4(R0, g0, l0),
                                           duration.epsilon_ref4(R1, g1, l1), length_km, alt)
        num.append(t_num), cor.append(t_cor), ref.append(t_ref)
        rows.append({"beta_sk_max_deg": b, "numeric_s": round(t_num, 2), "formula5_s": round(t_cor, 2),
                     "ref4_s": round(t_ref, 2), "err5_pct": round(100 * abs(t_cor - t_num) / t_num, 2),
                     "err4_pct": round(100 * abs(t_ref - t_num) / t_num, 1)})
    e5 = [r_["err5_pct"] for r_ in rows]
    e4 = [r_["err4_pct"] for r_ in rows]
    text += f"## Рисунок 1. Длительность наблюдения линии {length_km:.0f} км (боковое удаление {cross_km:.0f} км)\n\n"
    text += _md_table(["β_ск max, °", "Численно, с", "(1),(4) с ε по (5), с", "(1),(4) с ε по [4], с",
                       "Ошибка (5), %", "Ошибка [4], %"],
                      [list(r_.values()) for r_ in rows])
    text += (f"\nОшибка с исправленным ε: в среднем {np.mean(e5):.1f} % (максимум {np.max(e5):.1f} %); "
             f"с ε в виде [4]: в среднем {np.mean(e4):.0f} % (максимум {np.max(e4):.0f} %).\n")
    if out_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(betas, num, "k-o", ms=3, label="численно (вращающаяся Земля)")
        ax.plot(betas, cor, "b--s", ms=3, label="(1), (4), ε по (5)")
        ax.plot(betas, ref, "r:^", ms=3, label="(1), (4), ε по [4]")
        ax.set_xlabel("Предельный угол скоса β_ск max, °")
        ax.set_ylabel("Длительность наблюдения, с")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig1_duration.png"), dpi=150)
        plt.close(fig)
    _write(out_dir, "fig1_duration", text, rows)
    return text


# --- один пролёт: таблица 3 --------------------------------------------------------
def _single_pass_sets(n_sets, n_targets, seed0, sensor=ICEYE_CLASS, builder=None):
    _, pas, az = reference_pass(sensor)
    builder = builder or (lambda rng: targets.random_targets(rng, REGION_CENTER, az, n=n_targets))
    for s in range(n_sets):
        els, frs = builder(np.random.default_rng(seed0 + s))
        yield PassGeometry(pas, frs, sensor, center_ecef=geodetic_to_ecef(*REGION_CENTER, 0.0)), CoverageModel(els, frs)


def exp_table3(out_dir: Optional[str] = None, n_sets: int = 30, n_targets: int = 20,
               methods=("greedy_priority", "seq_analog2", "proposed", "proposed_cluster",
                        "proposed_full", "dp_additive")) -> str:
    res = {m: [] for m in methods}
    for g, model in _single_pass_sets(n_sets, n_targets, 1000):
        acc = model.new_acc()
        for m in methods:
            pl = plan_pass(g, model, acc, 0.0, range(len(model.frames)), m)
            res[m].append((pl.gain / model.total_alpha, len(pl.items), pl.runtime_s))
    rows, csv_rows = [], []
    for m in methods:
        a = np.array(res[m])
        rows.append([METHOD_LABELS[m], f"{a[:, 0].mean():.3f} ± {a[:, 0].std():.3f}",
                     f"{a[:, 1].mean():.1f}", f"{a[:, 2].mean():.3f}"])
        csv_rows.append({"method": m, "coverage_mean": a[:, 0].mean(), "coverage_std": a[:, 0].std(),
                         "frames": a[:, 1].mean(), "time_s": a[:, 2].mean()})
    cov = {m: np.array(res[m])[:, 0] for m in methods}
    text = f"## Таблица 3. Результаты для {n_sets} наборов по {n_targets} целей\n\n"
    text += _md_table(["Метод", "Покрытие", "Кадров", "Время, с"], rows)
    if "proposed" in cov:
        text += "\nПарный критерий Уилкоксона, предлагаемый (3) против:\n"
        for m in methods:
            if m != "proposed":
                d = 100 * (cov["proposed"].mean() - cov[m].mean())
                text += f"- {METHOD_LABELS[m]}: {d:+.1f} п. п., {_wilcoxon(cov['proposed'], cov[m])}\n"
    # эвристика против точного перебора на наборах по 12 кадров
    hit = []
    for g, model in _single_pass_sets(n_sets, 12, 5000):
        acc = model.new_acc()
        a = plan_pass(g, model, acc, 0.0, range(12), "proposed").gain
        b = plan_pass(g, model, acc, 0.0, range(12), "exact").gain
        d = plan_pass(g, model, acc, 0.0, range(12), "dp_additive").gain
        hit.append((abs(a - b) < 1e-9, abs(d - b) < 1e-9))
    hit = np.array(hit)
    text += (f"\nНа наборах из 12 кадров эвристика нашла оптимум (точный перебор) в "
             f"{100 * hit[:, 0].mean():.1f} % наборов, ДП — в {100 * hit[:, 1].mean():.1f} %.\n")
    _write(out_dir, "table3_single_pass", text, csv_rows)
    return text


# --- альтернативные кадры (дубли) ---------------------------------------------------
def exp_duplicates(out_dir: Optional[str] = None, n_sets: int = 30) -> str:
    _, pas, az = reference_pass()
    methods = ("dp_additive", "proposed", "greedy_priority_sat")
    res = {m: [] for m in methods}
    for g, model in _single_pass_sets(n_sets, 0, 2000,
                                      builder=lambda rng: targets.duplicate_targets(rng, REGION_CENTER, az)):
        acc = model.new_acc()
        for m in methods:
            res[m].append(plan_pass(g, model, acc, 0.0, range(len(model.frames)), m).gain / model.total_alpha)
    rows = [[METHOD_LABELS[m], f"{np.mean(res[m]):.3f} ± {np.std(res[m]):.3f}"] for m in methods]
    text = f"## Альтернативные кадры: 14 объектов с двумя кадрами-дублями и 10 одиночных ({n_sets} наборов)\n\n"
    text += _md_table(["Метод", "Покрытие"], rows)
    d = 100 * (np.mean(res["proposed"]) - np.mean(res["dp_additive"]))
    text += f"\nПредлагаемый против аддитивного ДП: {d:+.1f} п. п., {_wilcoxon(res['proposed'], res['dp_additive'])}.\n"
    _write(out_dir, "duplicates", text, [{"method": m, "coverage": float(np.mean(res[m]))} for m in methods])
    return text


# --- чувствительность ---------------------------------------------------------------
def exp_sensitivity(out_dir: Optional[str] = None, n_sets: int = 10) -> str:
    variants = [("t_вкл,1 = 2,5 с", dict(t_frame_s=2.5), 20), ("t_вкл,1 = 3,5 с", {}, 20),
                ("t_вкл,1 = 4,5 с", dict(t_frame_s=4.5), 20),
                ("Δβ max = 2°", dict(d_beta_sk_max=2.0, d_beta_um_max=2.0), 20),
                ("Δβ max = 3°", dict(d_beta_sk_max=3.0, d_beta_um_max=3.0), 20),
                ("Δβ max = 4°", dict(d_beta_sk_max=4.0, d_beta_um_max=4.0), 20),
                ("Δβ max = 6°", dict(d_beta_sk_max=6.0, d_beta_um_max=6.0), 20),
                ("15 целей", {}, 15), ("25 целей", {}, 25), ("30 целей", {}, 30)]
    methods = ("seq_analog2", "greedy_priority", "proposed", "proposed_cluster", "proposed_full")
    rows, csv_rows = [], []
    for label, kw, n in variants:
        sens = ICEYE_CLASS.with_(**kw)
        r = {m: [] for m in methods}
        nfr = []
        for g, model in _single_pass_sets(n_sets, n, 3000, sensor=sens):
            acc = model.new_acc()
            for m in methods:
                pl = plan_pass(g, model, acc, 0.0, range(len(model.frames)), m)
                r[m].append(pl.gain / model.total_alpha)
                if m == "proposed":
                    nfr.append(len(pl.items))
        mean = {m: np.mean(r[m]) for m in methods}
        rows.append([label, f"{100 * (mean['proposed'] - mean['seq_analog2']):+.1f}",
                     f"{100 * (mean['proposed_cluster'] - mean['seq_analog2']):+.1f}",
                     f"{100 * (mean['proposed_full'] - mean['seq_analog2']):+.1f}",
                     f"{100 * (mean['proposed'] - mean['greedy_priority']):+.1f}", f"{np.mean(nfr):.1f}"])
        csv_rows.append({"variant": label, **{m: mean[m] for m in methods}, "frames": np.mean(nfr)})
    text = f"## Чувствительность к параметрам ({n_sets} наборов на вариант), п. п. покрытия\n\n"
    text += _md_table(["Вариант", "(3) − (2)", "(4) − (2)", "(5) − (2)", "(3) − (1)", "Кадров (3)"], rows)
    _write(out_dir, "sensitivity", text, csv_rows)
    return text


# --- вклад модели допустимости: таблица 4 -------------------------------------------
def exp_table4(out_dir: Optional[str] = None, n_sets: int = 30,
               limits=((5, 5), (5, 10), (3, 10), (2.2, 10))) -> str:
    rows, csv_rows = [], []
    for sk_lim, um_lim in limits:
        sens = ICEYE_CLASS.with_(d_beta_sk_max=sk_lim, d_beta_um_max=um_lim)
        nm, nom, full, share = [], [], [], []
        for g, model in _single_pass_sets(n_sets, 20, 4000, sensor=sens):
            acc = model.new_acc()
            cand = range(len(model.frames))
            p0 = plan_pass(g, model, acc, 0.0, cand, "proposed", ignore_deflection=True)
            real = admissible_items(p0, g)
            nm.append(model.set_value(acc, [k for k, _ in real], 0.0) / model.total_alpha)
            share.append(len(real) / max(len(p0.items), 1))
            nom.append(plan_pass(g, model, acc, 0.0, cand, "proposed").gain / model.total_alpha)
            full.append(plan_pass(g, model, acc, 0.0, cand, "proposed_full").gain / model.total_alpha)
        a, b, c = np.mean(nm), np.mean(nom), np.mean(full)
        rows.append([f"{sk_lim:g}° / {um_lim:g}°", f"{a:.3f}", f"{b:.3f} ({100 * (b - a):+.1f})",
                     f"{c:.3f} ({100 * (c - a):+.1f})", f"{100 * np.mean(share):.0f} %"])
        csv_rows.append({"d_beta_sk_max": sk_lim, "d_beta_um_max": um_lim, "no_model": a,
                         "model_nominal": b, "model_full": c, "admissible_share": np.mean(share)})
    text = f"## Таблица 4. Покрытие планов без модели и по модели допустимости ({n_sets} наборов)\n\n"
    text += _md_table(["Пределы Δβ_ск max / Δβ_ум max", "Без модели", "Модель, номинальная точка",
                       "Модель, перебор точек", "Снимаемых кадров в плане без модели"], rows)
    _write(out_dir, "table4_admissibility", text, csv_rows)
    return text


# --- текущая коррекция: таблица 5 ----------------------------------------------------
def rolling_passes(sensor=ICEYE_CLASS, epoch=EPOCH, horizon_h=48.0):
    """Четыре пролёта горизонта (центр района в полосе 20…55°): до срочной заявки, между событиями и два после снижения бюджета."""
    sat = make_sso_satellite("ROLL", 99101, epoch, sensor.alt_km, 10.5, 0.0, sensor.incl_deg)
    c = geodetic_to_ecef(*REGION_CENTER, 0.0)
    ps = find_passes(sat, c, epoch, epoch + timedelta(hours=horizon_h), look_min_deg=sensor.beta_um_min,
                     look_max_deg=sensor.beta_um_max, squint_max_deg=sensor.beta_sk_max, dt_s=sensor.dt_s)
    def usable(p):
        sk, lk = geometry.squint_look(p.r, p.v, c[None])
        return sensor.beta_um_min <= lk[np.argmin(np.abs(sk))] <= sensor.beta_um_max

    ps = [p for p in ps if usable(p)]
    hours = [p.hours_since(epoch) for p in ps]
    chosen = []
    for lo, n in ((2.0, 1), (16.0, 1), (31.0, 2)):
        idx = [i for i, h in enumerate(hours) if h >= lo and i not in chosen][:n]
        chosen += idx
    chosen = sorted(chosen)[:4]
    return [ps[i] for i in chosen], [hours[i] for i in chosen]


def exp_table5(out_dir: Optional[str] = None, n_scen: int = 12, kappas=(0.0, 0.05, 0.1, 0.2)) -> str:
    sens = ICEYE_CLASS
    passes, hours = rolling_passes(sens)
    c = geodetic_to_ecef(*REGION_CENTER, 0.0)
    az = _ground_azimuth(passes[0], len(passes[0].t) // 2)
    results: Dict[str, list] = {}
    for sc in range(n_scen):
        rng = np.random.default_rng(7000 + sc)
        base = targets.rolling_targets(rng, REGION_CENTER, az)
        urg = targets.urgent_request(rng, REGION_CENTER, az)
        els, frs = targets.merge(base, urg)
        model = CoverageModel(els, frs)
        geoms = [PassGeometry(p, frs, sens, center_ecef=c) for p in passes]
        events = [RollingEvent(15.0, "urgent"),
                  RollingEvent(30.0, "budget", {"passes": [len(passes) - 2, len(passes) - 1], "budget_s": 40.0})]
        loss = rng.random((len(frs), len(passes))) < 0.15
        rp = RollingPlanner(model, geoms, hours, events, loss, hidden=["URGENT"])
        out = [rp.run(None, replan=False)] + [rp.run(k) for k in kappas]
        for r in out:
            results.setdefault(r.strategy, []).append(r)
    rows, csv_rows = [], []
    labels = {"static": "Без перепланирования"}
    for name, rs in results.items():
        cov = np.array([r.coverage for r in rs])
        ont = np.array([r.on_time for r in rs])
        urg = np.array([r.urgent for r in rs])
        ch = np.array([r.changes_share for r in rs])
        tm = np.array([r.cycle_time_s for r in rs])
        lab = labels.get(name, ("Перепланирование, " if name == "kappa=0.0" else "") + name.replace("kappa", "κ "))
        rows.append([lab, f"{cov.mean():.3f} ± {cov.std():.3f}", f"{ont.mean():.3f} ± {ont.std():.3f}",
                     f"{urg.mean():.2f} ± {urg.std():.2f}",
                     "—" if name == "static" else f"{np.nanmean(ch):.2f} ± {np.nanstd(ch):.2f}",
                     "—" if name == "static" else f"{tm.mean():.2f}"])
        csv_rows.append({"strategy": name, "coverage": cov.mean(), "on_time": ont.mean(), "urgent": urg.mean(),
                         "changes_share": float(np.nanmean(ch)) if name != "static" else "", "cycle_s": tm.mean()})
    text = (f"## Таблица 5. Результаты текущей коррекции ({n_scen} сценариев)\n\n"
            f"Пролёты: {', '.join(f'{h:.1f} ч' for h in hours)}; срочная заявка в 15 ч (3 кадра, α = 8, срок 30 ч); "
            f"в 30 ч бюджет двух последних пролётов 40 с; потеря кадра с вероятностью 0,15.\n\n")
    text += _md_table(["Стратегия", "Покрытие Э", "К сроку", "Срочная заявка", "Доля Д", "Время цикла, с"], rows)
    _write(out_dir, "table5_rolling", text, csv_rows)
    return text


# --- группировки: таблица 6, рис. 2 и таблица 7 ---------------------------------------
CONSTELLATIONS = {1: (1, 1), 3: (1, 3), 6: (2, 3), 12: (4, 3)}


def _completion_cases(sensor, sats_cfg, scenario, n_phases, n_arrivals, horizon_days, epoch=EPOCH,
                      method="proposed_cluster"):
    els, frs = scenario
    center = targets.region_center_of(frs)
    cache_frames = frs
    out = []
    n_planes, per_plane = sats_cfg
    for ph in range(n_phases):
        phase = ph * (360.0 / per_plane) / n_phases
        sats = build_constellation(n_planes, per_plane, epoch, sensor, phase_deg=phase)
        passes = constellation_passes(sats, center, epoch, epoch + timedelta(days=horizon_days + 1), sensor)
        cache = PassCache(cache_frames, sensor, center)
        for a in range(n_arrivals):
            t_req = to_dt64(epoch + timedelta(hours=24.0 * a / n_arrivals))
            model = CoverageModel(els, frs)
            out.append(time_to_complete(passes, model, cache, t_req, horizon_days * 24.0, method))
    return out


def exp_table6(out_dir: Optional[str] = None, n_phases: int = 6, n_arrivals: int = 8,
               horizon_days: int = 9, sizes=(1, 3, 6, 12)) -> str:
    scen = targets.dnestr_targets()
    rows, csv_rows, dist = [], [], {}
    for n in sizes:
        res = _completion_cases(ICEYE_CLASS, CONSTELLATIONS[n], scen, n_phases, n_arrivals, horizon_days)
        first = np.array([r.first_h for r in res])
        alls = np.array([r.all_h for r in res])
        frac = np.array([r.frac_12h for r in res])
        closed = alls[~np.isnan(alls)]
        dist[n] = closed
        q = lambda x, p: np.nanpercentile(x, p) if np.any(~np.isnan(x)) else np.nan
        rows.append([n, f"{q(first, 50):.1f} ({q(first, 90):.1f})",
                     f"{q(closed, 50):.1f} ({q(closed, 90):.1f})" + ("*" if len(closed) < len(res) else ""),
                     f"{np.mean(frac):.2f}", f"{int(np.sum(np.isnan(alls)))} из {len(res)}"])
        csv_rows.append({"n_sats": n, "first_median_h": q(first, 50), "first_p90_h": q(first, 90),
                         "all_median_h": q(closed, 50), "all_p90_h": q(closed, 90),
                         "frac_12h": float(np.mean(frac)), "not_closed": int(np.sum(np.isnan(alls))),
                         "max_frames_per_pass": int(max(r.max_per_pass for r in res))})
    text = (f"## Таблица 6. Время съёмки 40 кадров группировкой малых КА, ч "
            f"(медиана и 90-й процентиль по {n_phases * n_arrivals} случаям)\n\n")
    text += _md_table(["Число КА", "До первой съёмки", "До съёмки всех кадров", "Доля кадров за 12 ч",
                       "Не закрыто за горизонт"], rows)
    text += f"\n\\* По закрытым случаям (горизонт {horizon_days} сут).\n"
    if out_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        data = [dist[n] if len(dist[n]) else [np.nan] for n in sizes]
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(sizes) + 1), [str(n) for n in sizes])
        ax.set_yscale("log")
        ax.set_xlabel("Число КА в группировке")
        ax.set_ylabel("Время до съёмки всех кадров, ч")
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig2_constellation.png"), dpi=150)
        plt.close(fig)
    _write(out_dir, "table6_constellation", text, csv_rows)
    return text


def exp_table7(out_dir: Optional[str] = None, n_phases: int = 6, n_arrivals: int = 8,
               horizon_days: int = 20) -> str:
    cases = [
        ("Малый КА, гибридный режим (21 рамка 5×5 км)", ICEYE_CLASS, (1, 1), targets.dnestr_targets(21, 5.0, 0)),
        ("«Обзор-Р» (11 рамок 10×15 км)", OBZOR_R, (1, 1), targets.dnestr_targets(11, 10.0, 0)),
        ("«Кондор-ФКА» (11 рамок 10×10 км), один КА", KONDOR_FKA, (1, 1), targets.dnestr_targets(11, 10.0, 0)),
        ("«Кондор-ФКА» (11 рамок 10×10 км), два КА", KONDOR_FKA, (1, 2), targets.dnestr_targets(11, 10.0, 0)),
    ]
    rows, csv_rows = [], []
    for label, sens, cfg, scen in cases:
        res = _completion_cases(sens, cfg, scen, n_phases, n_arrivals, horizon_days)
        n = len(scen[1])
        per = [r.max_per_pass for r in res]
        first = np.array([r.first_h for r in res])
        alls = np.array([r.all_h for r in res])
        sector = (f"±{sens.d_beta_sk_max:g}°" if sens.mode == "electronic"
                  else f"скос до {sens.beta_sk_max:g}°, отклонение луча до {sens.d_beta_sk_max:g}°")
        rows.append([label, sector, f"{int(np.median(per))} из {n} (Δt = {sens.gap_s:g} с)",
                     f"первый пролёт {np.nanmedian(first):.1f} ч, вся цепочка {np.nanmedian(alls) / 24:.1f} сут "
                     f"(медиана; не закрыто {int(np.sum(np.isnan(alls)))} из {len(res)})"])
        csv_rows.append({"case": label, "frames_per_pass_median": float(np.median(per)), "n_frames": n,
                         "first_median_h": float(np.nanmedian(first)), "all_median_h": float(np.nanmedian(alls)),
                         "not_closed": int(np.sum(np.isnan(alls)))})
    text = "## Таблица 7. Цепочка вдоль Днестра для КА разных классов\n\n"
    text += _md_table(["КА (цепочка)", "Сектор по азимуту", "Кадров за пролёт (медиана лучшего)",
                       "Время съёмки цепочки"], rows)
    _write(out_dir, "table7_classes", text, csv_rows)
    return text


def exp_dnestr_single_pass(out_dir: Optional[str] = None) -> str:
    """Линия из 21 кадра за один пролёт и оценка (19)."""
    sens = ICEYE_CLASS
    els, frs = targets.dnestr_targets(21, 5.0, 0)
    center = targets.region_center_of(frs)
    sat = make_sso_satellite("LINE", 99201, EPOCH, sens.alt_km, 10.5, 0.0, sens.incl_deg)
    c = geodetic_to_ecef(center[0], center[1], 0.0)
    ps = find_passes(sat, c, EPOCH, EPOCH + timedelta(days=5), look_min_deg=sens.beta_um_min,
                     look_max_deg=sens.beta_um_max, squint_max_deg=sens.beta_sk_max)
    best = None
    for p in ps:
        g = PassGeometry(p, frs, sens, center_ecef=c)
        m = CoverageModel(els, frs)
        pl = plan_pass(g, m, m.new_acc(), 0.0, range(len(frs)), "proposed_cluster")
        if best is None or len(pl.items) > len(best[1].items):
            best = (g, pl)
    g, pl = best
    n = len(pl.items)
    on = n * sens.t_frame_s
    span = (pl.items[-1][1] + g.n_dur[pl.items[-1][0]] - pl.items[0][1]) * g.dt if n else 0
    win = g.union_length_s(pl.point)
    text = "## Линия из 21 кадра за один пролёт\n\n"
    text += (f"Снято {n} из 21 кадров: {on:.1f} с съёмки, {span:.1f} с с паузами, окно допустимости "
             f"{win:.1f} с; оценка (19): N_max = {plan_n_max(g, pl.point)}.\n")
    _write(out_dir, "line21_single_pass", text)
    return text


ALL = {
    "duration": exp_duration, "table3": exp_table3, "duplicates": exp_duplicates,
    "sensitivity": exp_sensitivity, "table4": exp_table4, "table5": exp_table5,
    "line21": exp_dnestr_single_pass, "table6": exp_table6, "table7": exp_table7,
}

QUICK = {
    "table3": dict(n_sets=8), "duplicates": dict(n_sets=8), "sensitivity": dict(n_sets=3),
    "table4": dict(n_sets=8), "table5": dict(n_scen=3), "table6": dict(n_phases=2, n_arrivals=2),
    "table7": dict(n_phases=2, n_arrivals=2),
}


def run_all(out_dir: str = "results", names: Sequence[str] = None, quick: bool = False) -> str:
    names = list(names or ALL)
    parts = ["# Результаты моделирования (sarplan)\n",
             f"Сформировано {datetime.now():%Y-%m-%d %H:%M}. Режим: {'быстрый' if quick else 'полный'}.\n"]
    for n in names:
        t = time.perf_counter()
        kw = QUICK.get(n, {}) if quick else {}
        parts.append(ALL[n](out_dir, **kw))
        print(f"[{n}] {time.perf_counter() - t:.1f} с", flush=True)
    report = "\n".join(parts)
    if out_dir and set(names) == set(ALL):
        with open(os.path.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
            f.write(report)
    return report
