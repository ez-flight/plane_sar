"""
Расчёт съёмки заданной цепочки кадров по пролётам: за сколько пролётов
снимаются все кадры и какие кадры (по номерам) снимаются на каждом витке.

Кадры берутся из GeoPackage (полигоны с полем seq), КА — синтетические (ССО)
или из файла TLE. На каждом пролёте методика выбирает подмножество
оставшихся кадров (формулы (16)–(19)), снятые кадры исключаются.
"""

import csv
import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

import numpy as np

from . import geometry
from .config import SensorParams
from .constellation import PassCache, constellation_passes
from .criterion import CoverageModel, Element, Frame
from .gis import read_gpkg_frames
from .orbit import Satellite
from .planner import plan_pass
from .targets import region_center_of


def chain_scenario(frames_info: Sequence[dict], chain_id: str = "CHAIN"):
    """Цепочка как один линейный элемент: каждый кадр даёт долю 1/N покрытия."""
    n = len(frames_info)
    els = [Element(chain_id, "line", 1.0)]
    frs = [Frame(str(f["num"]), f["lat"], f["lon"], {chain_id: 1.0 / n}) for f in frames_info]
    return els, frs


def plan_chain(frames_info: Sequence[dict], sats: Sequence[Satellite], sensor: SensorParams,
               t_start: datetime, days: float, method: Optional[str] = None) -> dict:
    if method is None:
        method = "strip" if sensor.mode == "stripmap" else "proposed_cluster"
    els, frs = chain_scenario(frames_info)
    center = region_center_of(frs)
    passes = constellation_passes(sats, center, t_start, t_start + timedelta(days=days), sensor)
    by_name = {s.name: s for s in sats}
    cache = PassCache(frs, sensor, center)
    model = CoverageModel(els, frs)
    acc = model.new_acc()
    remaining = set(range(len(frs)))
    rows, shots = [], {}
    for p in passes:
        h = p.hours_since(t_start)
        g = cache.get(p)
        plan = plan_pass(g, model, acc, h, sorted(remaining), method)
        if not plan.items:
            continue
        for k in plan.frames:
            model.add(acc, k, h)
            remaining.discard(k)
        sat = by_name[p.sat_name]
        sk, lk = geometry.squint_look(p.r, p.v, cache.center[None])
        ib = int(np.argmin(np.abs(sk)))
        nums = [frs[k].fid for k, _ in plan.items]
        t_first = p.time_at(plan.items[0][1])
        rows.append({
            "pass": len(rows) + 1, "sat": p.sat_name, "rev": sat.rev_number(t_first),
            "start_utc": str(t_first)[:19], "hours_from_start": round(h, 2),
            "direction": "восходящий" if p.ascending else "нисходящий",
            "look_center_deg": round(float(lk[ib]), 1),
            "swath_km": (g.point_description(plan.point).get("swath_center_km", "")),
            "n_frames": len(nums),
            "frames": " ".join(nums), "cumulative": len(frs) - len(remaining),
        })
        for k, s in plan.items:
            shots[frs[k].fid] = {"pass": rows[-1]["pass"], "sat": p.sat_name, "rev": rows[-1]["rev"],
                                 "shot_utc": str(p.time_at(s))[:19]}
        if not remaining:
            break
    return {"rows": rows, "shots": shots, "n_frames": len(frs), "n_passes_found": len(passes),
            "not_shot": sorted((frs[k].fid for k in remaining), key=int),
            "done_hours": rows[-1]["hours_from_start"] if rows and not remaining else None}


def chain_markdown(res: dict, title: str) -> str:
    out = [f"## {title}\n"]
    if res["done_hours"] is not None:
        out.append(f"Все {res['n_frames']} кадров снимаются за **{len(res['rows'])} пролёт(а/ов)**, "
                   f"последний — через {res['done_hours']:.1f} ч от начала расчёта.\n")
    else:
        out.append(f"За горизонт снято {res['n_frames'] - len(res['not_shot'])} из {res['n_frames']} кадров; "
                   f"не сняты: {', '.join(res['not_shot'])}.\n")
    strip = any(r["swath_km"] != "" for r in res["rows"])
    extra_h, extra_s = (" Центр полосы от трассы, км |", "---|") if strip else ("", "")
    out.append("| Пролёт | КА | Виток | Начало съёмки, UTC | Ветвь | Угол визир. центра, ° |" + extra_h +
               " Кадров | Номера кадров (порядок съёмки) | Всего снято |")
    out.append("|---|---|---|---|---|---|" + extra_s + "---|---|---|")
    for r in res["rows"]:
        ex = f" {abs(r['swath_km']):.1f} |" if strip else ""
        out.append(f"| {r['pass']} | {r['sat']} | {r['rev']} | {r['start_utc']} | {r['direction']} | "
                   f"{r['look_center_deg']} |{ex} {r['n_frames']} | {r['frames']} | {r['cumulative']} |")
    return "\n".join(out) + "\n"


def write_outputs(res: dict, src_gpkg: str, out_dir: str, name: str, title: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    md = chain_markdown(res, title)
    with open(os.path.join(out_dir, name + ".md"), "w", encoding="utf-8") as f:
        f.write(md)
    if res["rows"]:
        with open(os.path.join(out_dir, name + ".csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()))
            w.writeheader()
            w.writerows(res["rows"])
    # копия исходного GeoPackage с номером пролёта, КА, витком и временем съёмки у каждого кадра
    dst = os.path.join(out_dir, name + ".gpkg")
    shutil.copyfile(src_gpkg, dst)
    con = sqlite3.connect(dst)
    try:
        layer = con.execute("select table_name from gpkg_contents where data_type='features'").fetchone()[0]
        cols = [r[1] for r in con.execute(f'pragma table_info("{layer}")')]
        for c, t in (("pass_no", "INTEGER"), ("sat", "TEXT"), ("rev", "INTEGER"), ("shot_utc", "TEXT")):
            if c not in cols:
                con.execute(f'alter table "{layer}" add column "{c}" {t}')
        key = "seq" if "seq" in cols else "fid"
        for num, s in res["shots"].items():
            con.execute(f'update "{layer}" set pass_no=?, sat=?, rev=?, shot_utc=? where "{key}"=?',
                        (s["pass"], s["sat"], s["rev"], s["shot_utc"], int(num)))
        con.commit()
    finally:
        con.close()
    return md
