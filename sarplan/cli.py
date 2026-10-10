"""
Командная строка.

    python -m sarplan plan config/plan_example.yaml      # план съёмки СРО
    python -m sarplan experiments [имена ...] [--quick]  # таблицы 3–7, рис. 1–2
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import yaml

from . import experiments, targets
from .config import PRESETS
from .constellation import constellation_passes
from .criterion import CoverageModel
from .frames import geodetic_to_ecef, to_dt64
from .orbit import load_tle_file, make_sso_satellite
from .planner import PlanParams, plan_pass, plan_to_rows
from .windows import PassGeometry


def _parse_time(s) -> datetime:
    if isinstance(s, datetime):
        return s.replace(tzinfo=None)
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)


def build_scenario(cfg: dict):
    frame_km = float(cfg.get("frame_km", 5.0))
    parts = []
    for e in cfg["elements"]:
        kind = e.get("kind", "point")
        common = dict(alpha=float(e.get("priority", 1)), deadline_h=float(e.get("deadline_h", float("inf"))))
        if kind == "point":
            parts.append(targets.point_element(e["id"], e["lat"], e["lon"], n_looks=int(e.get("looks", 1)), **common))
        elif kind == "line":
            chord = False
            if "gpkg" in e:
                from .gis import cut_from_end, read_gpkg_lines
                from .frames import distance_km
                lines = read_gpkg_lines(e["gpkg"], e.get("layer"))
                start = tuple(e["from"])
                near = lambda l: float(np.min(distance_km(np.array(l)[:, 0], np.array(l)[:, 1], *start)))
                pts = cut_from_end(min(lines, key=near), start, float(e.get("length_km", 100.0)))
                chord = True
            else:
                pts = [tuple(p) for p in e["points"]]
            parts.append(targets.line_element(e["id"], pts, frame_km=frame_km, chord=chord,
                                              overlap_km=float(e.get("overlap_km", 0.1)),
                                              n_frames=e.get("max_frames"), **common))
        elif kind == "area":
            parts.append(targets.area_element(e["id"], e["lat"], e["lon"], int(e.get("n_along", 2)),
                                              int(e.get("n_across", 2)), float(e.get("azimuth_deg", 0.0)),
                                              frame_km=frame_km, **common))
        else:
            raise ValueError(f"Неизвестный тип элемента: {kind}")
    return targets.merge(*parts)


def build_satellites(cfg: dict, sensor, epoch: datetime):
    sc = cfg.get("satellites", {})
    sats = []
    for j, s in enumerate(sc.get("synthetic", []) or []):
        sats.append(make_sso_satellite(s.get("name", f"S{j + 1}"), 99001 + j, epoch,
                                       float(s.get("alt_km", sensor.alt_km)), float(s.get("ltan_h", 10.5)),
                                       float(s.get("mean_anomaly_deg", 0.0)), s.get("incl_deg")))
    if sc.get("tle_file"):
        ids = {str(i) for i in sc.get("norad_ids", [])}
        for s in load_tle_file(sc["tle_file"]):
            if not ids or s.line1[2:7].strip() in ids:
                sats.append(s)
    if not sats:
        raise ValueError("Не задано ни одного КА (satellites.synthetic или satellites.tle_file)")
    return sats


def run_plan(config_path: str) -> dict:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    sc = cfg.get("sensor", {})
    sensor = PRESETS[sc.get("preset", "iceye")].with_(**(sc.get("overrides") or {}))
    t0 = _parse_time(cfg["time"]["start"])
    horizon_h = float(cfg["time"].get("horizon_h", 48))
    els, frs = build_scenario(cfg)
    center = targets.region_center_of(frs)
    sats = build_satellites(cfg, sensor, t0)
    passes = constellation_passes(sats, center, t0, t0 + timedelta(hours=horizon_h), sensor)
    model = CoverageModel(els, frs)
    acc = model.new_acc()
    remaining = set(range(len(frs)))
    method = cfg.get("method", "proposed_cluster")
    rows, summary = [], []
    c = geodetic_to_ecef(center[0], center[1], 0.0)
    for p in passes:
        h = p.hours_since(t0)
        g = PassGeometry(p, frs, sensor, center_ecef=c)
        plan = plan_pass(g, model, acc, h, sorted(remaining), method, PlanParams())
        if not plan.items:
            continue
        for k in plan.frames:
            model.add(acc, k, h)
            remaining.discard(k)
        rows += plan_to_rows(plan, g)
        summary.append({"sat": p.sat_name, "start_utc": str(p.start), "hours": round(h, 2),
                        "frames": len(plan.items), "gain": round(plan.gain, 3),
                        "coverage_after": round(model.coverage(acc), 4)})
        if not remaining:
            break
    out_dir = cfg.get("output", {}).get("dir", "result/plan")
    os.makedirs(out_dir, exist_ok=True)
    if rows:
        with open(os.path.join(out_dir, "plan.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    result = {
        "sensor": sensor.name, "method": method, "n_elements": len(els), "n_frames": len(frs),
        "n_passes_found": len(passes), "passes": summary,
        "coverage": round(model.coverage(acc), 4),
        "elements": {e.eid: round(model.element_coverage(acc, e.eid), 3) for e in els},
        "not_shot": [frs[k].fid for k in sorted(remaining)],
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sarplan", description="Текущее планирование РСА-съёмки СРО")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan", help="построить план съёмки по YAML-конфигу")
    a.add_argument("config")
    b = sub.add_parser("experiments", help="воспроизвести таблицы 3–7 и рисунки 1–2 статьи")
    b.add_argument("names", nargs="*", help=f"из {', '.join(experiments.ALL)}; по умолчанию все")
    b.add_argument("--out", default="results")
    b.add_argument("--quick", action="store_true", help="меньше наборов (для быстрой проверки)")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        res = run_plan(args.config)
        print(f"{res['sensor']}: найдено пролётов {res['n_passes_found']}, использовано {len(res['passes'])}")
        for s in res["passes"]:
            print(f"  {s['start_utc']} {s['sat']}: кадров {s['frames']}, покрытие после пролёта {s['coverage_after']}")
        print(f"Итоговое покрытие Э/Σα = {res['coverage']}; не снято кадров: {len(res['not_shot'])}")
    else:
        unknown = [n for n in args.names if n not in experiments.ALL]
        if unknown:
            ap.error(f"неизвестные эксперименты: {unknown}")
        print(experiments.run_all(args.out, args.names or None, quick=args.quick))
    return 0


if __name__ == "__main__":
    sys.exit(main())
