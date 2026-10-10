#!/usr/bin/env python3
"""
Точка входа программного комплекса моделирования параметров РСА-съёмки высокого разрешения (ТЗ).

Загрузка конфигурации (YAML), баллистика (SGP4), поиск окон наблюдения, расчёт параметров
синтеза апертуры, формирование отчётов (таблицы 2 и 3, JSON/CSV) и графиков.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import yaml
from pyorbital.orbital import Orbital

from read_TBF import read_tle_base_file
from sar_model import ballistic, geometry, windows, synthesis, report, plots


def load_config(path: str) -> dict:
    """Загрузить YAML-конфиг."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_time(s: str) -> datetime:
    """Парсинг времени из ISO 8601."""
    if "T" in s:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    sc = cfg["spacecraft"]
    sar = cfg["sar"]
    slant = cfg["slant_range"]
    targets = cfg["targets"]
    time_cfg = cfg["time"]
    scan_cfg = cfg.get("scan", {})
    out_cfg = cfg.get("output", {})

    t_start = parse_time(time_cfg["start"])
    num_days = time_cfg["num_days"]
    t_end = t_start + timedelta(days=num_days)

    norad_id = sc["norad_id"]
    try:
        name_or_id, tle1, tle2 = read_tle_base_file(norad_id)
    except Exception as e:
        print(f"Ошибка загрузки TLE: {e}", file=sys.stderr)
        sys.exit(1)
    orb = Orbital("N", line1=tle1, line2=tle2)

    wavelength_m = sar["wavelength_m"]
    beamwidth_deg = sar["beamwidth_deg"]
    bandwidth_hz = sar["bandwidth_mhz"] * 1e6
    rho_x_m = cfg["target_azimuth_resolution_m"]
    time_step_s = scan_cfg.get("time_step_s", 10.0)
    min_dur_s = scan_cfg.get("min_window_duration_s", 30.0)

    all_windows = []
    all_synthesis = []
    config_summary = {
        "norad_id": norad_id,
        "wavelength_m": wavelength_m,
        "rho_x_m": rho_x_m,
        "t_start": t_start.isoformat(),
        "t_end": t_end.isoformat(),
        "num_days": num_days,
    }

    for target in targets:
        name = target.get("name", "target")
        lat = target["lat_deg"]
        lon = target["lon_deg"]
        height = target.get("height_km", 0.012)
        wins = windows.find_observation_windows(
            orb, lat, lon, height,
            t_start, t_end,
            time_step_s=time_step_s,
            R0_min_km=slant["min_km"],
            R0_max_km=slant["max_km"],
            incidence_min_deg=sar["incidence_angle_min_deg"],
            incidence_max_deg=sar["incidence_angle_max_deg"],
            squint_half_angle_deg=sar["scan_sector_deg"],
            min_window_duration_s=min_dur_s,
            wavelength_m=wavelength_m,
        )
        for w in wins:
            t_mid = w["start_time"] + (w["end_time"] - w["start_time"]) / 2
            R_s, V_s = ballistic.get_position_velocity(orb, t_mid)
            V_s_km_s = (sum(x**2 for x in V_s)) ** 0.5
            syn = synthesis.synthesis_parameters(
                w["R0_mean_km"], V_s_km_s, wavelength_m, rho_x_m, beamwidth_deg, bandwidth_hz
            )
            all_windows.append({**w, "target_name": name})
            all_synthesis.append(syn)

    table2 = report.table2_aggregate(
        all_windows,
        out_md=out_cfg.get("table2_md"),
        out_csv=None,
    )
    table3 = report.table3_characteristic_windows(
        all_windows, all_synthesis,
        out_md=out_cfg.get("table3_md"),
        out_csv=out_cfg.get("table3_csv"),
    )

    report_dict = report.build_report_json(
        config_summary, all_windows, all_synthesis, table2, table3
    )
    report_json_path = out_cfg.get("report_json")
    if report_json_path:
        os.makedirs(os.path.dirname(report_json_path) or ".", exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        print(f"Отчёт JSON: {report_json_path}")

    if out_cfg.get("report_csv"):
        import csv
        csv_path = out_cfg["report_csv"]
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            cw = csv.writer(f)
            cw.writerow(["window_id", "start_time", "end_time", "duration_s", "R0_mean_km", "T_c_s", "L_c_km", "delta_f_d_hz", "PRF_min_hz", "rcm_critical"])
            for i, (win, syn) in enumerate(zip(all_windows, all_synthesis)):
                cw.writerow([
                    i + 1, win["start_time"].isoformat(), win["end_time"].isoformat(),
                    win["duration_s"], win["R0_mean_km"], syn["T_c_s"], syn["L_c_km"],
                    syn["delta_f_d_hz"], syn["PRF_min_hz"], syn["rcm_critical"],
                ])
        print(f"Отчёт CSV: {csv_path}")

    print("\nТаблица 2 (геометрические параметры):\n" + table2)
    print("Таблица 3 (параметры синтеза для характерных окон):\n" + table3)

    plots_dir = out_cfg.get("plots_dir")
    if plots_dir and all_windows:
        fmt = out_cfg.get("format_plots", ["png"])
        first = all_windows[0]
        tgt = targets[0]
        series = plots.time_series_for_window(
            orb, tgt["lat_deg"], tgt["lon_deg"], tgt.get("height_km", 0.012),
            first["start_time"], first["end_time"], step_s=2.0, wavelength_m=wavelength_m,
        )
        for ext in fmt:
            path = os.path.join(plots_dir, f"window_params.{ext}")
            plots.plot_window_parameters(series, path)
            print(f"График: {path}")
    print(f"\nВсего окон наблюдения: {len(all_windows)}")


def main():
    parser = argparse.ArgumentParser(description="Моделирование параметров РСА-съёмки высокого разрешения")
    parser.add_argument("config", nargs="?", default="config/sar_model_config_example.yaml", help="Путь к YAML-конфигу")
    args = parser.parse_args()
    if not os.path.isfile(args.config):
        print(f"Файл конфигурации не найден: {args.config}", file=sys.stderr)
        sys.exit(1)
    run(args.config)


if __name__ == "__main__":
    main()
