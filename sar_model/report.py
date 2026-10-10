"""
Модуль анализа ограничений и формирования отчёта (ТЗ п. 2.6).

Проверка условий (T_c vs длительность окна, RCM vs c/(2B), PRF), формирование
таблиц 2 и 3, экспорт в JSON/CSV/Markdown.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List

import numpy as np


def table2_aggregate(windows: List[Dict], out_md: str = None, out_csv: str = None) -> str:
    """
    Таблица 2. Геометрические параметры режима съёмки (агрегат по всем окнам).

    Параметры: наклонная дальность до объекта (мин–макс), длительность окон (мин–макс, средняя).
    """
    if not windows:
        s = "Нет окон наблюдения.\n"
        if out_md:
            _write(out_md, s)
        return s
    R0_all = [w["R0_mean_km"] for w in windows]
    dur_all = [w["duration_s"] for w in windows]
    lines = [
        "| Параметр | Значение |",
        "|----------|----------|",
        f"| Наклонная дальность до объекта, км | {min(R0_all):.0f}-{max(R0_all):.0f} |",
        f"| Длительность отдельных окон наблюдения, с | {min(dur_all):.0f}-{max(dur_all):.0f} |",
        f"| Средняя длительность окна, с | {np.mean(dur_all):.0f} |",
    ]
    s = "\n".join(lines) + "\n"
    if out_md:
        _write(out_md, s)
    if out_csv:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("parameter,value\n")
            f.write(f"slant_range_km,{min(R0_all):.0f}-{max(R0_all):.0f}\n")
            f.write(f"window_duration_s,{min(dur_all):.0f}-{max(dur_all):.0f}\n")
            f.write(f"mean_window_duration_s,{np.mean(dur_all):.0f}\n")
    return s


def table3_characteristic_windows(
    windows: List[Dict],
    synthesis_list: List[Dict],
    out_md: str = None,
    out_csv: str = None,
) -> str:
    """
    Таблица 3. Расчётные параметры пристального прожекторного режима для характерных окон
    (минимум, средняя, максимальная наклонная дальность).

    synthesis_list[i] соответствует windows[i] (те же индексы).
    """
    if not windows or not synthesis_list or len(windows) != len(synthesis_list):
        s = "Недостаточно данных для таблицы 3.\n"
        if out_md:
            _write(out_md, s)
        return s
    # Выбираем три характерных окна: min R0, ближайшее к среднему, max R0
    R0_means = [w["R0_mean_km"] for w in windows]
    idx_min = int(np.argmin(R0_means))
    idx_max = int(np.argmax(R0_means))
    idx_mid = int(np.argmin(np.abs(np.array(R0_means) - np.mean(R0_means))))
    indices = [idx_min, idx_mid, idx_max]
    labels = ["Окно 1 (R₀ мин)", "Окно 2 (R₀ ср)", "Окно 3 (R₀ макс)"]
    rows = []
    for i, (idx, label) in enumerate(zip(indices, labels)):
        w = windows[idx]
        s = synthesis_list[idx]
        rows.append({
            "label": label,
            "R0_km": w["R0_mean_km"],
            "T_c_s": s["T_c_s"],
            "L_c_km": s["L_c_km"],
            "delta_f_d_hz": s["delta_f_d_hz"],
            "PRF_min_hz": s["PRF_min_hz"],
            "range_cell_m": s["range_cell_m"],
            "rcm_critical": s["rcm_critical"],
            "L_az_km": s["L_az_km"],
        })
    lines = [
        "| Параметр | " + " | ".join(labels) + " |",
        "|----------|" + "|".join(["----------"] * 3) + "|",
        f"| R₀, км | " + " | ".join(f"{r['R0_km']:.0f}" for r in rows) + " |",
        f"| Время синтеза, с | " + " | ".join(f"{r['T_c_s']:.2f}" for r in rows) + " |",
        f"| Длина синтеза апертуры, км | " + " | ".join(f"{r['L_c_km']:.3f}" for r in rows) + " |",
        f"| Доплеровская полоса, Гц | " + " | ".join(f"{r['delta_f_d_hz']:.1f}" for r in rows) + " |",
        f"| Требуемая PRF, Гц | " + " | ".join(f"{r['PRF_min_hz']:.0f}" for r in rows) + " |",
        f"| Ширина дальностного канала, м | " + " | ".join(f"{r['range_cell_m']:.2f}" for r in rows) + " |",
        f"| Критичность RCM | " + " | ".join("Да" if r["rcm_critical"] else "Нет" for r in rows) + " |",
        f"| Размер кадра по азимуту, км | " + " | ".join(f"{r['L_az_km']:.3f}" for r in rows) + " |",
    ]
    s = "\n".join(lines) + "\n"
    if out_md:
        _write(out_md, s)
    if out_csv:
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("parameter," + ",".join(labels) + "\n")
            for key in ["R0_km", "T_c_s", "L_c_km", "delta_f_d_hz", "PRF_min_hz", "range_cell_m", "L_az_km"]:
                f.write(key + "," + ",".join(str(rows[i][key]) for i in range(3)) + "\n")
            f.write("rcm_critical," + ",".join("yes" if rows[i]["rcm_critical"] else "no" for i in range(3)) + "\n")
    return s


def build_report_json(
    config_summary: Dict,
    windows: List[Dict],
    synthesis_per_window: List[Dict],
    table2_text: str,
    table3_text: str,
) -> Dict[str, Any]:
    """Собрать единый отчёт для экспорта в JSON."""
    def _dt_iso(dt):
        return dt.isoformat() if isinstance(dt, datetime) else str(dt)
    windows_ser = []
    for w in windows:
        wc = w.copy()
        wc["start_time"] = _dt_iso(w["start_time"])
        wc["end_time"] = _dt_iso(w["end_time"])
        windows_ser.append(wc)
    return {
        "config": config_summary,
        "n_windows": len(windows),
        "windows": windows_ser,
        "synthesis_per_window": synthesis_per_window,
        "table2_md": table2_text,
        "table3_md": table3_text,
    }


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
