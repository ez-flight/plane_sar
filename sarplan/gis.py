"""
Чтение линейных объектов из GeoPackage (.gpkg) без GDAL: SQLite + разбор
геометрии GPKG/WKB (LineString, MultiLineString), координаты WGS-84.
"""

import sqlite3
import struct
from typing import List, Optional, Tuple

import numpy as np

from .frames import distance_km

Line = List[Tuple[float, float]]          # [(lat, lon), ...]


def _parse_wkb(buf: bytes, pos: int = 0):
    order = "<" if buf[pos] == 1 else ">"
    gtype = struct.unpack_from(order + "I", buf, pos + 1)[0]
    base = gtype % 1000
    dims = {0: 2, 1: 3, 2: 3, 3: 4}[gtype // 1000]
    if gtype & 0x80000000 or gtype & 0x40000000:        # EWKB-флаги Z/M
        dims = 2 + bool(gtype & 0x80000000) + bool(gtype & 0x40000000)
        base = gtype & 0xFF
    pos += 5
    if base == 2:                                        # LineString
        n = struct.unpack_from(order + "I", buf, pos)[0]
        pos += 4
        xy = np.frombuffer(buf, dtype=order + "f8", count=n * dims, offset=pos).reshape(n, dims)
        pos += 8 * n * dims
        return [[(float(y), float(x)) for x, y in xy[:, :2]]], pos
    if base == 5:                                        # MultiLineString
        n = struct.unpack_from(order + "I", buf, pos)[0]
        pos += 4
        out = []
        for _ in range(n):
            parts, pos = _parse_wkb(buf, pos)
            out += parts
        return out, pos
    raise ValueError(f"Неподдерживаемый тип геометрии WKB: {gtype}")


def parse_gpkg_geometry(blob: bytes) -> List[Line]:
    """Геометрия GPKG (заголовок 'GP' + WKB) -> список линий [(lat, lon)]."""
    if blob[:2] != b"GP":
        raise ValueError("Не геометрия GeoPackage")
    flags = blob[3]
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0x07]
    lines, _ = _parse_wkb(blob, 8 + env)
    return lines


def read_gpkg_lines(path: str, layer: Optional[str] = None) -> List[Line]:
    """Все линии слоя (по умолчанию первого векторного слоя) в WGS-84."""
    con = sqlite3.connect(path)
    try:
        if layer is None:
            layer = con.execute("select table_name from gpkg_contents where data_type='features'").fetchone()[0]
        srs, col = con.execute("select srs_id, column_name from gpkg_geometry_columns where table_name=?",
                               (layer,)).fetchone()
        if srs != 4326:
            raise ValueError(f"Ожидается WGS-84 (EPSG:4326), в слое {layer} srs_id={srs}")
        lines = []
        for (blob,) in con.execute(f'select "{col}" from "{layer}"'):
            if blob:
                lines += parse_gpkg_geometry(blob)
        return lines
    finally:
        con.close()


def merge_lines(lines: List[Line], tol_km: float = 0.05) -> List[Line]:
    """Склеить части мультилинии в цепочки по совпадающим концам."""
    parts = [list(l) for l in lines if len(l) > 1]
    merged = []
    while parts:
        cur = parts.pop(0)
        grown = True
        while grown:
            grown = False
            for i, p in enumerate(parts):
                for cand, how in ((p, "end"), (p[::-1], "end"), (p, "start"), (p[::-1], "start")):
                    a, b = (cur[-1], cand[0]) if how == "end" else (cand[-1], cur[0])
                    if float(distance_km(a[0], a[1], b[0], b[1])) <= tol_km:
                        cur = cur + cand[1:] if how == "end" else cand[:-1] + cur
                        parts.pop(i)
                        grown = True
                        break
                if grown:
                    break
        merged.append(cur)
    return sorted(merged, key=polyline_length_km, reverse=True)


def polyline_length_km(line: Line) -> float:
    a = np.array(line)
    return float(np.sum(distance_km(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1])))


def cut_from_end(line: Line, start_point: Tuple[float, float], length_km: float) -> Line:
    """
    Участок линии длиной length_km, начиная от конца, ближайшего к start_point
    (например от устья вверх по течению).
    """
    a = np.array(line)
    d0 = float(distance_km(*a[0], *start_point))
    d1 = float(distance_km(*a[-1], *start_point))
    if d1 < d0:
        a = a[::-1]
    seg = distance_km(a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    j = int(np.searchsorted(cum, length_km))
    out = [tuple(map(float, p)) for p in a[:j]]
    if j < len(a):
        t = (length_km - cum[j - 1]) / max(seg[j - 1], 1e-12)
        p = a[j - 1] + t * (a[j] - a[j - 1])
        out.append((float(p[0]), float(p[1])))
    return out
