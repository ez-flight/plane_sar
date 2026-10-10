"""
Тесты sarplan: формулы статьи и планировщик.
Запуск: python -m pytest tests
"""

import itertools
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calc_cord import get_xyzv_from_latlon
from sar_model import geometry as base_geometry
from sarplan import duration, frames, geometry, orbit, planner, targets
from sarplan.config import ICEYE_CLASS, REGION_CENTER
from sarplan.criterion import CoverageModel, Element, Frame
from sarplan.windows import PassGeometry

EPOCH = datetime(2026, 6, 1)


# --- длительность наблюдения ------------------------------------------------------
def test_closed_form_matches_integral_and_kepler():
    v = duration.verify_closed_form(1500.0, 500.0)
    assert v["max_rel_error"] < 1e-12
    assert v["t_correct_s"] == pytest.approx(v["t_kepler_s"], rel=1e-12)
    assert v["t_no_correction_s"] < 0          # без поправки при ϑ₁ > π результат неверен


def test_theta0_and_circular_limit():
    p, e = duration.orbit_p_e(1500.0, 500.0)
    th = duration.theta0_from_height(p, e, 1000.0)
    r = p / (1 + e * np.cos(th))
    assert r == pytest.approx(frames.R_MEAN + 1000.0, rel=1e-12)
    # круговая орбита: t_s = Δϑ / ω
    t = duration.duration_circular(0.01, 0.02, 0.0, 560.0)
    assert t == pytest.approx(0.03 / np.sqrt(frames.MU / (frames.R_MEAN + 560.0) ** 3))


def test_epsilon_corrected_vs_ref4():
    # при нулевом крене ε по [4] равен нулю, исправленный — нет (теорема синусов)
    assert duration.epsilon_ref4(700.0, 0.0, 10.0) == 0.0
    eps = duration.epsilon_corrected(700.0, 0.0, 10.0)
    assert np.sin(eps) == pytest.approx(700.0 * np.sin(np.deg2rad(10.0)) / frames.R_MEAN)


def test_n_frames_formulas():
    assert duration.n_frames_mechanical(60.0, 10.0, 5.0, 5.0) == 3
    assert duration.n_frames_max(75.0, 3.5, 85.5, 0.5) == 21


# --- координаты и орбиты ------------------------------------------------------------
def test_tle_roundtrip():
    sat = orbit.make_sso_satellite("T", 99001, EPOCH, 560.0, 10.5, 30.0)
    assert len(sat.line1) == 69 and len(sat.line2) == 69
    assert int(sat.line1[-1]) == orbit.tle_checksum(sat.line1)
    r, v = sat.states_eci(np.array([np.datetime64("2026-06-01T00:00:00")]))
    assert 6900 < np.linalg.norm(r) < 6960
    assert orbit.sso_inclination_deg(560.0) == pytest.approx(97.64, abs=0.05)


def test_ecef_consistent_with_calc_cord():
    t = datetime(2026, 6, 1, 3, 0, 0)
    eci, _ = get_xyzv_from_latlon(t, 30.0, 46.5, 0.1)
    ecef = frames.geodetic_to_ecef(46.5, 30.0, 0.1)
    assert np.allclose(frames.ecef_to_eci(ecef, np.datetime64(t)), eci, atol=1e-6)
    la, lo, h = frames.ecef_to_geodetic(ecef)
    assert (la, lo, h) == pytest.approx((46.5, 30.0, 0.1), abs=1e-8)


# --- геометрия (10)–(15) ------------------------------------------------------------
@pytest.fixture(scope="module")
def ref_pass():
    sat = orbit.make_sso_satellite("T", 99001, EPOCH, 560.0, 10.5, 0.0)
    c = frames.geodetic_to_ecef(*REGION_CENTER, 0.0)
    ps = orbit.find_passes(sat, c, EPOCH, EPOCH + timedelta(days=1), look_min_deg=20, look_max_deg=55,
                           squint_max_deg=20)
    def look(p):
        sk, lk = geometry.squint_look(p.r, p.v, c[None])
        return lk[np.argmin(np.abs(sk))]
    return sat, min(ps, key=lambda p: abs(look(p) - 37.5))


def test_doppler_matches_base_model(ref_pass):
    sat, p = ref_pass
    i = len(p.t) // 3
    t = p.time_at(i)
    c = frames.geodetic_to_ecef(*REGION_CENTER, 0.0)
    sk, _ = geometry.squint_look(p.r[i], p.v[i], c)
    fd = geometry.doppler_hz(sk, np.linalg.norm(p.v[i]), 0.0314)
    r_eci, v_eci = sat.states_eci(np.array([t]))
    rt, vt = get_xyzv_from_latlon(frames.dt64_to_datetime(t), REGION_CENTER[1], REGION_CENTER[0], 0.0)
    fd_base = base_geometry.doppler_frequency_hz(r_eci[0], v_eci[0], np.array(rt), np.array(vt), 0.0314)
    assert abs(fd) == pytest.approx(abs(fd_base), rel=1e-6)   # знаки (8) и sar_model противоположны


def test_deflection_zero_on_pointing_target(ref_pass):
    _, p = ref_pass
    c = frames.geodetic_to_ecef(*REGION_CENTER, 0.0)
    dsk, dum = geometry.beam_deflection(p.r, p.v, c[None], c[None])
    assert np.allclose(dsk, 0, atol=1e-9) and np.allclose(dum, 0, atol=1e-9)


def test_ray_ellipsoid_on_surface(ref_pass):
    _, p = ref_pass
    c = frames.geodetic_to_ecef(*REGION_CENTER, 0.0)
    spot = geometry.ray_ellipsoid(p.r[0], c - 200 * c / np.linalg.norm(c))
    x, y, z = spot
    assert (x ** 2 + y ** 2) / frames.R_EQ ** 2 + z ** 2 / frames.R_POL ** 2 == pytest.approx(1.0, abs=1e-9)


def test_observation_window_close_to_article(ref_pass):
    _, p = ref_pass
    els, frs = targets.point_element("C", *REGION_CENTER)
    g = PassGeometry(p, frs, ICEYE_CLASS)
    assert 60 <= g.vis[0].sum() * g.dt <= 85        # статья: около 72 с


# --- критерий и планировщик -------------------------------------------------------
def test_saturation():
    els = [Element("A", alpha=2.0)]
    frs = [Frame("A1", 46.5, 30.0, {"A": 1.0}), Frame("A2", 46.5, 30.01, {"A": 1.0})]
    m = CoverageModel(els, frs)
    acc = m.new_acc()
    assert m.gain(acc, 0, 0.0) == 2.0
    m.add(acc, 0, 0.0)
    assert m.gain(acc, 1, 0.0) == 0.0               # дубль не добавляет покрытия
    assert m.additive_value(1, 0.0) == 2.0          # а в аддитивном критерии — добавляет


def test_deadline_decay():
    m = CoverageModel([Element("A", deadline_h=10.0, decay=0.3)], [Frame("A", 46.5, 30.0, {"A": 1.0})])
    assert m.gain(m.new_acc(), 0, 5.0) == 1.0
    assert m.gain(m.new_acc(), 0, 20.0) == pytest.approx(0.3)


def _instance(ref_pass, n, seed):
    _, p = ref_pass
    sk, _ = geometry.squint_look(p.r, p.v, frames.geodetic_to_ecef(*REGION_CENTER, 0.0)[None])
    i = int(np.argmin(np.abs(sk)))
    la1, lo1, _ = frames.ecef_to_geodetic(p.r[i - 4])
    la2, lo2, _ = frames.ecef_to_geodetic(p.r[i + 4])
    az = frames.azimuth_between(la1, lo1, la2, lo2)
    els, frs = targets.random_targets(np.random.default_rng(seed), REGION_CENTER, az, n=n)
    return PassGeometry(p, frs, ICEYE_CLASS), CoverageModel(els, frs)


def _feasible(plan, g):
    s = g.sensor
    items = sorted(plan.items, key=lambda x: x[1])
    for (k, a), (l, b) in zip(items[:-1], items[1:]):
        assert a + g.n_dur[k] + g.n_gap <= b                       # нет наложений
    for k, a in items:
        assert g.start_ok[plan.point, k, a]                         # t_k ∈ Ω_k
    assert sum(g.frames[k].duration(s.t_frame_s) for k, _ in items) <= s.budget_s + 1e-9


@pytest.mark.parametrize("method", ["greedy_priority", "seq_analog2", "proposed", "proposed_cluster",
                                    "dp_additive", "exact"])
def test_plans_are_feasible(ref_pass, method):
    g, m = _instance(ref_pass, 12, 1)
    pl = planner.plan_pass(g, m, m.new_acc(), 0.0, range(12), method)
    _feasible(pl, g)


def test_exact_is_optimal_and_dominates(ref_pass):
    # маленький пример с сокращённым бюджетом — сверяем с перебором перестановок
    g, m = _instance(ref_pass, 7, 2)
    params = planner.PlanParams(budget_s=14.0)
    ex = planner.plan_pass(g, m, m.new_acc(), 0.0, range(7), "exact", params)
    best = 0.0
    ctx = planner._Ctx(g, g.nominal, m, m.new_acc(), 0.0, range(7), params)
    for r in range(1, 5):
        for perm in itertools.permutations(ctx.cand, r):
            if not ctx.resources_ok(perm):
                continue
            occ, ok = ctx.new_occ(), True
            e = 0
            for k in perm:
                s = ctx.earliest(k, occ, after=e)
                if s < 0:
                    ok = False
                    break
                ctx.occupy(occ, k, s)
                e = s + ctx.n_dur[k] + ctx.n_gap
            if ok:
                best = max(best, ctx.value(perm))
    assert ex.gain == pytest.approx(best)
    gr = planner.plan_pass(g, m, m.new_acc(), 0.0, range(7), "proposed", params)
    assert gr.gain <= ex.gain + 1e-9


def test_saturation_beats_additive_dp_on_duplicates(ref_pass):
    _, p = ref_pass
    els, frs = targets.duplicate_targets(np.random.default_rng(3), REGION_CENTER, 195.0)
    g, m = PassGeometry(p, frs, ICEYE_CLASS), CoverageModel(els, frs)
    a = planner.plan_pass(g, m, m.new_acc(), 0.0, range(len(frs)), "proposed").gain
    d = planner.plan_pass(g, m, m.new_acc(), 0.0, range(len(frs)), "dp_additive").gain
    assert a > d


# --- профиль Днестра из GeoPackage -------------------------------------------------
@pytest.mark.skipif(not os.path.exists(targets.DNESTR_GPKG), reason="нет Днестр.gpkg")
def test_dnestr_gpkg_axis_and_chain():
    from sarplan import gis
    lines = gis.read_gpkg_lines(targets.DNESTR_GPKG)
    assert sum(gis.polyline_length_km(l) for l in lines) == pytest.approx(1383, rel=0.02)
    axis = targets.dnestr_axis("gpkg")
    assert frames.distance_km(*axis[0], *targets.BLACK_SEA_POINT) < 40      # начинается у устья
    els, frs = targets.dnestr_targets(source="gpkg")
    chain = [f for f in frs if f.fid.startswith("DNESTR")]
    assert len(chain) == 21 and len(frs) == 40
    gaps = [float(frames.distance_km(a.lat, a.lon, b.lat, b.lon)) for a, b in zip(chain[:-1], chain[1:])]
    assert min(gaps) >= 4.85 and max(gaps) < 5.2                             # шаг рамок 4,9 км по прямой
    assert frames.distance_km(chain[-1].lat, chain[-1].lon, *targets.TIRASPOL) < 20
