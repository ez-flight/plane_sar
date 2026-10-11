"""
Параметры КА и РСА (статья, таблица 2) и пресеты для КА большого класса.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SensorParams:
    name: str = "Малый КА РСА (класс ICEYE)"
    mode: str = "hybrid"            # hybrid — вращение антенны + электронное сканирование;
                                    # electronic — только электронный сектор (антенна по траверзу);
                                    # stripmap — маршрутный режим: полоса swath_km при крене, выбранном на пролёт
    alt_km: float = 560.0
    incl_deg: float = 97.6
    wavelength_m: float = 0.0314
    d_beta_sk_max: float = 5.0      # Δβ_ск max, °
    d_beta_um_max: float = 5.0      # Δβ_ум max, °
    beta_sk_max: float = 20.0       # β_ск max, °
    beta_um_min: float = 20.0       # β_ум min, °
    beta_um_max: float = 55.0       # β_ум max, °
    t_frame_s: float = 3.5          # t_вкл,1 — один вид, 0,5 м
    budget_s: float = 75.0          # T_вкл max за пролёт
    gap_s: float = 0.5              # Δt — пауза (коммутация луча)
    q_max: float = float("inf")     # Q_max — объём данных за пролёт (в единицах кадра)
    dt_s: float = 0.5               # шаг временной сетки пролёта
    swath_km: float = 0.0           # ширина полосы маршрутного режима
    frame_km: float = 5.0           # размер кадра поперёк трассы (кадр должен целиком лечь в полосу)

    def with_(self, **kw) -> "SensorParams":
        return replace(self, **kw)


ICEYE_CLASS = SensorParams()

# «Обзор-Р»: цифровая АФАР без механического вращения, сектор ±3° по азимуту,
# углы визирования 20...46°, время кадра 1 м около 3 с, интервал 1 с (допущение).
OBZOR_R = SensorParams(
    name="Обзор-Р", mode="electronic", alt_km=580.0, incl_deg=97.7,
    wavelength_m=0.031, d_beta_sk_max=3.0, d_beta_um_max=90.0, beta_sk_max=3.0,
    beta_um_min=20.0, beta_um_max=46.0, t_frame_s=3.0, budget_s=600.0, gap_s=1.0,
)

# «Кондор-ФКА»: зеркальная антенна, электронный сектор 4,1° (±2,05°),
# минимальное время между съёмками 66 с.
KONDOR_FKA = SensorParams(
    name="Кондор-ФКА", mode="electronic", alt_km=519.0, incl_deg=97.4,
    wavelength_m=0.1, d_beta_sk_max=2.05, d_beta_um_max=90.0, beta_sk_max=2.05,
    beta_um_min=24.0, beta_um_max=55.0, t_frame_s=3.0, budget_s=600.0, gap_s=66.0,
)

# «АИСТ-СТ» (Самарский университет, 16U, запуск 28.12.2025): X-диапазон, маршрутный режим,
# разрешение не хуже 5 м, полоса не менее 30 км, ФАР 0,32 м², до 150 Вт. Орбита ССО ~500 км,
# i ≈ 97,55° (по предстартовым TLE попутных кубсатов). Не опубликованы и приняты: углы
# визирования 20...50°, длительность маршрута за пролёт 60 с (энергетика 16U), время кадра 4 с,
# полуширина ДН по азимуту 1° (окно съёмки около траверза).
AIST_ST = SensorParams(
    name="АИСТ-СТ", mode="stripmap", alt_km=500.0, incl_deg=97.55, wavelength_m=0.031,
    d_beta_sk_max=1.0, d_beta_um_max=90.0, beta_sk_max=1.0, beta_um_min=20.0, beta_um_max=50.0,
    t_frame_s=4.0, budget_s=60.0, gap_s=0.0, swath_km=30.0, frame_km=5.0,
)

# «АИСТ-СТ», прожекторный режим (будет испытываться на следующем этапе, ожидаемое разрешение 2 м):
# луч удерживается на кадре разворотом всего КА, поэтому окно съёмки задаётся сектором скоса
# относительно траверза, а между кадрами нужен перенацеливание КА. Не опубликованы и приняты:
# сектор скоса ±20°, кадр 5×5 км за 5 с (синтез 2 м в X-диапазоне ≈ 0,7 с плюс успокоение),
# перенацеливание между кадрами 5 с (по данным разработчиков), суммарное время излучения за пролёт 60 с.
AIST_ST_SPOT = AIST_ST.with_(
    name="АИСТ-СТ (прожекторный)", mode="electronic", d_beta_sk_max=20.0, beta_sk_max=20.0,
    t_frame_s=5.0, gap_s=5.0, budget_s=60.0, swath_km=0.0,
)

PRESETS = {"iceye": ICEYE_CLASS, "obzor-r": OBZOR_R, "kondor-fka": KONDOR_FKA, "aist-st": AIST_ST,
           "aist-st-spot": AIST_ST_SPOT}

# Район моделирования: окрестность Днестра
REGION_CENTER = (46.5, 30.0)
