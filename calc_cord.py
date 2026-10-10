import math
from datetime import datetime, timedelta

import numpy as np
from pyorbital.orbital import XKMPER, F, astronomy

# Константы WGS84
EARTH_EQUATORIAL_RADIUS = 6378.137  # км
EARTH_FLATTENING = 1/298.257223563
EARTH_ECCENTRICITY_SQ = 2*EARTH_FLATTENING - EARTH_FLATTENING**2
MFACTOR = 7.292115e-5  # Угловая скорость вращения Земли (рад/с)

def get_xyzv_from_latlon(time, lon, lat, alt_km):
    """Преобразование географических координат в ECI (Earth-Centered Inertial)
    
    Аргументы:
        time (datetime): Время UTC
        lon (float): Долгота в градусах
        lat (float): Широта в градусах
        alt_km (float): Высота над эллипсоидом в километрах
        
    Возвращает:
        tuple: (x, y, z) в километрах
        tuple: (vx, vy, vz) в км/с
    """
    # Преобразование в радианы
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    
    # Звездное время
    theta = (astronomy.gmst(time) + lon_rad) % (2 * np.pi)
    
    # Параметры эллипсоида
    N = EARTH_EQUATORIAL_RADIUS / np.sqrt(1 - EARTH_ECCENTRICITY_SQ*np.sin(lat_rad)**2)
    
    # Расчет координат
    x = (N + alt_km) * np.cos(lat_rad) * np.cos(theta)
    y = (N + alt_km) * np.cos(lat_rad) * np.sin(theta)
    z = (N*(1 - EARTH_ECCENTRICITY_SQ) + alt_km) * np.sin(lat_rad)
    
    # Расчет скорости вращения
    vx = -MFACTOR * y
    vy = MFACTOR * x
    vz = 0.0
    
    return (x, y, z), (vx, vy, vz)

def get_lonlatalt(pos_km, utc_time):
    """Преобразование ECI координат в географические
    
    Аргументы:
        pos_km (np.ndarray): Вектор позиции [x, y, z] в километрах
        utc_time (datetime): Время UTC
        
    Возвращает:
        tuple: (Долгота, Широта, Высота) в градусах и километрах
    """
    # Конвертация в метры
    pos_m = pos_km * 1000  
    x, y, z = pos_m
    
    # Расчет долготы
    lon_rad = (np.arctan2(y, x) - astronomy.gmst(utc_time)) % (2 * np.pi)
    if lon_rad > np.pi:
        lon_rad -= 2*np.pi
    
    # Итерационный расчет широты
    p = np.sqrt(x**2 + y**2)
    lat_rad = np.arctan(z / (p * (1 - EARTH_ECCENTRICITY_SQ)))
    
    for _ in range(10):
        N = EARTH_EQUATORIAL_RADIUS*1000 / np.sqrt(1 - EARTH_ECCENTRICITY_SQ*np.sin(lat_rad)**2)
        h = p / np.cos(lat_rad) - N
        lat_new = np.arctan(z / (p * (1 - EARTH_ECCENTRICITY_SQ*(N/(N + h)))))
        if np.abs(lat_new - lat_rad) < 1e-15:
            break
        lat_rad = lat_new
    
    # Расчет высоты
    N = EARTH_EQUATORIAL_RADIUS*1000 / np.sqrt(1 - EARTH_ECCENTRICITY_SQ*np.sin(lat_rad)**2)
    h = p / np.cos(lat_rad) - N
    
    return (
        np.rad2deg(lon_rad),
        np.rad2deg(lat_rad),
        h / 1000  # Конвертация в километры
    )

def _test():
    """Тестирование преобразований координат"""
    # Тестовые параметры (Москва)
    test_lon = 37.6173    
    test_lat = 55.75583
    test_alt = 0.155  # 155 метров
    
    # Генерация тестового времени
    utc_time = datetime(2024, 2, 21, 12, 0, 0)
    
    # Прямое преобразование
    eci_pos, eci_vel = get_xyzv_from_latlon(utc_time, test_lon, test_lat, test_alt)
    
    # Обратное преобразование
    reconstructed_lon, reconstructed_lat, reconstructed_alt = get_lonlatalt(
        np.array(eci_pos), utc_time)
    
    # Проверка точности
    assert np.isclose(test_lon, reconstructed_lon, atol=1e-8), f"Ошибка долготы: {test_lon} vs {reconstructed_lon}"
    assert np.isclose(test_lat, reconstructed_lat, atol=1e-8), f"Ошибка широты: {test_lat} vs {reconstructed_lat}"
    assert np.isclose(test_alt, reconstructed_alt, atol=1e-6), f"Ошибка высоты: {test_alt} vs {reconstructed_alt}"
    print("Все тесты пройдены успешно!")

if __name__ == "__main__":
    _test()

