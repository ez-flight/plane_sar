#!/usr/bin/env python3
import logging
import os
from datetime import datetime

import spacetrack.operators as op
from dotenv import load_dotenv
from sgp4.api import Satrec
from spacetrack import SpaceTrackClient
from spacetrack.base import AuthenticationError

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)  # override=True чтобы переменные из .env перезаписывали системные

# Имя пользователя и пароль сейчас опишем как константы
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

# Проверка наличия переменных окружения
if not USERNAME or not PASSWORD:
    logger.warning("USERNAME или PASSWORD не найдены в переменных окружения")

def get_spacetrack_tle(sat_id):
    """Получение TLE с Space-Track с обработкой ошибок"""
    try:
        st = SpaceTrackClient(identity=USERNAME, password=PASSWORD)
        data = st.tle_latest(
            norad_cat_id=sat_id, 
            orderby="epoch desc", 
            limit=1, 
            format="tle"
        )
        if not data:
            logger.warning("Данные не найдены на Space-Track")
            return None, None
            
        return data[:69].strip(), data[70:139].strip()
        
    except AuthenticationError as e:
        logger.error("Ошибка аутентификации на Space-Track")
    except Exception as e:
        logger.error(f"Ошибка при запросе к Space-Track: {str(e)}")
    
    return None, None

def read_tle_from_file(norad_id):
    """Чтение TLE из локального файла"""
    try:
        with open("backup.tle", "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for i in range(len(lines)-2):
            if lines[i+1].startswith('1') and lines[i+2].startswith('2'):
                name = lines[i]
                tle1 = lines[i+1]
                tle2 = lines[i+2]
                if str(norad_id) in tle1.split()[1]:
                    return name, tle1, tle2
                    
        logger.warning("NORAD ID не найден в файле")
        return None, None, None
        
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {str(e)}")
        return None, None, None

def read_tle_base_file(norad_id):
    """Основная функция с приоритетом онлайн-данных"""
    # Пытаемся получить данные онлайн
    tle1, tle2 = get_spacetrack_tle(norad_id)
    if tle1 and tle2:
        # Валидация полученных TLE
        try:
            satellite = Satrec.twoline2rv(tle1, tle2)
            if satellite.satnum == norad_id:
                return norad_id, tle1, tle2
        except Exception as e:
            logger.warning(f"Ошибка валидации онлайн-TLE: {str(e)}")
    
    # Если онлайн-данные недоступны - пробуем файл
    name, tle1, tle2 = read_tle_from_file(norad_id)
    
    if tle1 and tle2:
        try:
            satellite = Satrec.twoline2rv(tle1, tle2)
            if satellite.satnum == norad_id:
                return name, tle1, tle2
        except Exception as e:
            logger.error("Ошибка валидации локальных TLE")
    
    raise ValueError(f"Не удалось получить TLE для NORAD ID {norad_id}")

def _test():
    try:
        # Тест для спутника "Космос-2552" (NORAD ID 56756)
        name, tle1, tle2 = read_tle_base_file(56756)
        print(f"Источник: {name}")
        print(tle1)
        print(tle2)
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    _test()

