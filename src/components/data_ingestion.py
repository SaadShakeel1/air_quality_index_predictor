import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from src.logger import logging
from src.exception import CustomException

# Load environment variables
load_dotenv()

# Configuration from environment variables
LAT = float(os.getenv("LATITUDE", "24.8607"))
LON = float(os.getenv("LONGITUDE", "67.0011"))
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "")
if not OPENWEATHER_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable is required. Please set it in .env file")

def safe_request(url):
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logging.error(f"Error fetching: {url[:100]}... {e}")
        return {}

def fetch_openmeteo_data(start_date, end_date):
    all_frames = []
    current_start = start_date
    logging.info(f"Fetching Open-Meteo data from {start_date} to {end_date}")
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=6), end_date)
        url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={current_start}&end_date={current_end}"
            f"&hourly=pm2_5,pm10,no2,so2,o3,co,us_aqi"
        )
        data = safe_request(url)
        if "hourly" in data:
            df = pd.DataFrame(data["hourly"])
            df["datetime"] = pd.to_datetime(df["time"])
            df.drop(columns=["time"], inplace=True)
            all_frames.append(df)
            logging.info(f"Fetched {len(df)} rows ({current_start} - {current_end})")
        else:
            logging.warning(f"No data for {current_start} - {current_end}")
        current_start = current_end + timedelta(days=1)
    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames).sort_values("datetime").reset_index(drop=True)
    logging.info(f"Open-Meteo rows: {len(result)}")
    return result

def fetch_openweather_data(start_date, end_date):
    all_frames = []
    current_start = datetime.combine(start_date, datetime.min.time())
    current_end_date = datetime.combine(end_date, datetime.min.time())
    logging.info(f"Fetching OpenWeather data from {start_date} to {end_date}")
    while current_start < current_end_date:
        current_end = min(current_start + timedelta(days=4, hours=23), current_end_date)
        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())
        url = (
            f"https://api.openweathermap.org/data/2.5/air_pollution/history?"
            f"lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_KEY}"
        )
        data = safe_request(url)
        records = data.get("list", [])
        if records:
            df = pd.DataFrame([
                {
                    "datetime": pd.to_datetime(r["dt"], unit="s"),
                    "ow_aqi": r["main"]["aqi"],
                    **r["components"]
                } for r in records
            ])
            all_frames.append(df)
            logging.info(f"Fetched {len(df)} rows ({current_start.date()} - {current_end.date()})")
        else:
            logging.warning(f"No data for {current_start.date()} - {current_end.date()}")
        current_start = current_end + timedelta(days=1)
    if not all_frames:
        return pd.DataFrame()
    result = pd.concat(all_frames).sort_values("datetime").reset_index(drop=True)
    logging.info(f"OpenWeather rows: {len(result)}")
    return result