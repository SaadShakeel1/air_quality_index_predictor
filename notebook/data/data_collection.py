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
OUTPUT_FILE = os.getenv("DATA_OUTPUT_PATH", "notebook/data/merged_aqi_data.csv")
FETCH_DAYS = int(os.getenv("FETCH_DAYS", "90"))

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

def merge_and_save(new_pollutant_df, new_weather_df):
    if "datetime" not in new_pollutant_df.columns:
        new_pollutant_df["datetime"] = pd.to_datetime([])
    if "datetime" not in new_weather_df.columns:
        new_weather_df["datetime"] = pd.to_datetime([])
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, parse_dates=["datetime"])
        logging.info(f"Existing file found: {len(existing)} rows")
    else:
        existing = pd.DataFrame()
        logging.info("No existing dataset found, starting fresh")
    if new_pollutant_df.empty and new_weather_df.empty:
        logging.warning("No new data fetched from either source.")
        print("No new data fetched.")
        return
    merged_new = pd.merge(new_pollutant_df, new_weather_df, on="datetime", how="outer").sort_values("datetime")
    logging.info(f"New merged data rows: {len(merged_new)}")
    final_df = pd.concat([existing, merged_new]).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved updated dataset to {OUTPUT_FILE} ({len(final_df)} rows)")
    print(f"Saved updated dataset → {OUTPUT_FILE} ({len(final_df)} rows)")

def main():
    try:
        logging.info("Starting full data collection pipeline...")
        today = date.today()
        if os.path.exists(OUTPUT_FILE):
            existing_df = pd.read_csv(OUTPUT_FILE, parse_dates=["datetime"])
            last_date = existing_df["datetime"].max().date()
            logging.info(f"Last record found at {last_date}")
            start_date = last_date + timedelta(days=1)
        else:
            start_date = today - timedelta(days=FETCH_DAYS)
            logging.info(f"No file found, starting from {start_date}")
        end_date = today
        if start_date >= end_date:
            logging.info("Data already up to date, no new fetch required")
            print("Data already up to date.")
            return
        pollutant_data = fetch_openmeteo_data(start_date, end_date)
        weather_data = fetch_openweather_data(start_date, end_date)
        merge_and_save(pollutant_data, weather_data)
        logging.info("Data collection completed successfully")
    except Exception as e:
        logging.error("Error during data collection")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
