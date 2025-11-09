"""
Configuration file for AQI Prediction Project
Loads configuration from environment variables with sensible defaults
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable is required. Please set it in .env file")

# Hopsworks Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")

# Location Configuration (default: Karachi, Pakistan)
LATITUDE = float(os.getenv("LATITUDE", "24.8607"))
LONGITUDE = float(os.getenv("LONGITUDE", "67.0011"))

# Data Collection Settings
FETCH_DAYS = int(os.getenv("FETCH_DAYS", "90"))
DATA_OUTPUT_PATH = os.getenv("DATA_OUTPUT_PATH", "notebook/data/merged_aqi_data.csv")

# Feature Store Configuration
USE_HOPSWORKS = os.getenv("USE_HOPSWORKS", "false").lower() == "true"
FEATURE_STORE_CSV_PATH = Path("notebook/data/feature_store.csv")

# Model Configuration
MODEL_DIR = Path("notebook/models")
OUTPUT_DIR = Path("notebook/output")

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_STORE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

