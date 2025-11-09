"""
Feature Pipeline
Fetches raw data, computes features, and stores in Feature Store
"""
import sys
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.components.data_ingestion import (
    fetch_openmeteo_data,
    fetch_openweather_data
)
from src.components.feature_store import FeatureStore
from src.logger import logging
from src.exception import CustomException

def run_feature_pipeline(use_hopsworks=False):
    """
    Run the feature pipeline:
    1. Fetch raw data from APIs
    2. Compute features
    3. Store features in Feature Store
    """
    try:
        logging.info("Starting feature pipeline...")
        
        # Initialize Feature Store
        fs = FeatureStore(use_hopsworks=use_hopsworks)
        
        # Step 1: Fetch raw data
        today = date.today()
        # Check if we need to backfill or just update
        if fs.csv_path.exists():
            existing_features = fs.get_features()
            if not existing_features.empty:
                latest_feature_date = existing_features['datetime'].max().date()
                start_date = latest_feature_date + timedelta(days=1)
                logging.info(f"Latest feature date found: {latest_feature_date}. Fetching new data from {start_date}")
            else:
                start_date = today - timedelta(days=FETCH_DAYS)
                logging.info(f"No existing features found. Fetching data for last {FETCH_DAYS} days from {start_date}")
        else:
            start_date = today - timedelta(days=FETCH_DAYS)
            logging.info(f"Feature store not found. Fetching data for last {FETCH_DAYS} days from {start_date}")
        
        end_date = today
        
        if start_date >= end_date:
            logging.info("Feature data already up to date, no new fetch required.")
            print("Feature data already up to date.")
            return True
        
        logging.info(f"Fetching raw data from {start_date} to {end_date}")
        pollutant_data = fetch_openmeteo_data(start_date, end_date)
        weather_data = fetch_openweather_data(start_date, end_date)
        
        if pollutant_data.empty and weather_data.empty:
            logging.warning("No raw data fetched")
            return False
        
        # Merge dataframes
        if "datetime" not in pollutant_data.columns:
            pollutant_data["datetime"] = pd.to_datetime([])
        if "datetime" not in weather_data.columns:
            weather_data["datetime"] = pd.to_datetime([])
        
        raw_data = pd.merge(pollutant_data, weather_data, on="datetime", how="outer").sort_values("datetime")
        
        if raw_data.empty:
            logging.warning("No merged raw data available")
            return False
        
        # Step 2: Compute features
        logging.info("Computing features from raw data...")
        features_df = fs.compute_features(raw_data)
        
        if features_df.empty:
            logging.warning("No features computed")
            return False
        
        logging.info(f"Computed {len(features_df)} feature records")
        
        # Step 3: Store features in Feature Store
        logging.info("Storing features in Feature Store...")
        success = fs.store_features(features_df)
        
        if success:
            logging.info("Feature pipeline completed successfully")
            return True
        else:
            logging.error("Failed to store features")
            return False
            
    except Exception as e:
        logging.error(f"Error in feature pipeline: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Load config to check if Hopsworks should be used
    from config import USE_HOPSWORKS
    run_feature_pipeline(use_hopsworks=USE_HOPSWORKS)

