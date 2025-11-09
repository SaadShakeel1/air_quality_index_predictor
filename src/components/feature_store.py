"""
Feature Store Integration Module
Supports Hopsworks Feature Store (with fallback to CSV)
"""
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging

import logging

# Load environment variables
load_dotenv()

# Fix for Python 3.13 compatibility with Hopsworks
if sys.version_info >= (3, 12):
    try:
        # Create compatibility shim for deprecated 'imp' module
        import types
        if 'imp' not in sys.modules:
            imp_module = types.ModuleType('imp')
            imp_module.find_module = lambda name, path=None: None
            imp_module.load_module = lambda name: __import__(name)
            sys.modules['imp'] = imp_module
    except Exception:
        pass

# Try to import Hopsworks
try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    HOPSWORKS_AVAILABLE = False
    logging.warning(f"Hopsworks not available: {e}. Using CSV fallback for Feature Store.")

class FeatureStore:
    """Feature Store wrapper - supports Hopsworks with CSV fallback"""
    
    def __init__(self, use_hopsworks=False):
        self.use_hopsworks = use_hopsworks and HOPSWORKS_AVAILABLE
        self.csv_path = Path("data/feature_store.csv")
        
        if self.use_hopsworks:
            try:
                # Get API key from environment
                api_key = os.getenv("HOPSWORKS_API_KEY", "")
                if not api_key:
                    logging.warning("HOPSWORKS_API_KEY not found. Using CSV fallback.")
                    self.use_hopsworks = False
                else:
                    # Initialize Hopsworks connection
                    project = hopsworks.login(api_key_value=api_key)
                    self.fs = project.get_feature_store()
                    logging.info("Connected to Hopsworks Feature Store")
            except Exception as e:
                logging.warning(f"Failed to connect to Hopsworks: {e}. Using CSV fallback.")
                self.use_hopsworks = False
        
        if not self.use_hopsworks:
            logging.info("Using CSV-based Feature Store")
            # Ensure directory exists
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    def compute_features(self, raw_data_df):
        """
        Compute features from raw data
        Features: pollutants + time-based features + derived features
        """
        if raw_data_df.empty:
            return pd.DataFrame()
        
        df = raw_data_df.copy()
        
        # Ensure datetime column
        if 'datetime' not in df.columns:
            logging.error("datetime column not found in raw data")
            return pd.DataFrame()
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Time-based features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Derived features: AQI change rate
        if 'ow_aqi' in df.columns:
            df['aqi_change_rate'] = df['ow_aqi'].diff().fillna(0)
            df['aqi_change_rate_pct'] = (df['ow_aqi'].pct_change() * 100).fillna(0)
            # Rolling averages
            df['aqi_rolling_3h'] = df['ow_aqi'].rolling(window=3, min_periods=1).mean()
            df['aqi_rolling_24h'] = df['ow_aqi'].rolling(window=24, min_periods=1).mean()
        
        # Pollutant rolling averages
        pollutant_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
        for col in pollutant_cols:
            if col in df.columns:
                df[f'{col}_rolling_3h'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_24h'] = df[col].rolling(window=24, min_periods=1).mean()
        
        # Target variable (AQI)
        if 'ow_aqi' in df.columns:
            df['target_aqi'] = df['ow_aqi']
            df['target_aqi_category'] = df['target_aqi'].apply(self._get_aqi_category)
        
        return df
    
    def _get_aqi_category(self, aqi_value):
        """Convert AQI value to category"""
        if pd.isna(aqi_value):
            return None
        if aqi_value <= 50:
            return 1
        elif aqi_value <= 100:
            return 2
        elif aqi_value <= 200:
            return 3
        elif aqi_value <= 300:
            return 4
        else:
            return 5
    
    def store_features(self, features_df):
        """Store features in Feature Store"""
        if features_df.empty:
            logging.warning("Empty features dataframe, nothing to store")
            return False
        
        try:
            if self.use_hopsworks:
                return self._store_hopsworks(features_df)
            else:
                return self._store_csv(features_df)
        except Exception as e:
            logging.error(f"Error storing features: {e}")
            return False
    
    def _store_hopsworks(self, features_df):
        """Store features in Hopsworks Feature Store"""
        try:
            # Get or create feature group
            feature_group_name = "aqi_features"
            try:
                fg = self.fs.get_feature_group(name=feature_group_name, version=1)
            except:
                # Create feature group if it doesn't exist
                fg = self.fs.create_feature_group(
                    name=feature_group_name,
                    version=1,
                    description="AQI prediction features",
                    primary_key=["datetime"],
                    event_time="datetime"
                )
            
            # Insert features
            fg.insert(features_df)
            logging.info(f"Stored {len(features_df)} features in Hopsworks")
            return True
        except Exception as e:
            logging.error(f"Error storing in Hopsworks: {e}")
            return False
    
    def _store_csv(self, features_df):
        """Store features in CSV (fallback)"""
        try:
            # Load existing features if any
            if self.csv_path.exists():
                existing = pd.read_csv(self.csv_path, parse_dates=["datetime"])
                # Merge and deduplicate
                combined = pd.concat([existing, features_df]).drop_duplicates(
                    subset=["datetime"], keep="last"
                ).sort_values("datetime").reset_index(drop=True)
            else:
                combined = features_df.copy()
            
            # Save to CSV
            combined.to_csv(self.csv_path, index=False)
            logging.info(f"Stored {len(features_df)} features in CSV ({len(combined)} total)")
            return True
        except Exception as e:
            logging.error(f"Error storing in CSV: {e}")
            return False
    
    def get_features(self, start_date=None, end_date=None):
        """Retrieve features from Feature Store"""
        try:
            if self.use_hopsworks:
                return self._get_hopsworks(start_date, end_date)
            else:
                return self._get_csv(start_date, end_date)
        except Exception as e:
            logging.error(f"Error retrieving features: {e}")
            return pd.DataFrame()
    
    def _get_hopsworks(self, start_date=None, end_date=None):
        """Retrieve features from Hopsworks"""
        try:
            fg = self.fs.get_feature_group(name="aqi_features", version=1)
            
            # Build query
            query = fg.select_all()
            if start_date:
                query = query.filter(fg.datetime >= start_date)
            if end_date:
                query = query.filter(fg.datetime <= end_date)
            
            # Read features
            df = query.read()
            logging.info(f"Retrieved {len(df)} features from Hopsworks")
            return df
        except Exception as e:
            logging.error(f"Error retrieving from Hopsworks: {e}")
            return pd.DataFrame()
    
    def _get_csv(self, start_date=None, end_date=None):
        """Retrieve features from CSV"""
        try:
            if not self.csv_path.exists():
                logging.warning(f"Feature store CSV not found: {self.csv_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(self.csv_path, parse_dates=["datetime"])
            
            # Filter by date if specified
            if start_date:
                df = df[df['datetime'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['datetime'] <= pd.to_datetime(end_date)]
            
            df = df.sort_values("datetime").reset_index(drop=True)
            logging.info(f"Retrieved {len(df)} features from CSV")
            return df
        except Exception as e:
            logging.error(f"Error retrieving from CSV: {e}")
            return pd.DataFrame()

