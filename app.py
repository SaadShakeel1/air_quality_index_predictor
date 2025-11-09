"""
Streamlit App for AQI Prediction - 3-Day Forecast Dashboard
Uses historical data to forecast pollutants, then predicts AQI using trained models
"""
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

# Import AQI alert system
try:
    from src.components.aqi_alerts import check_aqi_alerts, AQIALertSystem
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AQI Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .aqi-gauge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    .day-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
MODEL_DIR = Path("models")
CLASSIFIER_PATH = MODEL_DIR / "final_classifier.pkl"
REGRESSOR_PATH = MODEL_DIR / "final_regressor.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
REG_SCALER_PATH = MODEL_DIR / "reg_scaler.pkl"

# AQI category mapping
AQI_CATEGORIES = {
    1: {"name": "Good", "color": "aqi-good", "range": "0-50", "description": "Air quality is satisfactory"},
    2: {"name": "Satisfactory", "color": "aqi-satisfactory", "range": "51-100", "description": "Acceptable air quality"},
    3: {"name": "Moderate", "color": "aqi-moderate", "range": "101-200", "description": "Sensitive people may experience minor breathing discomfort"},
    4: {"name": "Poor", "color": "aqi-poor", "range": "201-300", "description": "Everyone may begin to experience health effects"},
    5: {"name": "Very Poor", "color": "aqi-very-poor", "range": ">300", "description": "Health alert: everyone may experience serious health effects"}
}

CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

@st.cache_resource
def load_models():
    """Load all trained models and scalers from Model Registry or local storage"""
    # Try Model Registry first
    try:
        from src.components.model_registry import ModelRegistry
        registry = ModelRegistry(use_mlflow=False)  # Use local registry
        
        classifier = registry.get_model("aqi_classifier")
        regressor = registry.get_model("aqi_regressor")
        scaler = registry.get_model("scaler")
        reg_scaler = registry.get_model("reg_scaler")
        
        if all([classifier, regressor, scaler, reg_scaler]):
            return classifier, regressor, scaler, reg_scaler
    except Exception as e:
        pass  # Fallback to local files
    
    # Fallback to local files
    missing_files = []
    if not CLASSIFIER_PATH.exists():
        missing_files.append("final_classifier.pkl")
    if not REGRESSOR_PATH.exists():
        missing_files.append("final_regressor.pkl")
    if not SCALER_PATH.exists():
        missing_files.append("scaler.pkl")
    if not REG_SCALER_PATH.exists():
        missing_files.append("reg_scaler.pkl")
    
    if missing_files:
        return None, None, None, None
    
    try:
        classifier = joblib.load(CLASSIFIER_PATH)
        regressor = joblib.load(REGRESSOR_PATH)
        scaler = joblib.load(SCALER_PATH)
        reg_scaler = joblib.load(REG_SCALER_PATH)
        return classifier, regressor, scaler, reg_scaler
    except Exception as e:
        return None, None, None, None

@st.cache_data
def load_historical_data(use_hopsworks=False):
    """Load historical data from Feature Store or CSV fallback"""
    try:
        # Try Feature Store first
        from src.components.feature_store import FeatureStore
        fs = FeatureStore(use_hopsworks=use_hopsworks)
        
        df = fs.get_features()
        if not df.empty:
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        
        # Fallback to raw data CSV
        csv_path = Path("data/merged_aqi_data.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def calculate_aqi_change_rate(df):
    """Calculate AQI change rate (hourly difference)"""
    if 'ow_aqi' not in df.columns or len(df) < 2:
        return df
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate change rate (difference from previous hour)
    df['aqi_change_rate'] = df['ow_aqi'].diff()
    
    # Fill first row with 0 (no previous value)
    df['aqi_change_rate'] = df['aqi_change_rate'].fillna(0)
    
    # Calculate percentage change rate
    df['aqi_change_rate_pct'] = (df['ow_aqi'].pct_change() * 100).fillna(0)
    
    return df

def forecast_pollutants(historical_df, hours=72):
    """
    Forecast future pollutant values using historical patterns and time series methods
    Uses the model's understanding of temporal patterns to predict pollutants
    """
    if historical_df.empty:
        return pd.DataFrame()
    
    now = datetime.now()
    forecast_data = []
    
    # Get recent data for pattern analysis (last 2 weeks for better patterns)
    recent_data = historical_df.tail(336).copy()  # Last 2 weeks (14 days * 24 hours)
    
    if recent_data.empty:
        return pd.DataFrame()
    
    # Calculate AQI change rate for historical data
    if 'ow_aqi' in recent_data.columns:
        recent_data = calculate_aqi_change_rate(recent_data)
    
    # Extract temporal features from historical data
    recent_data['hour'] = recent_data['datetime'].dt.hour
    recent_data['day_of_week'] = recent_data['datetime'].dt.dayofweek
    recent_data['month'] = recent_data['datetime'].dt.month
    
    # Pollutant columns to forecast
    pollutant_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    
    # Generate forecast for next 72 hours
    for i in range(hours):
        forecast_time = now + timedelta(hours=i)
        # Convert to pandas Timestamp for dayofweek access
        forecast_time_pd = pd.Timestamp(forecast_time)
        hour = forecast_time_pd.hour
        day_of_week = forecast_time_pd.dayofweek
        month = forecast_time_pd.month
        
        forecast_row = {'datetime': forecast_time}
        
        # For each pollutant, use multiple forecasting strategies
        for col in pollutant_cols:
            if col not in recent_data.columns:
                forecast_row[col] = 0
                continue
            
            # Strategy 1: Match by hour and day of week (weekly pattern)
            similar = recent_data[
                (recent_data['hour'] == hour) & 
                (recent_data['day_of_week'] == day_of_week)
            ]
            
            if len(similar) < 3:
                # Strategy 2: Match by hour only (daily pattern)
                similar = recent_data[recent_data['hour'] == hour]
            
            if len(similar) < 3:
                # Strategy 3: Use all recent data
                similar = recent_data
            
            # Use weighted average: more weight to recent data
            if len(similar) > 0:
                # Calculate trend (simple linear trend from last 24 hours)
                last_24h = recent_data.tail(24)
                if len(last_24h) > 1:
                    trend = (last_24h[col].iloc[-1] - last_24h[col].iloc[0]) / len(last_24h)
                    base_value = similar[col].median()
                    # Apply trend with decay (trend weakens over time)
                    forecast_value = base_value + (trend * i * 0.5)  # Decay factor
                else:
                    forecast_value = similar[col].median()
                
                # Ensure non-negative
                forecast_value = max(0, forecast_value)
            else:
                # Fallback to overall median
                forecast_value = recent_data[col].median() if len(recent_data) > 0 else 0
            
            forecast_row[col] = forecast_value
        
        forecast_data.append(forecast_row)
    
    return pd.DataFrame(forecast_data)

def get_aqi_category(aqi_value):
    """Convert AQI value to category"""
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

def predict_aqi_batch(features_df, classifier, regressor, scaler, reg_scaler):
    """Make predictions for a batch of features using trained models"""
    # Prepare features in correct order (as used in training)
    feature_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'year', 'month', 'day', 'hour']
    
    # Extract features
    X = features_df[feature_cols].values
    
    # Classification predictions
    X_scaled = scaler.transform(X)
    class_preds = classifier.predict(X_scaled)
    class_probas = classifier.predict_proba(X_scaled) if hasattr(classifier, "predict_proba") else None
    
    # Regression predictions
    X_reg_scaled = reg_scaler.transform(X)
    reg_preds = regressor.predict(X_reg_scaled)
    
    # Create results dataframe
    results = features_df[['datetime']].copy()
    results['predicted_aqi'] = reg_preds
    results['aqi_category'] = [get_aqi_category(aqi) for aqi in reg_preds]
    results['predicted_class'] = [INV_CLASS_MAP[int(cp)] for cp in class_preds]
    results['category_name'] = [AQI_CATEGORIES[cat]["name"] for cat in results['aqi_category']]
    
    # Add pollutant values for display
    for col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
        if col in features_df.columns:
            results[col] = features_df[col].values
    
    return results


def get_aqi_color(aqi):
    """Get color based on AQI value"""
    if aqi <= 50:
        return '#00e400'  # Green
    elif aqi <= 100:
        return '#ffff00'  # Yellow
    elif aqi <= 200:
        return '#ff7e00'  # Orange
    elif aqi <= 300:
        return '#ff0000'  # Red
    else:
        return '#8f3f97'  # Purple

def create_chart(results_df):
    """Create AQI time series chart"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(results_df['datetime'], results_df['predicted_aqi'], 
             linewidth=2.5, color='#667eea', marker='o', markersize=4)
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=100, color='yellow', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=200, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=300, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.fill_between(results_df['datetime'], results_df['predicted_aqi'], alpha=0.15, color='#667eea')
    ax.set_ylabel('AQI', fontsize=11, fontweight='bold')
    ax.set_title('3-Day AQI Forecast Trend', fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    return fig

def main():
    # Load models
    classifier, regressor, scaler, reg_scaler = load_models()
    if classifier is None:
        st.error("Models not found.")
        st.stop()
    
    # Load historical data
    historical_df = load_historical_data(use_hopsworks=True)
    if historical_df.empty:
        st.error("No data available.")
        st.stop()
    
    # Generate forecast
    forecast_pollutants_df = forecast_pollutants(historical_df, hours=72)
    if forecast_pollutants_df.empty:
        st.error("Could not generate forecast.")
        st.stop()
    
    # Prepare features
    forecast_pollutants_df['year'] = forecast_pollutants_df['datetime'].dt.year
    forecast_pollutants_df['month'] = forecast_pollutants_df['datetime'].dt.month
    forecast_pollutants_df['day'] = forecast_pollutants_df['datetime'].dt.day
    forecast_pollutants_df['hour'] = forecast_pollutants_df['datetime'].dt.hour
    
    required_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
    for col in required_cols:
        if col not in forecast_pollutants_df.columns:
            forecast_pollutants_df[col] = 0
        forecast_pollutants_df[col] = forecast_pollutants_df[col].fillna(0)
    
    # Calculate AQI change rate for forecast
    # First, we need to predict initial AQI to calculate change rate
    # For now, use historical average change rate pattern
    if 'aqi_change_rate' not in forecast_pollutants_df.columns:
        # Get historical change rate pattern
        historical_with_aqi = historical_df.copy()
        if 'ow_aqi' in historical_with_aqi.columns:
            historical_with_aqi = calculate_aqi_change_rate(historical_with_aqi)
            # Use average change rate by hour
            historical_with_aqi['hour'] = historical_with_aqi['datetime'].dt.hour
            avg_change_rate = historical_with_aqi.groupby('hour')['aqi_change_rate'].mean().to_dict()
            
            # Apply pattern to forecast
            forecast_pollutants_df['aqi_change_rate'] = forecast_pollutants_df['hour'].map(avg_change_rate).fillna(0)
        else:
            forecast_pollutants_df['aqi_change_rate'] = 0
    
    # Predict AQI
    forecast_results = predict_aqi_batch(forecast_pollutants_df, classifier, regressor, scaler, reg_scaler)
    
    # Update AQI change rate based on actual predictions
    forecast_results = forecast_results.sort_values('datetime').reset_index(drop=True)
    forecast_results['aqi_change_rate'] = forecast_results['predicted_aqi'].diff().fillna(0)
    forecast_results['aqi_change_rate_pct'] = (forecast_results['predicted_aqi'].pct_change() * 100).fillna(0)
    
    # Filter to next 3 days
    now = datetime.now()
    end_time = now + timedelta(hours=72)
    forecast_results = forecast_results[
        (forecast_results['datetime'] >= now) & 
        (forecast_results['datetime'] <= end_time)
    ].head(72)
    
    if forecast_results.empty:
        st.error("No forecast data available.")
        st.stop()
    
    # Title
    st.markdown('<h1 style="text-align: center; margin-bottom: 1rem;">🌬️ Air Quality Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Next 3 Days Forecast</p>', unsafe_allow_html=True)
    
    # Current AQI Gauge
    current_aqi = forecast_results.iloc[0]['predicted_aqi']
    current_category = forecast_results.iloc[0]['category_name']
    aqi_color = get_aqi_color(current_aqi)
    
    # Get AQI change rate
    current_change_rate = forecast_results.iloc[0].get('aqi_change_rate', 0)
    current_change_pct = forecast_results.iloc[0].get('aqi_change_rate_pct', 0)
    change_direction = "↑" if current_change_rate > 0 else "↓" if current_change_rate < 0 else "→"
    change_color = "#ff0000" if current_change_rate > 5 else "#00e400" if current_change_rate < -5 else "#666"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {aqi_color} 0%, {aqi_color}dd 100%);
                    padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0; font-size: 3rem; font-weight: bold;">{current_aqi:.0f}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem;">{current_category}</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: rgba(255,255,255,0.9);">
                Change: <span style="color: {change_color};">{change_direction} {abs(current_change_rate):.1f}</span>
                ({current_change_pct:+.1f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced AQI Alerts Section
    if ALERTS_AVAILABLE:
        st.markdown("### 🚨 AQI Alert System")
        
        # Check for AQI alerts
        try:
            # Get current AQI and forecast data
            current_aqi_value = float(current_aqi)
            forecast_aqi_values = forecast_results['predicted_aqi'].tolist()
            
            # Check for alerts
            alerts_triggered = check_aqi_alerts(current_aqi_value, forecast_aqi_values)
            
            if alerts_triggered:
                alert_system = AQIALertSystem()
                
                # Display active alerts
                st.markdown("#### 🚨 Active Alerts")
                for alert in alerts_triggered:
                    alert_color = '#ff0000' if alert.level.value >= 3 else '#ff6600'
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {alert_color}15 0%, {alert_color}25 100%);
                                padding: 1rem; border-radius: 10px; border-left: 5px solid {alert_color};
                                margin-bottom: 1rem;">
                        <h5 style="margin: 0 0 0.5rem 0; color: {alert_color}; font-size: 1.1rem;">
                            {alert.level.name.replace('_', ' ').title()}
                        </h5>
                        <p style="margin: 0; color: #333; font-size: 0.9rem;">
                            <strong>Rule:</strong> {alert.rule.description}<br>
                            <strong>Current AQI:</strong> {alert.triggered_value:.0f}<br>
                            <strong>Threshold:</strong> {alert.rule.threshold}<br>
                            <strong>Triggered at:</strong> {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add health recommendations
                    if alert.level.value >= 3:  # Severe or higher
                        st.warning("⚠️ **Health Recommendations:**")
                        st.write("• Avoid outdoor activities, especially for sensitive groups")
                        st.write("• Keep windows closed and use air purifiers indoors")
                        st.write("• Wear N95 masks if going outside is necessary")
                        st.write("• Monitor for respiratory symptoms")
            else:
                st.success("✅ No active AQI alerts - Air quality is within safe limits")
                
        except Exception as e:
            st.error(f"Error checking AQI alerts: {str(e)}")
    
    # Health Alerts Section (Original)
    high_aqi_periods = forecast_results[forecast_results['aqi_category'] >= 4]
    if not high_aqi_periods.empty:
        st.markdown("### ⚠️ Health Alerts")
        alert_color = '#ff0000'
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {alert_color}15 0%, {alert_color}25 100%);
                    padding: 1.5rem; border-radius: 10px; border-left: 5px solid {alert_color};
                    margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: {alert_color}; font-size: 1.2rem;">⚠️ High AQI Detected</h4>
            <p style="margin: 0; color: #333;">
                <strong>{len(high_aqi_periods)} hours</strong> over the next 3 days have AQI ≥ 200 (Poor/Very Poor).
                Sensitive groups should limit outdoor activities during these periods.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show worst periods
        worst_periods = high_aqi_periods.nlargest(3, 'predicted_aqi')
        st.markdown("**Worst Periods:**")
        for idx, row in worst_periods.iterrows():
            period_time = row['datetime'].strftime('%Y-%m-%d %H:00')
            st.markdown(f"""
            <div style="background: #fff5f5; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;
                        border-left: 3px solid {alert_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #333;">{period_time}</span>
                    <span style="font-weight: bold; color: {alert_color}; font-size: 1.1rem;">
                        AQI: {row['predicted_aqi']:.0f} ({row['category_name']})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Pollutant Cards
    st.markdown("### Current Pollutant Levels")
    pollutant_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
    pollutant_names = {'pm2_5': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO₂', 'so2': 'SO₂', 'co': 'CO', 'o3': 'O₃'}
    pollutant_units = {'pm2_5': 'µg/m³', 'pm10': 'µg/m³', 'no2': 'µg/m³', 'so2': 'µg/m³', 'co': 'mg/m³', 'o3': 'µg/m³'}
    
    # Pollutant safety thresholds (WHO guidelines)
    pollutant_thresholds = {
        'pm2_5': 25,  # µg/m³ (24-hour mean)
        'pm10': 50,   # µg/m³ (24-hour mean)
        'no2': 25,    # µg/m³ (24-hour mean)
        'so2': 20,    # µg/m³ (24-hour mean)
        'co': 10,     # mg/m³ (24-hour mean)
        'o3': 100     # µg/m³ (8-hour mean)
    }
    
    cols = st.columns(6)
    current_data = forecast_results.iloc[0]
    for idx, col_name in enumerate(pollutant_cols):
        if col_name in current_data:
            value = current_data[col_name]
            unit = pollutant_units.get(col_name, '')
            threshold = pollutant_thresholds.get(col_name, float('inf'))
            
            # Determine if pollutant is high
            is_high = value > threshold
            card_color = '#ffcccc' if is_high else 'white'
            border_color = '#ff0000' if is_high else 'transparent'
            
            with cols[idx]:
                st.markdown(f"""
                <div style="background: {card_color}; padding: 1rem; border-radius: 10px; 
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
                            border: 2px solid {border_color};">
                    <p style="margin: 0; font-size: 0.9rem; color: #666;">{pollutant_names[col_name]}</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; font-weight: bold; 
                              color: {'#ff0000' if is_high else '#333'};">
                        {value:.1f}
                    </p>
                    <p style="margin: 0; font-size: 0.7rem; color: #999;">{unit}</p>
                    {f'<p style="margin: 0.3rem 0 0 0; font-size: 0.7rem; color: #ff0000; font-weight: bold;">⚠️ High</p>' if is_high else ''}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3-Day Forecast Cards
    st.markdown("### 3-Day Forecast")
    forecast_results['date'] = forecast_results['datetime'].dt.date
    forecast_results['hour'] = forecast_results['datetime'].dt.hour
    unique_dates = sorted(forecast_results['date'].unique())[:3]
    
    # Create 3 columns for 3 day cards
    day_cols = st.columns(3)
    
    for idx, date in enumerate(unique_dates):
        day_data = forecast_results[forecast_results['date'] == date].copy()
        day_name = day_data['datetime'].iloc[0].strftime('%A')
        day_date = day_data['datetime'].iloc[0].strftime('%B %d')
        avg_aqi = day_data['predicted_aqi'].mean()
        day_color = get_aqi_color(avg_aqi)
        
        with day_cols[idx]:
            # Day Card
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 15px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; border-top: 4px solid {day_color};">
                <h3 style="margin: 0 0 0.5rem 0; color: #333; font-size: 1.3rem;">{day_name}</h3>
                <p style="margin: 0 0 1rem 0; color: #666; font-size: 0.9rem;">{day_date}</p>
                <div style="background: {day_color}15; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                    <p style="margin: 0; color: #666; font-size: 0.8rem;">Average AQI</p>
                    <p style="margin: 0.3rem 0 0 0; color: {day_color}; font-size: 2rem; font-weight: bold;">{avg_aqi:.0f}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Time periods
            def get_time_period(hour):
                if 6 <= hour < 12:
                    return "Morning"
                elif 12 <= hour < 18:
                    return "Afternoon"
                elif 18 <= hour < 22:
                    return "Evening"
                else:
                    return "Night"
            
            day_data['period'] = day_data['hour'].apply(get_time_period)
            periods = ['Morning', 'Afternoon', 'Evening', 'Night']
            
            for period in periods:
                period_data = day_data[day_data['period'] == period]
                if not period_data.empty:
                    period_avg = period_data['predicted_aqi'].mean()
                    period_color = get_aqi_color(period_avg)
                    
                    st.markdown(f"""
                    <div style="background: {period_color}10; padding: 0.8rem; border-radius: 8px; 
                                margin-bottom: 0.5rem; border-left: 3px solid {period_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: bold; color: #333;">{period}</span>
                            <span style="font-weight: bold; color: {period_color}; font-size: 1.1rem;">{period_avg:.0f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Day chart
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.plot(day_data['hour'], day_data['predicted_aqi'], 
                     linewidth=2, color=day_color, marker='o', markersize=3)
            ax.fill_between(day_data['hour'], day_data['predicted_aqi'], alpha=0.2, color=day_color)
            ax.set_xlabel('Hour', fontsize=8)
            ax.set_ylabel('AQI', fontsize=8)
            ax.set_title('Hourly Forecast', fontsize=9, fontweight='bold', pad=5)
            ax.grid(True, alpha=0.2, linestyle='--')
            ax.set_xticks(range(0, 24, 6))
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Feature Importance Section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Feature Importance Analysis")
    
    if SHAP_AVAILABLE:
        try:
            # Get sample data for SHAP
            sample_data = forecast_results.iloc[:5].copy()
            feature_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'year', 'month', 'day', 'hour']
            
            # Prepare features
            X_sample = sample_data[feature_cols].values
            X_scaled = scaler.transform(X_sample)
            
            # Create SHAP explainer
            with st.spinner("Computing SHAP values..."):
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_scaled[:5])
            
            # Feature importance plot
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Feature Importance (Summary)**")
                # Get mean absolute SHAP values
                if isinstance(shap_values, list):
                    mean_shap = np.abs(shap_values[0]).mean(0)
                else:
                    mean_shap = np.abs(shap_values).mean(0)
                
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': mean_shap
                }).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#667eea')
                ax.set_xlabel('Mean |SHAP Value|', fontsize=10, fontweight='bold')
                ax.set_title('Feature Importance (SHAP)', fontsize=12, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.2, axis='x', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**Top Contributing Features**")
                top_features = feature_importance_df.tail(5)
                for idx, row in top_features.iterrows():
                    st.markdown(f"""
                    <div style="background: #f0f0f0; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-weight: bold;">{row['Feature']}</span>
                            <span style="color: #667eea; font-weight: bold;">{row['Importance']:.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.warning(f"SHAP computation failed: {str(e)}")
            # Fallback to model feature importance
            if hasattr(classifier, 'feature_importances_'):
                feature_importance = classifier.feature_importances_
                feature_names = feature_cols
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=True)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'], color='#667eea')
                ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
                ax.set_title('Feature Importance (Model)', fontsize=12, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.2, axis='x', linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    else:
        st.info("SHAP not available. Using model feature importance instead.")
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = classifier.feature_importances_
            feature_names = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'year', 'month', 'day', 'hour']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='#667eea')
            ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
            ax.set_title('Feature Importance (Model)', fontsize=12, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.2, axis='x', linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

if __name__ == "__main__":
    main()
