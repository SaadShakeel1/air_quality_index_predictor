# Air Quality Index (AQI) Prediction Project

A machine learning project to predict Air Quality Index (AQI) using weather and pollutant data with an interactive Streamlit dashboard.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd data_science_project
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

## 📊 Running the Streamlit App

### Step 1: Verify Models Exist

Check if trained models are available:
```bash
python check_models.py
```

If models are missing, you need to train them first (see Training Models section below).

### Step 2: Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Using the App

1. **Enter Pollutant Values** (in sidebar):
   - CO (Carbon Monoxide) - µg/m³
   - NO (Nitric Oxide) - µg/m³
   - NO₂ (Nitrogen Dioxide) - µg/m³
   - O₃ (Ozone) - µg/m³
   - SO₂ (Sulfur Dioxide) - µg/m³
   - PM₂.₅ (Fine Particles) - µg/m³
   - PM₁₀ (Coarse Particles) - µg/m³
   - NH₃ (Ammonia) - µg/m³

2. **Set Date and Time**:
   - Year, Month, Day, Hour

3. **Click "🔮 Predict AQI"** to get predictions

### App Features

- ✅ Real-time AQI prediction (regression)
- ✅ AQI category classification (Good/Satisfactory/Moderate/Poor/Very Poor)
- ✅ Health alerts for hazardous levels
- ✅ Probability distributions
- ✅ Feature importance visualization
- ✅ Color-coded AQI categories

## 🎯 Training Models

If models don't exist, train them first:

1. **Open the training notebook:**
```bash
jupyter notebook notebook/Model_Training.ipynb
```

2. **Run all cells** (Cell → Run All)

3. **Wait for completion** (5-15 minutes)

4. **Verify models were created:**
```bash
python check_models.py
```

Required model files:
- `final_classifier.pkl`
- `final_regressor.pkl`
- `scaler.pkl`
- `reg_scaler.pkl`

## 📁 Project Structure

```
data_science_project/
├── app.py                      # Streamlit dashboard
├── check_models.py             # Helper to verify models
├── config.py                   # Configuration (for future use)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
│
├── src/                        # Source code utilities
│   ├── logger.py               # Logging (used by data collection)
│   ├── exception.py            # Exception handling (used by data collection)
│   ├── components/              # (for future use)
│   └── pipeline/               # (for future use)
│
└── notebook/                   # Notebooks and data
    ├── EDA_AQI.ipynb          # Exploratory Data Analysis
    ├── Model_Training.ipynb   # Model training
    ├── data/
    │   ├── data_collection.py # Data fetching script
    │   └── merged_aqi_data.csv # Dataset
    ├── models/                 # Trained models (.pkl files)
    └── output/                 # Results and visualizations
```

## 🔧 Data Collection

To fetch new data:

```bash
python notebook/data/data_collection.py
```

This will:
- Fetch data from Open-Meteo and OpenWeather APIs
- Merge data from both sources
- Save to `notebook/data/merged_aqi_data.csv`
- Only fetch new data (incremental updates)

## 📚 Features

### Data Collection
- Fetches from Open-Meteo Air Quality API
- Fetches from OpenWeather Air Pollution API
- Merges and deduplicates data
- Incremental updates

### Model Training
- **Classification Models**: Decision Tree, KNN, Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting, LightGBM, XGBoost
- **Regression Model**: Gradient Boosting Regressor
- Hyperparameter tuning with Optuna
- SMOTE for class imbalance handling
- Model evaluation with multiple metrics

### Exploratory Data Analysis
- Data overview and missing value analysis
- AQI distribution and trends
- Pollutant correlation analysis
- Temporal patterns (hourly, monthly, seasonal)
- Outlier detection

## 🛠️ Troubleshooting

### Models Not Found
**Error:** `Model file not found`

**Solution:**
1. Train models: `jupyter notebook notebook/Model_Training.ipynb`
2. Run all cells
3. Verify: `python check_models.py`

### ModuleNotFoundError
**Error:** `No module named 'streamlit'`

**Solution:**
```bash
pip install -r requirements.txt
```

### Port Already in Use
**Error:** Port 8501 is already in use

**Solution:**
```bash
streamlit run app.py --server.port 8502
```

## 📝 Dependencies

Key dependencies:
- `streamlit` - Web dashboard
- `pandas`, `numpy` - Data processing
- `scikit-learn`, `xgboost`, `lightgbm` - Machine learning
- `joblib` - Model serialization
- `matplotlib`, `seaborn` - Visualization

See `requirements.txt` for complete list.

## 🎓 Model Information

- **Classification**: Predicts AQI category (1-5)
- **Regression**: Predicts continuous AQI value
- **Features**: CO, NO, NO₂, O₃, SO₂, PM₂.₅, PM₁₀, NH₃, Year, Month, Day, Hour
- **Best Models**: XGBoost (classification), Gradient Boosting (regression)

## 🔐 Authentication & Setup

**Quick Answer**: 
- ✅ **No login required** for local usage (CSV fallback works)
- ⚠️ **Hopsworks**: Optional - requires Python 3.12 and API key
- ⚠️ **GitHub Actions**: Optional - requires GitHub account + secrets
- ✅ **OpenWeather API**: Required for fetching new data (configured in `.env`)

**API Keys**: Set in `.env` file:
- `OPENWEATHER_API_KEY`: Required for data collection
- `HOPSWORKS_API_KEY`: Optional (for cloud Feature Store)

## 📄 License

This project is for educational purposes.

## 👤 Author

Saad

---

**Quick Command Reference:**
```bash
# Check models
python check_models.py

# Run app
streamlit run app.py

# Train models (if needed)
jupyter notebook notebook/Model_Training.ipynb

# Collect data
python notebook/data/data_collection.py
```
