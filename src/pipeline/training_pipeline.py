"""
Training Pipeline
Fetches features from Feature Store, trains models, and registers in Model Registry
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.components.feature_store import FeatureStore
from src.components.model_registry import ModelRegistry
from src.components.feature_importance import FeatureImportanceAnalyzer
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,  # Capture all log levels
    filemode='w'  # Overwrite existing log file
)

# Also add a console handler to see logs in real-time
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier
import joblib

def run_training_pipeline(use_feature_store=True, use_model_registry=True, use_hopsworks=False, use_mlflow=False):
    """
    Run the training pipeline:
    1. Fetch features from Feature Store
    2. Train models
    3. Evaluate models
    4. Register models in Model Registry
    """
    try:
        logging.info("Starting training pipeline...")
        
        # Step 1: Load features from Feature Store
        if use_feature_store:
            fs = FeatureStore(use_hopsworks=use_hopsworks)
            df = fs.get_features()
        else:
            # Fallback to CSV
            csv_path = Path("data/merged_aqi_data.csv")
            if not csv_path.exists():
                logging.error("No data file found")
                return False
            df = pd.read_csv(csv_path, parse_dates=["datetime"])
        
        if df.empty:
            logging.error("No features available")
            return False
        
        logging.info(f"Loaded {len(df)} feature records")
        
        # Step 2: Prepare data
        # Compute features if not already computed
        if 'year' not in df.columns:
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            df['hour'] = df['datetime'].dt.hour
        
        # Prepare features and targets
        feature_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'year', 'month', 'day', 'hour']
        X = df[feature_cols].copy()
        
        # Prepare targets
        if 'ow_aqi' in df.columns:
            df['ow_aqi_orig'] = df['ow_aqi'].astype(float)
            df['ow_aqi_round'] = df['ow_aqi_orig'].round().astype(int)
            df = df[df['ow_aqi_round'].between(1, 5)]
            
            # Map AQI categories (1-5) to class indices (0-4)
            class_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
            y_class = df['ow_aqi_round'].map(class_map)
            y_reg = df['ow_aqi_orig'].copy()
            
            # Check for missing classes and handle them
            unique_classes = sorted(y_class.unique())
            if len(unique_classes) < 2:
                logging.error(f"Not enough classes for classification. Found: {unique_classes}")
                return False
            
            # XGBoost requires classes to start from 0 and be consecutive
            # If we have classes [2, 3, 4], we need to remap to [0, 1, 2]
            if unique_classes[0] != 0 or len(unique_classes) != unique_classes[-1] + 1:
                # Remap to consecutive classes starting from 0
                class_remap = {old: new for new, old in enumerate(unique_classes)}
                y_class = y_class.map(class_remap)
                logging.info(f"Remapped classes: {class_remap}")
            
            # Ensure X and y have same index after filtering
            X = X.loc[df.index].copy()
            y_class = y_class.loc[df.index].copy()
            y_reg = y_reg.loc[df.index].copy()
        else:
            logging.error("Target variable 'ow_aqi' not found")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        reg_scaler = StandardScaler()
        X_train_reg = reg_scaler.fit_transform(X_train)
        X_test_reg = reg_scaler.transform(X_test)
        
        # Step 3: Train classifier
        logging.info("Training classifier...")
        classifier = XGBClassifier(n_estimators=200, random_state=42, eval_metric="mlogloss")
        classifier.fit(X_train_scaled, y_train)
        
        # Evaluate classifier
        y_pred_clf = classifier.predict(X_test_scaled)
        clf_accuracy = accuracy_score(y_test, y_pred_clf)
        clf_f1 = f1_score(y_test, y_pred_clf, average="macro")
        
        logging.info(f"Classifier - Accuracy: {clf_accuracy:.4f}, F1: {clf_f1:.4f}")
        
        # Step 4: Train regressor
        logging.info("Training regressor...")
        regressor = GradientBoostingRegressor(n_estimators=300, random_state=42)
        regressor.fit(X_train_reg, y_reg_train)
        
        # Evaluate regressor
        y_pred_reg = regressor.predict(X_test_reg)
        reg_mae = mean_absolute_error(y_reg_test, y_pred_reg)
        reg_rmse = mean_squared_error(y_reg_test, y_pred_reg)
        reg_r2 = r2_score(y_reg_test, y_pred_reg)
        
        logging.info(f"Regressor - MAE: {reg_mae:.4f}, RMSE: {reg_rmse:.4f}, R²: {reg_r2:.4f}")
        
        # Step 5: Feature Importance Analysis
        logging.info("Analyzing feature importance...")
        try:
            # Define model directory relative to the project root
            project_root = Path(__file__).parent.parent.parent
            model_dir = project_root / "models"
            model_dir.mkdir(exist_ok=True)

            logging.info(f"Saving models to: {model_dir}")
            joblib.dump(classifier, model_dir / "final_classifier.pkl")
            joblib.dump(regressor, model_dir / "final_regressor.pkl")
            joblib.dump(scaler, model_dir / "scaler.pkl")
            joblib.dump(reg_scaler, model_dir / "reg_scaler.pkl")
            logging.info("Models and scalers saved successfully.")

            logging.info("Starting feature importance analysis for classifier...")
            # Analyze classifier
            clf_analyzer = FeatureImportanceAnalyzer()
            clf_analyzer.load_model(str(model_dir / "final_classifier.pkl"))
            clf_analyzer.create_shap_explainer(X_train, "tree")
            shap_values, X_sample = clf_analyzer.compute_shap_values(X_test, sample_size=100)
            clf_importance = clf_analyzer.get_feature_importance_summary(shap_values, X_sample)
            logging.info("Classifier feature importance analysis complete.")
            
            # Save classifier importance results
            clf_importance_dir = model_dir / "feature_importance" / "classifier"
            clf_analyzer.save_importance_results(clf_importance, str(clf_importance_dir))
            clf_analyzer.plot_feature_importance_bar(clf_importance, save_path=str(clf_importance_dir / "plots" / "feature_importance_bar.png"))
            
            logging.info("Starting feature importance analysis for regressor...")
            # Analyze regressor
            reg_analyzer = FeatureImportanceAnalyzer()
            reg_analyzer.load_model(str(model_dir / "final_regressor.pkl"))
            reg_analyzer.create_shap_explainer(X_train, "tree")
            reg_shap_values, reg_X_sample = reg_analyzer.compute_shap_values(X_test, sample_size=100)
            reg_importance = reg_analyzer.get_feature_importance_summary(reg_shap_values, reg_X_sample)
            logging.info("Regressor feature importance analysis complete.")
            
            # Save regressor importance results
            reg_importance_dir = model_dir / "feature_importance" / "regressor"
            reg_analyzer.save_importance_results(reg_importance, str(reg_importance_dir))
            reg_analyzer.plot_feature_importance_bar(reg_importance, save_path=str(reg_importance_dir / "plots" / "feature_importance_bar.png"))
            
            logging.info("Feature importance analysis completed")
            
        except Exception as e:
            logging.warning(f"Feature importance analysis failed: {e}")
        
        # Step 6: Register models
        if use_model_registry:
            registry = ModelRegistry(use_mlflow=use_mlflow)
            
            # Register classifier
            registry.register_model(
                model=classifier,
                model_name="aqi_classifier",
                model_type="classifier",
                metrics={"accuracy": clf_accuracy, "f1_score": clf_f1},
                params={"n_estimators": 200, "model": "XGBClassifier"}
            )
            
            # Register regressor
            registry.register_model(
                model=regressor,
                model_name="aqi_regressor",
                model_type="regressor",
                metrics={"mae": reg_mae, "rmse": reg_rmse, "r2": reg_r2},
                params={"n_estimators": 300, "model": "GradientBoostingRegressor"}
            )
            
            # Register scalers
            registry.register_model(
                model=scaler,
                model_name="scaler",
                model_type="preprocessor",
                metrics={},
                params={"type": "StandardScaler"}
            )
            
            registry.register_model(
                model=reg_scaler,
                model_name="reg_scaler",
                model_type="preprocessor",
                metrics={},
                params={"type": "StandardScaler"}
            )
        

        
        logging.info("Training pipeline completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        return False

if __name__ == "__main__":
    # Load config to check if Hopsworks should be used
    from config import USE_HOPSWORKS
    run_training_pipeline(
        use_feature_store=True,
        use_model_registry=True,
        use_hopsworks=USE_HOPSWORKS,
        use_mlflow=False
    )

