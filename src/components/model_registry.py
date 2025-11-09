"""
Model Registry Integration Module
Supports MLflow Model Registry (with fallback to local storage)
"""
import os
import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Try to import MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Using local storage for Model Registry.")

class ModelRegistry:
    """Model Registry wrapper - supports MLflow with local storage fallback"""
    
    def __init__(self, use_mlflow=False):
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.registry_path = Path("models/registry")
        self.metadata_path = self.registry_path / "metadata.json"
        
        if self.use_mlflow:
            try:
                # Initialize MLflow
                mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Local SQLite DB
                mlflow.set_experiment("AQI_Prediction")
                logging.info("Connected to MLflow Model Registry")
            except Exception as e:
                logging.warning(f"Failed to initialize MLflow: {e}. Using local storage.")
                self.use_mlflow = False
        
        if not self.use_mlflow:
            logging.info("Using local storage for Model Registry")
            # Ensure directory exists
            self.registry_path.mkdir(parents=True, exist_ok=True)
            # Initialize metadata file if it doesn't exist
            if not self.metadata_path.exists():
                self._init_metadata()
    
    def _init_metadata(self):
        """Initialize metadata file"""
        metadata = {
            "models": [],
            "latest_version": {}
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def register_model(self, model, model_name, model_type, metrics=None, params=None, version=None):
        """
        Register a model in the Model Registry
        
        Args:
            model: Trained model object
            model_name: Name of the model (e.g., 'classifier', 'regressor')
            model_type: Type of model ('classifier' or 'regressor')
            metrics: Dictionary of metrics (e.g., {'accuracy': 0.95, 'f1': 0.92})
            params: Dictionary of hyperparameters
            version: Model version (auto-incremented if not provided)
        """
        try:
            if self.use_mlflow:
                return self._register_mlflow(model, model_name, model_type, metrics, params, version)
            else:
                return self._register_local(model, model_name, model_type, metrics, params, version)
        except Exception as e:
            logging.error(f"Error registering model: {e}")
            return False
    
    def _register_mlflow(self, model, model_name, model_type, metrics, params, version):
        """Register model in MLflow"""
        try:
            with mlflow.start_run():
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=model_name,
                    registered_model_name=f"{model_name}_{model_type}"
                )
                
                # Log metadata
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("model_name", model_name)
                
                logging.info(f"Registered {model_name} in MLflow")
                return True
        except Exception as e:
            logging.error(f"Error registering in MLflow: {e}")
            return False
    
    def _register_local(self, model, model_name, model_type, metrics, params, version):
        """Register model in local storage"""
        try:
            # Load existing metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {"models": [], "latest_version": {}}
            
            # Get or increment version
            if version is None:
                latest = metadata["latest_version"].get(model_name, 0)
                version = latest + 1
            
            # Save model
            model_path = self.registry_path / f"{model_name}_v{version}.pkl"
            joblib.dump(model, model_path)
            
            # Create model entry
            model_entry = {
                "model_name": model_name,
                "model_type": model_type,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "metrics": metrics or {},
                "params": params or {}
            }
            
            # Add to metadata
            metadata["models"].append(model_entry)
            metadata["latest_version"][model_name] = version
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Registered {model_name} v{version} in local registry")
            return True
        except Exception as e:
            logging.error(f"Error registering locally: {e}")
            return False
    
    def get_model(self, model_name, version=None):
        """
        Retrieve a model from the registry
        
        Args:
            model_name: Name of the model
            version: Version number (default: latest)
        
        Returns:
            Model object or None
        """
        try:
            if self.use_mlflow:
                return self._get_mlflow(model_name, version)
            else:
                return self._get_local(model_name, version)
        except Exception as e:
            logging.error(f"Error retrieving model: {e}")
            return None
    
    def _get_mlflow(self, model_name, version):
        """Retrieve model from MLflow"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/latest"
            
            model = mlflow.sklearn.load_model(model_uri)
            logging.info(f"Retrieved {model_name} from MLflow")
            return model
        except Exception as e:
            logging.error(f"Error retrieving from MLflow: {e}")
            return None
    
    def _get_local(self, model_name, version):
        """Retrieve model from local storage"""
        try:
            # Load metadata
            if not self.metadata_path.exists():
                logging.warning("Model registry metadata not found")
                return None
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Find model entry
            if version is None:
                version = metadata["latest_version"].get(model_name)
                if version is None:
                    logging.warning(f"No version found for {model_name}")
                    return None
            
            # Find model entry
            model_entry = None
            for entry in metadata["models"]:
                if entry["model_name"] == model_name and entry["version"] == version:
                    model_entry = entry
                    break
            
            if model_entry is None:
                logging.warning(f"Model {model_name} v{version} not found")
                return None
            
            # Load model
            model_path = Path(model_entry["model_path"])
            if not model_path.exists():
                logging.warning(f"Model file not found: {model_path}")
                return None
            
            model = joblib.load(model_path)
            logging.info(f"Retrieved {model_name} v{version} from local registry")
            return model
        except Exception as e:
            logging.error(f"Error retrieving locally: {e}")
            return None
    
    def list_models(self):
        """List all registered models"""
        try:
            if self.use_mlflow:
                return self._list_mlflow()
            else:
                return self._list_local()
        except Exception as e:
            logging.error(f"Error listing models: {e}")
            return []
    
    def _list_mlflow(self):
        """List models from MLflow"""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            models = client.search_registered_models()
            return [{"name": m.name, "versions": len(m.latest_versions)} for m in models]
        except Exception as e:
            logging.error(f"Error listing from MLflow: {e}")
            return []
    
    def _list_local(self):
        """List models from local storage"""
        try:
            if not self.metadata_path.exists():
                return []
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Group by model name
            models = {}
            for entry in metadata["models"]:
                name = entry["model_name"]
                if name not in models:
                    models[name] = []
                models[name].append({
                    "version": entry["version"],
                    "timestamp": entry["timestamp"],
                    "metrics": entry["metrics"]
                })
            
            return [{"name": k, "versions": v} for k, v in models.items()]
        except Exception as e:
            logging.error(f"Error listing locally: {e}")
            return []
    
    def get_model_metadata(self, model_name, version=None):
        """Get metadata for a specific model version"""
        try:
            if not self.metadata_path.exists():
                return None
            
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if version is None:
                version = metadata["latest_version"].get(model_name)
            
            for entry in metadata["models"]:
                if entry["model_name"] == model_name and entry["version"] == version:
                    return entry
            
            return None
        except Exception as e:
            logging.error(f"Error getting metadata: {e}")
            return None

