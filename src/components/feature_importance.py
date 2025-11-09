"""
Feature importance analysis using SHAP and LIME
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union

# SHAP and LIME imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

import logging

from src.exception import CustomException

class FeatureImportanceAnalyzer:
    """Analyze feature importance using SHAP and LIME"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the analyzer
        
        Args:
            model_path: Path to the trained model (optional)
        """
        self.model = None
        self.explainer = None
        self.lime_explainer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            self.model = joblib.load(model_path)
            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise CustomException(e)
    
    def create_shap_explainer(self, X_train: pd.DataFrame, model_type: str = "tree"):
        """
        Create SHAP explainer
        
        Args:
            X_train: Training data
            model_type: Type of model ("tree", "linear", "kernel")
        """
        if not SHAP_AVAILABLE:
            logging.warning("SHAP not available. Install with: pip install shap")
            return
        
        try:
            if model_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(self.model, X_train)
            elif model_type == "kernel":
                self.explainer = shap.KernelExplainer(self.model.predict, X_train.sample(100))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logging.info(f"Created SHAP {model_type} explainer")
            
        except Exception as e:
            logging.error(f"Failed to create SHAP explainer: {e}")
            raise CustomException(e)
    
    def create_lime_explainer(self, X_train: pd.DataFrame, mode: str = "classification"):
        """
        Create LIME explainer
        
        Args:
            X_train: Training data
            mode: "classification" or "regression"
        """
        if not LIME_AVAILABLE:
            logging.warning("LIME not available. Install with: pip install lime")
            return
        
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=["Good", "Fair", "Moderate", "Poor", "Very Poor"] if mode == "classification" else None,
                mode=mode,
                verbose=False
            )
            logging.info(f"Created LIME {mode} explainer")
            
        except Exception as e:
            logging.error(f"Failed to create LIME explainer: {e}")
            raise CustomException(e)
    
    def compute_shap_values(self, X_test: pd.DataFrame, sample_size: int = 100) -> np.ndarray:
        """
        Compute SHAP values
        
        Args:
            X_test: Test data
            sample_size: Number of samples to use for explanation
            
        Returns:
            SHAP values array
        """
        if not self.explainer:
            raise ValueError("SHAP explainer not created. Call create_shap_explainer first.")
        
        try:
            # Use a sample for efficiency
            X_sample = X_test.sample(min(sample_size, len(X_test)))
            shap_values = self.explainer.shap_values(X_sample)
            
            logging.info(f"Computed SHAP values for {len(X_sample)} samples")
            return shap_values, X_sample
            
        except Exception as e:
            logging.error(f"Failed to compute SHAP values: {e}")
            raise CustomException(e)
    
    def get_feature_importance_summary(self, shap_values: np.ndarray, X_sample: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance summary from SHAP values
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data used for SHAP computation
            
        Returns:
            DataFrame with feature importance summary
        """
        try:
            # For multi-class classification, use absolute mean of SHAP values
            if isinstance(shap_values, list):
                # Multi-class case
                feature_importance = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
            else:
                # Binary classification or regression
                feature_importance = np.abs(shap_values).mean(axis=0)
            
            # Create summary DataFrame
            importance_df = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logging.error(f"Failed to create feature importance summary: {e}")
            raise CustomException(e)
    
    def explain_instance_lime(self, instance: pd.Series, num_features: int = 10) -> Dict:
        """
        Explain a single instance using LIME
        
        Args:
            instance: Single data instance
            num_features: Number of features to show in explanation
            
        Returns:
            Dictionary with LIME explanation
        """
        if not self.lime_explainer:
            raise ValueError("LIME explainer not created. Call create_lime_explainer first.")
        
        try:
            explanation = self.lime_explainer.explain_instance(
                instance.values,
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=num_features
            )
            
            # Extract explanation data
            lime_data = {
                'predicted_class': explanation.predict_proba.argmax() if hasattr(explanation, 'predict_proba') else None,
                'feature_contributions': explanation.as_list()
            }
            
            return lime_data
            
        except Exception as e:
            logging.error(f"Failed to create LIME explanation: {e}")
            raise CustomException(e)
    
    def plot_shap_summary(self, shap_values: np.ndarray, X_sample: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot SHAP summary plot
        
        Args:
            shap_values: SHAP values array
            X_sample: Sample data
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(12, 8))
            
            if isinstance(shap_values, list):
                # Multi-class case - plot for first class
                shap.summary_plot(shap_values[0], X_sample, show=False)
            else:
                # Binary classification or regression
                shap.summary_plot(shap_values, X_sample, show=False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"SHAP summary plot saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logging.error(f"Failed to plot SHAP summary: {e}")
            raise CustomException(e)
    
    def plot_feature_importance_bar(self, importance_df: pd.DataFrame, top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance bar chart
        
        Args:
            importance_df: Feature importance DataFrame
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Get top N features
            top_features = importance_df.head(top_n)
            
            # Create horizontal bar plot
            plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Feature importance bar plot saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logging.error(f"Failed to plot feature importance bar chart: {e}")
            raise CustomException(e)
    
    def save_importance_results(self, importance_df: pd.DataFrame, output_dir: str = "models/feature_importance"):
        """
        Save feature importance results
        
        Args:
            importance_df: Feature importance DataFrame
            output_dir: Directory to save results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            csv_path = output_path / "feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)
            
            # Save as JSON for easy loading
            json_path = output_path / "feature_importance.json"
            importance_df.to_json(json_path, orient='records', indent=2)
            
            logging.info(f"Feature importance results saved to {output_path}")
            
        except Exception as e:
            logging.error(f"Failed to save feature importance results: {e}")
            raise CustomException(e)
    
    def run_complete_analysis(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                            model_type: str = "tree", output_dir: str = "models/feature_importance"):
        """
        Run complete feature importance analysis
        
        Args:
            X_train: Training data
            X_test: Test data
            model_type: Type of model ("tree", "linear", "kernel")
            output_dir: Directory to save results
        """
        try:
            logging.info("Starting complete feature importance analysis...")
            
            # Create explainers
            self.create_shap_explainer(X_train, model_type)
            
            # Compute SHAP values
            shap_values, X_sample = self.compute_shap_values(X_test)
            
            # Get feature importance summary
            importance_df = self.get_feature_importance_summary(shap_values, X_sample)
            
            # Save results
            self.save_importance_results(importance_df, output_dir)
            
            # Create plots
            plots_dir = Path(output_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            self.plot_shap_summary(shap_values, X_sample, str(plots_dir / "shap_summary.png"))
            self.plot_feature_importance_bar(importance_df, save_path=str(plots_dir / "feature_importance_bar.png"))
            
            logging.info("Feature importance analysis completed successfully")
            return importance_df
            
        except Exception as e:
            logging.error(f"Failed to run complete feature importance analysis: {e}")
            raise CustomException(e)

# Convenience function for easy integration with training pipeline
def analyze_model_features(model_path: str, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          model_type: str = "tree", output_dir: str = "models/feature_importance"):
    """
    Analyze feature importance for a trained model
    
    Args:
        model_path: Path to the trained model
        X_train: Training data
        X_test: Test data
        model_type: Type of model ("tree", "linear", "kernel")
        output_dir: Directory to save results
        
    Returns:
        DataFrame with feature importance results
    """
    analyzer = FeatureImportanceAnalyzer(model_path)
    return analyzer.run_complete_analysis(X_train, X_test, model_type, output_dir)