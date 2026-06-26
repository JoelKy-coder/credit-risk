import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Union, List, Tuple
from src.preprocessing import engineer_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CreditRiskPredictor")

class CreditRiskPredictor:
    """
    Production-ready prediction module for Credit Risk Prediction.
    Loads trained models, pipelines, and historical feature statistics to predict new customer default risks.
    """
    def __init__(self, models_dir: str | os.PathLike[str] | None = None) -> None:
        """
        Initializes the predictor by loading the saved model, pipeline, and training statistics.

        Args:
            models_dir (str): Directory containing the serialized models and pipelines.
        """
        project_root = Path(__file__).resolve().parents[1]
        self.models_dir = Path(models_dir) if models_dir is not None else project_root / "models"
        self.model_path = self.models_dir / "best_model.pkl"
        self.pipeline_path = self.models_dir / "pipeline.pkl"
        self.stats_path = self.models_dir / "training_stats.pkl"
        
        # Validate paths
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Best model file not found at {self.model_path}")
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found at {self.pipeline_path}")
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(f"Training stats file not found at {self.stats_path}")
            
        try:
            logger.info("Loading model, pipeline, and training stats...")
            self.model = joblib.load(self.model_path)
            self.pipeline = joblib.load(self.pipeline_path)
            self.stats = joblib.load(self.stats_path)
            logger.info("All model artifacts loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {str(e)}")
            raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

    def _validate_input(self, df: pd.DataFrame) -> None:
        """
        Validates that the input DataFrame has the necessary raw columns for feature engineering.

        Args:
            df (pd.DataFrame): Input DataFrame.
        """
        required_cols = [
            'customer_id', 'lender_id', 'loan_type', 'Total_Amount', 
            'Total_Amount_to_Repay', 'disbursement_date', 'due_date', 
            'duration', 'New_versus_Repeat', 'Amount_Funded_By_Lender', 
            'Lender_portion_Funded', 'Lender_portion_to_be_repaid'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input is missing the following required columns: {missing_cols}")

    def predict(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts loan default class and probabilities for the input data.

        Args:
            input_data: A single input dictionary, a list of dictionaries, or a pandas DataFrame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted classes (0 or 1) and default probabilities.
        """
        # Convert inputs to DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a dict, a list of dicts, or a pandas DataFrame.")
            
        try:
            # Validate columns
            self._validate_input(df)
            
            # Formulate macroeconomic variables defaults if missing
            # Kenya macro values for 2023 (latest available) as default fallback
            macro_fallbacks = {
                'inflation_rate': 7.67139634029402,
                'exchange_rate': 139.846383759617,
                'real_interest_rate': 6.54651706101945,
                'average_precipitation': 630.0,
                'deposit_interest_rate': 9.16769017629068,
                'lending_interest_rate': 13.588501716128,
                'interest_rate_spread': 4.42081153983732,
                'unemployment_rate': 5.682
            }
            
            # Apply macro defaults if columns are missing
            for col, val in macro_fallbacks.items():
                if col not in df.columns:
                    df[col] = val
                    
            # Sort chronologically or parse dates
            df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
            df['due_date'] = pd.to_datetime(df['due_date'])
            
            # 1. Feature Engineering
            df_eng = engineer_features(df, stats=self.stats, is_train=False)
            
            # 2. Pipeline Transform
            X_preprocessed = self.pipeline.transform(df_eng[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
            feature_names = getattr(self.model, "feature_name_", None)
            if feature_names is not None:
                X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)
            
            # 3. Model Predict
            preds = self.model.predict(X_preprocessed)
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_preprocessed)[:, 1]
            elif hasattr(self.model, "decision_function"):
                dec_func = self.model.decision_function(X_preprocessed)
                probs = 1 / (1 + np.exp(-dec_func))
            else:
                probs = np.full(preds.shape, np.nan)
                
            return preds, probs
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise RuntimeError(f"Failed to generate prediction: {str(e)}")
