"""
Model Loading & Real-time Inference
"""

import joblib

from app.logger import logger
from utils import get_short_path


class PricePredictor:

    def __init__(self, model_path: str, scaler_path: str):
        self.model = self._load_artifact(model_path, "Model")
        self.scaler = self._load_artifact(scaler_path, "Scaler")

    def _load_artifact(self, path: str, artifact_type: str):

        logger.info(f"Loading {artifact_type} artifact from: {get_short_path(path)}")

        try:
            artifact = joblib.load(path)
            logger.info(f"{artifact_type} artifact loaded successfully.")
            return artifact
        except FileNotFoundError:
            logger.error(f"Critical error: {artifact_type} file not found at {path}")
            raise
        except Exception as e:
            logger.error(f"Failed to deserialize {artifact_type}: {str(e)}")
            raise RuntimeError(f"Artifact loading error: {str(e)}")

    def predict(self, feature_df):

        try:
            scaled_features = self.scaler.transform(feature_df)
            predictions = self.model.predict(scaled_features)
            logger.info(f"Generated {len(predictions)} predictions successfully.")

            return predictions

        except Exception as e:
            logger.error(f"Inference pipeline execution error: {str(e)}")
            raise ValueError(f"Inference error: {str(e)}")
