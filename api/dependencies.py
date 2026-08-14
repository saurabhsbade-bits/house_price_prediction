"""
Shared API Dependencies
"""

from fastapi import HTTPException, status

from api.config import settings
from app.logger import logger
from app.model_inference import PricePredictor

_predictor_instance = None


def get_predictor() -> PricePredictor:
    """Singleton getter for the PricePredictor service."""

    global _predictor_instance

    if _predictor_instance is None:
        try:
            _predictor_instance = PricePredictor(
                model_path=settings.MODEL_PATH, scaler_path=settings.SCALER_PATH
            )
        except Exception as err:
            logger.critical(f"Failed to initialize PricePredictor dependency: {err}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model inference service is currently unavailable.",
            )
    return _predictor_instance
