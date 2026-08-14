"""
Health check endpoints
"""

from fastapi import APIRouter, Depends, status

from api.config import settings
from api.dependencies import get_predictor
from app.model_inference import PricePredictor

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check(predictor: PricePredictor = Depends(get_predictor)):
    return {"status": "healthy", "service": settings.PROJECT_NAME}
