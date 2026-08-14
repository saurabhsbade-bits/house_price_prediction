"""
api/routers/predict.py - Inference endpoints
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_predictor
from api.schemas import HouseFeaturesRequest, PredictionResponse
from app.logger import logger
from app.model_inference import PricePredictor

router = APIRouter(tags=["Inference"])

FEATURE_ORDER = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_INLAND",
    "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
    "RoomsPerHousehold",
    "BedroomsPerRoom",
    "PopulationPerHousehold",
]


@router.post(
    "/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK
)
def predict_price(
    payload: HouseFeaturesRequest, predictor: PricePredictor = Depends(get_predictor)
):
    try:
        data = {
            "longitude": [payload.longitude],
            "latitude": [payload.latitude],
            "housing_median_age": [payload.housing_median_age],
            "total_rooms": [payload.total_rooms],
            "total_bedrooms": [payload.total_bedrooms],
            "population": [payload.population],
            "households": [payload.households],
            "median_income": [payload.median_income],
            "ocean_proximity_INLAND": [1 if payload.ocean_proximity == "INLAND" else 0],
            "ocean_proximity_ISLAND": [1 if payload.ocean_proximity == "ISLAND" else 0],
            "ocean_proximity_NEAR BAY": [
                1 if payload.ocean_proximity == "NEAR BAY" else 0
            ],
            "ocean_proximity_NEAR OCEAN": [
                1 if payload.ocean_proximity == "NEAR OCEAN" else 0
            ],
        }
        df = pd.DataFrame(data)
        df["RoomsPerHousehold"] = df["total_rooms"] / df["households"]
        df["BedroomsPerRoom"] = df["total_bedrooms"] / df["total_rooms"]
        df["PopulationPerHousehold"] = df["population"] / df["households"]

        df = df[FEATURE_ORDER]

        prediction = predictor.predict(df)[0]

        return PredictionResponse(
            predicted_median_house_value=round(float(prediction), 2), status_code=200
        )

    except Exception as exc:
        logger.error(f"Inference error handling REST request: {str(exc)}")

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference calculation failed: {str(exc)}",
        )
