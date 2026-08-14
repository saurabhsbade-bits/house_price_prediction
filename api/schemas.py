"""
Request and Response Data Models
"""

from pydantic import BaseModel, Field


class HouseFeaturesRequest(BaseModel):
    longitude: float = Field(..., example=-122.23, description="Longitude coordinate")
    latitude: float = Field(..., example=37.88, description="Latitude coordinate")
    housing_median_age: float = Field(
        ..., example=41.0, ge=0, description="Median age of house"
    )
    total_rooms: float = Field(
        ..., example=880.0, gt=0, description="Total rooms in block"
    )
    total_bedrooms: float = Field(
        ..., example=129.0, gt=0, description="Total bedrooms in block"
    )
    population: float = Field(
        ..., example=322.0, gt=0, description="Population in block"
    )
    households: float = Field(
        ..., example=126.0, gt=0, description="Households in block"
    )
    median_income: float = Field(
        ..., example=8.3252, ge=0, description="Median income (tens of thousands USD)"
    )
    ocean_proximity: str = Field(
        ..., example="NEAR BAY", description="Location category"
    )


class PredictionResponse(BaseModel):
    predicted_median_house_value: float
    status_code: int
    model_version: str = "RandomForest_v2"
