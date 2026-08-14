"""
API Configuration and Settings
"""

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Metadata
    PROJECT_NAME: str = "House Price Prediction API"
    DESCRIPTION: str = "Production REST service for estimating median house value."
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # Server Host & Port Configuration
    HOST: str = (
        "127.0.0.1"  # Default to localhost for local testing (use 0.0.0.0 for Docker)
    )
    PORT: int = 8000

    # File Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH: str = os.path.join(BASE_DIR, "models", "random_forest.joblib")
    SCALER_PATH: str = os.path.join(BASE_DIR, "models", "scaler.joblib")


settings = Settings()
