"""
Data Preparation & Feature Engineering
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.logger import logger
from utils import get_short_path


class DataPreparation:

    def __init__(self, models_dir: str = "models"):
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.models_dir = os.path.abspath(models_dir)
        os.makedirs(self.models_dir, exist_ok=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning phase...")
        df = df.drop_duplicates()

        # Impute missing total_bedrooms with median (only column with nulls)
        if "total_bedrooms" in df.columns:
            median_val = df["total_bedrooms"].median()
            df["total_bedrooms"] = df["total_bedrooms"].fillna(median_val)
            logger.info(f"Imputed missing total_bedrooms with median: {median_val}")

        # Cap extreme outliers in target (winsorize at 1st/99th percentile)
        if "MedianHouseValue" in df.columns:
            low, high = df["MedianHouseValue"].quantile([0.01, 0.99])
            df["MedianHouseValue"] = df["MedianHouseValue"].clip(low, high)
            logger.info(
                f"Clipped MedianHouseValue outliers between [{low:.2f}, {high:.2f}]"
            )

        # One-hot encoding the categorical feature
        if "ocean_proximity" in df.columns:
            df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
            logger.info("One-hot encoded 'ocean_proximity' column.")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Executing feature engineering...")
        df["RoomsPerHousehold"] = df["total_rooms"] / df["households"].replace(
            0, np.nan
        )
        df["BedroomsPerRoom"] = df["total_bedrooms"] / df["total_rooms"].replace(
            0, np.nan
        )
        df["PopulationPerHousehold"] = df["population"] / df["households"].replace(
            0, np.nan
        )
        df = df.fillna(df.median(numeric_only=True))
        return df

    def fit_transform_scaler(self, X_train: pd.DataFrame) -> np.ndarray:
        logger.info("Fitting and applying StandardScaler on training dataset...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.feature_cols = X_train.columns.tolist()

        # Save scaler immediately after fitting
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        try:
            joblib.dump(self.scaler, scaler_path)
            logger.info(
                f"Scaler artifact successfully dumped to: {get_short_path(scaler_path)}"
            )
        except Exception as e:
            logger.error(f"Failed to dump scaler artifact: {str(e)}")
            raise

        return X_train_scaled

    def transform_scaler(self, X_test: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X_test)
