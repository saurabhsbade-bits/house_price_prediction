import os

import joblib
import numpy as np
import pandas as pd

from app.feature_engineering import DataPreparation
from app.model_inference import PricePredictor
from app.model_training import ModelTrainer


def make_tiny_dataset():
    return pd.DataFrame(
        {
            "longitude": [-122.23, -122.22, -122.24, -122.25],
            "latitude": [37.88, 37.86, 37.85, 37.87],
            "housing_median_age": [41, 40, 39, 38],
            "total_rooms": [880, 900, 850, 870],
            "total_bedrooms": [129, 130, 120, 125],
            "population": [322, 300, 310, 305],
            "households": [126, 128, 125, 127],
            "median_income": [8.3252, 8.0, 7.8, 8.1],
            "ocean_proximity": ["NEAR BAY", "NEAR BAY", "INLAND", "NEAR OCEAN"],
            "MedianHouseValue": [452600, 358500, 352100, 412700],
        }
    )


def test_overfit_on_small_batch(tmp_path):
    """Train a model on a very small dataset and expect near-perfect fit (overfitting).

    This verifies the training loop works and a model can memorize a tiny batch.
    """
    df = make_tiny_dataset()

    # Prepare directories inside tmp_path
    models_dir = str(tmp_path / "models")
    figures_dir = str(tmp_path / "figures")

    # Data preparation
    dp = DataPreparation(models_dir=models_dir)
    df_clean = dp.clean_data(df.copy())
    df_feat = dp.engineer_features(df_clean.copy())

    target_col = "MedianHouseValue"
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    # Fit scaler and transform
    X_scaled = dp.fit_transform_scaler(X)

    # Train a small random forest with controlled params (fast during tests)
    trainer = ModelTrainer(models_dir=models_dir, figures_dir=figures_dir)
    trainer.train_random_forest(
        X_scaled, y, n_estimators=20, max_depth=8, random_state=42
    )

    # Evaluate on the same tiny training set (expect high R2)
    _, metrics = trainer.evaluate_model("random_forest", X_scaled, y)

    # R2 is stored as a rounded float value; expect a strong fit on tiny dataset
    assert metrics["R2"] >= 0.6


def test_inference_output_shape_and_range(tmp_path):
    """Train a model, save artifacts, and validate the PricePredictor outputs."""
    df = make_tiny_dataset()

    models_dir = str(tmp_path / "models")

    dp = DataPreparation(models_dir=models_dir)
    df_clean = dp.clean_data(df.copy())
    df_feat = dp.engineer_features(df_clean.copy())

    target_col = "MedianHouseValue"
    X = df_feat.drop(columns=[target_col])
    y = df_feat[target_col]

    X_scaled = dp.fit_transform_scaler(X)

    trainer = ModelTrainer(models_dir=models_dir)
    rf = trainer.train_random_forest(
        X_scaled, y, n_estimators=20, max_depth=8, random_state=42
    )

    # Save artifacts to explicit paths used by PricePredictor
    model_path = os.path.join(models_dir, "test_rf.joblib")
    scaler_path = os.path.join(models_dir, "test_scaler.joblib")
    joblib.dump(rf, model_path)
    joblib.dump(dp.scaler, scaler_path)

    # Load predictor and run a single-row inference
    predictor = PricePredictor(model_path=model_path, scaler_path=scaler_path)

    single_row = X.iloc[[0]]
    preds = predictor.predict(single_row)

    assert len(preds) == 1
    assert preds[0] >= 0
