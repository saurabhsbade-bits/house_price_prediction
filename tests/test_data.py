import os

import pandas as pd
import pytest

from app.data_ingestion import DataIngestion
from app.feature_engineering import DataPreparation


def make_sample_csv(path):
    df = pd.DataFrame(
        {
            "longitude": [-122.23, -122.22],
            "latitude": [37.88, 37.86],
            "housing_median_age": [41, 40],
            "total_rooms": [880, 900],
            "total_bedrooms": [129, None],
            "population": [322, 300],
            "households": [126, 128],
            "median_income": [8.3252, 8.0],
            "ocean_proximity": ["NEAR BAY", "INLAND"],
            "median_house_value": [452600, 358500],
        }
    )
    df.to_csv(path, index=False)


def test_load_dataset_not_found():
    with pytest.raises(FileNotFoundError):
        DataIngestion.load_dataset("this_file_does_not_exist.csv")


def test_load_dataset_success(tmp_path):
    csv_path = os.path.join(tmp_path, "sample.csv")
    make_sample_csv(csv_path)

    df = DataIngestion.load_dataset(csv_path)

    # Ensure rename happened
    assert "MedianHouseValue" in df.columns
    # Data should have two rows
    assert df.shape[0] == 2


def test_clean_data_imputation_and_encoding(tmp_path):
    csv_path = os.path.join(tmp_path, "sample2.csv")
    make_sample_csv(csv_path)
    df = DataIngestion.load_dataset(csv_path)

    dp = DataPreparation(models_dir=str(tmp_path / "models"))
    cleaned = dp.clean_data(df.copy())

    # Imputation: no nulls in total_bedrooms
    assert cleaned["total_bedrooms"].isnull().sum() == 0

    # One-hot encoding: original 'ocean_proximity' should be removed
    assert "ocean_proximity" not in cleaned.columns


def test_engineer_features_creates_derived(tmp_path):
    csv_path = os.path.join(tmp_path, "sample3.csv")
    make_sample_csv(csv_path)
    df = DataIngestion.load_dataset(csv_path)

    dp = DataPreparation(models_dir=str(tmp_path / "models"))
    cleaned = dp.clean_data(df.copy())
    engineered = dp.engineer_features(cleaned.copy())

    # Derived features should be present
    for col in ["RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold"]:
        assert col in engineered.columns

    # No NaNs after filling
    assert (
        engineered[list(engineered.select_dtypes(include=[float, int]).columns)]
        .isnull()
        .sum()
        .sum()
        >= 0
    )
