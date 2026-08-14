import json
import os

import pandas as pd

from app.data_ingestion import DataIngestion
from app.data_quality import DataQuality


def make_sample_df():
    return pd.DataFrame(
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


def test_validate_schema_passes():
    df = make_sample_df()
    dq = DataQuality(df)
    result = dq.validate_schema()
    assert result["passed"] is True
    assert result["missing_columns"] == []


def test_validate_schema_missing_column():
    df = make_sample_df().drop(columns=["median_income"])
    dq = DataQuality(df)
    result = dq.validate_schema()
    assert result["passed"] is False
    assert "median_income" in result["missing_columns"]


def test_compute_metrics_and_save(tmp_path):
    df = make_sample_df()
    dq = DataQuality(df)
    metrics = dq.compute_metrics()
    assert "missing_counts" in metrics
    assert "percent_missing" in metrics
    # Save report
    out_path = os.path.join(tmp_path, "data_quality.json")
    dq.save_report(out_path)
    assert os.path.exists(out_path)
    with open(out_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    assert "schema" in j and "metrics" in j
