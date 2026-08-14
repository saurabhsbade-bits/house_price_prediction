"""Data quality checks and metrics collection."""

import json
from typing import Any, Dict, List

import pandas as pd


class DataQuality:
    """Compute data-quality metrics and perform simple schema validation."""

    # Define expected schema for the housing dataset
    EXPECTED_COLUMNS: List[str] = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def validate_schema(self) -> Dict[str, Any]:
        """Check presence of expected columns and basic type checks.

        Returns a dict with keys: passed (bool), missing_columns (list), extra_columns (list).
        """
        cols = list(self.df.columns)
        missing = [c for c in self.EXPECTED_COLUMNS if c not in cols]
        extra = [
            c
            for c in cols
            if c not in self.EXPECTED_COLUMNS
            and c != "median_house_value"
            and c != "MedianHouseValue"
        ]

        passed = len(missing) == 0

        return {"passed": passed, "missing_columns": missing, "extra_columns": extra}

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute simple data quality metrics: missing counts, percent missing, unique counts, basic stats."""
        metrics: Dict[str, Any] = {}

        missing = self.df.isnull().sum().to_dict()
        percent_missing = (self.df.isnull().mean() * 100).round(2).to_dict()
        unique_counts = self.df.nunique(dropna=True).to_dict()

        # Descriptive stats for numeric columns
        numeric_stats = self.df.select_dtypes(include=["number"]).describe().to_dict()

        metrics["missing_counts"] = missing
        metrics["percent_missing"] = percent_missing
        metrics["unique_counts"] = unique_counts
        metrics["numeric_stats"] = numeric_stats

        return metrics

    def save_report(self, path: str) -> None:
        """Save schema validation and metrics into a JSON file at `path`."""
        report = {
            "schema": self.validate_schema(),
            "metrics": self.compute_metrics(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                report,
                f,
                indent=2,
                default=lambda o: list(o) if hasattr(o, "__iter__") else o,
            )
