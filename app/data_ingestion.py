"""
Data Ingestion
"""

import os

import pandas as pd

from app.logger import logger
from utils import get_short_path


class DataIngestion:
    """
    Handles loading and initial processing of housing dataset.

    This class provides methods for ingesting and preparing dataset files
    for model training and prediction.
    """

    @staticmethod
    def load_dataset(file_path: str) -> pd.DataFrame:
        """
        Load and validate a housing dataset from a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file containing the housing dataset.

        Returns
        -------
        pd.DataFrame
            Loaded dataset with 'median_house_value' column renamed to 'MedianHouseValue'.

        Raises
        ------
        FileNotFoundError
            If the file does not exist at the specified path.
        ValueError
            If the loaded dataset is empty.
        Exception
            If CSV parsing fails for any other reason.

        Notes
        -----
        - Logs the dataset shape upon successful loading
        - Renames the target column for consistency
        - Validates that the dataset is not empty before returning
        """

        logger.info(f"Attempting to load dataset from: {get_short_path(file_path)}")

        if not os.path.exists(file_path):
            logger.error(f"Dataset file missing at path: {file_path}")
            raise FileNotFoundError(f"Missing required input file: {file_path}")

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                logger.warning(f"Loaded dataset from {file_path} is empty!")
                raise ValueError("Dataset is empty.")

            logger.info(f"Successfully loaded dataset with shape: {df.shape}")

            return df.rename(columns={"median_house_value": "MedianHouseValue"})

        except Exception as e:
            logger.error(f"Failed to parse CSV file: {str(e)}")
            raise
