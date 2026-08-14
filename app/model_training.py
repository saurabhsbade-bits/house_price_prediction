"""
Modular Model Building, Evaluation, Persistence & Visualization
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless server execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.logger import logger
from utils import get_short_path


class ModelTrainer:
    """Handles training, evaluating, persisting regression models, and generating report figures."""

    def __init__(self, models_dir: str = "models", figures_dir: str = "figures"):
        self.models_dir = os.path.abspath(models_dir)
        self.figures_dir = os.path.abspath(figures_dir)

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        logger.info(
            f"Initialized ModelTrainer - Models Dir: {get_short_path(self.models_dir)}"
        )
        logger.info(
            f"Initialized ModelTrainer - Figures Dir: {get_short_path(self.figures_dir)}"
        )

        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, np.ndarray] = {}  # Stores evaluation predictions

    def train_linear_regression(
        self, X_train: np.ndarray, y_train: pd.Series
    ) -> LinearRegression:
        logger.info("Training Linear Regression model...")
        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            self.models["linear_regression"] = model
            logger.info("Linear Regression training completed successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to train Linear Regression model: {str(e)}")
            raise

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        n_estimators: int = 200,
        max_depth: int = 12,
        random_state: int = 42,
    ) -> RandomForestRegressor:
        logger.info("Training Random Forest Regressor model...")
        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            self.models["random_forest"] = model
            logger.info("Random Forest training completed successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to train Random Forest model: {str(e)}")
            raise

    def evaluate_model(
        self, model_name: str, X_test: np.ndarray, y_test: pd.Series
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        if model_name not in self.models:
            logger.error(f"Model '{model_name}' has not been trained yet.")
            raise ValueError(f"Model '{model_name}' is not available in trainer.")

        logger.info(f"Evaluating model performance: {model_name}")
        model = self.models[model_name]
        predictions = model.predict(X_test)

        # Save predictions inside class state for visualization methods
        self.predictions[model_name] = predictions

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        metrics_dict = {
            "model": model_name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        }

        self.metrics[model_name] = metrics_dict
        logger.info(
            f"[{model_name}] -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}"
        )

        return predictions, metrics_dict

    def save_models_and_metrics(self) -> None:
        if not self.models:
            logger.warning("No models found to dump/persist.")
            return

        logger.info(
            f"Saving trained models to directory: {get_short_path(self.models_dir)}"
        )
        for name, model in self.models.items():
            file_path = os.path.join(self.models_dir, f"{name}.joblib")
            try:
                joblib.dump(model, file_path)
                logger.info(f"Dumped model artifact: {get_short_path(file_path)}")
            except Exception as e:
                logger.error(f"Failed to save model '{name}': {str(e)}")
                raise

        if self.metrics:
            metrics_path = os.path.join(self.models_dir, "metrics.json")
            try:
                metrics_list = list(self.metrics.values())
                with open(metrics_path, "w") as f:
                    json.dump(metrics_list, f, indent=2)
                logger.info(
                    f"Saved evaluation metrics report to: {get_short_path(metrics_path)}"
                )
            except Exception as e:
                logger.error(f"Failed to write metrics report: {str(e)}")
                raise

    def generate_visualizations(
        self,
        df_cleaned: pd.DataFrame,
        feature_cols: List[str],
        y_test: pd.Series,
        rf_preds: Optional[np.ndarray] = None,
        target_col: str = "MedianHouseValue",
    ) -> None:
        """
        VISUALIZATION (evidence / screenshots for report)
        Generates and saves performance evaluation plots and data distributions.
        """

        logger.info("Generating visualization figures...")

        try:
            plt.style.use("seaborn-v0_8-whitegrid")

            # Fallback to stored evaluation predictions if not passed explicitly
            if rf_preds is None:
                rf_preds = self.predictions.get("random_forest")

            # (a) Actual vs Predicted - Random Forest
            if rf_preds is not None:
                plt.figure(figsize=(6, 6))
                plt.scatter(y_test, rf_preds, alpha=0.3, s=12, color="#2E86AB")
                plt.plot(
                    [y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    "r--",
                    lw=2,
                )
                plt.xlabel("Actual Median House Value (US$)")
                plt.ylabel("Predicted Median House Value (US$)")
                plt.title("Random Forest: Actual vs Predicted House Prices")
                plt.tight_layout()
                rf_actual_vs_pred_path = os.path.join(
                    self.figures_dir, "actual_vs_predicted_rf.png"
                )
                plt.savefig(rf_actual_vs_pred_path, dpi=150)
                plt.close()
                logger.info(f"Saved plot: {get_short_path(rf_actual_vs_pred_path)}")

            # (b) Feature importance
            if "random_forest" in self.models:
                rf_model = self.models["random_forest"]
                importances = pd.Series(
                    rf_model.feature_importances_, index=feature_cols
                )
                importances = importances.sort_values(ascending=True)

                csv_importance_path = os.path.join(
                    self.figures_dir, "feature_importance.csv"
                )
                importances.to_csv(csv_importance_path)

                plt.figure(figsize=(7, 5))
                importances.plot(kind="barh", color="#2E86AB")
                plt.title("Random Forest Feature Importance")
                plt.xlabel("Importance")
                plt.tight_layout()
                img_importance_path = os.path.join(
                    self.figures_dir, "feature_importance.png"
                )
                plt.savefig(img_importance_path, dpi=150)
                plt.close()
                logger.info(
                    f"Saved feature importances: {get_short_path(img_importance_path)}"
                )

            # (c) Model comparison bar chart
            if self.metrics:
                metrics_df = pd.DataFrame(list(self.metrics.values())).set_index(
                    "model"
                )
                plt.figure(figsize=(6, 4))
                metrics_df[["MAE", "RMSE"]].plot(
                    kind="bar", color=["#2E86AB", "#F26419"]
                )
                plt.title("Model Comparison: MAE & RMSE (lower is better)")
                plt.ylabel("Error (US$)")
                plt.xticks(rotation=0)
                plt.tight_layout()
                comp_path = os.path.join(self.figures_dir, "model_comparison.png")
                plt.savefig(comp_path, dpi=150)
                plt.close()
                logger.info(
                    f"Saved model comparison chart: {get_short_path(comp_path)}"
                )

            # (d) Target distribution after cleaning
            if target_col in df_cleaned.columns:
                plt.figure(figsize=(6, 4))
                df_cleaned[target_col].hist(bins=40, color="#2E86AB")
                plt.title("Distribution of Median House Value (post-cleaning)")
                plt.xlabel("Median House Value (US$)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                dist_path = os.path.join(self.figures_dir, "target_distribution.png")
                plt.savefig(dist_path, dpi=150)
                plt.close()
                logger.info(
                    f"Saved target distribution chart: {get_short_path(dist_path)}"
                )

            logger.info(
                f"All figures successfully saved to: {get_short_path(self.figures_dir)}"
            )

        except Exception as e:
            logger.error(f"Failed to generate visualization figures: {str(e)}")
            raise
