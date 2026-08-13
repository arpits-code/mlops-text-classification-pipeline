import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

import logging
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

def load_env():
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"

    if dotenv_path.exists():
        load_dotenv(dotenv_path)
    else:
        load_dotenv()


# ============================================================
# GLOBAL DAGSHUB / MLFLOW CONNECTION
# ============================================================

load_env()


DAGSHUB_USERNAME = (
    os.getenv("DAGSHUB_USERNAME")
    or os.getenv("dagshubusername")
)


DAGSHUB_TOKEN = (
    os.getenv("DAGSHUB_TOKEN")
    or os.getenv("dagshubtoken")
)


DAGSHUB_REPO_OWNER = os.getenv(
    "DAGSHUB_REPO_OWNER",
    "arpits-code"
)


DAGSHUB_REPO_NAME = os.getenv(
    "DAGSHUB_REPO_NAME",
    "mlops-text-classification-pipeline"
)


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"https://dagshub.com/"
    f"{DAGSHUB_REPO_OWNER}/"
    f"{DAGSHUB_REPO_NAME}.mlflow"
)


# ============================================================
# VALIDATE DAGSHUB CREDENTIALS
# ============================================================

if not DAGSHUB_USERNAME:
    raise ValueError(
        "DAGSHUB_USERNAME or dagshubusername is not set"
    )


if not DAGSHUB_TOKEN:
    raise ValueError(
        "DAGSHUB_TOKEN or dagshubtoken is not set"
    )


# ============================================================
# GLOBAL / CI-CD MLFLOW CONNECTION
# ============================================================

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME

os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


logging.info(
    "MLflow tracking URI configured: %s",
    MLFLOW_TRACKING_URI
)


logging.info(
    "DagsHub authentication configured"
)


# ============================================================
# LOCAL CONNECTION OPTION
# ============================================================

# For local development, you can use the following
# DagsHub connection instead of the CI/CD connection above.
#
# Make sure your .env contains:
#
# dagshubusername=your_username
# dagshubtoken=your_token
#
# ------------------------------------------------------------
#
# import dagshub
#
# mlflow.set_tracking_uri(
#     "https://dagshub.com/"
#     "arpits-code/"
#     "mlops-text-classification-pipeline.mlflow"
# )
#
# dagshub.init(
#     repo_owner="arpits-code",
#     repo_name="mlops-text-classification-pipeline",
#     mlflow=True
# )
#
# ============================================================


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(file_path: str):
    """Load the trained model from a file."""

    try:

        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logging.info(
            "Model loaded from %s",
            file_path
        )

        return model

    except FileNotFoundError:

        logging.error(
            "File not found: %s",
            file_path
        )

        raise

    except Exception as e:

        logging.error(
            "Unexpected error occurred while loading "
            "the model: %s",
            e
        )

        raise


# ============================================================
# LOAD DATA
# ============================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    try:

        df = pd.read_csv(file_path)

        logging.info(
            "Data loaded from %s",
            file_path
        )

        return df

    except pd.errors.ParserError as e:

        logging.error(
            "Failed to parse the CSV file: %s",
            e
        )

        raise

    except Exception as e:

        logging.error(
            "Unexpected error occurred while loading "
            "the data: %s",
            e
        )

        raise


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """Evaluate the model and return evaluation metrics."""

    try:

        y_pred = clf.predict(X_test)

        y_pred_proba = clf.predict_proba(
            X_test
        )[:, 1]


        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        precision = precision_score(
            y_test,
            y_pred
        )


        recall = recall_score(
            y_test,
            y_pred
        )


        auc = roc_auc_score(
            y_test,
            y_pred_proba
        )


        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }


        logging.info(
            "Model evaluation metrics calculated"
        )


        return metrics_dict

    except Exception as e:

        logging.error(
            "Error during model evaluation: %s",
            e
        )

        raise


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: dict,
    file_path: str
) -> None:
    """Save evaluation metrics to a JSON file."""

    try:

        with open(file_path, "w") as file:

            json.dump(
                metrics,
                file,
                indent=4
            )


        logging.info(
            "Metrics saved to %s",
            file_path
        )


    except Exception as e:

        logging.error(
            "Error occurred while saving the metrics: %s",
            e
        )

        raise


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

def save_model_info(
    run_id: str,
    model_path: str,
    file_path: str
) -> None:
    """Save model run ID and path to JSON."""

    try:

        model_info = {
            "run_id": run_id,
            "model_path": model_path
        }


        with open(file_path, "w") as file:

            json.dump(
                model_info,
                file,
                indent=4
            )


        logging.debug(
            "Model info saved to %s",
            file_path
        )


    except Exception as e:

        logging.error(
            "Error occurred while saving the model info: %s",
            e
        )

        raise


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CREATE / SELECT MLFLOW EXPERIMENT
    # --------------------------------------------------------

    mlflow.set_experiment(
        "my-dvc-pipeline"
    )


    # --------------------------------------------------------
    # START MLFLOW RUN
    # --------------------------------------------------------

    with mlflow.start_run() as run:

        try:

            # ------------------------------------------------
            # LOAD MODEL
            # ------------------------------------------------

            clf = load_model(
                "./models/model.pkl"
            )


            # ------------------------------------------------
            # LOAD TEST DATA
            # ------------------------------------------------

            test_data = load_data(
                "./data/processed/test_bow.csv"
            )


            # ------------------------------------------------
            # SPLIT FEATURES AND TARGET
            # ------------------------------------------------

            X_test = test_data.iloc[:, :-1].values

            y_test = test_data.iloc[:, -1].values


            # ------------------------------------------------
            # EVALUATE MODEL
            # ------------------------------------------------

            metrics = evaluate_model(
                clf,
                X_test,
                y_test
            )


            # ------------------------------------------------
            # SAVE METRICS
            # ------------------------------------------------

            save_metrics(
                metrics,
                "reports/metrics.json"
            )


            # ------------------------------------------------
            # LOG METRICS TO MLFLOW
            # ------------------------------------------------

            for metric_name, metric_value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value
                )


            # ------------------------------------------------
            # LOG MODEL PARAMETERS
            # ------------------------------------------------

            if hasattr(clf, "get_params"):

                params = clf.get_params()


                for param_name, param_value in params.items():

                    mlflow.log_param(
                        param_name,
                        param_value
                    )


            # ------------------------------------------------
            # LOG MODEL TO MLFLOW
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                clf,
                "model"
            )


            # ------------------------------------------------
            # SAVE MODEL INFORMATION
            # ------------------------------------------------

            save_model_info(
                run.info.run_id,
                "model",
                "reports/experiment_info.json"
            )


            # ------------------------------------------------
            # LOG METRICS FILE
            # ------------------------------------------------

            mlflow.log_artifact(
                "reports/metrics.json"
            )


            logging.info(
                "Model evaluation completed successfully"
            )


        except Exception as e:

            logging.error(
                "Failed to complete the model evaluation "
                "process: %s",
                e
            )

            print(
                f"Error: {e}"
            )

            raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()