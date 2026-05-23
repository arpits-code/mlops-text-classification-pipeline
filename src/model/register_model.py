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

import mlflow
import mlflow.sklearn
import dagshub

from dotenv import load_dotenv

import logging


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        '..'
    )
)

import sys

if PROJECT_ROOT not in sys.path:

    sys.path.insert(0, PROJECT_ROOT)


def load_env():

    dotenv_path = (
        Path(__file__).resolve().parents[2] / '.env'
    )

    if dotenv_path.exists():

        load_dotenv(
            dotenv_path=dotenv_path,
            override=True
        )

    else:

        load_dotenv(
            override=True
        )


load_env()

dagshub_token = os.getenv(
    "dagshubtoken"
)

dagshub_username = os.getenv(
    "dagshubusername"
)

dagshub_url = "https://dagshub.com"

repo_owner = os.getenv(
    "DAGSHUB_REPO_OWNER",
    "arpits-code"
)

repo_name = os.getenv(
    "DAGSHUB_REPO_NAME",
    "mlops-text-classification-pipeline"
)

mlflow_tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI"
) or f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"


if not dagshub_username:

    raise ValueError(
        "dagshubusername missing in .env"
    )

if not dagshub_token:

    raise ValueError(
        "dagshubtoken missing in .env"
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = (
    str(dagshub_username)
)

os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    str(dagshub_token)
)

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True
)

mlflow.set_tracking_uri(
    mlflow_tracking_uri
)


def load_model(
    file_path: str
):

    try:

        with open(file_path, 'rb') as file:

            model = pickle.load(file)

        logging.info(
            'Model loaded from %s',
            file_path
        )

        return model

    except Exception as e:

        logging.error(
            'Error loading model: %s',
            e
        )

        raise


def load_data(
    file_path: str
) -> pd.DataFrame:

    try:

        df = pd.read_csv(file_path)

        logging.info(
            'Data loaded from %s',
            file_path
        )

        return df

    except Exception as e:

        logging.error(
            'Error loading data: %s',
            e
        )

        raise


def evaluate_model(
    clf,
    X_test,
    y_test
):

    try:

        y_pred = clf.predict(X_test)

        y_pred_proba = clf.predict_proba(
            X_test
        )[:, 1]

        metrics = {

            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred
            ),

            "recall": recall_score(
                y_test,
                y_pred
            ),

            "auc": roc_auc_score(
                y_test,
                y_pred_proba
            )
        }

        logging.info(
            'Evaluation metrics calculated'
        )

        return metrics

    except Exception as e:

        logging.error(
            'Error during evaluation: %s',
            e
        )

        raise


def save_metrics(
    metrics,
    file_path
):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    with open(file_path, 'w') as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    logging.info(
        'Metrics saved to %s',
        file_path
    )


def save_model_info(
    run_id,
    model_path,
    file_path
):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    model_info = {

        "run_id": run_id,
        "model_path": model_path
    }

    with open(file_path, 'w') as file:

        json.dump(
            model_info,
            file,
            indent=4
        )

    logging.info(
        'Experiment info saved'
    )


def main():

    try:

        experiment_name = os.getenv(
            "EXPERIMENT_NAME"
        )

        mlflow.set_experiment(
            experiment_name
        )

        with mlflow.start_run() as run:

            clf = load_model(
                './models/model.pkl'
            )

            test_data = load_data(
                './data/processed/test_bow.csv'
            )

            X_test = (
                test_data.iloc[:, :-1].values
            )

            y_test = (
                test_data.iloc[:, -1].values
            )

            metrics = evaluate_model(
                clf,
                X_test,
                y_test
            )

            save_metrics(
                metrics,
                'reports/metrics.json'
            )

            for (
                metric_name,
                metric_value
            ) in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value
                )

            if hasattr(clf, 'get_params'):

                params = clf.get_params()

                for (
                    param_name,
                    param_value
                ) in params.items():

                    mlflow.log_param(
                        param_name,
                        param_value
                    )

            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model"
            )

            print(
                "\nMODEL LOGGED SUCCESSFULLY\n"
            )

            print(
                f"RUN ID: {run.info.run_id}"
            )

            save_model_info(
                run.info.run_id,
                "model",
                'reports/experiment_info.json'
            )

            mlflow.log_artifact(
                'reports/metrics.json'
            )

            mlflow.log_artifact(
                'reports/experiment_info.json'
            )

            logging.info(
                'Model evaluation completed successfully'
            )

    except Exception as e:

        logging.error(
            'Failed model evaluation: %s',
            e
        )

        raise


if __name__ == '__main__':

    main()