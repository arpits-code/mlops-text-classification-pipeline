# data ingestion

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# =========================================================
# PANDAS CONFIG
# =========================================================

pd.set_option('future.no_silent_downcasting', True)

# =========================================================
# PROJECT ROOT SETUP
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# IMPORT S3 OPERATIONS
# =========================================================

from src.connections import s3_operations

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

def load_env():
    """Load environment variables from .env if present."""

    dotenv_path = (
        Path(__file__).resolve().parents[2] / '.env'
    )

    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        load_dotenv(override=True)

# =========================================================
# LOAD PARAMS
# =========================================================

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)

        logging.debug(
            'Parameters retrieved from %s',
            params_path
        )

        return params

    except FileNotFoundError:
        logging.error(
            'File not found: %s',
            params_path
        )
        raise

    except yaml.YAMLError as e:
        logging.error(
            'YAML error: %s',
            e
        )
        raise

    except Exception as e:
        logging.error(
            'Unexpected error: %s',
            e
        )
        raise

# =========================================================
# LOAD DATA
# =========================================================

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""

    try:
        df = pd.read_csv(data_url)

        logging.info(
            'Data loaded from %s',
            data_url
        )

        return df

    except pd.errors.ParserError as e:
        logging.error(
            'Failed to parse the CSV file: %s',
            e
        )
        raise

    except Exception as e:
        logging.error(
            'Unexpected error occurred while loading the data: %s',
            e
        )
        raise

# =========================================================
# PREPROCESS DATA
# =========================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""

    try:
        logging.info('Pre-processing data...')

        final_df = df[
            df['sentiment'].isin([
                'positive',
                'negative'
            ])
        ].copy()

        final_df['sentiment'] = final_df[
            'sentiment'
        ].replace({
            'positive': 1,
            'negative': 0
        })

        logging.info(
            'Data preprocessing completed'
        )

        return final_df

    except KeyError as e:
        logging.error(
            'Missing column in the dataframe: %s',
            e
        )
        raise

    except Exception as e:
        logging.error(
            'Unexpected error during preprocessing: %s',
            e
        )
        raise

# =========================================================
# SAVE DATA
# =========================================================

def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str
) -> None:
    """Save the train and test datasets."""

    try:
        raw_data_path = os.path.join(
            data_path,
            'raw'
        )

        os.makedirs(
            raw_data_path,
            exist_ok=True
        )

        train_data.to_csv(
            os.path.join(raw_data_path, 'train.csv'),
            index=False
        )

        test_data.to_csv(
            os.path.join(raw_data_path, 'test.csv'),
            index=False
        )

        logging.info(
            'Train and test data saved to %s',
            raw_data_path
        )

    except Exception as e:
        logging.error(
            'Unexpected error occurred while saving the data: %s',
            e
        )
        raise

# =========================================================
# MAIN
# =========================================================

def main():

    try:
        # -----------------------------------------------------
        # LOAD ENV
        # -----------------------------------------------------

        load_env()

        # -----------------------------------------------------
        # LOAD PARAMS
        # -----------------------------------------------------

        params = load_params(
            params_path='params.yaml'
        )

        test_size = (
            params['data_ingestion']['test_size']
            if 'data_ingestion' in params
            else 0.2
        )

        # -----------------------------------------------------
        # AWS CREDENTIALS
        # Works for BOTH:
        # 1. Local .env
        # 2. GitHub Actions Secrets
        # -----------------------------------------------------

        access_key = (
            os.getenv('Accesskey')
            or os.getenv('AWS_ACCESS_KEY_ID')
        )

        secret_key = (
            os.getenv('Secretkey')
            or os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        if not access_key or not secret_key:
            raise ValueError(
                'AWS access key and secret key must be set '
                'in environment variables or the .env file.'
            )

        # Make credentials available globally
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key

        # -----------------------------------------------------
        # LOAD DATA FROM S3
        # -----------------------------------------------------

        s3 = s3_operations(
            'text-classification-982005835553-eu-north-1-an',
            access_key,
            secret_key
        )

        df = s3.fetch_file_from_s3(
            'IMDB.csv'
        )

        # -----------------------------------------------------
        # PREPROCESS DATA
        # -----------------------------------------------------

        final_df = preprocess_data(df)

        # -----------------------------------------------------
        # SPLIT DATA
        # -----------------------------------------------------

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        # -----------------------------------------------------
        # SAVE DATA
        # -----------------------------------------------------

        save_data(
            train_data,
            test_data,
            data_path='./data'
        )

        logging.info(
            'Data ingestion completed successfully'
        )

    except Exception as e:
        logging.error(
            'Failed to complete the data ingestion process: %s',
            e
        )

        print(f'Error: {e}')

        raise

# =========================================================
# RUN
# =========================================================

if __name__ == '__main__':
    main()