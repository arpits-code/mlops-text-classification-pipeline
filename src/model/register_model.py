import os
import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd

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

# =========================================================
# PROJECT ROOT SETUP
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

dotenv_path = PROJECT_ROOT / '.env'

if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    load_dotenv(override=True)

# =========================================================
# OPTION 1: LOCAL CONNECTION (COMMENTED)
# =========================================================

# dagshub_username = 'arpits-code'
# dagshub_token = 'YOUR_DAGSHUB_TOKEN'
# repo_owner = 'arpits-code'
# repo_name = 'mlops-text-classification-pipeline'

# =========================================================
# OPTION 2: GLOBAL .ENV CONNECTION
# =========================================================

dagshub_username = os.getenv("dagshubusername")
dagshub_token = os.getenv("dagshubtoken")

repo_owner = os.getenv(
    "DAGSHUB_REPO_OWNER",
    "arpits-code"
)

repo_name = os.getenv(
    "DAGSHUB_REPO_NAME",
    "mlops-text-classification-pipeline"
)

# =========================================================
# DAGSHUB + MLFLOW CONFIG
# =========================================================

dagshub_url = 'https://dagshub.com'

mlflow_tracking_uri = (
    f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow'
)

# =========================================================
# VALIDATE VARIABLES
# =========================================================

if not dagshub_username:
    raise ValueError(
        'DAGSHUB_USERNAME missing in .env'
    )

if not dagshub_token:
    raise ValueError(
        'DAGSHUB_TOKEN missing in .env'
    )

# =========================================================
# SET GLOBAL MLFLOW AUTH
# =========================================================

os.environ['MLFLOW_TRACKING_USERNAME'] = (
    dagshub_username
)

os.environ['MLFLOW_TRACKING_PASSWORD'] = (
    dagshub_token
)

os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '120'

# =========================================================
# INITIALIZE DAGSHUB
# =========================================================

dagshub.init(
    repo_owner=repo_owner,
    repo_name=repo_name,
    mlflow=True
)

# =========================================================
# SET TRACKING URI
# =========================================================

mlflow.set_tracking_uri(
    mlflow_tracking_uri
)

print('\nMLFLOW TRACKING URI:')
print(mlflow_tracking_uri)
print(f'REPOSITORY: {repo_owner}/{repo_name}')
print(f'USER: {dagshub_username}')