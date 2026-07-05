from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)

import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import numpy as np

from dotenv import load_dotenv

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

# =========================================================
# TEXT PREPROCESSING FUNCTIONS
# =========================================================

def lemmatization(text):

    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [
        lemmatizer.lemmatize(word)
        for word in text
    ]

    return " ".join(text)


def remove_stop_words(text):

    stop_words = set(stopwords.words("english"))

    text = [
        word
        for word in str(text).split()
        if word not in stop_words
    ]

    return " ".join(text)


def removing_numbers(text):

    text = ''.join(
        [
            char
            for char in text
            if not char.isdigit()
        ]
    )

    return text


def lower_case(text):

    text = text.split()

    text = [
        word.lower()
        for word in text
    ]

    return " ".join(text)


def removing_punctuations(text):

    text = re.sub(
        '[%s]' % re.escape(string.punctuation),
        ' ',
        text
    )

    text = text.replace('؛', "")

    text = re.sub(
        '\s+',
        ' ',
        text
    ).strip()

    return text


def removing_urls(text):

    url_pattern = re.compile(
        r'https?://\S+|www\.\S+'
    )

    return url_pattern.sub(r'', text)


def normalize_text(text):

    text = lower_case(text)

    text = remove_stop_words(text)

    text = removing_numbers(text)

    text = removing_punctuations(text)

    text = removing_urls(text)

    text = lemmatization(text)

    return text

mlflow.set_tracking_uri("https://dagshub.com/arpits-code/mlops-text-classification-pipeline.mlflow")

dagshub.init(
    repo_owner="arpits-code",
    repo_name="mlops-text-classification-pipeline",
    mlflow=True
)
# =========================================================
# DAGSHUB + MLFLOW CONFIGURATION
# =========================================================

# dagshubtoken = os.getenv("dagshubtoken")

# if not dagshubtoken:

#     raise EnvironmentError(
#         "dagshubtoken environment variable is not set"
#     )

# os.environ["MLFLOW_TRACKING_USERNAME"] = "arpits-code"

# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshubtoken

# os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "120"

# dagshub_url = "https://dagshub.com"

# repo_owner = "arpits-code"

# repo_name = "mlops-text-classification-pipeline"

# mlflow.set_tracking_uri(
#     f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
# )

# =========================================================
# CONNECT DAGSHUB
# =========================================================

try:

    dagshub.init(
        repo_owner=repo_owner,
        repo_name=repo_name,
        mlflow=True
    )

    print("DagsHub connected successfully!")

except Exception as e:

    print("DagsHub connection failed")
    print(e)

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# PROMETHEUS METRICS
# =========================================================

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction counts",
    ["prediction"],
    registry=registry
)

# =========================================================
# MODEL LOADING
# =========================================================

model_name = "my_model"

try:

    client = mlflow.MlflowClient()

    versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if len(versions) == 0:

        raise Exception(
            f"No model versions found for {model_name}"
        )

    latest_version = max(
        versions,
        key=lambda v: int(v.version)
    )

    model_version = latest_version.version

    model_uri = f"models:/{model_name}/{model_version}"

    print(f"Loading model from: {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    print("MLflow model loaded successfully!")

except Exception as e:

    print("MLflow loading failed")
    print(e)

    print("Loading local model.pkl")

    model = pickle.load(
        open("models/model.pkl", "rb")
    )

    print("Local model loaded successfully!")

# =========================================================
# LOAD VECTORIZER
# =========================================================

vectorizer = pickle.load(
    open("models/vectorizer.pkl", "rb")
)

print("Vectorizer loaded successfully!")

# =========================================================
# ROUTES
# =========================================================

@app.route("/")
def home():

    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/"
    ).inc()

    start_time = time.time()

    response = render_template(
        "index.html",
        result=None
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(time.time() - start_time)

    return response


@app.route("/predict", methods=["POST"])
def predict():

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict"
    ).inc()

    start_time = time.time()

    text = request.form["text"]

    # PREPROCESS
    text = normalize_text(text)

    # VECTORIZE
    features = vectorizer.transform([text])

    features_df = pd.DataFrame(
        features.toarray(),
        columns=[
            str(i)
            for i in range(features.shape[1])
        ]
    )

    # PREDICT
    result = model.predict(features_df)

    prediction = result[0]

    # METRICS
    PREDICTION_COUNT.labels(
        prediction=str(prediction)
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint="/predict"
    ).observe(time.time() - start_time)

    return render_template(
        "index.html",
        result=prediction
    )

# =========================================================
# PROMETHEUS METRICS ROUTE
# =========================================================

@app.route("/metrics")
def metrics():

    return (
        generate_latest(registry),
        200,
        {
            "Content-Type": CONTENT_TYPE_LATEST
        }
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )