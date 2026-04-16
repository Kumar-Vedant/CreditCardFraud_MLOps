from fastapi import FastAPI
import numpy as np
from contextlib import asynccontextmanager
from threading import Thread
import time

from kafka import KafkaConsumer
import json

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from pydantic import BaseModel
from typing import List

import mlflow
import mlflow.sklearn

class Transaction(BaseModel):
    features: List[float]

# load model
# MODEL_PATH = "model.pkl"
# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)


# metrics for Prometheus
REQUEST_COUNT = Counter("requests_total", "Total prediction requests")
FRAUD_COUNT = Counter("fraud_detected_total", "Total fraud detected")

# internal stats
stats = {
    "total_processed": 0,
    "fraud_detected": 0
}

model = None
# lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model


    # Ensure MLflow uses the proxy for artifact downloads
    import os
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    os.environ["MLFLOW_ARTIFACT_URI"] = "http://mlflow:5000"
    mlflow.set_tracking_uri("http://mlflow:5000")

    # load latest model
    from mlflow.tracking import MlflowClient
    client = MlflowClient("http://mlflow:5000")

    experiment = None
    while experiment is None:
        try:
            experiment = client.get_experiment_by_name("Default")
        except Exception as e:
            print("MLflow not ready, retrying in 2 seconds...", e)
            time.sleep(2)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise Exception("No MLflow runs found!")

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    model = None
    while model is None:
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"Model not ready, retrying in 2s... {e}")
            time.sleep(2)
    
    # start Kafka consumer
    start_kafka_thread()

    yield

app = FastAPI(lifespan=lifespan)


def predict_transaction(features: list):
    features_array = np.array(features).reshape(1, -1)

    prob = model.predict_proba(features_array)[0][1]
    is_fraud = prob > 0.5

    # update metrics
    REQUEST_COUNT.inc()
    stats["total_processed"] += 1

    if is_fraud:
        FRAUD_COUNT.inc()
        stats["fraud_detected"] += 1

    return prob, is_fraud

@app.post("/predict")
def predict(txn: Transaction):
    prob, is_fraud = predict_transaction(txn.features)

    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(is_fraud)
    }

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "kafka_connected": kafka_running
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


kafka_running = False

# Kafka consumer (background thread)
KAFKA_TOPIC = "transactions"
KAFKA_SERVER = "kafka:9092"

def kafka_listener():
    global kafka_running

    consumer = None

    # retry until Kafka is available
    while consumer is None:
        try:
            # create Kafka consumer
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVER,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="fraud-detection-group"
            )
            kafka_running = True
            print("Kafka connected")

        except Exception as e:
            kafka_running = False
            print("Kafka not ready, retrying in 2 seconds...", e)
            time.sleep(2)

    for message in consumer:
        txn = message.value
        features = txn["features"]

        prob, is_fraud = predict_transaction(features)

        print(f"Fraud: {is_fraud} | Prob: {prob:.4f}")

# Kafka thread
def start_kafka_thread():
    thread = Thread(target=kafka_listener, daemon=True)
    thread.start()
