import pandas as pd
import json
import time
from kafka import KafkaProducer

KAFKA_TOPIC = "transactions"
KAFKA_SERVER = "localhost:9092"
STREAM_PATH = "data/stream.csv"

# create producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# load data
df = pd.read_csv(STREAM_PATH)

print(f"Loaded {len(df)} transactions")

# stream data
for i, row in df.iterrows():
    # convert row to feature list
    features = row.drop("Class").values.tolist()

    message = {
        "features": features
    }

    # send to Kafka
    producer.send(KAFKA_TOPIC, value=message)

    print(f"Sent transaction {i}")

    # simulate real-time delay
    time.sleep(0.1)

# flush & close
producer.flush()
producer.close()

print("Finished streaming.")