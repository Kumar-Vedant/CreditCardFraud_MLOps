import pandas as pd
import json
import time
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

KAFKA_TOPIC = "transactions"
KAFKA_SERVER = "kafka:9092"
STREAM_PATH = "data/stream.csv"

producer = None

# retry until Kafka is available
while producer is None:
    try:
        # create producer
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8")
        )
        print("Connected to Kafka!")
    except NoBrokersAvailable:
        print("Kafka not ready, retrying in 2 seconds...")
        time.sleep(2)

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