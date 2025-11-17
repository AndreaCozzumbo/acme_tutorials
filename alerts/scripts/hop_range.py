from hop import stream
import json

# Kafka topic and Hop username
TOPIC = "igwn.gwistat.H1.range_history"
HOP_USER = "samuele.ronchini-5b5f5b4e"   # replace if needed

# Output files
OUTPUT_FILE = "range_history_H1.txt"


# Use stream.open() instead of Stream context manager
with stream.open(f"kafka://{HOP_USER}@kafka.scimma.org/{TOPIC}", "r") as s, \
     open(OUTPUT_FILE, "a") as f:

    for message in s:
        # message.content is a dict, convert to JSON string
        data_json = json.dumps(message.content)
        f.write(data_json + "\n")
        f.flush()  # make sure it's written immediately
        print(data_json)  # optional: print to terminal