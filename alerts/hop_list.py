from hop import stream, Stream
from hop.io import StartPosition
from hop.auth import Auth

# Kafka topic and Hop username
TOPIC = "igwn.gwalert"
HOP_USER = "samuele.ronchini-4ffd4374"   # replace if needed

# Output files
RAW_FILE = "gwalert_messages.txt"
PARSED_FILE = "gwalert_parsed.txt"

auth = Auth(HOP_USER, '784IXpNxz7b3pyHDf6onglym7lSbOtre')

stream = Stream(auth=auth, start_at=StartPosition.LATEST)

with stream.open("kafka://kafka.scimma.org/igwn.gwalert", "r") as s:

    for message in s:

        alert = message.content

        print(alert['superevent_id'])
        print(alert['alert_type'])