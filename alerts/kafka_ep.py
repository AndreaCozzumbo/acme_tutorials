from gcn_kafka import Consumer
from confluent_kafka import TopicPartition
import os
import json
import datetime 
import xmltodict
from astropy.coordinates import SkyCoord

# Connect as a consumer (client "Mac MOC")
# Warning: don't share the client secret with others.
consumer = Consumer(client_id='5h42prl0v1gtqh60b2p2a1ltho',
                    client_secret='icq2qm2nelng5ucelm71dh9kf3l0jgq0b0omthv457i5h74im0v')


def voevent_to_json(text_data, output_dir="/Users/sjs8171/Desktop/alert_acme/gcn_alerts"):
    """Parse GCN JSON format and extract position and trigger time, then save to JSON."""
    
    try:
        # Parse the JSON data
        data = json.loads(text_data)
        
        # Extract only position and trigger time
        alert_info = {
            "trigger_time": data.get("trigger_time"),
            "ra": data.get("ra"),
            "dec": data.get("dec")
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on trigger time and ID
        trigger_id = data.get("id", ["unknown"])[0]
        filename = f"einstein_probe_{trigger_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(alert_info, f, indent=2)
        
        print(f"Saved alert to {filepath}")
        return alert_info
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        print(f"Error processing alert: {e}")
        return None


# Time ranges are specified as Kafka times (milliseconds since the unix epoch)

timestamp1 = int((datetime.datetime.now() - datetime.timedelta(days=20)).timestamp() * 1000)
timestamp2 = timestamp1 + 10*86400000 # +1 days

print(f"Fetching messages between {timestamp1} and {timestamp2}")

# Subscribe to topics and receive alerts
topics = 'gcn.notices.einstein_probe.wxt.alert'


start = consumer.offsets_for_times(
    [TopicPartition(topics, 0, timestamp1)])
end = consumer.offsets_for_times(
    [TopicPartition(topics, 0, timestamp2)])


consumer.assign(start)

# Calculate the number of messages to consume, ensuring it's within valid range
num_messages = end[0].offset - start[0].offset

print(end[0].offset, start[0].offset)

for message in consumer.consume(abs(num_messages), timeout=1):
    
# while True:
#     for message in consumer.consume(timeout=1):
        if message.error():
            print(message.error())
            continue
        # Print the topic and message ID
        print(f'topic={message.topic()}, offset={message.offset()}')
        value = message.value()
        # xml_dict = xmltodict.parse(value)
        voevent_to_json(value)