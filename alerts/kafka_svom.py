from gcn_kafka import Consumer
import json
import datetime 
import xmltodict
from astropy.coordinates import SkyCoord

# Connect as a consumer (client "Mac MOC")
# Warning: don't share the client secret with others.
consumer = Consumer(client_id='5h42prl0v1gtqh60b2p2a1ltho',
                    client_secret='icq2qm2nelng5ucelm71dh9kf3l0jgq0b0omthv457i5h74im0v')


def voevent_to_json(xml_file):

    # Convert to normal dict and JSON
    json_data = json.loads(json.dumps(xml_file))

    # Navigate through the XML structure safely
    voevent = json_data.get("voe:VOEvent", {})
    wherewhen = voevent.get("WhereWhen", {})

    try:
        obs_location = wherewhen.get("ObsDataLocation", {}).get("ObservationLocation", {})
        astro_coords = obs_location.get("AstroCoords", {})
        pos = astro_coords.get("Position2D", {}).get("Value2", {})
        c1 = pos.get("C1")
        c2 = pos.get("C2")
        
        # Convert from galactic (l, b) to equatorial (RA, Dec)
        import astropy.units as u
        
        coord = SkyCoord(l=float(c1)*u.degree, b=float(c2)*u.degree, frame='galactic')
        equatorial = coord.icrs
        ra = equatorial.ra.degree
        dec = equatorial.dec.degree
        
        print("RA:", ra)
        print("Dec:", dec)
    
    except Exception as e:
        print("Error processing coordinates:", e)

    trigger_time = astro_coords.get("Time", {}).get("TimeInstant", {}).get("ISOTime")

    print("Trigger time:", trigger_time)

    return json_data


# get messages occurring 7 days ago
timestamp1 = int((datetime.datetime.now() - datetime.timedelta(days=7)).timestamp() * 1000)
timestamp2 = timestamp1 + 4*86400000 # +4 days

print(f"Fetching messages between {timestamp1} and {timestamp2}")

# Subscribe to topics and receive alerts
topics = 'gcn.notices.svom.voevent.eclairs'

from confluent_kafka import TopicPartition

start = consumer.offsets_for_times(
    [TopicPartition(topics, 0, timestamp1)])
end = consumer.offsets_for_times(
    [TopicPartition(topics, 0, timestamp2)])


consumer.assign(start)
for message in consumer.consume(end[0].offset - start[0].offset, timeout=1):
    
# while True:
#     for message in consumer.consume(timeout=1):
        if message.error():
            print(message.error())
            continue
        # Print the topic and message ID
        print(f'topic={message.topic()}, offset={message.offset()}')
        value = message.value()
        xml_dict = xmltodict.parse(value)
        voevent_to_json(xml_dict)