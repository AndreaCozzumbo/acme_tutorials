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
    """Parse GCN text format and extract relevant information, then save to JSON."""
    
    # Convert bytes to string if necessary
    if isinstance(text_data, bytes):
        text_data = text_data.decode('utf-8')
    
    # Initialize result dictionary
    result = {}
    
    try:
        lines = text_data.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract RA (J2000)
            if line.startswith('GRB_RA:'):
                # Extract the decimal degrees value
                ra_str = line.split()[1].rstrip('d')
                result['ra'] = float(ra_str)
                print("RA:", result['ra'])
            
            # Extract Dec (J2000)
            elif line.startswith('GRB_DEC:'):
                # Extract the decimal degrees value
                dec_str = line.split()[1].rstrip('d')
                result['dec'] = float(dec_str)
                print("Dec:", result['dec'])
            
            # Extract trigger time
            elif line.startswith('GRB_TIME:'):
                # Extract the time in SOD (seconds of day) and UT format
                parts = line.split()
                sod_time = float(parts[1])  # seconds of day
                # Extract UT time from the line (format: {HH:MM:SS.SS})
                ut_time = line.split('{')[1].split('}')[0]
                result['grb_time_sod'] = sod_time
                result['grb_time_ut'] = ut_time
                print("GRB Time (SOD):", sod_time)
                print("GRB Time (UT):", ut_time)

            # Extract GRB date
            elif line.startswith('GRB_DATE:'):
                # Extract TJD, DOY, and calendar date
                parts = line.split()
                tjd = parts[1]
                doy = parts[3].rstrip(';')
                date_str = parts[5]  # format: YY/MM/DD
                result['grb_date_tjd'] = tjd
                result['grb_date_doy'] = doy
                result['grb_date'] = date_str
                print("GRB Date:", date_str)
            # Extract trigger number
            elif line.startswith('TRIGGER_NUM:'):
                trigger_num = line.split()[1].rstrip(',')
                result['trigger_num'] = trigger_num
        
        # Save to JSON file if we have data
        if result:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a filtered dictionary with only trigger name, time, and position
            filtered_result = {}
            
            # Add trigger number if available
            if 'trigger_num' in result:
                filtered_result['trigger_num'] = result['trigger_num']
            
            # Add time information if available (convert to ISO format)
            if 'grb_date' in result and 'grb_time_ut' in result:
                # Combine date and time into ISO format
                date_parts = result['grb_date'].split('/')  # YY/MM/DD
                time_str = result['grb_time_ut']  # HH:MM:SS.SS
                
                # Convert YY to full year (assuming 20YY)
                year = f"20{date_parts[0]}"
                month = date_parts[1]
                day = date_parts[2]
                
                iso_datetime = f"{year}-{month}-{day}T{time_str}"
                filtered_result['grb_time_iso'] = iso_datetime
            elif 'grb_time_ut' in result:
                filtered_result['grb_time_ut'] = result['grb_time_ut']
            
            # Add position if available
            if 'ra' in result:
                filtered_result['ra'] = result['ra']
            if 'dec' in result:
                filtered_result['dec'] = result['dec']
            
            # Create filename using trigger number or timestamp
            if 'trigger_num' in filtered_result:
                filename = f"GRB_{filtered_result['trigger_num']}.json"
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"GRB_{timestamp}.json"
            
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(filtered_result, f, indent=2)
            
            print(f"Saved to: {filepath}")
    
    except Exception as e:
        print("Error processing GCN text notice:", e)
    
    return result

# Time ranges are specified as Kafka times (milliseconds since the unix epoch)

timestamp1 = int((datetime.datetime.now() - datetime.timedelta(days=10)).timestamp() * 1000)
timestamp2 = timestamp1 + 7*86400000 # +7 days

print(f"Fetching messages between {timestamp1} and {timestamp2}")

# Subscribe to topics and receive alerts
topics = 'gcn.classic.text.FERMI_GBM_ALERT'


start = consumer.offsets_for_times(
    [TopicPartition(topics, 0, timestamp1)])
end = consumer.offsets_for_times(
    [TopicPartition(topics, 0, timestamp2)])


consumer.assign(start)

# Calculate the number of messages to consume, ensuring it's within valid range
num_messages = end[0].offset - start[0].offset

print(num_messages)

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