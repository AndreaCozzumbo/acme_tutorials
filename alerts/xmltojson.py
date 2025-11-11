import xmltodict
import json
from astropy.coordinates import SkyCoord

def voevent_to_json(xml_file):
    # Parse the XML into an ordered dict
    with open(xml_file, 'r') as f:
        data = xmltodict.parse(f.read())

    # Convert to normal dict and JSON
    json_data = json.loads(json.dumps(data))

    # Navigate through the XML structure safely
    voevent = json_data.get("voe:VOEvent", {})
    wherewhen = voevent.get("WhereWhen", {})
    if not wherewhen:
        print("No WhereWhen information found.")
        return json_data
    else:
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

    trigger_time = astro_coords.get("Time", {}).get("TimeInstant", {}).get("ISOTime")

    print("Trigger time:", trigger_time)

    return json_data


if __name__ == "__main__":
    xml_file = "eclairs.xml"  # change this to your XML file path
    json_data = voevent_to_json(xml_file)

    # Optionally save to JSON file
    with open("voevent_output.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print("\nFull VOEvent converted to JSON saved as 'voevent_output.json'")