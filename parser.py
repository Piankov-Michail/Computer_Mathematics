import struct
import pandas as pd
from datetime import datetime, timezone
import os, glob

def parse_messenger_odf(label_path: str):
    import xml.etree.ElementTree as ET

    tree = ET.parse(label_path)
    root = tree.getroot()
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

    orbit_table = None
    for tbl in root.findall(".//pds:Table_Binary", ns):
        name_elem = tbl.find("pds:name", ns)
        if name_elem is not None and name_elem.text.strip() == "ODF Orbit Data Group Data":
            orbit_table = tbl
            break

    if orbit_table is None:
        raise ValueError("'ODF Orbit Data Group Data' table not found in XML.")

    offset_elem = orbit_table.find("pds:offset", ns)
    records_elem = orbit_table.find("pds:records", ns)
    offset = int(offset_elem.text)
    n_records = int(records_elem.text)

    dat_path = label_path.replace(".xml", ".dat")
    with open(dat_path, "rb") as f:
        f.seek(offset)
        raw = f.read(36 * n_records)

    if len(raw) != 36 * n_records:
        raise ValueError(f"Data truncated: expected {36 * n_records} bytes, got {len(raw)}")

    records = []
    ref_epoch = datetime(1950, 1, 1, tzinfo=timezone.utc)
    C = 299_792_458.0

    for i in range(n_records):
        start = i * 36
        rec = raw[start:start+36]

        t_int = struct.unpack('>I', rec[0:4])[0]
        packed_23 = struct.unpack('>I', rec[4:8])[0]
        t_frac_ms = (packed_23 >> 22) & 0x3FF
        total_sec = t_int + t_frac_ms / 1000.0
        time_utc = ref_epoch + pd.Timedelta(seconds=total_sec)

        obs_int = struct.unpack('>i', rec[8:12])[0]
        obs_frac = struct.unpack('>i', rec[12:16])[0]
        observable_raw = obs_int + obs_frac * 1e-9

        packed_6_14 = struct.unpack('>I', rec[16:20])[0]
        station_id = (packed_6_14 >> 22) & 0x7F
        data_type = (packed_6_14 >> 7) & 0x3F
        valid = (packed_6_14 & 0x1) == 0

        record = {
            "time_utc": time_utc,
            "data_type": data_type,
            "station_id": station_id,
            "valid": valid,
            "observable_raw": observable_raw,
            "doppler_hz": None,
            "range_m": None,
            "range_type": None
        }

        if not valid:
            records.append(record)
            continue

        if data_type in (11, 12, 13):
            record["doppler_hz"] = observable_raw
        elif data_type == 37:
            record["range_m"] = observable_raw
            record["range_type"] = "sequential"
        elif data_type == 41:
            record["range_m"] = C * observable_raw / 2e9
            record["range_type"] = "re"

        records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    input_dir = "downloaded_files"
    output_dir = "data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_pattern = os.path.join(input_dir, "mess_rs_*.xml")
    xml_files = glob.glob(file_pattern)
    
    if not xml_files:
        print("No files found.")
    else:
        for xml_file in xml_files:
            df = parse_messenger_odf(xml_file)
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            csv_file = os.path.join(output_dir, base_name + ".csv")
            df.to_csv(csv_file, index=False)