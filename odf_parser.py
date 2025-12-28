# odf_parser.py

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
import glob
import struct
from typing import List
from pds4_tools import pds4_read

REF_EPOCH = datetime(1950, 1, 1, tzinfo=timezone.utc)

def get_dat_path_from_structure(xml_path: str) -> str:
    dir_name = os.path.dirname(xml_path)
    base_name = os.path.basename(xml_path).rsplit('.', 1)[0]
    for ext in ['.dat', '.DAT']:
        path = os.path.join(dir_name, base_name + ext)
        if os.path.exists(path):
            return path
    return None

def parse_ramp_group(structure, xml_path: str) -> pd.DataFrame:
    if structure is None:
        return pd.DataFrame()

    try:
        offset = structure.meta_data['offset']
        records = structure.meta_data['records']
    except KeyError:
        return pd.DataFrame()

    dat_path = get_dat_path_from_structure(xml_path)
    if not dat_path:
        return pd.DataFrame()

    ramp_records = []
    with open(dat_path, 'rb') as f:
        f.seek(offset)
        for _ in range(records):
            raw = f.read(36)
            if len(raw) < 36:
                break

            start_int   = struct.unpack('>I', raw[0:4])[0]
            start_frac  = struct.unpack('>I', raw[4:8])[0]
            rate_int    = struct.unpack('>i', raw[8:12])[0]
            rate_frac   = struct.unpack('>i', raw[12:16])[0]
            items56     = struct.unpack('>I', raw[16:20])[0]
            freq_int    = struct.unpack('>I', raw[20:24])[0]
            freq_frac   = struct.unpack('>I', raw[24:28])[0]
            end_int     = struct.unpack('>I', raw[28:32])[0]
            end_frac    = struct.unpack('>I', raw[32:36])[0]

            freq_ghz = (items56 >> 10) & 0x3FFFFF

            start_time = REF_EPOCH + pd.Timedelta(seconds=start_int) + pd.Timedelta(nanoseconds=start_frac)
            end_time   = REF_EPOCH + pd.Timedelta(seconds=end_int)   + pd.Timedelta(nanoseconds=end_frac)

            f0 = freq_ghz * 1e9 + freq_int + freq_frac * 1e-9
            ramp_rate = rate_int + rate_frac * 1e-9

            ramp_records.append({
                "ramp_start_time": pd.to_datetime(start_time),
                "ramp_end_time": pd.to_datetime(end_time),
                "ramp_start_freq_hz": f0,
                "ramp_rate_hz_s": ramp_rate,
                "duration_s": (end_time - start_time).total_seconds()
            })

    return pd.DataFrame(ramp_records)

def parse_data_group(structure, xml_path: str) -> pd.DataFrame:
    if structure is None:
        return pd.DataFrame()

    try:
        offset = structure.meta_data['offset']
        records = structure.meta_data['records']
    except KeyError:
        return pd.DataFrame()

    dat_path = get_dat_path_from_structure(xml_path)
    if not dat_path:
        return pd.DataFrame()

    data_records = []
    with open(dat_path, 'rb') as f:
        f.seek(offset)
        for idx in range(records):
            raw = f.read(36)
            if len(raw) < 36:
                break

            t_int         = struct.unpack('>I', raw[0:4])[0]
            items23       = struct.unpack('>I', raw[4:8])[0]
            obs_int       = struct.unpack('>i', raw[8:12])[0]
            obs_frac      = struct.unpack('>i', raw[12:16])[0]
            items614      = struct.unpack('>I', raw[16:20])[0]
            items1519     = struct.unpack('>Q', raw[20:28])[0]
            items2022     = struct.unpack('>Q', raw[28:36])[0]

            t_frac_ms = (items23 >> 22) & 0x3FF
            receive_time = REF_EPOCH + pd.Timedelta(seconds=t_int) + pd.Timedelta(milliseconds=t_frac_ms)

            observable = obs_int + obs_frac * 1e-9

            station_id           = (items614 >> 22) & 0x7F
            transmitting_station = (items614 >> 15) & 0x7F
            data_type_id         = (items614 >> 7)  & 0x3F
            data_valid           = (items614 & 0x1) == 0

            if data_type_id != 12 or not data_valid:
                continue

            receiver_channel = (items1519 >> 57) & 0x7F
            ramp_indicator   = (items1519 >> 46) & 0x1
            ref_freq_high    = (items1519 >> 24) & 0x3FFFFF
            ref_freq_low     = items1519 & 0xFFFFFF
            reference_freq   = (ref_freq_high << 24 | ref_freq_low) / 1000.0

            compression_time = ((items2022 >> 22) & 0x3FFFFF) * 0.01
            uplink_delay_s   = (items2022 & 0x3FFFFF) * 1e-9

            data_records.append({
                "receive_time_utc": pd.to_datetime(receive_time),
                "station_id": int(station_id),
                "transmitting_station_id": int(transmitting_station),
                "observable_hz": observable,
                "reference_frequency_hz": reference_freq,
                "ramp_indicator": bool(ramp_indicator),
                "receiver_channel": int(receiver_channel),
                "downlink_delay_ns": items23 & 0x3FFFFF,
                "uplink_delay_s": uplink_delay_s,
                "compression_time_s": compression_time,
                "record_index": idx
            })

    return pd.DataFrame(data_records)

def attach_ramps_to_data(data_df: pd.DataFrame, ramp_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if data_df.empty:
        return data_df
    data_df = data_df.copy()
    data_df['ramp_active'] = False
    data_df['ramp_start_freq_hz'] = data_df['reference_frequency_hz']
    data_df['ramp_rate_hz_s'] = 0.0
    data_df['ramp_start_time'] = data_df['receive_time_utc']

    all_ramps = pd.concat(ramp_dfs, ignore_index=True) if ramp_dfs else pd.DataFrame()
    if all_ramps.empty:
        return data_df

    for idx, row in data_df.iterrows():
        t = row['receive_time_utc']
        matching = all_ramps[(all_ramps['ramp_start_time'] <= t) & (t < all_ramps['ramp_end_time'])]
        if not matching.empty:
            ramp = matching.iloc[0]
            data_df.at[idx, 'ramp_active'] = True
            data_df.at[idx, 'ramp_start_freq_hz'] = ramp['ramp_start_freq_hz']
            data_df.at[idx, 'ramp_rate_hz_s'] = ramp['ramp_rate_hz_s']
            data_df.at[idx, 'ramp_start_time'] = ramp['ramp_start_time']

    return data_df

def parse_messenger_odf_pds4(xml_path: str) -> pd.DataFrame:
    print(f"Обработка: {os.path.basename(xml_path)}")
    try:
        structures = pds4_read(xml_path, lazy_load=True, quiet=True)

        data_struct = None
        ramp_structs = []

        for struct in structures:
            name = struct.meta_data.get('name', '')
            if 'Data Group' in name and 'ODF' in name:
                data_struct = struct
            elif 'Ramp Group' in name and 'ODF' in name:
                ramp_structs.append(struct)

        if data_struct is None:
            print("  Нет ODF Data Group")
            return pd.DataFrame()

        data_df = parse_data_group(data_struct, xml_path)
        ramp_dfs = [parse_ramp_group(rs, xml_path) for rs in ramp_structs]
        final_df = attach_ramps_to_data(data_df, ramp_dfs)

        print(f"  Готово: {len(final_df)} записей, {final_df['ramp_active'].sum()} с ramp")
        return final_df
    except Exception as e:
        print(f"Ошибка: {e}")
        return pd.DataFrame()

def process_all_odf_files(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    successful = 0
    for xml_file in xml_files:
        df = parse_messenger_odf_pds4(xml_file)
        if not df.empty:
            base = os.path.basename(xml_file).rsplit('.', 1)[0]
            csv_path = os.path.join(output_dir, f"{base}_doppler.csv")
            df.to_csv(csv_path, index=False)
            successful += 1
    print(f"\nУспешно: {successful}/{len(xml_files)}")

if __name__ == "__main__":
    process_all_odf_files("downloaded_files", "processed_data")