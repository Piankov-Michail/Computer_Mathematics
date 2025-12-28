import pandas as pd
import glob
import os
from stations_positions import DSN_STATIONS

def safe_parse_time(time_input):
    if isinstance(time_input, str):
        try:
            return pd.to_datetime(time_input, utc=True)
        except (ValueError, TypeError):
            time_clean = time_input.strip()
            if ' ' in time_clean and '.' not in time_clean and '+' in time_clean:
                time_clean = time_clean.replace('+', '.000000+')
            if '.' in time_clean:
                base, rest = time_clean.split('.', 1)
                if '+' in rest:
                    micro_part, tz_part = rest.split('+', 1)
                    if len(micro_part) > 6:
                        micro_part = micro_part[:6]
                    time_clean = f"{base}.{micro_part}+{tz_part}"
            return pd.to_datetime(time_clean, utc=True, errors='coerce')
    
    parsed = pd.to_datetime(time_input, utc=True, errors='coerce')
    
    if isinstance(parsed, pd.Series) and parsed.isna().sum() > len(parsed) * 0.5:
        time_series_clean = time_input.astype(str).str.strip()
        
        mask_no_micro = (time_series_clean.str.contains(' ') & 
                        (~time_series_clean.str.contains('\\.')) & 
                        time_series_clean.str.contains('\\+'))
        if mask_no_micro.any():
            time_series_clean = time_series_clean.where(
                ~mask_no_micro, 
                time_series_clean.str.replace('\\+', '.000000+', regex=True)
            )
        
        mask_long_micro = time_series_clean.str.contains('\\.')
        if mask_long_micro.any():
            def truncate_microseconds(ts):
                if pd.isna(ts) or not isinstance(ts, str):
                    return ts
                if '.' in ts:
                    parts = ts.split('.', 1)
                    base = parts[0]
                    rest = parts[1]
                    if '+' in rest:
                        micro_part, tz_part = rest.split('+', 1)
                        if len(micro_part) > 6:
                            micro_part = micro_part[:6]
                        return f"{base}.{micro_part}+{tz_part}"
                return ts
            
            time_series_clean = time_series_clean.apply(truncate_microseconds)
        
        parsed = pd.to_datetime(time_series_clean, utc=True, errors='coerce')
    
    return parsed

def load_messenger_doppler_data(data_dir: str = './processed_data') -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, '*_doppler.csv'))
    if not csv_files:
        print("Нет файлов в processed_data/")
        return pd.DataFrame()

    print(f"Найдено {len(csv_files)} файлов")
    all_data = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, parse_dates=['receive_time_utc'])
            if df.empty:
                continue

            df = df.rename(columns={'receive_time_utc': 'time_utc'})

            filtered_df = df[df['observable_hz'] < 1e6]

            df = filtered_df.copy()

            df['transmit_frequency_hz'] = df['reference_frequency_hz']
            mask = df['ramp_active']
            df.loc[mask, 'transmit_frequency_hz'] = df.loc[mask, 'ramp_start_freq_hz']

            df['station_name'] = df['station_id'].map(lambda x: DSN_STATIONS.get(x, {}).get('name', f'Station {x}'))

            cols = ['time_utc', 'station_id', 'station_name', 'transmitting_station_id',
                    'observable_hz', 'transmit_frequency_hz', 'ramp_active', 'ramp_rate_hz_s',
                    'ramp_start_time', 'reference_frequency_hz', 'downlink_delay_ns',
                    'uplink_delay_s', 'compression_time_s', 'receiver_channel']

            df = df[[c for c in cols if c in df.columns]]
            all_data.append(df)
            print(f"{os.path.basename(file_path)}: {len(df)} записей")
        except Exception as e:
            print(f"Ошибка {file_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('time_utc').reset_index(drop=True)
    print(f"\nВсего: {len(combined)} записей от {combined['time_utc'].min()} до {combined['time_utc'].max()}")
    return combined