import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time
import astropy.units as u
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

c = 299792.458
MESSENGER_XBAND_HIGH_FREQ = 8445.734e6
RAMPED_DOPPLER_DATA_TYPE = 12

DSN_STATIONS = {
    14: {'name': 'Goldstone DSS-14', 'x': -2353621.3988, 'y': -4641341.4050, 'z': 3677052.2385},
    15: {'name': 'Goldstone DSS-15', 'x': -2353538.9600, 'y': -4641649.4082, 'z': 3676669.9412},
    24: {'name': 'Goldstone DSS-24', 'x': -2354906.7148, 'y': -4646840.0771, 'z': 3669242.2852},
    25: {'name': 'Goldstone DSS-25', 'x': -2355022.0181, 'y': -4646953.1865, 'z': 3669040.5265},
    26: {'name': 'Goldstone DSS-26', 'x': -2354890.8011, 'y': -4647166.3107, 'z': 3668871.7147},
    34: {'name': 'Canberra DSS-34', 'x': -4461147.1937, 'y': 2682439.2835, 'z': -3674392.9674},
    35: {'name': 'Canberra DSS-35', 'x': -4461273.1915, 'y': 2682568.9709, 'z': -3674151.9281},
    43: {'name': 'Canberra DSS-43', 'x': -4460894.9746, 'y': 2682361.5266, 'z': -3674747.9625},
    45: {'name': 'Canberra DSS-45', 'x': -4460935.6766, 'y': 2682765.7043, 'z': -3674380.8142},
    54: {'name': 'Madrid DSS-54', 'x': 4849434.4781, 'y': -360723.8954, 'z': 4114618.8518},
    55: {'name': 'Madrid DSS-55', 'x': 4849525.2463, 'y': -360606.0885, 'z': 4114495.1006},
    63: {'name': 'Madrid DSS-63', 'x': 4849092.4565, 'y': -360180.3397, 'z': 4115109.2231},
    65: {'name': 'Madrid DSS-65', 'x': 4849339.6208, 'y': -360427.6502, 'z': 4114750.7302}
}

from horizons_parser import *
from interpolators import *
from messener_orbit import *
from create_graphs import *
from parse_csv import *
from stations_positions import *

def utc_to_tdb_seconds(utc_time):
    if isinstance(utc_time, str):
        if utc_time.endswith('+00:00'):
            utc_time = utc_time.replace('+00:00', 'Z')
        elif '+00:00' in utc_time:
            utc_time = utc_time.replace('+00:00', '')
    
    t_utc = Time(utc_time, format='iso', scale='utc')
    
    t_tt = t_utc.tt
    t_tdb = t_tt.tdb
    
    tdb_seconds = (t_tdb.jd - 2451545.0) * 86400.0
    
    return tdb_seconds

def get_transponder_ratios(uplink_band, downlink_band):
    ratios = {
        (1, 1): {'M2': (240, 221), 'M2R': (240, 221)},
        (1, 2): {'M2': (880, 221), 'M2R': (880, 221)},
        (1, 3): {'M2': (3344, 221), 'M2R': (3344, 221)},
        (2, 1): {'M2': (240, 749), 'M2R': (240, 749)},
        (2, 2): {'M2': (880, 749), 'M2R': (880, 749)},
        (2, 3): {'M2': (3344, 749), 'M2R': (3344, 749)},
        (3, 1): {'M2': (240, 3599), 'M2R': (240, 3599)},
        (3, 2): {'M2': (880, 3599), 'M2R': (880, 3599)},
        (3, 3): {'M2': (3344, 3599), 'M2R': (3344, 3599)}
    }
    
    key = (uplink_band, downlink_band)
    if key in ratios:
        ratio_data = ratios[key]
        M2 = ratio_data['M2'][0] / ratio_data['M2'][1]
        M2R = ratio_data['M2R'][0] / ratio_data['M2R'][1]
        return M2, M2R
    else:
        raise ValueError(f"Неизвестная комбинация полос: uplink={uplink_band}, downlink={downlink_band}")

def get_ramp_band_constants(band_id):
    constants = {
        1: {'T1': 240, 'T2': 221, 'T3': 96, 'T4': 0},
        2: {'T1': 240, 'T2': 749, 'T3': 32, 'T4': 6.5e9},
        3: {'T1': 14, 'T2': 15, 'T3': 1000, 'T4': 1.0e10},
    }
    return constants.get(band_id, constants[2])

def compute_ramped_frequency(t, ramp_table, station_id):
    for ramp in ramp_table:
        if ramp['station_id'] == station_id and ramp['start_time'] <= t <= ramp['end_time']:
            f0 = ramp['start_freq']
            f_dot = ramp['ramp_rate']
            t0 = ramp['start_time']
            return f0 + f_dot * (t - t0)
    
    return None

def integrate_ramped_frequency(t_start, t_end, ramp_table, station_id, row_data=None):
    W = t_end - t_start
    
    station_ramps = [r for r in ramp_table if r['station_id'] == station_id]
    
    if not station_ramps:
        base_freq = 7165.0e6
        ramp_rate = 0.0
        if row_data and 'ramp_rate_hz_s' in row_data and pd.notna(row_data['ramp_rate_hz_s']):
            ramp_rate = float(row_data['ramp_rate_hz_s'])
        
        result = base_freq * W + 0.5 * ramp_rate * W**2
        return result
    
    station_ramps.sort(key=lambda x: x['start_time'])
    
    first_ramp = station_ramps[0]
    t0 = first_ramp['start_time']
    f0_initial = first_ramp['start_freq']
    f_dot_initial = first_ramp['ramp_rate']
    
    if t_start >= t0:
        ts = t_start
        f0_ts = f0_initial + f_dot_initial * (ts - t0)
    else:
        ts = t_start
        f0_ts = f0_initial + f_dot_initial * (ts - t0)
    
    ramps_to_use = []
    current_time = t_start

    for idx, ramp in enumerate(station_ramps):
        if ramp['end_time'] <= t_start:
            continue
        if ramp['start_time'] >= t_end:
            break
            
        segment = {
            'start_time': max(ramp['start_time'], t_start),
            'end_time': min(ramp['end_time'], t_end),
            'start_freq': ramp['start_freq'],
            'ramp_rate': ramp['ramp_rate'],
            'station_id': ramp['station_id']
        }
        
        if segment['start_time'] > ramp['start_time']:
            time_offset = segment['start_time'] - ramp['start_time']
            segment['start_freq'] += segment['ramp_rate'] * time_offset
        
        ramps_to_use.append(segment)
        current_time = segment['end_time']
        
        if current_time >= t_end:
            break
    
    if not ramps_to_use:
        closest_ramp = min(station_ramps, key=lambda r: abs((r['start_time'] + r['end_time'])/2 - (t_start + t_end)/2))
        ramp_rate = closest_ramp['ramp_rate']
        t_mid = (t_start + t_end) / 2
        f_mid = closest_ramp['start_freq'] + ramp_rate * (t_mid - closest_ramp['start_time'])
        result = f_mid * W + 0.5 * ramp_rate * W**2
        return result

    Wi_list = []
    
    for i in range(len(ramps_to_use) - 1):
        segment = ramps_to_use[i]
        Wi = segment['end_time'] - segment['start_time']
        Wi_list.append(Wi)

    if len(ramps_to_use) > 1:
        total_width = sum(Wi_list)
        Wn = W - total_width
        Wi_list.append(Wn)
    else:
        Wn = W
        Wi_list = [Wn]

    integral = 0.0

    for i, segment in enumerate(ramps_to_use):
        if i >= len(Wi_list) or Wi_list[i] <= 0:
            continue
        
        Wi = Wi_list[i]
        f0_segment = segment['start_freq']
        f_dot_segment = segment['ramp_rate']
        
        fi = f0_segment + 0.5 * f_dot_segment * Wi
        
        segment_integral = fi * Wi
        integral += segment_integral
        
    total_integral = integral
    return total_integral

def compute_doppler_reference_frequency(f_T, uplink_band, downlink_band, f_type='transmitter'):
    M2, M2R = get_transponder_ratios(uplink_band, downlink_band)
    
    if f_type == 'transmitter':
        return M2 * f_T
    elif f_type == 'receiver':
        return M2R * f_T
    else:
        raise ValueError(f"Неизвестный тип частоты: {f_type}")

def compute_two_way_doppler_ramped(t_receive_utc, Tc, station_id, uplink_band, downlink_band, \
                                   ramp_table, body_interpolators, body_vel_interpolators, sc_pos_interp,\
                                      sc_vel_interp, GM_params, row_data=None):
    M2, M2R = get_transponder_ratios(uplink_band, downlink_band)
    
    light_time_solution = solve_two_way_light_time_with_intervals(
        t_receive_utc, Tc, station_id,
        body_interpolators, body_vel_interpolators,
        sc_pos_interp, sc_vel_interp, GM_params
    )

    t1_start = light_time_solution['t1_start_tdb']
    t1_end = light_time_solution['t1_end_tdb']
    t3_start = light_time_solution['t3_start_tdb']
    t3_end = light_time_solution['t3_end_tdb']

    integral_fT_t1 = integrate_ramped_frequency(t1_start, t1_end, ramp_table, station_id, row_data)
    avg_fT_t1 = integral_fT_t1 / (t1_end - t1_start)

    integral_fT_t3 = integrate_ramped_frequency(t3_start, t3_end, ramp_table, station_id, row_data)
    avg_fT_t3 = integral_fT_t3 / (t3_end - t3_start)

    term1 = (M2R / Tc) * integral_fT_t3
    term2 = (M2 / Tc) * integral_fT_t1
    
    F_theory = term1 - term2

    return {
        'theoretical_doppler_hz': F_theory,
        'light_time_s': light_time_solution['total_light_time'],
        'distance_km': light_time_solution['distance_km'],
        'range_rate_kms': light_time_solution['range_rate_kms'],
        'M2': M2,
        'M2R': M2R,
        'transmit_freq_hz': avg_fT_t1,
        'receive_freq_hz': avg_fT_t3,
        'integral_fT_t1': integral_fT_t1,
        'integral_fT_t3': integral_fT_t3,
        't1_start_tdb': t1_start,
        't1_end_tdb': t1_end,
        't3_start_tdb': t3_start,
        't3_end_tdb': t3_end
    }

def compute_one_way_doppler(t_receive_utc, Tc, station_id, downlink_band, \
                            spacecraft_freq_params, ramp_table):
    f_T0 = spacecraft_freq_params.get('f_T0', 0)
    delta_f_T0 = spacecraft_freq_params.get('delta_f_T0', 0)
    f_T1 = spacecraft_freq_params.get('f_T1', 0)
    f_T2 = spacecraft_freq_params.get('f_T2', 0)
    t0 = spacecraft_freq_params.get('t0', 0)
    
    C2_values = {1: 1, 2: 880/749, 3: 3344/240}
    C2 = C2_values.get(downlink_band, 1)
    
    TT = t_receive_utc
    t3_s = TT - Tc/2
    t3_e = TT + Tc/2
    
    t3_s_tai = Time(t3_s, scale='utc').tai.jd * 86400
    t3_e_tai = Time(t3_e, scale='utc').tai.jd * 86400
    
    def spacecraft_frequency(t):
        t_rel = t - t0
        f_sc = f_T0 + delta_f_T0 + f_T1 * t_rel + f_T2 * t_rel**2
        return C2 * f_sc
    
    t_mid = (t3_s_tai + t3_e_tai) / 2
    f_mid = spacecraft_frequency(t_mid)

    F1 = C2 * f_T0 - (1/Tc) * f_mid * (t3_e_tai - t3_s_tai)
    
    return F1

def process_doppler_with_exact_model(doppler_df, body_interpolators, body_vel_interpolators, \
                                     sc_pos_interp, sc_vel_interp, GM_params, ramp_table):
    results = []
    
    test_df = doppler_df
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Processing Doppler"):
        try:
            theoretical_result = compute_two_way_doppler_ramped(
                t_receive_utc=row['time_utc'],
                Tc=row['compression_time_s'],
                station_id=int(row['station_id']),
                uplink_band=int(row.get('uplink_band', 2)),
                downlink_band=int(row.get('downlink_band', 2)),
                ramp_table=ramp_table,
                body_interpolators=body_interpolators,
                body_vel_interpolators=body_vel_interpolators,
                sc_pos_interp=sc_pos_interp,
                sc_vel_interp=sc_vel_interp,
                GM_params=GM_params,
                row_data=row.to_dict()
            )
            
            theoretical_doppler = theoretical_result['theoretical_doppler_hz']
            measured_doppler = row['observable_hz']
            residual = abs(measured_doppler - theoretical_doppler)
            
            result = {
                'time_utc': row['time_utc'],
                'station_id': row['station_id'],
                'measured_doppler_hz': float(measured_doppler),
                'theoretical_doppler_hz': float(theoretical_doppler),
                'doppler_residual_hz': float(residual),
                'light_time_s': float(theoretical_result.get('light_time_s', 0)),
            }
            
            results.append(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    return results_df

def prepare_ramp_table_from_data(df):
    ramp_table = []
    
    required_cols = ['ramp_active', 'ramp_rate_hz_s', 'ramp_start_time', 'station_id']

    if 'ramp_start_time' in df.columns and df['ramp_start_time'].notna().any():
        df['ramp_key'] = df['station_id'].astype(str) + '_' + df['ramp_start_time'].astype(str)
        
        unique_ramps = df[df['ramp_active'] == True].drop_duplicates('ramp_key')
        
        for idx, row in unique_ramps.iterrows():
            try:
                start_freq = None
                if 'ramp_start_freq_hz' in row and pd.notna(row['ramp_start_freq_hz']):
                    start_freq = float(row['ramp_start_freq_hz'])
                elif 'transmit_frequency_hz' in row and pd.notna(row['transmit_frequency_hz']):
                    start_freq = float(row['transmit_frequency_hz'])
                elif 'reference_frequency_hz' in row and pd.notna(row['reference_frequency_hz']):
                    start_freq = float(row['reference_frequency_hz'])
                else:
                    start_freq = 7165.0e6
                
                if start_freq < 1e7:
                    start_freq *= 1e6
                elif start_freq < 1e10:
                    pass
                else:
                    start_freq = 7165.0e6
                
                ramp_rate = float(row['ramp_rate_hz_s']) if pd.notna(row['ramp_rate_hz_s']) else 0.0
                ramp_start_time = row['ramp_start_time']
                
                ramp_start_tdb = utc_to_tdb_seconds(ramp_start_time)

                ramp_end_tdb = ramp_start_tdb + 3600
                
                ramp_table.append({
                    'station_id': int(row['station_id']),
                    'start_time': ramp_start_tdb,
                    'end_time': ramp_end_tdb,
                    'start_freq': start_freq,
                    'ramp_rate': ramp_rate
                })
            except Exception as e:
                print(f"Ошибка обработки ramp для станции {row['station_id']}: {e}")
                continue

    ramp_table.sort(key=lambda x: (x['station_id'], x['start_time']))
    
    if ramp_table:
        sample = ramp_table[0]

    if not ramp_table and not df.empty:
        for station_id in df['station_id'].unique():
            if pd.isna(station_id) or station_id not in DSN_STATIONS:
                continue
                
            mid_time = df['time_utc'].iloc[len(df)//2]
            ramp_start_tdb = utc_to_tdb_seconds(mid_time)
            
            start_freq = 7165.0e6
            
            ramp_table.append({
                'station_id': int(station_id),
                'start_time': ramp_start_tdb - 86400,
                'end_time': ramp_start_tdb + 86400,
                'start_freq': start_freq,
                'ramp_rate': 0.0
            })
    
    return ramp_table

def solve_two_way_light_time_with_intervals(t_receive_utc, Tc, station_id, \
            body_interpolators, body_vel_interpolators,sc_pos_interp, sc_vel_interp, GM_params):
    t3_tdb = utc_to_tdb_seconds(t_receive_utc)

    t3_center = t3_tdb
    t3_start = t3_center - Tc/2
    t3_end = t3_center + Tc/2
    
    r_station_t3 = get_station_barycentric_pos(t3_center, station_id, body_interpolators)
    
    r_sc_approx = sc_pos_interp(t3_center)
    
    distance_approx = np.linalg.norm(r_sc_approx - r_station_t3)
    tau_D_approx = distance_approx / c

    t2_approx = t3_center - tau_D_approx

    tau_D_prev = 0
    max_iter = 10
    tol = 1e-9
    
    for iter_num in range(max_iter):
        
        r_sc_t2 = sc_pos_interp(t2_approx)

        rho_D = r_sc_t2 - r_station_t3
        geometric_distance = np.linalg.norm(rho_D)

        geometric_tau_D = geometric_distance / c
        
        delta_tau_D = compute_light_time_corrections(
            t2_approx, t3_center, r_sc_t2, r_station_t3, 
            body_interpolators, GM_params, leg='down'
        )
        
        tau_D = geometric_tau_D + delta_tau_D

        t2_new = t3_center - tau_D
        
        if abs(tau_D - tau_D_prev) < tol:
            break
        
        tau_D_prev = tau_D
        t2_approx = t2_new
    
    t2_center = t2_new
    r_sc_t2 = sc_pos_interp(t2_center)
    v_sc_t2 = sc_vel_interp(t2_center)

    tau_U_approx = tau_D
   
    t1_approx = t2_center - tau_U_approx

    tau_U_prev = 0
    
    for iter_num in range(max_iter):
        r_station_t1 = get_station_barycentric_pos(t1_approx, station_id, body_interpolators)

        rho_U = r_sc_t2 - r_station_t1
        geometric_distance_up = np.linalg.norm(rho_U)

        geometric_tau_U = geometric_distance_up / c

        delta_tau_U = compute_light_time_corrections(
            t1_approx, t2_center, r_station_t1, r_sc_t2,
            body_interpolators, GM_params, leg='up'
        )

        tau_U = geometric_tau_U + delta_tau_U

        t1_new = t2_center - tau_U

        if abs(tau_U - tau_U_prev) < tol:
            break
        
        tau_U_prev = tau_U
        t1_approx = t1_new
    
    t1_center = t1_new
    
    t1_start = t1_center - Tc/2
    t1_end = t1_center + Tc/2
    
    r_vec = r_station_t3 - r_sc_t2
    distance = np.linalg.norm(r_vec)
    r_hat = r_vec / distance if distance > 0 else np.zeros(3)
    
    v_station_t3 = get_station_velocity(t3_center, station_id, body_vel_interpolators)
    range_rate = np.dot(v_station_t3 - v_sc_t2, r_hat)
    
    total_light_time = tau_D + tau_U

    return {
        't0_tdb': t1_center,
        't1_start_tdb': t1_start,
        't1_end_tdb': t1_end,
        't1_center_tdb': t1_center,
        't2_center_tdb': t2_center,
        't3_start_tdb': t3_start,
        't3_center_tdb': t3_center,
        't3_end_tdb': t3_end,
        't3_tdb': t3_center,
        'light_time_up': tau_U,
        'light_time_down': tau_D,
        'total_light_time': total_light_time,
        'distance_km': distance,
        'range_rate_kms': range_rate,
        'r_sc_transmit': r_sc_t2,
        'sc_velocity': v_sc_t2,
        'r_station_receive': r_station_t3,
        'station_velocity': v_station_t3
    }

def compute_light_time_corrections(t_emit, t_obs, r_emit, r_obs, body_interpolators, GM_params, leg='up'):
    total_correction = 0.0
    
    gamma = 1.0
    
    for body_name, gm in GM_params.items():
        if body_name == 'messenger':
            continue
            
        t_mid = (t_emit + t_obs) / 2
        r_body = body_interpolators[body_name](t_mid)
        
        r1 = np.linalg.norm(r_emit - r_body)
        r2 = np.linalg.norm(r_obs - r_body)
        r12 = np.linalg.norm(r_obs - r_emit)
        
        if r1 + r2 - r12 > 1e-10:
            mu_term = (1 + gamma) * gm / (c**2)
            numerator = r1 + r2 + r12 + mu_term
            denominator = r1 + r2 - r12 + mu_term
            
            if denominator > 0:
                shapiro_delay = ((1 + gamma) * gm / (c**3)) * np.log(numerator / denominator)
                total_correction += shapiro_delay

    if leg == 'up':
        solar_corona_correction = estimate_solar_corona_delay(t_emit, t_obs, r_emit, r_obs, body_interpolators)
        total_correction += solar_corona_correction
    
    if leg in ['up', 'down']:
        troposphere_correction = 0.002
        ionosphere_correction = 0.001
        atmospheric_correction = (troposphere_correction + ionosphere_correction) / 1000.0
        total_correction += atmospheric_correction
    
    return total_correction

def estimate_solar_corona_delay(t_emit, t_obs, r_emit, r_obs, body_interpolators, frequency=8.4e9):
    t_mid = (t_emit + t_obs) / 2
    r_sun = body_interpolators['sun'](t_mid)

    d_emit = np.linalg.norm(r_emit - r_sun)
    d_obs = np.linalg.norm(r_obs - r_sun)

    r_vec = r_obs - r_emit
    sun_vec = r_sun - (r_emit + r_obs) / 2

    min_distance = np.min([d_emit, d_obs])
    
    if min_distance < 10 * 1.496e8:
        elongation_factor = 1.0 / (1.0 + (min_distance / 1.496e8)**2)
        base_delay = 10.0
        delay = base_delay * elongation_factor * (8.4e9 / frequency)**2 / 1000.0
        return delay
    
    return 0.0

def solve_one_way_light_time(t_obs_tdb, r_obs, sc_pos_interp, sc_vel_interp, body_interpolators,
                             GM_params, max_iter=20, tol=1e-6):
    r_sc = sc_pos_interp(t_obs_tdb)
    dt_guess = np.linalg.norm(r_obs - r_sc) / c
    t_emit = t_obs_tdb - dt_guess
    prev_dt = dt_guess
    
    for i in range(max_iter):
        r_sc = sc_pos_interp(t_emit)
        geometric_dist = np.linalg.norm(r_obs - r_sc)
        t_mid = (t_obs_tdb + t_emit) / 2
        shapiro = shapiro_delay(r_sc, r_obs, t_mid, body_interpolators, GM_params)
        total_dt = geometric_dist / c + shapiro
        t_emit_new = t_obs_tdb - total_dt
        if abs(total_dt - prev_dt) < tol:
            break
        prev_dt = total_dt
        t_emit = t_emit_new

    r_sc_final = sc_pos_interp(t_emit)
    v_sc_final = sc_vel_interp(t_emit) / 86400.0
    
    return {
        't_emit': t_emit,
        'light_time': prev_dt,
        'r_sc': r_sc_final,
        'v_sc': v_sc_final
    }

def shapiro_delay(r_emit, r_obs, t_mid, body_interpolators, GM_params, body='sun'):
    r_body = body_interpolators[body](t_mid)
    r1 = np.linalg.norm(r_emit - r_body)
    r2 = np.linalg.norm(r_obs - r_body)
    r12 = np.linalg.norm(r_obs - r_emit)
    gm = GM_params[body]
    if r1 + r2 - r12 > 1e-10:
        delay = (2 * gm / (c ** 3)) * np.log((r1 + r2 + r12) / (r1 + r2 - r12))
    else:
        delay = 0.0
    return delay

def main():
    '''
    print("1. ЗАГРУЗКА ДАННЫХ DOPPLER MESSENGER...")
    doppler_df = load_messenger_doppler_data('./processed_data')
    
    if doppler_df.empty:
        print("Нет данных для обработки. Выход.")
        return

    raw_output_file = 'raw_messenger_doppler_data.csv'
    doppler_df.to_csv(raw_output_file, index=False)
    exit(0)'''
    raw_output_file = 'raw_messenger_doppler_data.csv'
    doppler_df = pd.read_csv(raw_output_file)
    doppler_df = doppler_df[:5000]
    #doppler_df = doppler_df.sample(1000)

    #print("3. АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ДАННЫХ...")
    
    #plot_files = plot_messenger_doppler_data(doppler_df, DSN_STATIONS)

    #print("4. ЗАГРУЗКА ЭФЕМЕРИД HORIZONS...")
    
    ephemeris_files = {
                'sun': 'horizons_results_sun.txt',
                'earth': 'horizons_results_earth.txt',
                'venus': 'horizons_results_venus.txt',
                'mercury': 'horizons_results_mercury.txt',
                'jupiter': 'horizons_results_jupiter.txt',
                'mars': 'horizons_results_mars.txt',
                'messenger': 'horizons_results_messenger.txt'
            }

    body_interpolators = {}
    body_vel_interpolators = {}
    body_times = {}
    gms_data = {}
    radius_data = {}

    for body_name, filename in ephemeris_files.items():
        try:
            times, positions, velocities, gms_data[body_name], radius_data[body_name] = load_horizons_data(filename)

            pos_interp, vel_interp = create_interpolators(times, positions, velocities)
            
            body_interpolators[body_name] = pos_interp
            body_vel_interpolators[body_name] = vel_interp
            body_times[body_name] = times
            
        except Exception as e:
            print(f"Ошибка загрузки {body_name}: {e}")

    '''
    # Шаг 5: Определение временного интервала
    print("5. ОПРЕДЕЛЕНИЕ ВРЕМЕННОГО ИНТЕРВАЛА...")
    
    # Используем время из данных Doppler
    min_time_utc = doppler_df['time_utc'].min()
    max_time_utc = doppler_df['time_utc'].max()
    
    # Преобразование в TDB
    min_time_tdb = (min_time_utc.to_julian_date() - 2451545.0) * 86400.0
    max_time_tdb = (max_time_utc.to_julian_date() - 2451545.0) * 86400.0
    
    # Добавляем запас по времени
    t_span = [min_time_tdb, max_time_tdb]
    t_eval = np.linspace(t_span[0], t_span[1], 100000)
    
    print(f"Временной интервал данных Doppler: {min_time_utc} - {max_time_utc}")
    print(f"Интервал интегрирования: {t_span[0]:.0f} - {t_span[1]:.0f} s TDB")
    print(f"Длительность: {(t_span[1]-t_span[0])/3600:.1f} часов")
    
    # Шаг 6: Получение начальных условий и интегрирование
    print("6. ИНТЕГРИРОВАНИЕ ОРБИТЫ MESSENGER...")
    
    # Используем среднее время как начальное
    avg_time_utc = min_time_utc + (max_time_utc - min_time_utc) / 2
    avg_time_tdb = (avg_time_utc.to_julian_date() - 2451545.0) * 86400.0
    
    
    initial_state = get_initial_conditions_from_horizons(
        avg_time_tdb,
        body_interpolators['mercury'],
        body_vel_interpolators['mercury']
    )
    
    
    orbit_result = integrate_messenger_orbit(
        t_span, t_eval, initial_state, body_interpolators, gms_data, radius_data
    )
    
    if not orbit_result['success']:
        print("Ошибка интегрирования орбиты. Выход.")
        return
    
    # Создание интерполяторов для позиции и скорости КА
    sc_pos_interp, sc_vel_interp = create_interpolators(orbit_result['times'], orbit_result['positions'], orbit_result['velocities'])'''
    
    sc_pos_interp = body_interpolators['messenger']
    sc_vel_interp = body_vel_interpolators['messenger']
    
    print("7. ВЫЧИСЛЕНИЕ THEORETICAL DOPPLER С LIGHT-TIME КОРРЕКЦИЕЙ...")
    
    gms_data.pop('messenger')

    ramp_table = prepare_ramp_table_from_data(doppler_df)

    results_df = process_doppler_with_exact_model(
        doppler_df,
        body_interpolators,
        body_vel_interpolators,
        sc_pos_interp,
        sc_vel_interp,
        gms_data,
        ramp_table
    )

    print("8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")

    print(f"Столбцы в results_df: {list(results_df.columns)}")
    
    results_columns = ['time_utc', 'station_id', 'theoretical_doppler_hz', 'light_time_s', 'range_km', 'range_rate_kms', 'doppler_residual_hz', 'measured_doppler_hz']
    
    available_columns = [col for col in results_columns if col in results_df.columns]
    print(f"Доступные столбцы для объединения: {available_columns}")

    final_df = pd.merge(
        doppler_df,
        results_df[available_columns],
        on=['time_utc', 'station_id'],
        how='inner'
    )
    
    output_file = 'messenger_doppler_final_results.csv'
    #final_df.to_csv(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")
    
    print("9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    
    plot_doppler_comparison(final_df, DSN_STATIONS)
    
    print("СВОДНАЯ СТАТИСТИКА:")
    print(f"Всего измерений: {len(final_df)}")
    
    if 'light_time_s' in final_df.columns:
        print(f"Средний light-time: {final_df['light_time_s'].mean():.1f} ± {final_df['light_time_s'].std():.1f} s")
    
    if 'range_km' in final_df.columns:
        print(f"Средняя дальность: {final_df['range_km'].mean()/1e6:.3f} ± {final_df['range_km'].std()/1e6:.3f} млн км")
    
    if 'doppler_residual_hz' in final_df.columns:
        print(f"Средний residual Doppler: {final_df['doppler_residual_hz'].mean():.1f} ± {final_df['doppler_residual_hz'].std():.1f} Hz")
    
    return final_df

if __name__ == "__main__":
    final_results = main()
    