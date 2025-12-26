import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from astropy.time import Time
import astropy.units as u
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from horizons_parser import *
from interpolators import *
from messener_orbit import *
from create_graphs import *
from parse_csv import *
from stations_positions import *

import logging
import traceback

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='doppler_debug.log',
    filemode='w'
)
logger = logging.getLogger('lt_tw_doppler')

c = 299792.458
MESSENGER_XBAND_HIGH_FREQ = 8445.734e6
RAMPED_DOPPLER_DATA_TYPE = 12

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
    """Integrate frequency over time interval with detailed logging"""
    logger.debug(f"Integrating frequency for station {station_id} from {t_start:.6f} to {t_end:.6f} seconds")
    W = t_end - t_start
    logger.debug(f"Integration window width: {W:.6f} seconds")
    
    station_ramps = [r for r in ramp_table if r['station_id'] == station_id]
    logger.debug(f"Found {len(station_ramps)} ramps for station {station_id}")
    
    if not station_ramps:
        logger.warning("No ramps found - using fallback frequency")
        base_freq = 7165.0e6
        ramp_rate = 0.0
        if row_data and 'ramp_rate_hz_s' in row_data and pd.notna(row_data['ramp_rate_hz_s']):
            ramp_rate = float(row_data['ramp_rate_hz_s'])
            logger.debug(f"Using ramp rate from row data: {ramp_rate:.3f} Hz/s")
        
        result = base_freq * W + 0.5 * ramp_rate * W**2
        logger.debug(f"Fallback integration result: {result:.3f} Hz·s")
        return result
    
    # Sort ramps by start time
    station_ramps.sort(key=lambda x: x['start_time'])
    logger.debug("Sorted station ramps by start time")
    
    # Find relevant ramps within integration window
    ramps_to_use = []
    current_time = t_start
    
    for ramp in station_ramps:
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
        
        # Adjust start frequency if segment starts after ramp start
        if segment['start_time'] > ramp['start_time']:
            time_offset = segment['start_time'] - ramp['start_time']
            segment['start_freq'] += segment['ramp_rate'] * time_offset
            logger.debug(f"Adjusted start frequency for partial ramp: {segment['start_freq']/1e6:.6f} MHz")
        
        ramps_to_use.append(segment)
        current_time = segment['end_time']
        
        if current_time >= t_end:
            break
    
    logger.debug(f"Using {len(ramps_to_use)} ramp segments for integration")
    
    if not ramps_to_use:
        logger.warning("No ramps fully within window - using closest ramp")
        closest_ramp = min(station_ramps, key=lambda r: abs((r['start_time'] + r['end_time'])/2 - (t_start + t_end)/2))
        ramp_rate = closest_ramp['ramp_rate']
        t_mid = (t_start + t_end) / 2
        f_mid = closest_ramp['start_freq'] + ramp_rate * (t_mid - closest_ramp['start_time'])
        result = f_mid * W + 0.5 * ramp_rate * W**2
        logger.debug(f"Closest ramp integration: {result:.3f} Hz·s")
        return result

    # Calculate segment widths
    Wi_list = []
    for i in range(len(ramps_to_use) - 1):
        segment = ramps_to_use[i]
        Wi = segment['end_time'] - segment['start_time']
        Wi_list.append(Wi)
        logger.debug(f"Segment {i} width: {Wi:.6f} s")

    if len(ramps_to_use) > 1:
        total_width = sum(Wi_list)
        Wn = W - total_width
        Wi_list.append(Wn)
        logger.debug(f"Final segment width: {Wn:.6f} s")
    else:
        Wn = W
        Wi_list = [Wn]
        logger.debug(f"Single segment width: {Wn:.6f} s")

    # Perform integration
    integral = 0.0
    for i, segment in enumerate(ramps_to_use):
        if i >= len(Wi_list) or Wi_list[i] <= 0:
            continue
        
        Wi = Wi_list[i]
        f0_segment = segment['start_freq']
        f_dot_segment = segment['ramp_rate']
        
        # Frequency at segment midpoint
        fi = f0_segment + 0.5 * f_dot_segment * Wi
        
        segment_integral = fi * Wi
        integral += segment_integral
        
        logger.debug(f"Segment {i}: start_freq={f0_segment/1e6:.6f} MHz, "
                    f"ramp_rate={f_dot_segment:.3f} Hz/s, width={Wi:.6f}s, "
                    f"integral={segment_integral:.3f} Hz·s")
    
    total_integral = integral
    logger.debug(f"Total integrated frequency: {total_integral:.3f} Hz·s")
    return total_integral

def compute_doppler_reference_frequency(f_T, uplink_band, downlink_band, f_type='transmitter'):
    M2, M2R = get_transponder_ratios(uplink_band, downlink_band)
    
    if f_type == 'transmitter':
        return M2 * f_T
    elif f_type == 'receiver':
        return M2R * f_T
    else:
        raise ValueError(f"Неизвестный тип частоты: {f_type}")

def compute_two_way_doppler_ramped(t_receive_utc, Tc, station_id, uplink_band, downlink_band, 
                                  ramp_table, body_interpolators, body_vel_interpolators, 
                                  sc_pos_interp, sc_vel_interp, GM_params, row_data=None):
    """Compute two-way Doppler with ramped frequencies and detailed logging"""
    logger.info(f"Computing two-way Doppler for station {station_id} at {t_receive_utc}")
    logger.debug(f"Parameters: Tc={Tc}, uplink_band={uplink_band}, downlink_band={downlink_band}")
    
    try:
        M2, M2R = get_transponder_ratios(uplink_band, downlink_band)
        logger.debug(f"Transponder ratios: M2={M2:.6f}, M2R={M2R:.6f}")
        
        # Log ramp table status
        logger.debug(f"Ramp table contains {len(ramp_table)} entries for station {station_id}")
        station_ramps = [r for r in ramp_table if r['station_id'] == station_id]
        if station_ramps:
            logger.debug(f"Found {len(station_ramps)} relevant ramps for station {station_id}")
            for ramp in station_ramps:
                logger.debug(f"Ramp: start={ramp['start_time']:.2f}, end={ramp['end_time']:.2f}, "
                            f"freq={ramp['start_freq']/1e6:.3f} MHz, rate={ramp['ramp_rate']:.3f} Hz/s")
        else:
            logger.warning(f"No ramps found for station {station_id}, using fallback values")
        
        # Solve light time problem
        logger.debug("Solving two-way light time problem...")
        light_time_solution = solve_two_way_light_time_with_intervals(
            t_receive_utc, Tc, station_id,
            body_interpolators, body_vel_interpolators,
            sc_pos_interp, sc_vel_interp, GM_params
        )
        
        # Log light time solution details
        logger.debug(f"Light time solution computed:")
        logger.debug(f"  T1 interval: [{light_time_solution['t1_start_tdb']:.6f}, {light_time_solution['t1_end_tdb']:.6f}]")
        logger.debug(f"  T3 interval: [{light_time_solution['t3_start_tdb']:.6f}, {light_time_solution['t3_end_tdb']:.6f}]")
        logger.debug(f"  Total light time: {light_time_solution['total_light_time']:.6f} s")
        logger.debug(f"  Distance: {light_time_solution['distance_km']:.3f} km")
        logger.debug(f"  Range rate: {light_time_solution['range_rate_kms']:.6f} km/s")
        
        # Compute frequency integrals
        logger.debug("Calculating frequency integrals for T1 interval...")
        integral_fT_t1 = integrate_ramped_frequency(
            light_time_solution['t1_start_tdb'], 
            light_time_solution['t1_end_tdb'], 
            ramp_table, station_id, row_data
        )
        avg_fT_t1 = integral_fT_t1 / (light_time_solution['t1_end_tdb'] - light_time_solution['t1_start_tdb'])
        logger.debug(f"T1 integral: {integral_fT_t1:.3f} Hz·s, avg freq: {avg_fT_t1/1e6:.6f} MHz")
        
        logger.debug("Calculating frequency integrals for T3 interval...")
        integral_fT_t3 = integrate_ramped_frequency(
            light_time_solution['t3_start_tdb'], 
            light_time_solution['t3_end_tdb'], 
            ramp_table, station_id, row_data
        )
        avg_fT_t3 = integral_fT_t3 / (light_time_solution['t3_end_tdb'] - light_time_solution['t3_start_tdb'])
        logger.debug(f"T3 integral: {integral_fT_t3:.3f} Hz·s, avg freq: {avg_fT_t3/1e6:.6f} MHz")
        
        # Calculate theoretical Doppler
        term1 = (M2R / Tc) * integral_fT_t3
        term2 = (M2 / Tc) * integral_fT_t1
        F_theory = term1 - term2
        logger.debug(f"Doppler terms: term1={term1:.6f}, term2={term2:.6f}, F_theory={F_theory:.6f} Hz")
        
        result = {
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
            't1_start_tdb': light_time_solution['t1_start_tdb'],
            't1_end_tdb': light_time_solution['t1_end_tdb'],
            't3_start_tdb': light_time_solution['t3_start_tdb'],
            't3_end_tdb': light_time_solution['t3_end_tdb']
        }
        
        logger.info(f"Successfully computed Doppler: {F_theory:.6f} Hz")
        return result
    
    except Exception as e:
        logger.error(f"Doppler computation failed at {t_receive_utc} for station {station_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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

def process_doppler_with_exact_model(doppler_df, body_interpolators, body_vel_interpolators, 
                                     sc_pos_interp, sc_vel_interp, GM_params, ramp_table):
    """Process Doppler data with detailed logging"""
    logger.info(f"Processing {len(doppler_df)} Doppler measurements")
    logger.debug(f"GM parameters: {GM_params}")
    
    results = []
    
    for idx, row in tqdm(doppler_df.iterrows(), total=doppler_df.shape[0], desc="Processing Doppler"):
        try:
            logger.info(f"Processing row {idx+1}/{len(doppler_df)} at {row['time_utc']}")
            logger.debug(f"Row data: {row.to_dict()}")
            
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
            
            logger.info(f"Measurement: {measured_doppler:.6f} Hz, "
                       f"Theoretical: {theoretical_doppler:.6f} Hz, "
                       f"Residual: {residual:.6f} Hz")
            
            result = {
                'time_utc': row['time_utc'],
                'station_id': row['station_id'],
                'measured_doppler_hz': float(measured_doppler),
                'theoretical_doppler_hz': float(theoretical_doppler),
                'doppler_residual_hz': float(residual),
                'light_time_s': float(theoretical_result.get('light_time_s', 0)),
                'distance_km': float(theoretical_result.get('distance_km', 0)),
                'range_rate_kms': float(theoretical_result.get('range_rate_kms', 0))
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to process row {idx}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    if not results:
        logger.error("No successful Doppler computations")
        return pd.DataFrame()
    
    logger.info(f"Successfully processed {len(results)}/{len(doppler_df)} measurements")
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

def solve_two_way_light_time_with_intervals(t_receive_utc, Tc, station_id, 
            body_interpolators, body_vel_interpolators, sc_pos_interp, sc_vel_interp, GM_params):
    """Solve two-way light time problem with detailed logging"""
    logger.debug(f"Solving light time for {t_receive_utc}, station {station_id}, Tc={Tc}")
    
    try:
        t3_tdb = utc_to_tdb_seconds(t_receive_utc)
        logger.debug(f"Receive time (TDB): {t3_tdb:.6f} seconds")
        
        t3_center = t3_tdb
        t3_start = t3_center - Tc/2
        t3_end = t3_center + Tc/2
        logger.debug(f"T3 interval: [{t3_start:.6f}, {t3_end:.6f}] seconds")
        
        # Get station position at receive time
        r_station_t3 = get_station_barycentric_pos(t3_center, station_id, body_interpolators)
        logger.debug(f"Station {station_id} position at T3: {r_station_t3} km")
        
        # Approximate spacecraft position
        r_sc_approx = sc_pos_interp(t3_center)
        logger.debug(f"Approx spacecraft position: {r_sc_approx} km")
        
        distance_approx = np.linalg.norm(r_sc_approx - r_station_t3)
        tau_D_approx = distance_approx / c
        logger.debug(f"Initial down-leg light time estimate: {tau_D_approx:.6f} s")
        
        # Down-leg iteration
        t2_approx = t3_center - tau_D_approx
        tau_D_prev = 0
        logger.debug("Starting down-leg light time iteration...")
        
        for iter_num in range(10):
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
            
            logger.debug(f"Iter {iter_num}: t2={t2_new:.6f}, tau_D={tau_D:.9f}, "
                        f"geometric={geometric_tau_D:.9f}, correction={delta_tau_D:.9f}")
            
            if abs(tau_D - tau_D_prev) < 1e-9:
                logger.debug(f"Down-leg converged in {iter_num} iterations")
                break
                
            tau_D_prev = tau_D
            t2_approx = t2_new
        else:
            logger.warning("Down-leg iteration reached max iterations without convergence")
        
        t2_center = t2_new
        r_sc_t2 = sc_pos_interp(t2_center)
        v_sc_t2 = sc_vel_interp(t2_center)
        logger.debug(f"Spacecraft state at T2: pos={r_sc_t2}, vel={v_sc_t2} km/s")
        
        # Up-leg iteration
        tau_U_approx = tau_D
        t1_approx = t2_center - tau_U_approx
        tau_U_prev = 0
        logger.debug("Starting up-leg light time iteration...")
        
        for iter_num in range(10):
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
            
            logger.debug(f"Iter {iter_num}: t1={t1_new:.6f}, tau_U={tau_U:.9f}, "
                        f"geometric={geometric_tau_U:.9f}, correction={delta_tau_U:.9f}")
            
            if abs(tau_U - tau_U_prev) < 1e-9:
                logger.debug(f"Up-leg converged in {iter_num} iterations")
                break
                
            tau_U_prev = tau_U
            t1_approx = t1_new
        else:
            logger.warning("Up-leg iteration reached max iterations without convergence")
        
        t1_center = t1_new
        t1_start = t1_center - Tc/2
        t1_end = t1_center + Tc/2
        
        # Calculate range rate
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
    
    except Exception as e:
        logger.error(traceback.format_exc())
        raise

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
    
    # Solar corona correction for up-leg
    if leg == 'up':
        solar_corona_correction = estimate_solar_corona_delay(t_emit, t_obs, r_emit, r_obs, body_interpolators)
        total_correction += solar_corona_correction
    
    # Atmospheric corrections
    troposphere_correction = 0.002  # seconds
    ionosphere_correction = 0.001  # seconds
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