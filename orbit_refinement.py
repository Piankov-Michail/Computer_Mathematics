import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from ls_manual import ManualLeastSquares
from lt_tw_doppler import compute_two_way_doppler_ramped
from messener_orbit import integrate_messenger_orbit
from lt_tw_doppler import utc_to_tdb_seconds, compute_two_way_doppler_ramped
from interpolators import create_interpolators
warnings.filterwarnings('ignore')
import pandas as pd
import glob
import os
from main import DSN_STATIONS

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

class OrbitRefinementLSQ:
    
    def __init__(self, doppler_df, body_interpolators, body_vel_interpolators, 
                 gms_data, ramp_table, dsn_stations, radius_data):
        self.doppler_df = doppler_df
        self.body_interpolators = body_interpolators
        self.body_vel_interpolators = body_vel_interpolators
        self.gms_data = gms_data
        self.ramp_table = ramp_table
        self.dsn_stations = dsn_stations
        self.radius_data = radius_data
        
        self.c = 299792.458  
        self.GMsun = gms_data.get('sun', 1.32712440018e11)  
        
        self.n_state_params = 6  
        self.n_params = self.n_state_params
        
        self.integration_method = 'DOP853'
        self.rtol = 1e-10
        self.atol = 1e-10
        
    def compute_residuals(self, params, return_components=False):
        initial_state = params[:self.n_state_params]
        t_min, t_max = float('inf'), float('-inf')
        sample_size = min(2000, len(self.doppler_df))
        sample_df = self.doppler_df.sample(sample_size, random_state=42) if len(self.doppler_df) > 2000 else self.doppler_df
        for utc_time in sample_df['time_utc']:
            t_tdb = utc_to_tdb_seconds(utc_time)
            t_min = min(t_min, t_tdb)
            t_max = max(t_max, t_tdb)
        min_time_utc = safe_parse_time(self.doppler_df['time_utc'].min())
        max_time_utc = safe_parse_time(self.doppler_df['time_utc'].max())
        min_time_tdb = (min_time_utc.to_julian_date() - 2451545.0) * 86400.0
        max_time_tdb = (max_time_utc.to_julian_date() - 2451545.0) * 86400.0
        
        t_span = [min_time_tdb, max_time_tdb]
        t_eval = np.linspace(t_span[0], t_span[1], 100000)
        
        try:
            orbit_result = integrate_messenger_orbit(
                t_span=t_span,
                t_eval=t_eval,
                initial_state=initial_state,
                body_interpolators=self.body_interpolators,
                gms_data=self.gms_data,
                radius_data=self.radius_data  
            )
            
            if not orbit_result['success']:
                print(f"Ошибка интегрирования: {orbit_result.get('message', 'Unknown')}")
                return np.ones(len(sample_df)) * 1e6
            
            times = orbit_result['times']
            positions = orbit_result['positions']  
            velocities = orbit_result['velocities']

            sc_pos_interp, sc_vel_interp = create_interpolators(orbit_result['times'], orbit_result['positions'], orbit_result['velocities'])
            
            residuals = []
            
            for idx, row in sample_df.iterrows():
                try:
                    theoretical_result = compute_two_way_doppler_ramped(
                        t_receive_utc=row['time_utc'],
                        Tc=row['compression_time_s'],
                        station_id=int(row['station_id']),
                        uplink_band=int(row.get('uplink_band', 2)),
                        downlink_band=int(row.get('downlink_band', 2)),
                        ramp_table=self.ramp_table,
                        body_interpolators=self.body_interpolators,
                        body_vel_interpolators=self.body_vel_interpolators,
                        sc_pos_interp=sc_pos_interp, 
                        sc_vel_interp=sc_vel_interp, 
                        GM_params=self.gms_data,
                        row_data=row.to_dict()
                    )
                    
                    theoretical_doppler = theoretical_result['theoretical_doppler_hz']
                    measured_doppler = row['observable_hz']
                    residual = measured_doppler - theoretical_doppler
                    residuals.append(residual)
                    
                except Exception as e:
                    print(f"Ошибка в расчёте доплера для точки {idx}: {e}")
                    residuals.append(1e6)
            
            residuals_array = np.array(residuals)
            self.sample_df = sample_df  
            
            return residuals_array
            
        except Exception as e:
            print(f"Критическая ошибка в compute_residuals: {e}")
            import traceback
            traceback.print_exc()
            return np.ones(len(sample_df)) * 1e6

    def solve_manual_lsq(self, initial_params, bounds=None, 
                        max_iter=50, verbose=True):
        
        solver = ManualLeastSquares(
            residual_func=self.compute_residuals,
            initial_params=initial_params,
            bounds=bounds
        )
        
        result = solver.optimize(
            max_iter=max_iter,
            cost_tol=1e-6,
            param_tol=1e-8,
            grad_tol=1e-6,
            verbose=verbose
        )
        
        final_residuals = self.compute_residuals(result['params'])
        rms = np.sqrt(np.mean(final_residuals**2))
        
        print(f"\nФинальный RMS: {rms:.3f} Гц")
        
        cov_matrix = solver.compute_covariance_matrix(result['params'])
        param_errors = np.sqrt(np.diag(cov_matrix))
        
        return {
            'success': True,
            'params': result['params'],
            'residuals': final_residuals,
            'rms': rms,
            'cov_matrix': cov_matrix,
            'param_errors': param_errors,
            'iterations': result['iterations'],
            'history': result['history']
        }
    
    def analyze_residuals(self, residuals):
        residuals = np.array(residuals)
        
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'rms': np.sqrt(np.mean(residuals**2)),
            'max_abs': np.max(np.abs(residuals)),
            'min_abs': np.min(np.abs(residuals)),
            'median': np.median(residuals)
        }
        
        return analysis