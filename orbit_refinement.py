import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from ls_manual import ManualLeastSquares
warnings.filterwarnings('ignore')

class OrbitRefinementLSQ:
    
    def __init__(self, doppler_df, body_interpolators, body_vel_interpolators, 
                 gms_data, ramp_table, dsn_stations):
        self.doppler_df = doppler_df
        self.body_interpolators = body_interpolators
        self.body_vel_interpolators = body_vel_interpolators
        self.gms_data = gms_data
        self.ramp_table = ramp_table
        self.dsn_stations = dsn_stations
        
        self.c = 299792.458  
        self.GMsun = gms_data.get('sun', 1.32712440018e11)  
        
        self.n_state_params = 6  
        self.n_params = self.n_state_params
        
        self.integration_method = 'DOP853'
        self.rtol = 1e-10
        self.atol = 1e-10
        
    def equations_of_motion(self, t, state):
        r = state[:3]
        r_norm = np.linalg.norm(r)
        
        acc = -self.GMsun * r / r_norm**3
        
        for body in ['mercury', 'venus', 'earth', 'mars', 'jupiter']:
            if body in self.body_interpolators:
                try:
                    r_body = self.body_interpolators[body](t)
                    r_to_body = r_body - r
                    r_to_body_norm = np.linalg.norm(r_to_body)
                    
                    GM_body = self.gms_data.get(body, 0)
                    
                    if GM_body > 0 and r_to_body_norm > 0:
                        acc_direct = GM_body * r_to_body / r_to_body_norm**3
                        
                        r_body_norm = np.linalg.norm(r_body)
                        acc_indirect = -GM_body * r_body / r_body_norm**3
                        
                        acc += acc_direct + acc_indirect
                except:
                    continue
        
        return np.concatenate([state[3:], acc])
    
    def propagate_orbit(self, initial_state, t_span, t_eval=None):
        try:
            if t_eval is None:
                t_eval = np.linspace(t_span[0], t_span[1], 20000)
            
            sol = solve_ivp(
                self.equations_of_motion,
                t_span,
                initial_state,
                t_eval=t_eval,
                method=self.integration_method,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=True
            )
            
            if sol.success:
                from scipy.interpolate import interp1d
                
                pos_interp = interp1d(
                    sol.t, sol.y[:3, :], 
                    kind='cubic', 
                    axis=1,
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                vel_interp = interp1d(
                    sol.t, sol.y[3:, :],
                    kind='cubic',
                    axis=1,
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                return {
                    'success': True,
                    'times': sol.t,
                    'positions': sol.y[:3, :].T,
                    'velocities': sol.y[3:, :].T,
                    'pos_interp': lambda t: pos_interp(t).T,
                    'vel_interp': lambda t: vel_interp(t).T,
                    'solution': sol
                }
            else:
                return {'success': False, 'message': sol.message}
                
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def compute_residuals(self, params, return_components=False):
        from lt_tw_doppler import utc_to_tdb_seconds
        
        initial_state = params[:self.n_state_params]
        
        t_min = float('inf')
        t_max = float('-inf')
        
        sample_size = min(2000, len(self.doppler_df))
        sample_df = self.doppler_df.sample(sample_size, random_state=42) if len(self.doppler_df) > 2000 else self.doppler_df
        
        for utc_time in sample_df['time_utc']:
            t_tdb = utc_to_tdb_seconds(utc_time)
            t_min = min(t_min, t_tdb)
            t_max = max(t_max, t_tdb)
        
        margin = 1800.0  
        t_span = (t_min - margin, t_max + margin)
        
        orbit_result = self.propagate_orbit(initial_state, t_span)
        
        if not orbit_result['success']:
            return np.ones(len(sample_df)) * 1e6
        
        sc_pos_interp = orbit_result['pos_interp']
        sc_vel_interp = orbit_result['vel_interp']
        
        residuals = []
        components = []
        
        from lt_tw_doppler import compute_two_way_doppler_ramped
        
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
                residuals.append(1e6)
        
        residuals_array = np.array(residuals)
        
        self.sample_df = sample_df
        
        if return_components:
            return residuals_array, components
        else:
            return residuals_array
    
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