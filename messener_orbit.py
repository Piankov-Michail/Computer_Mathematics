from typing import List, Dict
import numpy as np
from scipy.integrate import solve_ivp
from horizons_parser import load_horizons_data
from interpolators import create_interpolators

J2_mercury = 0.00006

J2000_JD = 2451545.0

def jd_to_seconds_since_j2000(jd: float) -> float:
    days_since_j2000 = jd - J2000_JD
    return days_since_j2000 * 86400.0

def create_initial_state_from_horizons_data(new_time) -> tuple:

    sc_times, sc_positions, sc_velocities, _, _ = load_horizons_data('horizons_results_messenger.txt')
    sc_position_interpolator, sc_velocity_interpolator = create_interpolators(sc_times, sc_positions, sc_velocities)

    sc_new_position = sc_position_interpolator(new_time)
    sc_new_velocity = sc_velocity_interpolator(new_time)

    initial_state = np.concatenate([sc_new_position, sc_new_velocity])
    
    return initial_state

def integrate_messenger_orbit(
    t_span: List[float],
    t_eval: np.ndarray,
    initial_state: np.ndarray,
    body_interpolators: dict,
    gms_data: dict,
    radius_data: dict
) -> Dict:

    try:
        solution = solve_ivp(
            fun=lambda t, state: equations_of_motion_corrected(
                t, state,
                body_interpolators['mercury'],
                body_interpolators['sun'],
                body_interpolators['earth'],
                body_interpolators['venus'],
                gms_data['sun'],
                gms_data['mercury'],
                gms_data['earth'],
                gms_data['venus'],
                radius_data['mercury']
            ),
            t_span=t_span,
            y0=initial_state,
            method='DOP853',
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9
        )
        
        if solution.success:
            return {
                'times': solution.t,
                'positions': solution.y[:3].T,
                'velocities': solution.y[3:6].T,
                'success': True
            }
        else:
            return {'success': False, 'message': solution.message}
            
    except Exception as e:
        print(f"Integration error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e)}

def equations_of_motion_corrected(
    t: float,
    state: np.ndarray,
    mercury_pos_interp, sun_pos_interp, earth_pos_interp, venus_pos_interp,
    GM_sun: float, GM_mercury: float, GM_earth: float, GM_venus: float, R_mercury: float) -> np.ndarray:
    
    r_sc = state[0:3]
    v_sc = state[3:6]
    
    r_mercury = mercury_pos_interp(t)
    r_sun = sun_pos_interp(t)
    r_earth = earth_pos_interp(t)
    r_venus = venus_pos_interp(t) if venus_pos_interp is not None else None

    r_sc_sun = r_sun - r_sc
    r_sc_mercury = r_mercury - r_sc
    r_sc_earth = r_earth - r_sc
    r_sc_venus = r_venus - r_sc if r_venus is not None else None

    norm_sun = np.linalg.norm(r_sc_sun)
    norm_mercury = np.linalg.norm(r_sc_mercury)
    norm_earth = np.linalg.norm(r_sc_earth)
    norm_venus = np.linalg.norm(r_sc_venus) if r_venus is not None else 1.0

    norm_sun = max(norm_sun, 1.0)
    norm_mercury = max(norm_mercury, 1.0)
    norm_earth = max(norm_earth, 1.0)
    norm_venus = max(norm_venus, 1.0) if r_venus is not None else 1.0

    a_sun = GM_sun * r_sc_sun / norm_sun**3
    a_mercury_central = GM_mercury * r_sc_mercury / norm_mercury**3
    a_earth = GM_earth * r_sc_earth / norm_earth**3
    a_venus = GM_venus * r_sc_venus / norm_venus**3 if r_venus is not None else np.zeros(3)

    a_mercury_j2 = np.zeros(3)
    if norm_mercury > R_mercury:
        r_vec = r_sc_mercury
        r_norm = norm_mercury
        z = r_vec[2]
        
        factor = (3/2) * GM_mercury * J2_mercury * (R_mercury**2) / (r_norm**5)
        a_mercury_j2[0] = factor * r_vec[0] * (5*(z/r_norm)**2 - 1)
        a_mercury_j2[1] = factor * r_vec[1] * (5*(z/r_norm)**2 - 1)
        a_mercury_j2[2] = factor * (5*(z/r_norm)**2 - 3) * z

    total_acceleration = (
        a_sun + 
        a_mercury_central + a_mercury_j2 +
        a_earth +
        a_venus
    )
    
    return np.concatenate([v_sc, total_acceleration])