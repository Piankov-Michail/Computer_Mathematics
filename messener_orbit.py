from typing import List, Dict, Callable, Optional, Tuple
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

def rk4_integrate(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
    max_step: float = 10.0,
    rtol: float = 1e-6,
    atol: float = 1e-9
) -> Dict[str, np.ndarray]:
    
    t0, tf = t_span
    y0 = np.asarray(y0, dtype=float)
    n_dim = y0.size
    
    if t_eval is None:
        num_points = int((tf - t0) / max_step) + 1
        t_eval = np.linspace(t0, tf, num_points)
    else:
        t_eval = np.sort(np.asarray(t_eval, dtype=float))
        if t_eval[0] > t0:
            t_eval = np.insert(t_eval, 0, t0)
        if t_eval[-1] < tf:
            t_eval = np.append(t_eval, tf)
    
    n_points = len(t_eval)
    y_result = np.zeros((n_dim, n_points))
    t_result = t_eval.copy()
    
    y_current = y0.copy()
    t_current = t0
    idx = 0
    
    if np.isclose(t_current, t_eval[0], atol=1e-12):
        y_result[:, idx] = y_current
        idx += 1
    
    while t_current < tf and idx < n_points:
        t_target = t_eval[idx]
        
        h = min(t_target - t_current, max_step)

        if h < 1e-12:
            y_result[:, idx] = y_current
            idx += 1
            continue

        k1 = fun(t_current, y_current)
        
        y_mid1 = y_current + h * k1 / 2.0
        k2 = fun(t_current + h/2.0, y_mid1)
        
        y_mid2 = y_current + h * k2 / 2.0
        k3 = fun(t_current + h/2.0, y_mid2)
        
        y_end = y_current + h * k3
        k4 = fun(t_current + h, y_end)
        
        y_next = y_current + h / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        
        t_current += h
        y_current = y_next

        while idx < n_points and t_eval[idx] <= t_current + 1e-12:
            if np.isclose(t_eval[idx], t_current, atol=1e-12):
                y_result[:, idx] = y_current
            else:
                if idx > 0:
                    t_prev = t_eval[idx-1]
                    y_prev = y_result[:, idx-1]
                    ratio = (t_eval[idx] - t_prev) / (t_current - t_prev)
                    y_result[:, idx] = y_prev + ratio * (y_current - y_prev)
                else:
                    y_result[:, idx] = y_current
            idx += 1

    while idx < n_points:
        y_result[:, idx] = y_current
        idx += 1
    
    return {
        't': t_result,
        'y': y_result,
        'success': True,
        'message': 'Интегрирование успешно завершено'
    }

def integrate_messenger_orbit(
    t_span: List[float],
    t_eval: np.ndarray,
    initial_state: np.ndarray,
    body_interpolators: dict,
    gms_data: dict,
    radius_data: dict
) -> Dict:

    try:
        solution = rk4_integrate(
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
            t_eval=t_eval,
            max_step=50,
            rtol=1e-6,
            atol=1e-9
        )
        
        if solution['success']:
            return {
                'times': solution['t'],
                'positions': solution['y'][:3].T,
                'velocities': solution['y'][3:6].T,
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