from typing import List, Dict
import numpy as np
from scipy.integrate import solve_ivp
from horizons_parser import load_horizons_data

J2_mercury = 0.00006

file = 'horizons_results_mercury.txt'

def integrate_messenger_orbit(t_span: List[float], t_eval: np.ndarray, initial_state: np.ndarray, body_interpolators, gms_data, radius_data) -> Dict:    
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
        print(f"Ошибка при интегрировании: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'message': str(e)}
    
def get_initial_conditions_from_horizons(t0_tdb: float, mercury_pos_interp, mercury_vel_interp) -> np.ndarray:

    _, _, _, GM_mercury, R_mercury = load_horizons_data(file)

    mercury_pos = mercury_pos_interp(t0_tdb)
    mercury_vel = mercury_vel_interp(t0_tdb)
    
    peri_altitude = 300
    r_peri = peri_altitude + R_mercury
    
    period_hours = 12
    period_sec = period_hours * 3600
    a = (GM_mercury * (period_sec / (2 * np.pi))**2)**(1/3)
    e = 1 - r_peri / a
    r_apo = a * (1 + e)
    apo_altitude = r_apo - R_mercury
    
    orbital_speed = np.sqrt(GM_mercury * (2 / r_peri - 1 / a))
    
    r_sc_mercury = np.array([r_peri, 0, 0])
    r_sc = mercury_pos + r_sc_mercury
    
    v_sc_mercury = np.array([0, orbital_speed, 0])

    incl = np.deg2rad(83.0)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(incl), -np.sin(incl)],
        [0, np.sin(incl), np.cos(incl)]
    ])
    v_sc_mercury = rotation_matrix @ v_sc_mercury

    v_sc = mercury_vel + v_sc_mercury
    
    '''
    print("НАЧАЛЬНЫЕ УСЛОВИЯ MESSENGER")
    print(f"Время TDB: {t0_tdb:.2f} s от J2000")
    print(f"Позиция Меркурия: [{mercury_pos[0]:.2f}, {mercury_pos[1]:.2f}, {mercury_pos[2]:.2f}] км")
    print(f"Скорость Меркурия: [{mercury_vel[0]:.6f}, {mercury_vel[1]:.6f}, {mercury_vel[2]:.6f}] км/с")
    print(f"Позиция КА: [{r_sc[0]:.2f}, {r_sc[1]:.2f}, {r_sc[2]:.2f}] км")
    print(f"Скорость КА: [{v_sc[0]:.6f}, {v_sc[1]:.6f}, {v_sc[2]:.6f}] км/с")
    print(f"Параметры орбиты вокруг Меркурия:")
    print(f"Высота перицентра: {peri_altitude:.2f} км")
    print(f"Высота апоцентра: {apo_altitude:.2f} км")
    print(f"Большая полуось: {a:.2f} км")
    print(f"Эксцентриситет: {e:.6f}")
    print(f"Орбитальный период: {period_hours:.2f} часов")
    print(f"Скорость в перицентре: {orbital_speed:.6f} км/с")
    print(f"Наклонение: 83°")'''
    
    return np.concatenate([r_sc, v_sc])

def equations_of_motion_corrected(t: float, state: np.ndarray, mercury_pos_interp, sun_pos_interp, earth_pos_interp, venus_pos_interp, \
                                  GM_sun, GM_mercury, GM_earth, GM_venus, R_mercury) -> np.ndarray:
    r_sc = state[0:3]
    v_sc = state[3:6]
    
    r_mercury = mercury_pos_interp(t)
    r_sun = sun_pos_interp(t)
    r_earth = earth_pos_interp(t)

    r_sc_sun = r_sun - r_sc
    r_sc_mercury = r_mercury - r_sc
    r_sc_earth = r_earth - r_sc

    norm_sun = max(np.linalg.norm(r_sc_sun), 1)
    norm_mercury = max(np.linalg.norm(r_sc_mercury), 1)
    norm_earth = max(np.linalg.norm(r_sc_earth), 1)

    a_sun = GM_sun / norm_sun**3 * r_sc_sun
    
    a_mercury_central = GM_mercury / norm_mercury**3 * r_sc_mercury
    
    if norm_mercury > R_mercury:
        unit_merc = r_sc_mercury / norm_mercury
        z_over_r = unit_merc[2]
        j2_factor = (3 / 2) * GM_mercury * J2_mercury * R_mercury**2 / norm_mercury**4
        a_mercury_j2 = j2_factor * np.array([
            unit_merc[0] * (5 * z_over_r**2 - 1),
            unit_merc[1] * (5 * z_over_r**2 - 1),
            unit_merc[2] * (5 * z_over_r**2 - 3)
        ])
    else:
        a_mercury_j2 = np.zeros(3)
    
    a_mercury = a_mercury_central + a_mercury_j2
     
    a_earth = GM_earth / norm_earth**3 * r_sc_earth
    
    a_venus = np.zeros(3)
    if venus_pos_interp is not None:
        r_venus = venus_pos_interp(t)
        r_sc_venus = r_venus - r_sc
        norm_venus = max(np.linalg.norm(r_sc_venus), 1)
        a_venus = GM_venus / norm_venus**3 * r_sc_venus
    
    total_acceleration = a_sun + a_mercury + a_earth + a_venus
    
    return np.concatenate([v_sc, total_acceleration])
