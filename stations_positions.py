from main import DSN_STATIONS
import numpy as np

def geodetic_to_geocentric(station_id, t_utc=None):
    station = DSN_STATIONS[station_id]

    X = station['x'] / 1000.0
    Y = station['y'] / 1000.0
    Z = station['z'] / 1000.0
    
    return np.array([X, Y, Z])

def compute_earth_rotation_matrix(t_tdb):
    jd = 2451545.0 + t_tdb / 86400.0
    T = (jd - 2451545.0) / 36525.0

    gmst_sec = (67310.54841 + (8640184.812866 + (0.093104 - 6.2e-6 * T) * T) * T)
    gmst_rad = np.deg2rad((gmst_sec / 240.0) % 360.0)

    cos_g = np.cos(gmst_rad)
    sin_g = np.sin(gmst_rad)
    return np.array([[cos_g, -sin_g, 0],[sin_g,  cos_g, 0], [    0,       0, 1]])

def get_geocentric_station_pos(station_id, t_tdb):
    geo = geodetic_to_geocentric(station_id)
    R   = compute_earth_rotation_matrix(t_tdb)
    return R @ geo

def get_station_barycentric_pos(t_tdb, station_id, body_interpolators):
    earth_bary = body_interpolators['earth'](t_tdb)
    station_geo = get_geocentric_station_pos(station_id, t_tdb)
    return earth_bary + station_geo

def get_station_velocity(t_tdb, station_id, body_vel_interpolators):
    """
    Правильное вычисление скорости станции в барицентрической системе
    """
    # 1. Скорость Земли относительно барицентра
    v_earth = body_vel_interpolators['earth'](t_tdb) / 86400.0  # км/с
    
    # 2. Геоцентрическая позиция станции
    r_geo = geodetic_to_geocentric(station_id)
    
    # 3. Угловая скорость вращения Земли (рад/с)
    omega = 7.2921151467e-5  # рад/с
    
    # 4. Матрица вращения Земли
    R = compute_earth_rotation_matrix(t_tdb)
    
    # 5. Позиция станции в инерциальной системе
    r_geo_inertial = R @ r_geo
    
    # 6. Скорость вращения: v = ω × r
    omega_vec = np.array([0, 0, omega])  # ось вращения Z
    v_rot_inertial = np.cross(omega_vec, r_geo_inertial)
    
    # 7. Полная скорость станции = скорость Земли + скорость вращения
    v_total = v_earth + v_rot_inertial
    
    return v_total