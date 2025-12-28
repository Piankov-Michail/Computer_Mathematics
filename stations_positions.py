import numpy as np

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