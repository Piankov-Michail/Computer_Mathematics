# Примеры для теста

from horizons_parser import load_horizons_data
from interpolators import create_interpolators
from messener_orbit import *
from create_graphs import *
from parse_csv import safe_parse_time

from stations_positions import *

from main import utc_to_tdb_seconds
from lt_tw_doppler import solve_two_way_light_time_with_intervals, prepare_ramp_table_from_data, compute_two_way_doppler_ramped

# Cчитывание данных для эферимидов с HORIZONS
def get_data_from_horizons(filename = 'horizons_results_venus.txt'):
    times, positions, velocities, gm, radius = load_horizons_data(filename)
    print(filename)
    print("GM:\n", gm, end='\n\n')
    print("Radius:\n", radius, end='\n\n')
    print("Times:\n", times[:5], ' ...', end='\n\n')
    print("Positions:\n", positions[:5], ' ...', end='\n\n')
    print("Velocities:\n", velocities[:5], ' ...', end='\n\n')


# График для станций в определённый период:
def plot_data_from_odf():
    pass


# Интерполяция положения и скорости эферемида
def interlotation(filename = 'horizons_results_venus.txt'):
    times, positions, velocities, gm, radius = load_horizons_data(filename)
    position_interpolator, velocity_interpolator = create_interpolators(times, positions, velocities)

    new_time = times[0] + (times[1] - times[0]) / 2

    new_position = position_interpolator(new_time)
    new_velocity = velocity_interpolator(new_time)

    print('New time:\n', new_time, end='\n\n')
    print('Inetrpolated position:\n', new_position, end='\n\n')
    print('Inetrpolated velocity:\n', new_velocity, end='\n\n')


# Пример расчёта орбиты для MESSENGER
def messenger_orbit():
    ephemeris_files = {
            'sun': 'horizons_results_sun.txt',
            'earth': 'horizons_results_earth.txt',
            'venus': 'horizons_results_venus.txt',
            'mercury': 'horizons_results_mercury.txt'
        }
        
    body_interpolators = {}
    body_vel_interpolators = {}
    body_times = {}
    gms_data = {}
    radius_data = {}

    _, _, _, _, mercury_radius = load_horizons_data(ephemeris_files['mercury'])

    for body_name, filename in ephemeris_files.items():
        try:
            times, positions, velocities, gms_data[body_name], radius_data[body_name] = load_horizons_data(filename)
            pos_interp, vel_interp = create_interpolators(times, positions, velocities)
            
            body_interpolators[body_name] = pos_interp
            body_vel_interpolators[body_name] = vel_interp
            body_times[body_name] = times
            
        except Exception as e:
            print(f"Ошибка загрузки {body_name}: {e}")

    min_time_utc = safe_parse_time('2014-05-02 21:00:03.500000+00:00')
    max_time_utc = safe_parse_time('2014-05-03 21:00:03.500000+00:00')

    # Преобразование в TDB
    min_time_tdb = (min_time_utc.to_julian_date() - 2451545.0) * 86400.0
    max_time_tdb = (max_time_utc.to_julian_date() - 2451545.0) * 86400.0

    t_span = [min_time_tdb, max_time_tdb]
    t_eval = np.linspace(t_span[0], t_span[1], 5000)

    initial_state = create_initial_state_from_horizons_data(t_eval[0])

    #print('Начальные условия (положение, скорость):\n', initial_state, '\n\n')

    orbit_result = integrate_messenger_orbit(t_span, t_eval, initial_state, body_interpolators, gms_data, radius_data)

    plot_mercury_orbit_detailed_corrected(orbit_result['times'], orbit_result['positions'], body_interpolators, radius_data['mercury'])


# Настоящая орбита MESSENGER из файла horizons_results_messenger.txt
def true_messenger_orbit():
    messenger_times, messenger_positions, messenger_velocities, _, _ = load_horizons_data('horizons_results_messenger.txt')
    sc_pos_interp, sc_vel_interp = create_interpolators(messenger_times, messenger_positions, messenger_velocities)

    min_time_utc = safe_parse_time('2014-05-02 21:00:03.500000+00:00')
    max_time_utc = safe_parse_time('2014-05-03 21:00:03.500000+00:00')

    min_time_tdb = (min_time_utc.to_julian_date() - 2451545.0) * 86400.0
    max_time_tdb = (max_time_utc.to_julian_date() - 2451545.0) * 86400.0

    t_span = [min_time_tdb, max_time_tdb]
    t_eval = np.linspace(t_span[0], t_span[1], 5000)

    new_sc_positions = sc_pos_interp(t_eval)

    ephemeris_files = {
                'sun': 'horizons_results_sun.txt',
                'earth': 'horizons_results_earth.txt',
                'venus': 'horizons_results_venus.txt',
                'mercury': 'horizons_results_mercury.txt'
            }
            
    body_interpolators = {}
    body_vel_interpolators = {}
    body_times = {}
    gms_data = {}
    radius_data = {}

    _, _, _, _, mercury_radius = load_horizons_data(ephemeris_files['mercury'])

    for body_name, filename in ephemeris_files.items():
        try:
            times, positions, velocities, gms_data[body_name], radius_data[body_name] = load_horizons_data(filename)
            pos_interp, vel_interp = create_interpolators(times, positions, velocities)
            
            body_interpolators[body_name] = pos_interp
            body_vel_interpolators[body_name] = vel_interp
            body_times[body_name] = times
            
        except Exception as e:
            print(f"Ошибка загрузки {body_name}: {e}")


    plot_mercury_orbit_detailed_corrected(t_eval, new_sc_positions, body_interpolators, radius_data['mercury'], True)


# Графики для эферимидов
def plot_ephirimides():

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

    plot_solar_system_full_data(body_interpolators, body_times)

    plot_mercury_messenger_24h_closeup(body_interpolators, body_times)


# Координаты станции
def test_station_coords():
    time = utc_to_tdb_seconds('2014-01-02 21:00:03.500000+00:00')

    times, positions, velocities, gm, radius = load_horizons_data('horizons_results_earth.txt')
    position_interpolator, velocity_interpolator = create_interpolators(times, positions, velocities)

    body_interpolator = {}
    body_interpolator['earth'] = position_interpolator

    earth_position = position_interpolator(time)
    position = get_station_barycentric_pos(time, 55, body_interpolator)

    print('Earth: ', earth_position)
    print('Station: ', position)
    print('Vector: ', earth_position - position)
    print('Vector norm < Earth radius: ', np.linalg.norm(earth_position - position), ' < ', 6371, end='\n\n')

    time = utc_to_tdb_seconds('2014-01-02 22:00:03.500000+00:00')

    times, positions, velocities, gm, radius = load_horizons_data('horizons_results_earth.txt')
    position_interpolator, velocity_interpolator = create_interpolators(times, positions, velocities)

    body_interpolator = {}
    body_interpolator['earth'] = position_interpolator

    earth_position = position_interpolator(time)
    position = get_station_barycentric_pos(time, 55, body_interpolator)

    print('Earth: ', earth_position)
    print('Station: ', position)
    print('Vector: ', earth_position - position)
    print('Vector norm < Earth radius: ', np.linalg.norm(earth_position - position), ' < ', 6371, end='\n\n')


# Получение Two-Way-Light-Time
def test_light_time():
    df = pd.read_csv('raw_messenger_doppler_data.csv')

    filtered_df = df[df['station_id'] == 25].reset_index(drop=True)

    #print(len(filtered_df))

    #print(filtered_df.head())

    times, positions, velocities, gm, radius = load_horizons_data('horizons_results_earth.txt')
    position_interpolator, velocity_interpolator = create_interpolators(times, positions, velocities)

    sun_times, sun_positions, sun_velocities, sun_gm, sun_radius = load_horizons_data('horizons_results_sun.txt')
    sun_position_interpolator, sun_velocity_interpolator = create_interpolators(times, positions, velocities)
    
    sc_times, sc_positions, sc_velocities, sc_gm, sc_radius = load_horizons_data('horizons_results_messenger.txt')
    sc_position_interpolator, sc_velocity_interpolator = create_interpolators(sc_times, sc_positions, sc_velocities)

    body_interpolator = {}
    body_interpolator['earth'] = position_interpolator
    body_velocity = {}
    body_velocity['earth'] = velocity_interpolator

    body_interpolator['sun'] = sun_position_interpolator
    body_velocity['sun'] = sun_velocity_interpolator

    for i in range(min(len(filtered_df), 1), min(len(filtered_df), 5)):
        time = filtered_df['time_utc'][i]
        station_id = filtered_df['station_id'][i]

        Tc = filtered_df['compression_time_s']

        print('Time: ', time)
        print('Station id: ', station_id)

        gm_data = {
            'earth': gm,
            'sun': sun_gm
                   }

        print(f'{solve_two_way_light_time_with_intervals(time, Tc, station_id, body_interpolator, body_velocity, sc_position_interpolator, sc_velocity_interpolator, gm_data)['total_light_time']:.3f} s', end='\n\n')


# Получение Two-Way-Dopple
def test_two_way_doppler():
    df = pd.read_csv('raw_messenger_doppler_data.csv')

    filtered_df = df[df['station_id'] == 25].reset_index(drop=True)

    filtered_df = filtered_df[:100]

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

    min_time_utc = safe_parse_time(filtered_df['time_utc'].min())
    max_time_utc = safe_parse_time(filtered_df['time_utc'].max())
    min_time_tdb = (min_time_utc.to_julian_date() - 2451545.0) * 86400.0
    max_time_tdb = (max_time_utc.to_julian_date() - 2451545.0) * 86400.0

    avg_time_utc = min_time_utc + (max_time_utc - min_time_utc) / 2
    avg_time_tdb = (avg_time_utc.to_julian_date() - 2451545.0) * 86400.0
    
    t_span = [min_time_tdb, max_time_tdb]
    t_eval = np.linspace(t_span[0], t_span[1], 100000)

    ramp_table = prepare_ramp_table_from_data(filtered_df)

    avg_time_utc = min_time_utc + (max_time_utc - min_time_utc) / 2
    avg_time_tdb = (avg_time_utc.to_julian_date() - 2451545.0) * 86400.0
    
    
    initial_state = create_initial_state_from_horizons_data(min_time_tdb)
    
    
    orbit_result = integrate_messenger_orbit(
        t_span, t_eval, initial_state, body_interpolators, gms_data, radius_data
    )
    
    if not orbit_result['success']:
        print("Ошибка интегрирования орбиты. Выход.")
        return
    
    sc_pos_interp, sc_vel_interp = create_interpolators(orbit_result['times'], orbit_result['positions'], orbit_result['velocities'])

    gms_data.pop('messenger')

    test_df = filtered_df

    for idx, row in test_df.iterrows():
        print(f"\n{'='*60}")
        print(f"Обработка измерения {idx+1}/{len(test_df)}")
        print(f"{'='*60}")
        print(f"Время: {row['time_utc']}")
        print(f"Станция: {row['station_id']}")
        print(f"Измеренный допплер: {row['observable_hz']:.3f} Hz")

        t_tdb_check = utc_to_tdb_seconds(row['time_utc'])
        
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
            GM_params=gms_data,
            row_data=row.to_dict()
        )
        
        theoretical_doppler = theoretical_result['theoretical_doppler_hz']
        measured_doppler = row['observable_hz']
        residual = abs(measured_doppler - theoretical_doppler)

        M2 = theoretical_result['M2']
        transmit_freq = theoretical_result['transmit_freq_hz']
        c_km_s = 299792.458
        
        measured_velocity = -measured_doppler * c_km_s / (2 * M2 * transmit_freq)
        theoretical_velocity = theoretical_result['range_rate_kms']
        
        print(f"\nРезультаты:")
        print(f"  Теоретический допплер: {theoretical_doppler:.3f} Hz")
        print(f"  Измеренный допплер: {measured_doppler:.3f} Hz")
        print(f"  Residual: {residual:.3f} Hz")
        print(f"  Радиальная скорость (теор): {theoretical_velocity:.6f} км/с")
        print(f"  Радиальная скорость (изм): {measured_velocity/1000:.6f} км/с")

if __name__ == '__main__':
    #get_data_from_horizons()

    #interlotation()

    #messenger_orbit()

    #true_messenger_orbit()

    #plot_ephirimides()

    #test_station_coords()

    #test_light_time()

    #test_two_way_doppler()

    pass