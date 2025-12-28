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
from messener_orbit import *
from create_graphs import *
from lt_tw_doppler import *

def main():
    print("1. ЗАГРУЗКА ДАННЫХ DOPPLER MESSENGER...")
    '''
    doppler_df, ramp_df = load_messenger_doppler_data('./processed_data')
    
    if doppler_df.empty:
        print("Нет данных для обработки. Выход.")
        return
    '''

    raw_output_file = 'raw_messenger_doppler_data.csv'
    raw_ramp_output_file = 'raw_messenger_ramp_data.csv'
    #doppler_df.to_csv(raw_output_file, index=False)
    #ramp_df.to_csv(raw_ramp_output_file, index=False)

    #exit(0)
    doppler_df = pd.read_csv(raw_output_file)
    filtered_doppler_df = doppler_df[doppler_df['time_utc'] <= '2014-01-07']
    doppler_df = filtered_doppler_df
    ramp_df = pd.read_csv(raw_ramp_output_file)
    filtered_ramp_df = ramp_df[ramp_df['station_id'] != 0]
    ramp_df = filtered_ramp_df[filtered_ramp_df['ramp_start_time'] <= '2014-01-07']
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
                'mars': 'horizons_results_mars.txt'
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

    print("5. ОПРЕДЕЛЕНИЕ ВРЕМЕННОГО ИНТЕРВАЛА...")
    
    min_time_utc = safe_parse_time(doppler_df['time_utc'].min())
    max_time_utc = safe_parse_time(doppler_df['time_utc'].max())
    min_time_tdb = (min_time_utc.to_julian_date() - 2451545.0) * 86400.0
    max_time_tdb = (max_time_utc.to_julian_date() - 2451545.0) * 86400.0
    
    t_span = [min_time_tdb, max_time_tdb]
    t_eval = np.linspace(t_span[0], t_span[1], 100000)
    
    print(f"Временной интервал данных Doppler: {min_time_utc} - {max_time_utc}")
    print(f"Интервал интегрирования: {t_span[0]:.0f} - {t_span[1]:.0f} s TDB")
    print(f"Длительность: {(t_span[1]-t_span[0])/3600:.1f} часов")
    
    print("6. ИНТЕГРИРОВАНИЕ ОРБИТЫ MESSENGER...")
    
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
    
    #sc_pos_interp = body_interpolators['messenger']
    #sc_vel_interp = body_vel_interpolators['messenger']
    
    print("7. ВЫЧИСЛЕНИЕ THEORETICAL DOPPLER С LIGHT-TIME КОРРЕКЦИЕЙ...")
    print("Подготовка Ramp")
    print("Размер Ramp", len(ramp_df))
    ramp_table = prepare_ramp_table_from_data(doppler_df, ramp_df)
    print("Подготовка Ramp завершена")
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
    
    #output_file = 'messenger_doppler_final_results.csv'
    #final_df.to_csv(output_file, index=False)
    #print(f"Результаты сохранены в {output_file}")
    
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
    