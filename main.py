from horizons_parser import *
from interpolators import *
from messener_orbit import *
from create_graphs import *
from parse_csv import *
from stations_positions import *
from lt_tw_doppler import *
from orbit_refinement import OrbitRefinementLSQ  

def refine_orbit_with_lsq(doppler_df, body_interpolators, body_vel_interpolators, 
                         gms_data, ramp_table, initial_state, sc_pos_interp, sc_vel_interp):
    print("УТОЧНЕНИЕ ОРБИТЫ MESSENGER МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ")
    
    refinement = OrbitRefinementLSQ(
        doppler_df=doppler_df,
        body_interpolators=body_interpolators,
        body_vel_interpolators=body_vel_interpolators,
        gms_data=gms_data,
        ramp_table=ramp_table,
        dsn_stations=DSN_STATIONS
    )
    
    print(f"\nНачальное состояние:")
    print(f"  Положение: {initial_state[:3]} км")
    print(f"  Скорость: {initial_state[3:]} км/с")
    
    print("\nВычисление невязок для начального состояния...")
    initial_residuals = refinement.compute_residuals(initial_state)
    initial_rms = np.sqrt(np.mean(initial_residuals**2))
    initial_max = np.max(np.abs(initial_residuals))
    initial_min = np.min(np.abs(initial_residuals))
    
    print(f"Начальные невязки:")
    print(f"  RMS: {initial_rms:.3f} Гц")
    print(f"  Максимальная: {initial_max:.3f} Гц")
    print(f"  Минимальная: {initial_min:.3f} Гц")
    
    position_uncertainty = 1e6  
    velocity_uncertainty = 10.0  
    
    lower_bounds = np.array([
        initial_state[0] - position_uncertainty,
        initial_state[1] - position_uncertainty,
        initial_state[2] - position_uncertainty,
        initial_state[3] - velocity_uncertainty,
        initial_state[4] - velocity_uncertainty,
        initial_state[5] - velocity_uncertainty
    ])
    
    upper_bounds = np.array([
        initial_state[0] + position_uncertainty,
        initial_state[1] + position_uncertainty,
        initial_state[2] + position_uncertainty,
        initial_state[3] + velocity_uncertainty,
        initial_state[4] + velocity_uncertainty,
        initial_state[5] + velocity_uncertainty
    ])
    
    bounds = (lower_bounds, upper_bounds)
    
    result = refinement.solve_lsq(
        initial_params=initial_state,
        method='trf',
        bounds=bounds,
        max_iter=50,   
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=2
    )
    
    if result['success']:
        print("\n" + "="*70)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ УТОЧНЕНИЯ")
        print("="*70)
        
        final_rms = result['rms']
        final_max = np.max(np.abs(result['residuals']))
        final_min = np.min(np.abs(result['residuals']))
        
        print(f"\nСРАВНЕНИЕ ДО И ПОСЛЕ УТОЧНЕНИЯ:")
        print("-" * 50)
        print(f"{'Метрика':<25} {'До уточнения':<15} {'После уточнения':<15} {'Изменение':<15}")
        print("-" * 50)
        print(f"{'RMS невязок (Гц)':<25} {initial_rms:<15.3f} {final_rms:<15.3f} {initial_rms-final_rms:<15.3f}")
        print(f"{'Макс. невязка (Гц)':<25} {initial_max:<15.3f} {final_max:<15.3f} {initial_max-final_max:<15.3f}")
        print(f"{'Мин. невязка (Гц)':<25} {initial_min:<15.3f} {final_min:<15.3f} {initial_min-final_min:<15.3f}")
        
        if initial_rms > 0:
            improvement_percent = ((initial_rms - final_rms) / initial_rms) * 100
            print(f"\nУлучшение RMS: {improvement_percent:.2f}%")
        analysis = refinement.analyze_residuals(result['residuals'])
        
        print(f"\nСТАТИСТИЧЕСКИЙ АНАЛИЗ ПОСЛЕ УТОЧНЕНИЯ:")
        print(f"Среднее невязок: {analysis['mean']:.3f} Гц")
        print(f"СКО невязок: {analysis['std']:.3f} Гц")
        print(f"RMS невязок: {analysis['rms']:.3f} Гц")
        print(f"Максимальная по модулю: {analysis['max_abs']:.3f} Гц")
        print(f"Медиана: {analysis['median']:.3f} Гц")
        
        print(f"\nСРАВНЕНИЕ ПАРАМЕТРОВ ОРБИТЫ:")
        param_names = ['rx (км)', 'ry (км)', 'rz (км)', 'vx (км/с)', 'vy (км/с)', 'vz (км/с)']
        
        print(f"\n{'Параметр':<15} {'Начальное':<20} {'Уточненное':<20} {'Разность':<20} {'Отн.изм.(%)':<15}")
        print("-" * 90)
        
        for i, name in enumerate(param_names):
            initial_val = initial_state[i]
            final_val = result['params'][i]
            diff = final_val - initial_val
            
            if abs(initial_val) > 1e-10:
                rel_change = (diff / abs(initial_val)) * 100
            else:
                rel_change = 0.0
            
            print(f"{name:<15} {initial_val:<20.6f} {final_val:<20.6f} {diff:<20.6f} {rel_change:<15.6f}")
        
        refined_state = result['params']
        print(f"\nУточненное состояние успешно получено")
        
        return refined_state, result, initial_rms
    
    else:
        print("Уточнение орбиты не удалось")
        return None, result, initial_rms

def main():
    raw_output_file = 'raw_messenger_doppler_data.csv'
    doppler_df = pd.read_csv(raw_output_file)
    doppler_df = doppler_df[10000:10010]

    print("3. АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ДАННЫХ...")
    
    print("4. ЗАГРУЗКА ЭФЕМЕРИД HORIZONS...")
    
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
    
    print("7. ВЫЧИСЛЕНИЕ THEORETICAL DOPPLER С LIGHT-TIME КОРРЕКЦИЕЙ...")

    ramp_table = prepare_ramp_table_from_data(doppler_df)

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

    if 'doppler_residual_hz' in results_df.columns:
        initial_rms_full = np.sqrt(np.mean(results_df['doppler_residual_hz']**2))
        print(f"\nНачальные невязки (полный набор):")
        print(f"  RMS: {initial_rms_full:.3f} Гц")
        print(f"  Максимальная: {results_df['doppler_residual_hz'].abs().max():.3f} Гц")
        print(f"  Минимальная: {results_df['doppler_residual_hz'].abs().min():.3f} Гц")
        print(f"  Стандартное отклонение: {results_df['doppler_residual_hz'].std():.3f} Гц")
    
    print("\n8. УТОЧНЕНИЕ ОРБИТЫ МЕТОДОМ НАИМЕНЬШИХ КВАДРАТОВ...")
    
    refined_state, lsq_result, initial_rms = refine_orbit_with_lsq(
        doppler_df=doppler_df,
        body_interpolators=body_interpolators,
        body_vel_interpolators=body_vel_interpolators,
        gms_data=gms_data,
        ramp_table=ramp_table,
        initial_state=initial_state,
        sc_pos_interp=sc_pos_interp,
        sc_vel_interp=sc_vel_interp
    )
    
    if refined_state is not None:
        print("\n9. ПЕРЕВЫЧИСЛЕНИЕ ОРБИТЫ С УТОЧНЕННЫМИ ПАРАМЕТРАМИ...")
        
        refined_orbit_result = integrate_messenger_orbit(
            t_span, t_eval, refined_state, body_interpolators, gms_data, radius_data
        )
        
        if refined_orbit_result['success']:
            refined_sc_pos_interp, refined_sc_vel_interp = create_interpolators(
                refined_orbit_result['times'], 
                refined_orbit_result['positions'], 
                refined_orbit_result['velocities']
            )
            
            print("\n10. ПЕРЕВЫЧИСЛЕНИЕ DOPPLER С УТОЧНЕННОЙ ОРБИТОЙ...")
            
            refined_results_df = process_doppler_with_exact_model(
                doppler_df,
                body_interpolators,
                body_vel_interpolators,
                refined_sc_pos_interp,
                refined_sc_vel_interp,
                gms_data,
                ramp_table
            )
            
            print("\n11. ИТОГОВОЕ СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
            
            if 'doppler_residual_hz' in refined_results_df.columns:
                final_rms_full = np.sqrt(np.mean(refined_results_df['doppler_residual_hz']**2))
                
                print(f"\nСРАВНЕНИЕ НА ПОЛНОМ НАБОРЕ ДАННЫХ ({len(refined_results_df)} измерений):")
                
                if 'doppler_residual_hz' in results_df.columns:
                    print(f"{'Метрика':<25} {'Начальная орбита':<20} {'Уточненная орбита':<20}")
                    print(f"{'RMS (Гц)':<25} {initial_rms_full:<20.3f} {final_rms_full:<20.3f}")
                    print(f"{'Среднее (Гц)':<25} {results_df['doppler_residual_hz'].mean():<20.3f} {refined_results_df['doppler_residual_hz'].mean():<20.3f}")
                    print(f"{'Стд.отклонение (Гц)':<25} {results_df['doppler_residual_hz'].std():<20.3f} {refined_results_df['doppler_residual_hz'].std():<20.3f}")
                    print(f"{'Максимум (Гц)':<25} {results_df['doppler_residual_hz'].abs().max():<20.3f} {refined_results_df['doppler_residual_hz'].abs().max():<20.3f}")
                    print(f"{'Минимум (Гц)':<25} {results_df['doppler_residual_hz'].abs().min():<20.3f} {refined_results_df['doppler_residual_hz'].abs().min():<20.3f}")
                    
                    if initial_rms_full > 0:
                        improvement_full = ((initial_rms_full - final_rms_full) / initial_rms_full) * 100
                        print(f"\nОбщее улучшение RMS на полном наборе: {improvement_full:.2f}%")
                
                print(f"\nАНАЛИЗ ПО СТАНЦИЯМ:")
                
                for station_id in refined_results_df['station_id'].unique():
                    station_name = DSN_STATIONS.get(station_id, {}).get('name', f'DSN-{station_id}')
                    
                    mask_initial = results_df['station_id'] == station_id
                    mask_final = refined_results_df['station_id'] == station_id
                    
                    if mask_initial.any() and mask_final.any():
                        rms_initial = np.sqrt(np.mean(results_df.loc[mask_initial, 'doppler_residual_hz']**2))
                        rms_final = np.sqrt(np.mean(refined_results_df.loc[mask_final, 'doppler_residual_hz']**2))
                        
                        improvement = ((rms_initial - rms_final) / rms_initial * 100) if rms_initial > 0 else 0
                        
                        print(f"{station_name:<20} RMS: {rms_initial:7.3f} → {rms_final:7.3f} Гц")
        else:
            print("Ошибка интегрирования уточненной орбиты")
    else:
        print("Уточнение орбиты не выполнено")

if __name__ == "__main__":
    main()