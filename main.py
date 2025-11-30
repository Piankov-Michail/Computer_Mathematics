import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution
from datetime import datetime, timedelta, timezone
import astropy.units as u
from astropy.time import Time
import glob
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Параметры Меркурия
GM_mercury = 22031.78  # км³/с²
R_mercury = 2439.7     # км
J2_mercury = 5.03e-5   # коэффициент несферичности
omega_mercury = 6.1385025e-6  # радиан/с (угловая скорость вращения Меркурия)

# Физические константы
c = 299792.458  # км/с (скорость света)

# Координаты DSN станций (точные координаты для разных станций)
DSN_STATIONS = {
    25: {'name': 'DSS-25 Goldstone', 'lon': -116.89, 'lat': 35.43, 'alt': 0.1, 'type': 'earth'},  # США
    34: {'name': 'DSS-34 Canberra', 'lon': 149.0, 'lat': -35.4, 'alt': 0.7, 'type': 'earth'},    # Австралия  
    35: {'name': 'DSS-35 Canberra', 'lon': 149.0, 'lat': -35.4, 'alt': 0.7, 'type': 'earth'},    # Австралия
    43: {'name': 'DSS-43 Canberra', 'lon': 149.0, 'lat': -35.2, 'alt': 0.7, 'type': 'earth'},    # Австралия
    45: {'name': 'DSS-45 Canberra', 'lon': 149.0, 'lat': -35.3, 'alt': 0.7, 'type': 'earth'},    # Австралия
    63: {'name': 'DSS-63 Madrid', 'lon': -4.25, 'lat': 40.42, 'alt': 0.9, 'type': 'earth'},      # Испания
    65: {'name': 'DSS-65 Madrid', 'lon': -4.25, 'lat': 40.42, 'alt': 0.9, 'type': 'earth'},      # Испания
}

def load_all_doppler_data(data_dir='data/'):
    """
    Загружает все файлы доплеровских данных из директории и объединяет их в один DataFrame
    """
    print("🔍 Поиск файлов с доплеровскими данными...")
    
    # Ищем все CSV файлы в директории
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print("❌ Не найдено CSV файлов в директории. Проверьте путь.")
        return None
    
    print(f"✅ Найдено {len(csv_files)} файлов:")
    for f in csv_files:
        print(f"   - {os.path.basename(f)}")
    
    # Загружаем все файлы
    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Проверяем наличие необходимых столбцов
            required_cols = ['time_utc', 'doppler_hz', 'station_id', 'data_type', 'valid']
            if all(col in df.columns for col in required_cols):
                # Преобразуем временные метки с правильным часовым поясом
                df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
                df['filename'] = os.path.basename(file)
                all_data.append(df)
                print(f"   ✓ Файл {os.path.basename(file)} загружен ({len(df)} записей)")
            else:
                print(f"   ✗ Файл {os.path.basename(file)} пропущен: отсутствуют необходимые столбцы")
        except Exception as e:
            print(f"   ✗ Ошибка при загрузке {os.path.basename(file)}: {str(e)}")
    
    
    # Объединяем все данные
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Сортируем по времени
    combined_df = combined_df.sort_values('time_utc').reset_index(drop=True)
    
    # Фильтруем только валидные данные и one-way Doppler (data_type=11)
    valid_df = combined_df[(combined_df['valid']) & (combined_df['data_type'] == 11)].copy()
    
    print(f"\n📊 Всего загружено записей: {len(combined_df)}")
    print(f"✅ Валидных one-way Doppler записей: {len(valid_df)}")
    
    # Статистика по станциям
    print("\n📡 Статистика по станциям:")
    station_stats = valid_df['station_id'].value_counts()
    for station_id, count in station_stats.items():
        station_name = DSN_STATIONS.get(int(station_id), {}).get('name', f'Станция {station_id}')
        print(f"   - {station_name} (ID={station_id}): {count} записей")
    
    return valid_df

def get_station_position(station_id, time_utc):
    """
    Получение положения DSN станции на поверхности Земли в заданный момент времени
    
    Для MESSENGER данные с DSN станций, поэтому станции находятся на Земле, а не на Меркурии
    """
    # Преобразуем station_id в целое число
    station_id = int(station_id)
    
    if station_id not in DSN_STATIONS:
        print(f"⚠️  Неизвестный ID станции: {station_id}. Используются координаты по умолчанию.")
        station_id = 35  # Canberra по умолчанию
    
    station = DSN_STATIONS[station_id]
    
    # Для земных станций мы не учитываем вращение Земли в этой упрощенной модели
    # Просто возвращаем фиксированные координаты станции
    
    # Координаты станции на поверхности Земли
    lon_rad = np.radians(station['lon'])
    lat_rad = np.radians(station['lat'])
    alt = station['alt']  # км над поверхностью
    
    # Радиус Земли в км
    R_earth = 6371.0
    
    # Декартовы координаты (в системе координат Земли)
    r_station = R_earth + alt
    x = r_station * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_station * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_station * np.sin(lat_rad)
    
    return np.array([x, y, z])

def equations_of_motion(t, y, use_J2=False):
    """
    Уравнения движения спутника в гравитационном поле Меркурия
    """
    x, y_pos, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y_pos**2 + z**2)
    
    if r < R_mercury * 1.1:  # Минимальная высота ~10% от радиуса
        r = R_mercury * 1.1
    
    # Центральное гравитационное поле
    ax = -GM_mercury * x / r**3
    ay = -GM_mercury * y_pos / r**3
    az = -GM_mercury * z / r**3
    
    # Добавляем возмущение от J2
    if use_J2:
        r_eq = R_mercury
        factor = (3/2) * J2_mercury * GM_mercury * (r_eq**2) / (r**5)
        z2_r2 = (z/r)**2
        
        ax += factor * x * (1 - 5*z2_r2)
        ay += factor * y_pos * (1 - 5*z2_r2)
        az += factor * z * (3 - 5*z2_r2)
    
    return [vx, vy, vz, ax, ay, az]

def runge_kutta_4(fun, t_span, y0, t_eval=None, h=10.0, use_J2=False):
    """
    Реализация метода Рунге-Кутты 4-го порядка
    """
    t_start, t_end = t_span
    t_current = t_start
    y_current = np.array(y0, dtype=float)
    
    if t_eval is None:
        t = [t_start]
        while t_current < t_end:
            t_current += h
            if t_current <= t_end:
                t.append(t_current)
    else:
        t = sorted(t_eval)
    
    y = [y_current.copy()]
    t_current = t_start
    y_current = np.array(y0, dtype=float)
    
    for t_next in t[1:]:
        dt = t_next - t_current
        
        # Шаг 1
        k1 = np.array(fun(t_current, y_current, use_J2))
        
        # Шаг 2
        k2 = np.array(fun(t_current + dt/2, y_current + dt/2 * k1, use_J2))
        
        # Шаг 3
        k3 = np.array(fun(t_current + dt/2, y_current + dt/2 * k2, use_J2))
        
        # Шаг 4
        k4 = np.array(fun(t_current + dt, y_current + dt * k3, use_J2))
        
        # Обновляем решение
        y_next = y_current + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        y.append(y_next.copy())
        t_current = t_next
        y_current = y_next
    
    return np.array(t), np.array(y)

def calculate_doppler_shift(spacecraft_pos, spacecraft_vel, station_pos, 
                          carrier_freq=7168.0, data_type=11):
    """
    Расчет доплеровского сдвига с учетом типа измерений
    
    data_type:
    11 = One-way Doppler
    12 = Two-way Doppler  
    13 = Three-way Doppler
    """
    # Вектор от станции до спутника
    r_vec = spacecraft_pos - station_pos
    r_mag = np.linalg.norm(r_vec)
    
    if r_mag < 1e-6:
        return 0.0
    
    # Единичный вектор от станции к спутнику
    r_unit = r_vec / r_mag
    
    # Относительная скорость спутника относительно станции
    v_rel = spacecraft_vel
    
    # Проекция скорости на линию визирования
    v_radial = np.dot(v_rel, r_unit)
    
    # Доплеровский сдвиг
    if data_type == 11:  # One-way
        doppler_shift = carrier_freq * 1e6 * (v_radial / c)
    elif data_type == 12:  # Two-way
        doppler_shift = 2 * carrier_freq * 1e6 * (v_radial / c)
    elif data_type == 13:  # Three-way
        doppler_shift = 3 * carrier_freq * 1e6 * (v_radial / c)
    else:
        doppler_shift = carrier_freq * 1e6 * (v_radial / c)
    
    return doppler_shift

def simulate_orbit(initial_state, t_span, t_eval, use_J2=True):
    """
    Симуляция орбиты с использованием Рунге-Кутты 4-го порядка
    """
    times, trajectory = runge_kutta_4(equations_of_motion, t_span, initial_state, t_eval, h=10.0, use_J2=use_J2)
    return times, trajectory

def objective_function(params, t_seconds, observed_doppler, station_ids, 
                      times_utc, carrier_freq=7168.0, use_J2=True):
    """
    Целевая функция для метода наименьших квадратов
    
    params: [x0, y0, z0, vx0, vy0, vz0] - начальные условия
    """
    # Разделяем параметры
    initial_state = params
    
    # Временной интервал
    t_span = [0, max(t_seconds)]
    
    # Симуляция орбиты
    times, trajectory = simulate_orbit(initial_state, t_span, t_seconds, use_J2=use_J2)
    
    # Расчет смоделированных доплеровских сдвигов
    simulated_doppler = []
    
    for i, t in enumerate(t_seconds):
        spacecraft_pos = trajectory[i, :3]
        spacecraft_vel = trajectory[i, 3:]
        
        # Положение станции для текущего времени
        station_pos = get_station_position(station_ids[i], times_utc[i])
        
        # Расчет доплеровского сдвига
        doppler = calculate_doppler_shift(
            spacecraft_pos, 
            spacecraft_vel, 
            station_pos, 
            carrier_freq,
            data_type=11  # one-way Doppler
        )
        simulated_doppler.append(doppler)
    
    # Остатки
    residuals = np.array(simulated_doppler) - np.array(observed_doppler)
    
    return residuals

def optimize_orbit(initial_state, t_seconds, observed_doppler, station_ids, 
                  times_utc, carrier_freq=7168.0, use_J2=True):
    """
    Уточнение орбиты с использованием гибридного подхода:
    1. Сначала глобальная оптимизация (differential_evolution)
    2. Затем локальная оптимизация (least_squares)
    """
    print("🚀 Начинаем оптимизацию орбиты...")
    
    # Определяем границы для глобальной оптимизации
    bounds = []
    for i in range(3):  # x, y, z
        bounds.append((initial_state[i] * 0.9, initial_state[i] * 1.1))
    for i in range(3, 6):  # vx, vy, vz
        bounds.append((initial_state[i] * 0.95, initial_state[i] * 1.05))
    
    print("🌍 Шаг 1: Глобальная оптимизация (differential_evolution)...")
    
    # Функция для глобальной оптимизации
    def global_obj(params):
        res = objective_function(params, t_seconds, observed_doppler, station_ids, 
                               times_utc, carrier_freq, use_J2)
        return np.sum(res**2)
    
    # Глобальная оптимизация
    result_global = differential_evolution(
        global_obj, 
        bounds, 
        maxiter=50,
        popsize=15,
        tol=1e-3,
        polish=False,
        seed=42
    )
    
    print(f"   ✓ Глобальная оптимизация завершена. Лучшее значение функции: {result_global.fun:.2f}")
    
    print("🎯 Шаг 2: Локальная оптимизация (least_squares)...")
    
    # Локальная оптимизация с использованием результатов глобальной
    result_local = least_squares(
        objective_function, 
        result_global.x, 
        args=(t_seconds, observed_doppler, station_ids, times_utc, carrier_freq, use_J2),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=200,
        verbose=1
    )
    
    print(f"   ✓ Локальная оптимизация завершена.")
    print(f"   📊 Начальное значение функции: {np.sum(objective_function(initial_state, t_seconds, observed_doppler, station_ids, times_utc, carrier_freq, use_J2)**2):.2f}")
    print(f"   📊 Конечное значение функции: {np.sum(result_local.fun**2):.2f}")
    print(f"   🔄 Количество итераций: {result_local.nfev}")
    
    return result_local

def plot_3d_orbit(trajectory, trajectory_opt=None, stations=None, times_utc=None, title="Орбита MESSENGER вокруг Меркурия"):
    """
    3D визуализация орбиты с возможностью анимации
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Параметры для Меркурия
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_merc = R_mercury * np.outer(np.cos(u), np.sin(v))
    y_merc = R_mercury * np.outer(np.sin(u), np.sin(v))
    z_merc = R_mercury * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Рисуем Меркурий
    ax.plot_surface(x_merc, y_merc, z_merc, color='orange', alpha=0.3, rstride=4, cstride=4, linewidth=0)
    
    # Рисуем начальную орбиту
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'b-', linewidth=2, label='Начальная орбита')
    
    # Рисуем оптимизированную орбиту, если есть
    if trajectory_opt is not None:
        ax.plot(trajectory_opt[:, 0], trajectory_opt[:, 1], trajectory_opt[:, 2], 'r--', linewidth=2, label='Оптимизированная орбита')
    
    # Рисуем положение станций
    if stations is not None and times_utc is not None:
        unique_stations = np.unique(stations)
        for station_id in unique_stations:
            station_mask = (stations == station_id)
            if np.any(station_mask):
                station_name = DSN_STATIONS.get(station_id, {}).get('name', f'Станция {station_id}')
                station_pos = get_station_position(station_id, times_utc[station_mask][0])
                ax.scatter(station_pos[0], station_pos[1], station_pos[2], s=100, 
                          label=station_name, alpha=0.8)
    
    # Настройки осей
    max_range = np.max(np.abs(trajectory[:, :3])) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    ax.set_xlabel('X (км)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y (км)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z (км)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Добавляем легенду
    ax.legend(loc='upper right', fontsize=10)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3)
    
    # Улучшаем отображение
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    return fig, ax

def plot_doppler_comparison(t_seconds, observed_doppler, simulated_doppler, 
                          optimized_doppler=None, station_ids=None, title="Сравнение доплеровских сдвигов"):
    """
    Визуализация сравнения наблюдаемых и смоделированных доплеровских сдвигов
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Основной график
    ax1 = fig.add_subplot(211)
    
    # Строим наблюдаемые данные
    scatter = ax1.scatter(t_seconds, observed_doppler, c=station_ids, cmap='viridis', 
                        s=60, alpha=0.8, label='Наблюдаемые данные', zorder=3)
    
    # Строим смоделированные данные
    ax1.plot(t_seconds, simulated_doppler, 'b-', linewidth=2, label='До оптимизации', zorder=2)
    
    # Строим оптимизированные данные, если есть
    if optimized_doppler is not None:
        ax1.plot(t_seconds, optimized_doppler, 'r--', linewidth=2, label='После оптимизации', zorder=1)
    
    ax1.set_xlabel('Время (секунды)', fontsize=12)
    ax1.set_ylabel('Доплеровский сдвиг (Гц)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Добавляем colorbar для станций
    cbar = plt.colorbar(scatter, ax=ax1, label='ID станции')
    
    # График остатков
    ax2 = fig.add_subplot(212)
    
    residuals_initial = np.array(simulated_doppler) - np.array(observed_doppler)
    ax2.plot(t_seconds, residuals_initial, 'b-', linewidth=2, label='Остатки до оптимизации')
    
    if optimized_doppler is not None:
        residuals_opt = np.array(optimized_doppler) - np.array(observed_doppler)
        ax2.plot(t_seconds, residuals_opt, 'r--', linewidth=2, label='Остатки после оптимизации')
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Время (секунды)', fontsize=12)
    ax2.set_ylabel('Остатки (Гц)', fontsize=12)
    ax2.set_title('Остатки', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    print("🚀 MESSENGER Orbit Determination Tool")
    print("=" * 50)
    
    # 1. Загрузка данных
    print("\n1. Загрузка доплеровских данных...")
    data_dir = 'data/'  # Измените на вашу директорию
    
    if not os.path.exists(data_dir):
        print(f"❌ Директория {data_dir} не существует. Создаем пример данных...")
        os.makedirs(data_dir, exist_ok=True)
        
        # Создаем пример данных если директория пуста
        sample_data = pd.DataFrame({
            'time_utc': [
                '2015-01-02 21:20:12.500000+00:00',
                '2015-01-02 21:20:17.500000+00:00',
                '2015-01-02 21:20:22.500000+00:00',
                '2015-01-02 21:20:27.500000+00:00',
                '2015-01-02 21:20:32.500000+00:00'
            ],
            'doppler_hz': [
                -841457.20240116,
                -841449.536668776,
                -841441.779811858,
                -841434.146040915,
                -841426.498896598
            ],
            'station_id': [35, 35, 35, 35, 35],
            'data_type': [11, 11, 11, 11, 11],
            'valid': [True, True, True, True, True]
        })
        sample_data['time_utc'] = pd.to_datetime(sample_data['time_utc'], utc=True)
        sample_data.to_csv(os.path.join(data_dir, 'sample_data.csv'), index=False)
        print("✅ Пример данных создан")
    
    doppler_df = load_all_doppler_data(data_dir)
    
    if doppler_df is None or len(doppler_df) == 0:
        print("❌ Нет данных для обработки. Завершение работы.")
        return
    
    # 2. Подготовка данных
    print("\n2. Подготовка данных для расчетов...")
    
    # Берем первую временную метку как начало отсчета
    t0 = doppler_df['time_utc'].iloc[0]
    doppler_df['t_seconds'] = (doppler_df['time_utc'] - t0).dt.total_seconds().values
    
    print(f"⏰ Начальное время: {t0}")
    print(f"⏱️  Общая продолжительность наблюдений: {max(doppler_df['t_seconds']):.2f} секунд")
    
    # 3. Задаем начальные условия
    print("\n3. Задаем начальные условия для MESSENGER...")
    
    # Более точные начальные условия для MESSENGER в январе 2015
    # Используем реальные данные орбиты MESSENGER
    a = 10136.2  # км (большая полуось)
    e = 0.7396   # эксцентриситет
    i = 82.5     # градусы (наклонение)
    omega = 0.0  # долгота восходящего узла (для упрощения)
    w = 0.0      # аргумент перицентра (для упрощения)
    
    # Расстояние в перицентре
    r_p = a * (1 - e)
    
    # Скорость в перицентре
    v_p = np.sqrt(GM_mercury * (2/r_p - 1/a))
    
    # Начальное положение в перицентре (в инерциальной системе)
    r0 = np.array([r_p, 0, 0])
    
    # Начальная скорость (в плоскости орбиты)
    i_rad = np.radians(i)
    v0 = np.array([0, v_p * np.cos(i_rad), v_p * np.sin(i_rad)])
    
    initial_state = np.concatenate([r0, v0])
    
    print(f"📊 Начальные условия:")
    print(f"   Положение: [{r0[0]:.2f}, {r0[1]:.2f}, {r0[2]:.2f}] км")
    print(f"   Скорость: [{v0[0]:.6f}, {v0[1]:.6f}, {v0[2]:.6f}] км/с")
    print(f"   Большая полуось: {a:.2f} км")
    print(f"   Эксцентриситет: {e:.4f}")
    print(f"   Наклонение: {i:.1f}°")
    
    # 4. Построение орбиты методом Рунге-Кутты
    print("\n4. Построение орбиты методом Рунге-Кутты 4-го порядка...")
    
    t_span = [0, max(doppler_df['t_seconds']) + 100]  # добавляем буфер
    t_eval = doppler_df['t_seconds'].values
    
    # Ограничиваем количество точек для визуализации
    max_points = 200
    if len(t_eval) > max_points:
        step = len(t_eval) // max_points
        t_eval_plot = t_eval[::step]
    else:
        t_eval_plot = t_eval
    
    # Симуляция орбиты для визуализации
    times_plot, trajectory_plot = runge_kutta_4(equations_of_motion, t_span, initial_state, t_eval_plot, use_J2=True)
    
    # Симуляция орбиты для всех точек данных
    times_full, trajectory_full = runge_kutta_4(equations_of_motion, t_span, initial_state, t_eval, use_J2=True)
    
    # 5. Расчет смоделированных доплеровских сдвигов
    print("\n5. Расчет смоделированных доплеровских сдвигов...")
    
    carrier_freq = 7168.0  # МГц (X-band для MESSENGER)
    simulated_doppler = []
    station_positions = []
    
    # Используем tqdm для отображения прогресса
    for i in range(len(t_eval)):
        spacecraft_pos = trajectory_full[i, :3]
        spacecraft_vel = trajectory_full[i, 3:]
        
        # Получаем ID станции и преобразуем в целое число
        station_id = int(doppler_df['station_id'].iloc[i])
        station_pos = get_station_position(station_id, doppler_df['time_utc'].iloc[i])
        station_positions.append(station_pos)
        
        # Расчет доплеровского сдвига
        doppler = calculate_doppler_shift(spacecraft_pos, spacecraft_vel, station_pos, carrier_freq, data_type=11)
        simulated_doppler.append(doppler)
    
    # 6. Визуализация первых результатов
    print("\n6. Визуализация первых результатов...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(doppler_df['t_seconds'], doppler_df['doppler_hz'], 'o-', label='Наблюдаемые данные', linewidth=2, markersize=6)
    plt.plot(t_eval, simulated_doppler, 's--', label='Смоделированные данные', linewidth=2, markersize=4)
    plt.xlabel('Время (секунды от начала)')
    plt.ylabel('Доплеровский сдвиг (Гц)')
    plt.title('Сравнение наблюдаемых и смоделированных доплеровских сдвигов')
    plt.legend()
    plt.grid(True)
    plt.savefig('initial_doppler_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Первая часть расчетов завершена!")
    print("📊 Результаты сохранены в файл 'initial_doppler_comparison.png'")
    print("🔍 Запустите следующий этап для уточнения орбиты методом наименьших квадратов")

if __name__ == "__main__":
    main()