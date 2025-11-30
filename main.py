import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import glob

# --- Константы ---
GM_mercury = 22031.86855  # км^3/с^2 (точное значение из данных Horizons)
GM_sun = 132712440041.9394  # км^3/с^2
GM_earth = 398600.4415  # км^3/с^2
J2_mercury = 0.00006
R_mercury = 2439.4  # км (точное значение из данных)
c = 299792.458  # км/с

# --- Функции загрузки данных Horizons (оставляем без изменений) ---
def load_horizons_data(filepath):
    """Загружает данные JPL Horizons из файла."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    start_idx = -1
    for i, line in enumerate(lines):
        if "$$SOE" in line:
            start_idx = i + 1
            break

    if start_idx == -1:
        raise ValueError(f"Не найдена строка начала данных ($$SOE) в файле {filepath}")

    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if "$$EOE" in lines[i]:
            end_idx = i
            break

    data_lines = lines[start_idx:end_idx]
    
    times_jd = []
    positions = []
    velocities = []
    
    i = 0
    while i < len(data_lines):
        line = data_lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        if re.match(r'^\d+\.\d+', line):
            try:
                jd = float(line.split()[0])
                
                i += 1
                if i >= len(data_lines):
                    break
                    
                pos_line = data_lines[i].strip()
                x_match = re.search(r'X\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)', pos_line)
                y_match = re.search(r'Y\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)', pos_line)
                z_match = re.search(r'Z\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)', pos_line)
                
                if not (x_match and y_match and z_match):
                    i += 1
                    continue
                    
                x = float(x_match.group(1))
                y = float(y_match.group(1))
                z = float(z_match.group(1))
                
                i += 1
                if i >= len(data_lines):
                    break
                    
                vel_line = data_lines[i].strip()
                vx_match = re.search(r'VX\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)', vel_line)
                vy_match = re.search(r'VY\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)', vel_line)
                vz_match = re.search(r'VZ\s*=\s*([-+]?\d*\.?\d+[eE]?[-+]?\d*)', vel_line)
                
                if not (vx_match and vy_match and vz_match):
                    i += 1
                    continue
                    
                vx = float(vx_match.group(1))
                vy = float(vy_match.group(1))
                vz = float(vz_match.group(1))
                
                i += 1
                
                times_jd.append(jd)
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Предупреждение: Ошибка парсинга данных в строке {i}: {e}")
                i += 1
                continue
        else:
            i += 1

    if len(times_jd) < 2:
        raise ValueError(f"Недостаточно данных для интерполяции в файле {filepath}. Найдено {len(times_jd)} точек.")

    times_jd = np.array(times_jd)
    positions = np.array(positions)
    velocities = np.array(velocities)

    times_tdb_s = (times_jd - 2451545.0) * 86400.0

    print(f"Загружено {len(times_jd)} точек из {filepath}")
    return times_tdb_s, positions, velocities

def create_interpolators(time_s, positions, velocities):
    """Создает интерполяторы для положения и скорости."""
    pos_interp = CubicSpline(time_s, positions, axis=0)
    vel_interp = CubicSpline(time_s, velocities, axis=0)
    return pos_interp, vel_interp

# --- Загрузка данных Horizons ---
try:
    sun_times, sun_pos, sun_vel = load_horizons_data('horizons_results_sun.txt')
    print(f"✅ Данные Солнца загружены: {len(sun_times)} точек.")
except Exception as e:
    print(f"❌ Ошибка загрузки данных Солнца: {e}")
    sun_times, sun_pos, sun_vel = None, None, None

try:
    earth_times, earth_pos, earth_vel = load_horizons_data('horizons_results_earth.txt')
    print(f"✅ Данные Земли загружены: {len(earth_times)} точек.")
except Exception as e:
    print(f"❌ Ошибка загрузки данных Земли: {e}")
    earth_times, earth_pos, earth_vel = None, None, None

try:
    mercury_times, mercury_pos, mercury_vel = load_horizons_data('horizons_results_mercury.txt')
    print(f"✅ Данные Меркурия загружены: {len(mercury_times)} точек.")
except Exception as e:
    print(f"❌ Ошибка загрузки данных Меркурия: {e}")
    mercury_times, mercury_pos, mercury_vel = None, None, None

if sun_times is None or earth_times is None or mercury_times is None:
    print("❌ Не все данные Horizons успешно загружены. Завершение.")
    exit()

# Создание интерполяторов
try:
    sun_pos_interp, sun_vel_interp = create_interpolators(sun_times, sun_pos, sun_vel)
    earth_pos_interp, earth_vel_interp = create_interpolators(earth_times, earth_pos, earth_vel)
    mercury_pos_interp, mercury_vel_interp = create_interpolators(mercury_times, mercury_pos, mercury_vel)
    print("✅ Интерполяторы созданы.")
except Exception as e:
    print(f"❌ Ошибка создания интерполяторов: {e}")
    exit()

# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ УРАВНЕНИЙ ДВИЖЕНИЯ ---
def equations_of_motion_corrected(t, state, mercury_pos_interp, sun_pos_interp, earth_pos_interp):
    """
    Уравнения движения КА в гелиоцентрической системе координат.
    
    state: [x, y, z, vx, vy, vz] - гелиоцентрические координаты (км, км/с)
    """
    x, y, z, vx, vy, vz = state
    r_sc_sun = np.array([x, y, z])  # положение КА относительно Солнца
    
    # Положения планет относительно Солнца
    mercury_pos = mercury_pos_interp(t)
    earth_pos = earth_pos_interp(t)
    
    # Векторы от КА к планетам
    r_mercury_sc = mercury_pos - r_sc_sun
    r_earth_sc = earth_pos - r_sc_sun
    
    # Расстояния
    r_mercury_norm = np.linalg.norm(r_mercury_sc)
    r_earth_norm = np.linalg.norm(r_earth_sc)
    r_sun_norm = np.linalg.norm(r_sc_sun)
    
    # Защита от деления на ноль
    if r_mercury_norm < 1: r_mercury_norm = 1
    if r_earth_norm < 1: r_earth_norm = 1
    if r_sun_norm < 1: r_sun_norm = 1
    
    # Ускорения
    # От Солнца
    a_sun = -GM_sun / r_sun_norm**3 * r_sc_sun
    
    # От Меркурия (включая J2)
    a_mercury_central = GM_mercury / r_mercury_norm**3 * r_mercury_sc
    
    # J2 поправка для Меркурия
    if r_mercury_norm > R_mercury:
        r_mercury_unit = r_mercury_sc / r_mercury_norm
        z_over_r = r_mercury_unit[2]
        
        j2_factor = (GM_mercury * J2_mercury * R_mercury**2) / r_mercury_norm**5
        a_mercury_j2_x = j2_factor * r_mercury_unit[0] * (1 - 5 * z_over_r**2)
        a_mercury_j2_y = j2_factor * r_mercury_unit[1] * (1 - 5 * z_over_r**2)
        a_mercury_j2_z = j2_factor * r_mercury_unit[2] * (3 - 5 * z_over_r**2)
        a_mercury_j2 = np.array([a_mercury_j2_x, a_mercury_j2_y, a_mercury_j2_z])
    else:
        a_mercury_j2 = np.zeros(3)
    
    a_mercury = a_mercury_central + a_mercury_j2
    
    # От Земли
    a_earth = GM_earth / r_earth_norm**3 * r_earth_sc
    
    # Суммарное ускорение
    total_acceleration = a_sun + a_mercury + a_earth
    
    derivatives = np.array([vx, vy, vz, total_acceleration[0], total_acceleration[1], total_acceleration[2]])
    return derivatives

# --- ПРАВИЛЬНЫЕ НАЧАЛЬНЫЕ УСЛОВИЯ ---
def get_initial_conditions_from_horizons(t0_tdb, mercury_pos_interp, mercury_vel_interp):
    """
    Получает начальные условия на основе данных Horizons для Меркурия
    и добавляет орбитальную скорость вокруг Меркурия.
    """
    # Положение и скорость Меркурия относительно Солнца
    mercury_pos = mercury_pos_interp(t0_tdb)
    mercury_vel = mercury_vel_interp(t0_tdb)
    
    # Для КА: начинаем с положения Меркурия + смещение (например, 5000 км по X)
    # и добавляем орбитальную скорость вокруг Меркурия
    offset_distance = 5000.0  # км от центра Меркурия
    orbital_speed = np.sqrt(GM_mercury / offset_distance)  # круговая орбитальная скорость
    
    # Начальное положение КА (в системе Меркурия)
    r_sc_mercury = np.array([offset_distance, 0, 0])
    
    # Преобразуем в гелиоцентрическую систему
    r_sc_sun = mercury_pos + r_sc_mercury
    
    # Орбитальная скорость вокруг Меркурия (перпендикулярно радиус-вектору)
    v_sc_mercury = np.array([0, orbital_speed, 0])
    
    # Преобразуем в гелиоцентрическую систему
    v_sc_sun = mercury_vel + v_sc_mercury
    
    initial_state = np.concatenate([r_sc_sun, v_sc_sun])
    
    print(f"🎯 Начальные условия:")
    print(f"   Положение Меркурия: [{mercury_pos[0]:.2f}, {mercury_pos[1]:.2f}, {mercury_pos[2]:.2f}] км")
    print(f"   Скорость Меркурия: [{mercury_vel[0]:.6f}, {mercury_vel[1]:.6f}, {mercury_vel[2]:.6f}] км/с")
    print(f"   Положение КА: [{r_sc_sun[0]:.2f}, {r_sc_sun[1]:.2f}, {r_sc_sun[2]:.2f}] км")
    print(f"   Скорость КА: [{v_sc_sun[0]:.6f}, {v_sc_sun[1]:.6f}, {v_sc_sun[2]:.6f}] км/с")
    print(f"   Высота над Меркурием: {offset_distance - R_mercury:.2f} км")
    print(f"   Орбитальная скорость вокруг Меркурия: {orbital_speed:.6f} км/с")
    
    return initial_state

# --- ОСНОВНОЙ РАСЧЕТ ---
print("\n🚀 Расчет правильной орбиты КА вокруг Меркурия...")

# Время начала (из данных или по умолчанию)
try:
    # Пробуем загрузить данные Doppler для определения времени начала
    data_dir = './data'
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if csv_files:
        df = pd.read_csv(csv_files[0])
        t0_utc = pd.to_datetime(df['time_utc'].iloc[0], utc=True)
    else:
        t0_utc = pd.to_datetime('2015-01-02 21:20:12.500000+00:00', utc=True)
except:
    t0_utc = pd.to_datetime('2015-01-02 21:20:12.500000+00:00', utc=True)

JD_t0 = t0_utc.to_julian_date()
TDB_start_from_J2000 = (JD_t0 - 2451545.0) * 86400.0

print(f"⏰ Начальное время: {t0_utc}")
print(f"🌍 Время TDB от J2000: {TDB_start_from_J2000:.2f} секунд")

# Получаем правильные начальные условия
initial_state = get_initial_conditions_from_horizons(
    TDB_start_from_J2000, 
    mercury_pos_interp, 
    mercury_vel_interp
)

# Временной интервал для интегрирования (1 день)
t_span = [TDB_start_from_J2000, TDB_start_from_J2000 + 86400.0]  # 1 день
t_eval = np.linspace(t_span[0], t_span[1], 1000)

print(f"\n🔄 Интегрирование на интервале {t_span[1]-t_span[0]:.0f} секунд ({((t_span[1]-t_span[0])/3600):.1f} часов)")

# Интегрирование
try:
    solution = solve_ivp(
        fun=lambda t, state: equations_of_motion_corrected(
            t, state, mercury_pos_interp, sun_pos_interp, earth_pos_interp
        ),
        t_span=t_span,
        y0=initial_state,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )

    # --- УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ ---
    if solution.success:
        print("✅ Интегрирование завершено успешно!")
        
        # Анализ результатов
        positions = solution.y[:3]
        times = solution.t
        
        # Вычисляем расстояние до Меркурия и относительные координаты
        mercury_positions = np.array([mercury_pos_interp(t) for t in times])
        distances_to_mercury = np.linalg.norm(positions.T - mercury_positions, axis=1)
        relative_positions = positions.T - mercury_positions
        
        min_distance = np.min(distances_to_mercury)
        max_distance = np.max(distances_to_mercury)
        eccentricity = (max_distance - min_distance) / (max_distance + min_distance)
        
        print(f"\n📊 Результаты:")
        print(f"   Минимальное расстояние до Меркурия: {min_distance:.2f} км")
        print(f"   Максимальное расстояние до Меркурия: {max_distance:.2f} км") 
        print(f"   Эксцентриситет орбиты: {eccentricity:.6f}")
        print(f"   Радиус Меркурия: {R_mercury} км")
        
        if min_distance < R_mercury:
            print("   ⚠️  КА упал на Меркурий!")
        else:
            print("   ✅ КА остается на стабильной орбите вокруг Меркурия")
        
        # УЛУЧШЕННАЯ ВИЗУАЛИЗАЦИЯ
        fig = plt.figure(figsize=(20, 5))
        
        # 1. ОТНОСИТЕЛЬНАЯ ТРАЕКТОРИЯ (система Меркурия) - ГЛАВНЫЙ ГРАФИК
        ax1 = fig.add_subplot(141)
        ax1.plot(relative_positions[:, 0], relative_positions[:, 1], 'b-', linewidth=2, label='Траектория КА')
        
        # Добавляем Меркурий в масштабе
        mercury_circle = plt.Circle((0, 0), R_mercury, color='red', alpha=0.7, label='Меркурий')
        ax1.add_patch(mercury_circle)
        
        # Начальная точка
        ax1.scatter([relative_positions[0, 0]], [relative_positions[0, 1]], 
                color='green', s=100, marker='o', label='Начало', zorder=5)
        
        ax1.set_xlabel('X относительно Меркурия (км)')
        ax1.set_ylabel('Y относительно Меркурия (км)')
        ax1.set_title('Орбита КА вокруг Меркурия\n(вид сверху)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Устанавливаем одинаковные пределы по осям
        max_range = max(np.abs(relative_positions[:, 0]).max(), 
                    np.abs(relative_positions[:, 1]).max()) * 1.1
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        
        # 2. 3D ОРБИТА В СИСТЕМЕ МЕРКУРИЯ
        ax2 = fig.add_subplot(142, projection='3d')
        ax2.plot(relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2], 
                'b-', linewidth=2, label='Орбита КА')
        
        # Сфера Меркурия
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = R_mercury * np.outer(np.cos(u), np.sin(v))
        y = R_mercury * np.outer(np.sin(u), np.sin(v)) 
        z = R_mercury * np.outer(np.ones(np.size(u)), np.cos(v))
        ax2.plot_surface(x, y, z, color='red', alpha=0.3, label='Меркурий')
        
        ax2.scatter([0], [0], [0], color='red', s=100, label='Центр Меркурия')
        ax2.scatter([relative_positions[0, 0]], [relative_positions[0, 1]], [relative_positions[0, 2]],
                color='green', s=100, marker='o', label='Начало')
        
        ax2.set_xlabel('X (км)')
        ax2.set_ylabel('Y (км)')
        ax2.set_zlabel('Z (км)')
        ax2.set_title('3D Орбита КА вокруг Меркурия')
        ax2.legend()
        
        # 3. РАССТОЯНИЕ ОТ МЕРКУРИЯ
        ax3 = fig.add_subplot(143)
        ax3.plot((times - times[0])/3600, distances_to_mercury, 'g-', linewidth=2)
        ax3.axhline(y=R_mercury, color='r', linestyle='--', linewidth=2, label='Радиус Меркурия')
        ax3.axhline(y=min_distance, color='orange', linestyle=':', linewidth=1, label=f'Мин: {min_distance:.1f} км')
        ax3.axhline(y=max_distance, color='purple', linestyle=':', linewidth=1, label=f'Макс: {max_distance:.1f} км')
        
        ax3.set_xlabel('Время (часы)')
        ax3.set_ylabel('Расстояние до Меркурия (км)')
        ax3.set_title('Высота орбиты КА над Меркурием')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ГЕЛИОЦЕНТРИЧЕСКАЯ СИСТЕМА (масштабированная)
        ax4 = fig.add_subplot(144)
        
        # Масштабируем для лучшего отображения
        scale_factor = 1e6  # переводим в миллионы км
        positions_scaled = positions / scale_factor
        mercury_positions_scaled = mercury_positions / scale_factor
        
        ax4.plot(mercury_positions_scaled[:, 0], mercury_positions_scaled[:, 1], 
                'r-', alpha=0.5, linewidth=1, label='Орбита Меркурия')
        ax4.plot(positions_scaled[:, 0], positions_scaled[:, 1], 
                'b-', alpha=0.7, linewidth=2, label='Траектория КА')
        
        # Показываем начальные положения
        ax4.scatter([mercury_positions_scaled[0, 0]], [mercury_positions_scaled[0, 1]], 
                color='red', s=50, label='Меркурий (начало)')
        ax4.scatter([positions_scaled[0, 0]], [positions_scaled[0, 1]], 
                color='blue', s=30, label='КА (начало)')
        
        ax4.set_xlabel('X (миллионы км)')
        ax4.set_ylabel('Y (миллионы км)')
        ax4.set_title('Гелиоцентрическая система\n(масштабированная)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ
        print(f"\n📈 Дополнительная информация об орбите:")
        print(f"   Большая полуось: {(max_distance + min_distance)/2:.2f} км")
        print(f"   Период обращения: {len(times)/1000 * 24:.2f} часов (приблизительно)")
        print(f"   Форма орбиты: {'круговая' if eccentricity < 0.1 else 'эллиптическая'}")
        
    else:
        print("❌ Интегрирование не удалось.")
        print(solution.message)

except Exception as e:
    print(f"❌ Ошибка при интегрировании: {e}")
    import traceback
    traceback.print_exc()