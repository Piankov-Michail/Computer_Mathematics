import re
import numpy as np
from typing import Tuple

dir = 'horizons_data/'

def load_horizons_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with open(dir+filepath, 'r') as f:
        lines = f.readlines()

    gm_value = None
    radius = None

    for line in lines:
        if ("GM (km^3/s^2)" in line) or ("GM, km^3/s^2" in line):
            gm_match = re.search(r'=\s*([0-9.]+)', line)
            if gm_match:
                gm_value = float(gm_match.group(1))

        if ("Vol. Mean Radius (km)" in line) or ("Vol. mean radius, km" in line) or ("Vol. mean radius (km)" in line):
            radius_match = re.search(r'=\s*([0-9.]+)', line)
            if radius_match:
                radius = float(radius_match.group(1))
        
        if gm_value is not None and radius is not None:
            break
    
    #if gm_value is None:
        #raise ValueError(f"Не найдено значение GM в файле {filepath}")

    #if radius is None:
        #raise ValueError(f"Не найдено значение Vol. Mean Radius (km) в файле {filepath}")

    start_idx, end_idx = -1, len(lines)
    for i, line in enumerate(lines):
        if "$$SOE" in line:
            start_idx = i + 1
        elif "$$EOE" in line:
            end_idx = i
            break
    
    if start_idx == -1:
        raise ValueError(f"Не найдено начало данных ($$SOE) в файле {filepath}")

    data_lines = lines[start_idx:end_idx]
    times_jd, positions, velocities = [], [], []
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

                times_jd.append(jd)
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                i += 1
                
            except (ValueError, IndexError, AttributeError) as e:
                i += 1
                continue
        else:
            i += 1
    
    if len(times_jd) < 2:
        raise ValueError(f"Недостаточно данных для интерполяции в {filepath}. Найдено {len(times_jd)} точек.")

    times_jd = np.array(times_jd)
    times_tdb_s = (times_jd - 2451545.0) * 86400.0
    
    return times_tdb_s, np.array(positions), np.array(velocities), gm_value, radius