import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import re
from datetime import datetime, timedelta
import astropy.time as astro_time
import astropy.units as u
import plotly.graph_objects as go

c = 299792.458  # km/s - speed of light
MESSENGER_XBAND_HIGH_FREQ = 8445.734e6  # Hz - X-band high frequency for MESSENGER
RAMPED_DOPPLER_DATA_TYPE = 12  # Ramped Two-way Doppler Shift (X-band)
GM_sun = 132712440041.9394  # km^3/s^2
GM_earth = 398600.4415  # km^3/s^2
GM_venus = 324858.592  # km^3/s^2
GM_mercury = 22031.86855  # km^3/s^2
J2_mercury = 0.00006
R_mercury = 2439.7  # km

DSN_STATIONS = {
    35: {'name': 'Goldstone DSS-14', 'lat': 35.420278, 'lon': -116.887222, 'height': 965.0},
    45: {'name': 'Canberra DSS-43', 'lat': -35.398333, 'lon': 148.981944, 'height': 728.0},
    55: {'name': 'Madrid DSS-65', 'lat': 40.426389, 'lon': -4.248889, 'height': 840.0},
    65: {'name': 'Tidbinbilla DSS-34', 'lat': -35.398333, 'lon': 148.981944, 'height': 728.0},
    15: {'name': 'Parkes Observatory', 'lat': -32.998333, 'lon': 148.261944, 'height': 386.0},
    25: {'name': 'Uranquinty', 'lat': -34.621667, 'lon': 146.741111, 'height': 280.0},
    24: {'name': 'Hartebeesthoek', 'lat': -25.889167, 'lon': 28.201667, 'height': 1400.0},
    54: {'name': 'Woomera', 'lat': -31.116667, 'lon': 136.816667, 'height': 176.0}
}

def safe_parse_time(time_series):
    parsed = pd.to_datetime(time_series, utc=True, errors='coerce')
    if parsed.isna().sum() > len(parsed) * 0.5:
        time_series_clean = time_series.astype(str).str.strip()
        mask_no_micro = (time_series_clean.str.contains(' ') & 
                        (~time_series_clean.str.contains('\\.')) & 
                        time_series_clean.str.contains('\\+'))
        time_series_clean = time_series_clean.where(
            ~mask_no_micro, 
            time_series_clean.str.replace('\\+', '.000000+', regex=True)
        )
        mask_long_micro = time_series_clean.str.contains('\\.')
        if mask_long_micro.any():
            def truncate_microseconds(ts):
                if '.' in ts:
                    base, rest = ts.split('.', 1)
                    if '+' in rest:
                        micro_part, tz_part = rest.split('+', 1)
                        if len(micro_part) > 6:
                            micro_part = micro_part[:6]
                        return f"{base}.{micro_part}+{tz_part}"
                return ts
            time_series_clean = time_series_clean.where(
                ~mask_long_micro,
                time_series_clean.apply(truncate_microseconds)
            )
        parsed = pd.to_datetime(time_series_clean, utc=True, errors='coerce')
    return parsed

def load_ramped_doppler_data(data_dir='./data'):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    all_doppler_data = []
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        required_columns = ['time_utc', 'data_type', 'station_id', 'valid', 'doppler_hz']
        if not all(col in df.columns for col in required_columns):
            continue
        df['time_utc'] = safe_parse_time(df['time_utc'])
        df = df.dropna(subset=['time_utc'])
        if len(df) == 0:
            continue
        doppler_mask = (df['data_type'] == RAMPED_DOPPLER_DATA_TYPE) & (df['valid'] == True)
        doppler_df = df[doppler_mask].copy()
        if len(doppler_df) == 0:
            continue
        doppler_df['frequency_hz'] = MESSENGER_XBAND_HIGH_FREQ
        doppler_df['radial_velocity_kms'] = -(
            c * doppler_df['doppler_hz'] / (2 * MESSENGER_XBAND_HIGH_FREQ)
        )
        doppler_df['station_name'] = doppler_df['station_id'].map(
            lambda x: DSN_STATIONS.get(x, {}).get('name', f'Unknown Station {x}')
        )
        all_doppler_data.append(doppler_df)
    combined_doppler = pd.concat(all_doppler_data, ignore_index=True)
    return combined_doppler

def validate_ramped_doppler_data(doppler_df):
    filtered_df = doppler_df.copy()
    doppler_min = -200000
    doppler_max = 200000
    valid_doppler_mask = (
        (filtered_df['doppler_hz'] >= doppler_min) & 
        (filtered_df['doppler_hz'] <= doppler_max)
    )
    filtered_df = filtered_df[valid_doppler_mask].copy()
    velocity_min = -3.0
    velocity_max = 3.0
    valid_velocity_mask = (
        (filtered_df['radial_velocity_kms'] >= velocity_min) & 
        (filtered_df['radial_velocity_kms'] <= velocity_max)
    )
    filtered_df = filtered_df[valid_velocity_mask].copy()
    filtered_df = filtered_df.sort_values('time_utc')
    return filtered_df

def plot_ramped_doppler_data(doppler_df):
    print("Generating Ramped Two-way Doppler analysis plots...")
    
    if len(doppler_df) == 0:
        print("No valid Ramped Doppler data available for plotting")
        return []
    
    os.makedirs('plots', exist_ok=True)
    
    doppler_df['date'] = doppler_df['time_utc'].dt.date
    
    plot_files = []
    
    for station_id in sorted(doppler_df['station_id'].unique()):
        station_data = doppler_df[doppler_df['station_id'] == station_id]
        station_name = DSN_STATIONS.get(station_id, {}).get('name', f'Station {station_id}')
        
        for date in sorted(station_data['date'].unique()):
            daily_data = station_data[station_data['date'] == date].copy()
            daily_data = daily_data.sort_values('time_utc')
            
            if len(daily_data) < 10:
                continue
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            ax1.plot(daily_data['time_utc'], daily_data['doppler_hz'], 
                    'b-', linewidth=2, alpha=0.8, marker='o', markersize=4)
            ax1.set_title(f'Ramped Two-way Doppler Shift - {station_name} - {date}\nMESSENGER X-band ({MESSENGER_XBAND_HIGH_FREQ/1e6:.2f} MHz)', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Doppler Shift (Hz)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            ax1.axhline(y=-200000, color='r', linestyle='--', alpha=0.5, label='Min valid (-200 kHz)')
            ax1.axhline(y=200000, color='r', linestyle='--', alpha=0.5, label='Max valid (+200 kHz)')
            ax1.legend(loc='upper right')
            
            doppler_stats = (
                f"N = {len(daily_data)}\n"
                f"Mean: {daily_data['doppler_hz'].mean():.1f} Hz\n"
                f"Std: {daily_data['doppler_hz'].std():.1f} Hz\n"
                f"Min: {daily_data['doppler_hz'].min():.1f} Hz\n"
                f"Max: {daily_data['doppler_hz'].max():.1f} Hz"
            )
            ax1.text(0.02, 0.95, doppler_stats, transform=ax1.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    fontsize=10, verticalalignment='top')
            
            ax2.plot(daily_data['time_utc'], daily_data['radial_velocity_kms'], 
                    'r-', linewidth=2, alpha=0.8, marker='o', markersize=4)
            ax2.set_title(f'Radial Velocity from Ramped Doppler - {station_name} - {date}', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (UTC)', fontsize=12)
            ax2.set_ylabel('Radial Velocity (km/s)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            ax2.axhline(y=-3.0, color='r', linestyle='--', alpha=0.5, label='Min valid (-3.0 km/s)')
            ax2.axhline(y=3.0, color='r', linestyle='--', alpha=0.5, label='Max valid (+3.0 km/s)')
            ax2.legend(loc='upper right')
            
            velocity_stats = (
                f"Mean vel: {daily_data['radial_velocity_kms'].mean():.4f} km/s\n"
                f"Std vel: {daily_data['radial_velocity_kms'].std():.4f} km/s\n"
                f"Min vel: {daily_data['radial_velocity_kms'].min():.4f} km/s\n"
                f"Max vel: {daily_data['radial_velocity_kms'].max():.4f} km/s"
            )
            ax2.text(0.02, 0.95, velocity_stats, transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    fontsize=10, verticalalignment='top')
            
            session_duration = (daily_data['time_utc'].max() - daily_data['time_utc'].min()).total_seconds() / 60
            plt.figtext(0.5, 0.01, f"Session duration: {session_duration:.1f} minutes", 
                       ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            plot_filename = f'plots/ramped_doppler_{station_id}_{date}.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plot_files.append(plot_filename)
            print(f"Saved plot: {plot_filename}")
            
            plt.close()
            
            print(f"Session summary for {station_name} on {date}:")
            print(f"Duration: {session_duration:.1f} minutes")
            print(f"Measurements: {len(daily_data)}")
            print(f"Doppler range: {daily_data['doppler_hz'].min():.1f} to {daily_data['doppler_hz'].max():.1f} Hz")
            print(f"Velocity range: {daily_data['radial_velocity_kms'].min():.4f} to {daily_data['radial_velocity_kms'].max():.4f} km/s")
    
    if not plot_files:
        print("No plots generated - insufficient data for visualization")
    
    return plot_files

def main_stage_1():
    doppler_df = load_ramped_doppler_data('./data')
    validated_df = validate_ramped_doppler_data(doppler_df)
    #plot_files = plot_ramped_doppler_data(validated_df)
    output_file = 'processed_ramped_doppler_data.csv'
    validated_df.to_csv(output_file, index=False)
    return True

def load_horizons_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start_idx = -1
    for i, line in enumerate(lines):
        if "$$SOE" in line:
            start_idx = i + 1
            break
    if start_idx == -1:
        raise ValueError(f"No start of data ($$SOE) found in file {filepath}")
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
            except (ValueError, IndexError, AttributeError):
                i += 1
                continue
        else:
            i += 1

    if len(times_jd) < 2:
        raise ValueError(f"Insufficient data for interpolation in file {filepath}. Found {len(times_jd)} points.")
    times_jd = np.array(times_jd)
    positions = np.array(positions)
    velocities = np.array(velocities)
    times_tdb_s = (times_jd - 2451545.0) * 86400.0
    return times_tdb_s, positions, velocities

def create_interpolators(time_s, positions, velocities):
    pos_interp = CubicSpline(time_s, positions, axis=0)
    vel_interp = CubicSpline(time_s, velocities, axis=0)
    return pos_interp, vel_interp

def equations_of_motion_corrected(t, state, mercury_pos_interp, sun_pos_interp, earth_pos_interp, venus_pos_interp=None):
    """
    Equations of motion for SC in barycentric system.
    
    state: [x, y, z, vx, vy, vz] - barycentric coordinates (km, km/s)
    """
    r_sc = np.array(state[0:3])
    v_sc = np.array(state[3:6])
    
    # Positions from interpolators (barycentric)
    r_mercury = mercury_pos_interp(t)
    r_sun = sun_pos_interp(t)
    r_earth = earth_pos_interp(t)
    
    # Vectors from SC to bodies
    r_sc_sun = r_sun - r_sc  # vector from SC to Sun
    r_sc_mercury = r_mercury - r_sc
    r_sc_earth = r_earth - r_sc
    
    # Norms
    norm_sun = np.linalg.norm(r_sc_sun)
    norm_mercury = np.linalg.norm(r_sc_mercury)
    norm_earth = np.linalg.norm(r_sc_earth)
    
    # Protection against division by zero
    if norm_sun < 1: norm_sun = 1
    if norm_mercury < 1: norm_mercury = 1
    if norm_earth < 1: norm_earth = 1
    
    # Accelerations (attraction towards bodies)
    a_sun = GM_sun / norm_sun**3 * r_sc_sun
    
    # Mercury central
    a_mercury_central = GM_mercury / norm_mercury**3 * r_sc_mercury
    
    # J2 correction for Mercury
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
    
    # Venus acceleration if available
    a_venus = np.zeros(3)
    if venus_pos_interp is not None:
        r_venus = venus_pos_interp(t)
        r_sc_venus = r_venus - r_sc
        norm_venus = np.linalg.norm(r_sc_venus)
        if norm_venus < 1: norm_venus = 1
        a_venus = GM_venus / norm_venus**3 * r_sc_venus
    
    # Total acceleration
    total_acceleration = a_sun + a_mercury + a_earth + a_venus
    
    derivatives = np.concatenate([v_sc, total_acceleration])
    return derivatives

def get_initial_conditions_from_horizons(t0_tdb, mercury_pos_interp, mercury_vel_interp):
    """
    Gets initial conditions based on Horizons data for Mercury
    and adds elliptic orbital parameters around Mercury for late 2014/early 2015.
    """
    # Position and velocity of Mercury (barycentric)
    mercury_pos = mercury_pos_interp(t0_tdb)
    mercury_vel = mercury_vel_interp(t0_tdb)
    
    # Approximate MESSENGER parameters in early 2015 (low-altitude phase)
    peri_altitude = 25.0  # km (approximate pre-boost)
    r_peri = peri_altitude + R_mercury
    
    # Calculate orbital parameters based on period (as in old_main.py)
    period_hours = 8 + 17/60  # ~8 h 17 m
    period_sec = period_hours * 3600
    a = (GM_mercury * (period_sec / (2 * np.pi))**2)**(1/3)
    e = 1 - r_peri / a
    r_apo = a * (1 + e)
    apo_altitude = r_apo - R_mercury
    
    # Velocity at periapsis for elliptic orbit
    orbital_speed = np.sqrt(GM_mercury * (2 / r_peri - 1 / a))
    
    # Initial position relative to Mercury (at periapsis, along X)
    r_sc_mercury = np.array([r_peri, 0, 0])
    
    # Barycentric position
    r_sc = mercury_pos + r_sc_mercury
    
    # Initial velocity relative to Mercury (perpendicular, in Y)
    v_sc_mercury = np.array([0, orbital_speed, 0])
    
    # Approximate inclination ~83 deg (rotate velocity)
    incl = np.deg2rad(83.0)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(incl), -np.sin(incl)],
        [0, np.sin(incl), np.cos(incl)]
    ])
    v_sc_mercury = rotation_matrix @ v_sc_mercury  # Apply to velocity for argument of peri ~0
    
    # Barycentric velocity
    v_sc = mercury_vel + v_sc_mercury
    
    initial_state = np.concatenate([r_sc, v_sc])
    
    print(f"Initial conditions:")
    print(f"Mercury position: [{mercury_pos[0]:.2f}, {mercury_pos[1]:.2f}, {mercury_pos[2]:.2f}] km")
    print(f"Mercury velocity: [{mercury_vel[0]:.6f}, {mercury_vel[1]:.6f}, {mercury_vel[2]:.6f}] km/s")
    print(f"SC position: [{r_sc[0]:.2f}, {r_sc[1]:.2f}, {r_sc[2]:.2f}] km")
    print(f"SC velocity: [{v_sc[0]:.6f}, {v_sc[1]:.6f}, {v_sc[2]:.6f}] km/s")
    print(f"Periapsis altitude: {peri_altitude:.2f} km")
    print(f"Apoapsis altitude: {apo_altitude:.2f} km")
    print(f"Semi-major axis: {a:.2f} km")
    print(f"Eccentricity: {e:.6f}")
    print(f"Orbital period: {period_hours:.2f} hours")
    print(f"Velocity at periapsis: {orbital_speed:.6f} km/s")
    print(f"Inclination: 83 deg")
    
    return initial_state

def plot_mercury_orbit_detailed_corrected(integration_times, spacecraft_positions, body_interpolators):
    mercury_positions = np.array([body_interpolators['mercury'](t) for t in integration_times])
    spacecraft_rel = spacecraft_positions - mercury_positions
    distances = np.linalg.norm(spacecraft_rel, axis=1)

    min_distance = np.min(distances)
    max_distance = np.max(distances)
    eccentricity = (max_distance - min_distance) / (max_distance + min_distance)

    print(f"\nMESSENGER Orbit Analysis:")
    print(f"Min distance to Mercury: {min_distance:.2f} km")
    print(f"Max distance to Mercury: {max_distance:.2f} km")
    print(f"Orbit eccentricity: {eccentricity:.6f}")
    print(f"Mercury radius: {R_mercury} km")

    if min_distance < R_mercury:
        print("SC crashed into Mercury!")
    else:
        print("SC remains in stable orbit around Mercury")

    fig_3d = go.Figure()

    fig_3d.add_trace(go.Scatter3d(
        x=spacecraft_rel[:, 0],
        y=spacecraft_rel[:, 1],
        z=spacecraft_rel[:, 2],
        mode='lines',
        line=dict(width=4, color='blue'),
        name='MESSENGER Orbit'
    ))

    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = R_mercury * np.outer(np.cos(u), np.sin(v))
    y_sphere = R_mercury * np.outer(np.sin(u), np.sin(v))
    z_sphere = R_mercury * np.outer(np.ones(np.size(u)), np.cos(v))

    fig_3d.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        opacity=0.6,
        colorscale='Reds',
        showscale=False,
        name='Mercury'
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=[spacecraft_rel[0, 0]],
        y=[spacecraft_rel[0, 1]],
        z=[spacecraft_rel[0, 2]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Start'
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=[spacecraft_rel[-1, 0]],
        y=[spacecraft_rel[-1, 1]],
        z=[spacecraft_rel[-1, 2]],
        mode='markers',
        marker=dict(size=8, color='darkred'),
        name='End'
    ))

    min_idx = np.argmin(distances)
    max_idx = np.argmax(distances)

    fig_3d.add_trace(go.Scatter3d(
        x=[spacecraft_rel[min_idx, 0]],
        y=[spacecraft_rel[min_idx, 1]],
        z=[spacecraft_rel[min_idx, 2]],
        mode='markers',
        marker=dict(size=8, color='orange', symbol='diamond'),
        name='Periapsis'
    ))

    fig_3d.add_trace(go.Scatter3d(
        x=[spacecraft_rel[max_idx, 0]],
        y=[spacecraft_rel[max_idx, 1]],
        z=[spacecraft_rel[max_idx, 2]],
        mode='markers',
        marker=dict(size=8, color='purple', symbol='diamond'),
        name='Apoapsis'
    ))

    fig_3d.update_layout(
        title="MESSENGER Orbit around Mercury (Interactive 3D)",
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data'
        ),
        width=900,
        height=700
    )

    plotly_file = 'mercury_orbit_3d_plotly.html'
    fig_3d.write_html(plotly_file)
    print(f"3D orbit plot saved as {plotly_file}")

    os.makedirs('plots', exist_ok=True)

    time_hours = (integration_times - integration_times[0]) / 3600

    fig_2d, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_hours, distances, 'g-', linewidth=2, label='Distance to Mercury')
    ax.axhline(y=R_mercury, color='r', linestyle='--', linewidth=2,
                label=f'Mercury Radius ({R_mercury} km)')
    ax.axhline(y=min_distance, color='orange', linestyle=':', linewidth=2,
                label=f'Min: {min_distance:.1f} km')
    ax.axhline(y=max_distance, color='purple', linestyle=':', linewidth=2,
                label=f'Max: {max_distance:.1f} km')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Distance to Mercury (km)')
    ax.set_title('MESSENGER Orbital Distance from Mercury over Time', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    matplotlib_file = 'plots/mercury_distance_over_time_matplotlib.png'
    plt.savefig(matplotlib_file, dpi=300, bbox_inches='tight')
    print(f"Distance plot saved as {matplotlib_file}")
    #plt.show()

    return fig_3d, fig_2d

def plot_solar_system_full_data(body_interpolators, body_times):
    fig = go.Figure()

    colors = {'sun': 'gold', 'mercury': 'gray', 'venus': 'orange', 'earth': 'blue'}
    labels = {'sun': 'Sun', 'mercury': 'Mercury', 'venus': 'Venus', 'earth': 'Earth'}

    for body_name in ['sun', 'mercury', 'venus', 'earth']:
        if body_name in body_interpolators and body_name in body_times:
            times = body_times[body_name]
            positions = np.array([body_interpolators[body_name](t) for t in times])
            positions_scaled = positions / 1e6

            fig.add_trace(go.Scatter3d(
                x=positions_scaled[:, 0],
                y=positions_scaled[:, 1],
                z=positions_scaled[:, 2],
                mode='lines',
                line=dict(width=4, color=colors[body_name]),
                name=labels[body_name]
            ))

            fig.add_trace(go.Scatter3d(
                x=[positions_scaled[0, 0]],
                y=[positions_scaled[0, 1]],
                z=[positions_scaled[0, 2]],
                mode='markers',
                marker=dict(size=5, color=colors[body_name]),
                name=f'{labels[body_name]} Start',
                showlegend=False
            ))

            fig.add_trace(go.Scatter3d(
                x=[positions_scaled[-1, 0]],
                y=[positions_scaled[-1, 1]],
                z=[positions_scaled[-1, 2]],
                mode='markers',
                marker=dict(size=7, color=colors[body_name], symbol='diamond'),
                name=f'{labels[body_name]} End',
                showlegend=False
            ))

    fig.update_layout(
        title="Solar System - 3D View (Full Ephemeris Data)<br>Barycentric Coordinates",
        scene=dict(
            xaxis_title='X (million km)',
            yaxis_title='Y (million km)',
            zaxis_title='Z (million km)',
            aspectmode='cube'
        ),
        width=900,
        height=700
    )

    plot_file = 'solar_system_full_ephemeris_plotly.html'
    fig.write_html(plot_file)
    print(f"Solar system plot created using FULL ephemeris data and saved as {plot_file}")

    return fig

def main_stage_2():
    
    # Load Horizons data
    ephemeris_files = {
        'sun': 'horizons_results_sun.txt',
        'earth': 'horizons_results_earth.txt', 
        'venus': 'horizons_results_venus.txt',
        'mercury': 'horizons_results_mercury.txt'
    }
    
    body_interpolators = {}
    body_vel_interpolators = {}
    body_times = {}  # Store original times for full data plotting
    
    for body_name, filename in ephemeris_files.items():
        try:
            times, positions, velocities = load_horizons_data(filename)
            pos_interp, vel_interp = create_interpolators(times, positions, velocities)
            body_interpolators[body_name] = pos_interp
            body_vel_interpolators[body_name] = vel_interp
            body_times[body_name] = times  # Store original times
            print(f"{body_name.capitalize()} data loaded: {len(times)} points.")
        except Exception as e:
            print(f"Error loading {body_name} data: {e}")
    
    if 'mercury' not in body_interpolators or 'sun' not in body_interpolators or 'earth' not in body_interpolators:
        print("Not all required Horizons data loaded successfully. Exiting.")
        return False
    
    # Create SOLAR SYSTEM plot using FULL data from files
    print("\nCreating solar system plot with FULL ephemeris data...")
    plot_solar_system_full_data(body_interpolators, body_times)
    
    # Set initial time for MESSENGER orbit simulation
    try:
        t0_utc = pd.to_datetime('2015-01-02 21:20:12.500000+00:00', utc=True)
    except:
        t0_utc = pd.to_datetime('2015-01-02 21:20:12.500000+00:00', utc=True)
    
    JD_t0 = t0_utc.to_julian_date()
    TDB_start_from_J2000 = (JD_t0 - 2451545.0) * 86400.0
    
    print(f"MESSENGER simulation start time: {t0_utc}")
    print(f"TDB time from J2000: {TDB_start_from_J2000:.2f} seconds")
    
    # Get initial conditions
    initial_state = get_initial_conditions_from_horizons(
        TDB_start_from_J2000, 
        body_interpolators['mercury'], 
        body_vel_interpolators['mercury']
    )
    
    # Time span for integration
    t_span = [TDB_start_from_J2000, TDB_start_from_J2000 + 86400.0]  # 1 day
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    print(f"\nIntegrating MESSENGER orbit over {t_span[1]-t_span[0]:.0f} seconds ({((t_span[1]-t_span[0])/3600):.1f} hours)")
    
    # Integration with corrected equations
    try:
        solution = solve_ivp(
            fun=lambda t, state: equations_of_motion_corrected(
                t, state, 
                body_interpolators['mercury'], 
                body_interpolators['sun'], 
                body_interpolators['earth'],
                body_interpolators.get('venus')
            ),
            t_span=t_span,
            y0=initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8
        )
        
        if solution.success:
            print("MESSENGER orbit integration successful!")
            
            spacecraft_positions = solution.y[:3].T
            integration_times = solution.t
            
            # Create the detailed Mercury orbit plot
            print("\nCreating detailed MESSENGER orbit plot...")
            plot_mercury_orbit_detailed_corrected(integration_times, spacecraft_positions, body_interpolators)
            
            return True
        else:
            print("Integration failed.")
            print(solution.message)
            return False
            
    except Exception as e:
        print(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    #main_stage_1()
    main_stage_2()