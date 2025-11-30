import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import glob

# --- Constants ---
GM_mercury = 22031.86855  # km^3/s^2 (from Horizons)
GM_sun = 132712440041.9394  # km^3/s^2
GM_earth = 398600.4415  # km^3/s^2
J2_mercury = 0.00006
R_mercury = 2439.7  # km (more accurate)
c = 299792.458  # km/s

# --- Functions for loading Horizons data ---
def load_horizons_data(filepath):
    """Loads JPL Horizons data from file."""
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
                
            except (ValueError, IndexError, AttributeError) as e:
                print(f"Warning: Parsing error at line {i}: {e}")
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

    print(f"Loaded {len(times_jd)} points from {filepath}")
    return times_tdb_s, positions, velocities

def create_interpolators(time_s, positions, velocities):
    """Creates interpolators for position and velocity."""
    pos_interp = CubicSpline(time_s, positions, axis=0)
    vel_interp = CubicSpline(time_s, velocities, axis=0)
    return pos_interp, vel_interp

# --- Loading Horizons data ---
try:
    sun_times, sun_pos, sun_vel = load_horizons_data('horizons_results_sun.txt')
    print(f"✅ Sun data loaded: {len(sun_times)} points.")
except Exception as e:
    print(f"❌ Error loading Sun data: {e}")
    sun_times, sun_pos, sun_vel = None, None, None

try:
    earth_times, earth_pos, earth_vel = load_horizons_data('horizons_results_earth.txt')
    print(f"✅ Earth data loaded: {len(earth_times)} points.")
except Exception as e:
    print(f"❌ Error loading Earth data: {e}")
    earth_times, earth_pos, earth_vel = None, None, None

try:
    mercury_times, mercury_pos, mercury_vel = load_horizons_data('horizons_results_mercury.txt')
    print(f"✅ Mercury data loaded: {len(mercury_times)} points.")
except Exception as e:
    print(f"❌ Error loading Mercury data: {e}")
    mercury_times, mercury_pos, mercury_vel = None, None, None

if sun_times is None or earth_times is None or mercury_times is None:
    print("❌ Not all Horizons data loaded successfully. Exiting.")
    exit()

# Create interpolators
try:
    sun_pos_interp, sun_vel_interp = create_interpolators(sun_times, sun_pos, sun_vel)
    earth_pos_interp, earth_vel_interp = create_interpolators(earth_times, earth_pos, earth_vel)
    mercury_pos_interp, mercury_vel_interp = create_interpolators(mercury_times, mercury_pos, mercury_vel)
    print("✅ Interpolators created.")
except Exception as e:
    print(f"❌ Error creating interpolators: {e}")
    exit()

# --- Corrected equations of motion ---
def equations_of_motion_corrected(t, state, mercury_pos_interp, sun_pos_interp, earth_pos_interp):
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
    
    # Total acceleration
    total_acceleration = a_sun + a_mercury + a_earth
    
    derivatives = np.concatenate([v_sc, total_acceleration])
    return derivatives

# --- Initial conditions for MESSENGER-like orbit ---
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
    
    print(f"🎯 Initial conditions:")
    print(f"   Mercury position: [{mercury_pos[0]:.2f}, {mercury_pos[1]:.2f}, {mercury_pos[2]:.2f}] km")
    print(f"   Mercury velocity: [{mercury_vel[0]:.6f}, {mercury_vel[1]:.6f}, {mercury_vel[2]:.6f}] km/s")
    print(f"   SC position: [{r_sc[0]:.2f}, {r_sc[1]:.2f}, {r_sc[2]:.2f}] km")
    print(f"   SC velocity: [{v_sc[0]:.6f}, {v_sc[1]:.6f}, {v_sc[2]:.6f}] km/s")
    print(f"   Periapsis altitude: {peri_altitude:.2f} km")
    print(f"   Apoapsis altitude: {apo_altitude:.2f} km")
    print(f"   Semi-major axis: {a:.2f} km")
    print(f"   Eccentricity: {e:.6f}")
    print(f"   Orbital period: {period_hours:.2f} hours")
    print(f"   Velocity at periapsis: {orbital_speed:.6f} km/s")
    print(f"   Inclination: 83 deg")
    
    return initial_state

# --- Main calculation ---
print("\n🚀 Calculating MESSENGER-like orbit around Mercury...")

# Time start (from Doppler data or default)
try:
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

print(f"⏰ Start time: {t0_utc}")
print(f"🌍 TDB time from J2000: {TDB_start_from_J2000:.2f} seconds")

# Get initial conditions
initial_state = get_initial_conditions_from_horizons(
    TDB_start_from_J2000, 
    mercury_pos_interp, 
    mercury_vel_interp
)

# Time span for integration (1 day)
t_span = [TDB_start_from_J2000, TDB_start_from_J2000 + 86400.0]  # 1 day
t_eval = np.linspace(t_span[0], t_span[1], 1000)

print(f"\n🔄 Integrating over {t_span[1]-t_span[0]:.0f} seconds ({((t_span[1]-t_span[0])/3600):.1f} hours)")

# Integration
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

    # --- Improved visualization ---
    if solution.success:
        print("✅ Integration successful!")
        
        # Analysis
        positions = solution.y[:3].T  # Note: .T for shape
        times = solution.t
        
        # Mercury positions over time
        mercury_positions = np.array([mercury_pos_interp(t) for t in times])
        distances_to_mercury = np.linalg.norm(positions - mercury_positions, axis=1)
        relative_positions = positions - mercury_positions
        
        min_distance = np.min(distances_to_mercury)
        max_distance = np.max(distances_to_mercury)
        eccentricity = (max_distance - min_distance) / (max_distance + min_distance)
        
        print(f"\n📊 Results:")
        print(f"   Min distance to Mercury: {min_distance:.2f} km")
        print(f"   Max distance to Mercury: {max_distance:.2f} km") 
        print(f"   Orbit eccentricity: {eccentricity:.6f}")
        print(f"   Mercury radius: {R_mercury} km")
        
        if min_distance < R_mercury:
            print("   ⚠️ SC crashed into Mercury!")
        else:
            print("   ✅ SC remains in stable orbit around Mercury")
        
        # Visualization
        fig = plt.figure(figsize=(20, 5))
        
        # 1. Relative trajectory (Mercury system) - Main plot
        ax1 = fig.add_subplot(141)
        ax1.plot(relative_positions[:, 0], relative_positions[:, 1], 'b-', linewidth=2, label='SC Trajectory')
        
        # Add Mercury as circle
        mercury_circle = plt.Circle((0, 0), R_mercury, color='red', alpha=0.7, label='Mercury')
        ax1.add_patch(mercury_circle)
        
        # Start point
        ax1.scatter([relative_positions[0, 0]], [relative_positions[0, 1]], 
                    color='green', s=100, marker='o', label='Start', zorder=5)
        
        ax1.set_xlabel('X relative to Mercury (km)')
        ax1.set_ylabel('Y relative to Mercury (km)')
        ax1.set_title('SC Orbit around Mercury\n(top view)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Set equal limits
        max_range = max(np.abs(relative_positions[:, 0]).max(), 
                        np.abs(relative_positions[:, 1]).max()) * 1.1
        ax1.set_xlim(-max_range, max_range)
        ax1.set_ylim(-max_range, max_range)
        
        # 2. 3D Orbit in Mercury system
        ax2 = fig.add_subplot(142, projection='3d')
        ax2.plot(relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2], 
                 'b-', linewidth=2, label='SC Orbit')
        
        # Mercury sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = R_mercury * np.outer(np.cos(u), np.sin(v))
        y = R_mercury * np.outer(np.sin(u), np.sin(v)) 
        z = R_mercury * np.outer(np.ones(np.size(u)), np.cos(v))
        ax2.plot_surface(x, y, z, color='red', alpha=0.3, label='Mercury')
        
        ax2.scatter([0], [0], [0], color='red', s=100, label='Mercury Center')
        ax2.scatter([relative_positions[0, 0]], [relative_positions[0, 1]], [relative_positions[0, 2]],
                    color='green', s=100, marker='o', label='Start')
        
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.set_zlabel('Z (km)')
        ax2.set_title('3D SC Orbit around Mercury')
        ax2.legend()
        
        # 3. Distance from Mercury
        ax3 = fig.add_subplot(143)
        ax3.plot((times - times[0])/3600, distances_to_mercury, 'g-', linewidth=2)
        ax3.axhline(y=R_mercury, color='r', linestyle='--', linewidth=2, label='Mercury Radius')
        ax3.axhline(y=min_distance, color='orange', linestyle=':', linewidth=1, label=f'Min: {min_distance:.1f} km')
        ax3.axhline(y=max_distance, color='purple', linestyle=':', linewidth=1, label=f'Max: {max_distance:.1f} km')
        
        ax3.set_xlabel('Time (hours)')
        ax3.set_ylabel('Distance to Mercury (km)')
        ax3.set_title('SC Orbit Altitude over Mercury')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Heliocentric system (scaled)
        ax4 = fig.add_subplot(144)
        
        # Scale to millions km
        scale_factor = 1e6
        positions_scaled = positions / scale_factor
        mercury_positions_scaled = mercury_positions / scale_factor
        
        ax4.plot(mercury_positions_scaled[:, 0], mercury_positions_scaled[:, 1], 
                 'r-', alpha=0.5, linewidth=1, label='Mercury Orbit')
        ax4.plot(positions_scaled[:, 0], positions_scaled[:, 1], 
                 'b-', alpha=0.7, linewidth=2, label='SC Trajectory')
        
        # Start positions
        ax4.scatter([mercury_positions_scaled[0, 0]], [mercury_positions_scaled[0, 1]], 
                    color='red', s=50, label='Mercury (start)')
        ax4.scatter([positions_scaled[0, 0]], [positions_scaled[0, 1]], 
                    color='blue', s=30, label='SC (start)')
        
        ax4.set_xlabel('X (millions km)')
        ax4.set_ylabel('Y (millions km)')
        ax4.set_title('Barycentric System\n(scaled)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Additional info
        print(f"\n📈 Additional orbit info:")
        print(f"   Semi-major axis: {(max_distance + min_distance)/2:.2f} km")
        print(f"   Approximate period: {len(times)/1000 * 24:.2f} hours")
        print(f"   Orbit shape: {'circular' if eccentricity < 0.1 else 'elliptical'}")
        
    else:
        print("❌ Integration failed.")
        print(solution.message)

except Exception as e:
    print(f"❌ Error during integration: {e}")
    import traceback
    traceback.print_exc()