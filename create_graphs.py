import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

dirname = 'plots'

def plot_mercury_orbit_detailed_corrected(integration_times, spacecraft_positions, body_interpolators, R_mercury, is_true_orbit=False):
    mercury_positions = np.array([body_interpolators['mercury'](t) for t in integration_times])
    spacecraft_rel = spacecraft_positions - mercury_positions

    distances = np.linalg.norm(spacecraft_rel, axis=1)

    min_distance = np.min(distances)
    max_distance = np.max(distances)
    eccentricity = (max_distance - min_distance) / (max_distance + min_distance)

    '''
    print(f"\nMESSENGER Orbit Analysis:")
    print(f"Min distance to Mercury: {min_distance:.2f} km")
    print(f"Max distance to Mercury: {max_distance:.2f} km")
    print(f"Orbit eccentricity: {eccentricity:.6f}")
    print(f"Mercury radius: {R_mercury} km")

    if min_distance < R_mercury:
        print("SC crashed into Mercury!")
    else:
        print("SC remains in stable orbit around Mercury")'''

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

    os.makedirs(dirname, exist_ok=True)
    
    plotly_file = f'{dirname}/true_mercury_orbit_3d_plotly.html' if is_true_orbit else f'{dirname}/mercury_orbit_3d_plotly.html'
    fig_3d.write_html(plotly_file)
    #print(f"3D orbit plot saved as {plotly_file}")

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
    matplotlib_file = f'{dirname}/mercury_distance_over_time_matplotlib.png'
    plt.savefig(matplotlib_file, dpi=300, bbox_inches='tight')
    #print(f"Distance plot saved as {matplotlib_file}")
    #plt.show()

    return fig_3d, fig_2d

def plot_solar_system_full_data(body_interpolators, body_times):
    fig = go.Figure()

    colors = {
        'sun': 'gold', 
        'mercury': 'gray', 
        'venus': 'orange', 
        'earth': 'blue',
        'mars': 'red',
        'jupiter': '#D4AF37'
    }

    labels = {
        'sun': 'Sun', 
        'mercury': 'Mercury', 
        'venus': 'Venus', 
        'earth': 'Earth',
        'mars': 'Mars',
        'jupiter': 'Jupiter'
    }

    bodies_to_plot = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter']

    all_positions = []
    
    for body_name in bodies_to_plot:
        if body_name in body_interpolators and body_name in body_times:
            times = body_times[body_name]
            positions = np.array([body_interpolators[body_name](t) for t in times])
            positions_scaled = positions / 1e6
            all_positions.append(positions_scaled)
            
            # Линия орбиты
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
                marker=dict(size=6, color=colors[body_name], symbol='circle'),
                name=f'{labels[body_name]} Start',
                showlegend=False
            ))

            fig.add_trace(go.Scatter3d(
                x=[positions_scaled[-1, 0]],
                y=[positions_scaled[-1, 1]],
                z=[positions_scaled[-1, 2]],
                mode='markers',
                marker=dict(size=8, color=colors[body_name], symbol='diamond', line=dict(width=2, color='white')),
                name=f'{labels[body_name]} End',
                showlegend=False
            ))

    if all_positions:
        all_positions_flat = np.vstack(all_positions)
        x_min, y_min, z_min = all_positions_flat.min(axis=0)
        x_max, y_max, z_max = all_positions_flat.max(axis=0)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2
        
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        padding = max_range * 0.1

        axis_range = max_range + padding
        axis_min = center_x - axis_range/2
        axis_max = center_x + axis_range/2

    fig.update_layout(
        title="Solar System - 3D View (Full Ephemeris Data)<br>Barycentric Coordinates",
        scene=dict(
            xaxis_title='X (million km)',
            yaxis_title='Y (million km)',
            zaxis_title='Z (million km)',
            aspectmode='cube',
            xaxis=dict(range=[axis_min, axis_max]),
            yaxis=dict(range=[axis_min, axis_max]),
            zaxis=dict(range=[axis_min, axis_max]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        width=1000,
        height=800,
        margin=dict(r=20, l=10, b=10, t=50)
    )

    plot_file = 'solar_system_full_ephemeris_plotly.html'
    fig.write_html(dirname+'/'+plot_file)
    #print(f"Solar system plot created using FULL ephemeris data and saved as {plot_file}")
    #print(f"Equal scale applied to all axes with range: [{axis_min:.1f}, {axis_max:.1f}] million km")

    return fig

def plot_messenger_doppler_data(doppler_df: pd.DataFrame, DSN_STATIONS):
    os.makedirs('plots', exist_ok=True)

    min_year = doppler_df['time_utc'].dt.year.min()
    max_year = doppler_df['time_utc'].dt.year.max()
    print(f"Диапазон лет в данных: {min_year} - {max_year}")

    start_time = doppler_df['time_utc'].min()
    doppler_df['days_from_start'] = (doppler_df['time_utc'] - start_time).dt.total_seconds() / 86400
    
    plot_files = []

    plt.figure(figsize=(15, 8))

    stations = sorted(doppler_df['station_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(stations)))
    
    for i, station_id in enumerate(stations):
        station_data = doppler_df[doppler_df['station_id'] == station_id]
        station_name = DSN_STATIONS.get(station_id, {}).get('name', f'Station {station_id}')
        
        if len(station_data) > 0:
            plt.scatter(station_data['days_from_start'], station_data['observable_hz'],
                       c=[colors[i]], s=10, alpha=0.5, label=station_name)
    
    plt.xlabel('Дни от начала наблюдений', fontsize=12)
    plt.ylabel('Doppler (Гц)', fontsize=12)
    plt.title(f'Doppler (Гц) MESSENGER ({start_time.date()} - {doppler_df["time_utc"].max().date()})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = 'plots/messenger_all_stations_doppler.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plot_files.append(plot_filename)
    print(f"Сохранен общий график: {plot_filename}")
    plt.close()

    for station_id in stations:
        station_data = doppler_df[doppler_df['station_id'] == station_id].copy()
        station_name = DSN_STATIONS.get(station_id, {}).get('name', f'Station {station_id}')
        
        if len(station_data) < 10:
            print(f"Пропуск станции {station_id}: недостаточно данных ({len(station_data)} записей)")
            continue

        station_data = station_data.sort_values('time_utc')

        station_min_year = station_data['time_utc'].dt.year.min()
        station_max_year = station_data['time_utc'].dt.year.max()

        fig, axes = plt.subplots(figsize=(15, 12), sharex=True)

        axes.plot(station_data['time_utc'], station_data['observable_hz'],
                    'b-', linewidth=1.5, alpha=0.7, marker='o', markersize=3)
        axes.set_title(f'Doppler смещение - {station_name}', fontsize=14, fontweight='bold')
        axes.set_ylabel('Doppler (Гц)', fontsize=12)
        axes.grid(True, alpha=0.3)

        axes.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d\n%H:%M'))
        axes.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_filename = f'plots/messenger_station_{station_id}_analysis.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plot_files.append(plot_filename)
        print(f"Сохранен график для станции {station_id}: {plot_filename}")
        plt.close()
    
    print(f"Всего создано графиков: {len(plot_files)}")
    return plot_files

def plot_mercury_messenger_24h_closeup(body_interpolators, body_times):
    fig = go.Figure()

    bodies_to_plot = ['mercury', 'messenger']
    colors = {'mercury': '#8A8A8A', 'messenger': '#00FFFF'}
    names = {'mercury': 'Mercury', 'messenger': 'MESSENGER'}

    all_times = []
    for body_name in bodies_to_plot:
        if body_name in body_times:
            all_times.extend(body_times[body_name])
    
    if not all_times:
        print("ERROR: No time data available")
        return fig
    
    max_time = max(all_times)
    min_time = max_time - 40000
    #print(f"Time window for 24-hour view (TDB seconds since J2000):")
    #print(f"  Start: {min_time:.3f}")
    #print(f"  End:   {max_time:.3f}")
    #print(f"  Duration: 24.0 hours")

    all_positions_24h = []
    body_data_24h = {}
    
    for body_name in bodies_to_plot:
        if body_name not in body_interpolators or body_name not in body_times:
            print(f"Warning: Data for {body_name} not available. Skipping.")
            continue

        times = np.array(body_times[body_name])
        mask = (times >= min_time) & (times <= max_time)
        filtered_times = times[mask]
        
        if len(filtered_times) < 2:
            #print(f"Warning: Insufficient data for {body_name} in 24h window. Using last 50 points instead.")
            filtered_times = times[-50:] if len(times) >= 50 else times

        positions = np.array([body_interpolators[body_name](float(t)) for t in filtered_times])
        positions_scaled = positions / 1e6  # млн км
        
        body_data_24h[body_name] = {
            'times': filtered_times,
            'positions': positions_scaled
        }
        all_positions_24h.append(positions_scaled)
        
        #print(f"\n{names[body_name]} data in 24h window:")
        #print(f"  Points: {len(positions_scaled)}")
        #print(f"  Time range: {filtered_times[0]:.3f} to {filtered_times[-1]:.3f} TDB seconds")
        #print(f"  Position range (million km):")
        #print(f"    X: [{positions_scaled[:,0].min():.6f}, {positions_scaled[:,0].max():.6f}]")
        #print(f"    Y: [{positions_scaled[:,1].min():.6f}, {positions_scaled[:,1].max():.6f}]")
        #print(f"    Z: [{positions_scaled[:,2].min():.6f}, {positions_scaled[:,2].max():.6f}]")
    
    if not body_data_24h:
        #print("ERROR: No valid data for 24-hour window")
        return fig

    if 'mercury' in body_data_24h:
        mercury_positions = body_data_24h['mercury']['positions']
        center_x = (mercury_positions[:, 0].min() + mercury_positions[:, 0].max()) / 2
        center_y = (mercury_positions[:, 1].min() + mercury_positions[:, 1].max()) / 2
        center_z = (mercury_positions[:, 2].min() + mercury_positions[:, 2].max()) / 2
        #print(f"\nCenter based on Mercury trajectory midpoint: ({center_x:.6f}, {center_y:.6f}, {center_z:.6f}) million km")
    else:
        all_positions_combined = np.vstack([data['positions'] for data in body_data_24h.values()])
        center_x = all_positions_combined[:, 0].mean()
        center_y = all_positions_combined[:, 1].mean()
        center_z = all_positions_combined[:, 2].mean()
        #print(f"\nCenter based on all positions: ({center_x:.6f}, {center_y:.6f}, {center_z:.6f}) million km")

    all_positions_combined = np.vstack([data['positions'] for data in body_data_24h.values()])
    distances = np.sqrt(
        (all_positions_combined[:, 0] - center_x) ** 2 +
        (all_positions_combined[:, 1] - center_y) ** 2 +
        (all_positions_combined[:, 2] - center_z) ** 2
    )
    
    max_distance = distances.max()
    #print(f"Maximum distance from center in 24h window: {max_distance:.6f} million km")

    padding = 0.4
    plot_range = max_distance * (1 + padding)

    plot_range = max(plot_range, 0.1)
    
    #print(f"Plot range (radius): {plot_range:.6f} million km")

    for body_name, data in body_data_24h.items():
        positions = data['positions']
        times = data['times']

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            line=dict(
                width=8 if body_name == 'messenger' else 5,
                color=colors[body_name],
                dash='dash' if body_name == 'messenger' else 'solid'
            ),
            name=f"{names[body_name]} (24h)",
            hovertemplate=(
                f"<b>{names[body_name]}</b><br>"
                "TDB time: %{text}<br>"
                "X: %{x:.6f}M km<br>"
                "Y: %{y:.6f}M km<br>"
                "Z: %{z:.6f}M km<extra></extra>"
            ),
            text=[f"{t:.3f}" for t in times]
        ))

        fig.add_trace(go.Scatter3d(
            x=[positions[0, 0]],
            y=[positions[0, 1]],
            z=[positions[0, 2]],
            mode='markers',
            marker=dict(
                size=10,
                color=colors[body_name],
                symbol='circle',
                line=dict(width=2, color='white' if body_name == 'mercury' else 'black')
            ),
            name=f"{names[body_name]} start",
            showlegend=False,
            hovertemplate=(
                f"<b>{names[body_name]} start</b><br>"
                f"TDB time: {times[0]:.3f}<br>"
                "X: %{x:.6f}M km<br>"
                "Y: %{y:.6f}M km<br>"
                "Z: %{z:.6f}M km<extra></extra>"
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=[positions[-1, 0]],
            y=[positions[-1, 1]],
            z=[positions[-1, 2]],
            mode='markers',
            marker=dict(
                size=12,
                color=colors[body_name],
                symbol='x' if body_name == 'messenger' else 'diamond',  # 'x' поддерживается
                line=dict(width=2, color='black' if body_name == 'messenger' else 'white')
            ),
            name=f"{names[body_name]} end",
            showlegend=False,
            hovertemplate=(
                f"<b>{names[body_name]} end</b><br>"
                f"TDB time: {times[-1]:.3f}<br>"
                "X: %{x:.6f}M km<br>"
                "Y: %{y:.6f}M km<br>"
                "Z: %{z:.6f}M km<extra></extra>"
            )
        ))

    fig.update_layout(
        title=(
            f"MESSENGER & Mercury - 24 Hour Close-up View<br>"
            f"Center: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f}) million km from Solar System Barycenter"
        ),
        scene=dict(
            xaxis_title='X (million km)',
            yaxis_title='Y (million km)',
            zaxis_title='Z (million km)',
            aspectmode='cube',
            xaxis=dict(range=[center_x - plot_range, center_x + plot_range]),
            yaxis=dict(range=[center_y - plot_range, center_y + plot_range]),
            zaxis=dict(range=[center_z - plot_range, center_z + plot_range]),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )
        ),
        width=1200,
        height=900,
        margin=dict(r=20, l=10, b=10, t=70),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=14)
        )
    )

    scale_note = (
        f"24-HOUR VIEW<br>"
        f"Center: Mercury trajectory midpoint<br>"
        f"Range: ±{plot_range:.4f} million km<br>"
        f"1 million km = 1,000,000 km"
    )
    
    fig.add_annotation(
        x=0.02,
        y=0.02,
        xref="paper",
        yref="paper",
        text=scale_note,
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.7)",
        bordercolor="gray",
        borderwidth=1
    )

    fig.add_trace(go.Scatter3d(
        x=[center_x],
        y=[center_y],
        z=[center_z],
        mode='markers',
        marker=dict(
            size=15,
            color='yellow',
            symbol='circle',
            line=dict(width=2, color='orange')
        ),
        name='Trajectory center',
        hovertemplate=(
            "<b>Trajectory center</b><br>"
            f"X: {center_x:.6f}M km<br>"
            f"Y: {center_y:.6f}M km<br>"
            f"Z: {center_z:.6f}M km<extra></extra>"
        )
    ))
    
    plot_file = dirname + '/mercury_messenger_24h_closeup.html'
    fig.write_html(plot_file)
        
    return fig

def plot_doppler_comparison(final_df: pd.DataFrame, DSN_STATIONS, output_dir: str = 'plots'):
    os.makedirs(output_dir, exist_ok=True)

    df = final_df.copy()
    if df['time_utc'].dtype == 'object' or df['time_utc'].dtype == 'string':
        try:
            df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
        except:
            try:
                df['time_utc'] = df['time_utc'].str.replace('+00:00', 'Z', regex=False)
                df['time_utc'] = pd.to_datetime(df['time_utc'], utc=True)
            except:
                try:
                    df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce', utc=True)
                except Exception as e:
                    print(f"Ошибка при преобразовании времени: {e}")
                    return
    
    for station_id in df['station_id'].unique():
        station_data = df[df['station_id'] == station_id].copy()
        station_name = DSN_STATIONS.get(station_id, {}).get('name', f'Station {station_id}')
        
        if len(station_data) < 10:
            continue

        station_data = station_data.sort_values('time_utc')

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        dpi_value = 100
        
        max_plot_points = 1000
        if len(station_data) > max_plot_points:
            plot_indices = np.linspace(0, len(station_data)-1, max_plot_points, dtype=int)
            plot_data = station_data.iloc[plot_indices]
        else:
            plot_data = station_data
        
        if 'measured_doppler_hz' in station_data.columns and 'theoretical_doppler_hz' in station_data.columns:
            axes[0].plot(plot_data['time_utc'], plot_data['measured_doppler_hz'], 
                         'b-', linewidth=0.5, alpha=0.7, label='Измеренный')
            axes[0].plot(plot_data['time_utc'], plot_data['theoretical_doppler_hz'],
                         'r-', linewidth=1, alpha=0.8, label='Теоретический')
            axes[0].set_title(f'Doppler Shift - {station_name}', fontsize=12)
            axes[0].set_ylabel('Doppler Shift (Hz)', fontsize=10)
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'Нет данных Doppler для сравнения',
                         ha='center', va='center', fontsize=10)
            axes[0].set_title(f'Doppler Shift - {station_name}', fontsize=12)

        if 'doppler_residual_hz' in station_data.columns:
            axes[1].plot(plot_data['time_utc'], plot_data['doppler_residual_hz'],
                         'g-', linewidth=0.5, alpha=0.7)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
            axes[1].set_title('Doppler Residuals', fontsize=12)
            axes[1].set_ylabel('Residual (Hz)', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            
            mean_res = station_data['doppler_residual_hz'].mean()
            std_res = station_data['doppler_residual_hz'].std()
            axes[1].text(0.02, 0.98, f'Mean: {mean_res:.2f} Hz\nStd: {std_res:.2f} Hz',
                         transform=axes[1].transAxes, fontsize=8,
                         verticalalignment='top')
        else:
            axes[1].text(0.5, 0.5, 'Нет данных residuals Doppler',
                         ha='center', va='center', fontsize=10)
            axes[1].set_title('Doppler Residuals', fontsize=12)

        if 'light_time_s' in station_data.columns:
            axes[2].plot(plot_data['time_utc'], plot_data['light_time_s'],
                         'm-', linewidth=1, alpha=0.7)
            axes[2].set_xlabel('Time (UTC)', fontsize=10)
            axes[2].set_ylabel('Light-time (s)', fontsize=10, color='m')
            axes[2].tick_params(axis='y', labelcolor='m')
            axes[2].set_title('Light-time', fontsize=12)
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Нет данных о light-time',
                         ha='center', va='center', fontsize=10)
            axes[2].set_title('Light-time', fontsize=12)

        if len(plot_data) > 0 and not plot_data['time_utc'].isna().all():
            valid_times = plot_data['time_utc'].dropna()
            if len(valid_times) > 1:
                time_range = valid_times.max() - valid_times.min()
                
                if time_range.total_seconds() < 3600:
                    num_ticks = 5
                    time_format = '%H:%M:%S'
                elif time_range.total_seconds() < 86400:
                    num_ticks = 6
                    time_format = '%H:%M\n%d/%m'
                elif time_range.total_seconds() < 604800:
                    num_ticks = 7
                    time_format = '%d/%m\n%H:%M'
                else:
                    num_ticks = 8
                    time_format = '%Y-%m-%d'

                for ax in axes:
                    ax.xaxis.set_major_locator(MaxNLocator(num_ticks))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        filename = f'{output_dir}/doppler_comparison_station_{station_id}.png'
        try:
            plt.savefig(filename, dpi=dpi_value, bbox_inches='tight')
            print(f"Сохранен график: {filename} (размер: {fig.get_size_inches()}, DPI: {dpi_value})")
        except Exception as e:
            print(f"Ошибка при сохранении графика {filename}: {e}")
            plt.savefig(filename, dpi=72, bbox_inches='tight')
            print(f"Сохранен с DPI=72")
        
        plt.close()

    if 'doppler_residual_hz' in df.columns and not df['time_utc'].isna().all():
        plt.figure(figsize=(14, 7))
        
        plot_every = max(1, len(df) // 10000)
        
        for station_id in df['station_id'].unique():
            station_data = df[df['station_id'] == station_id].copy()
            if len(station_data) > plot_every:
                station_data = station_data.iloc[::plot_every]

            station_data = station_data.dropna(subset=['time_utc', 'doppler_residual_hz'])
            
            if len(station_data) < 2:
                continue
                
            station_name = DSN_STATIONS.get(station_id, {}).get('name', f'St{station_id}')
            short_name = ''.join([word[0] for word in station_name.split() if word[0].isupper()])
            if not short_name:
                short_name = f'St{station_id}'
            
            plt.scatter(station_data['time_utc'], station_data['doppler_residual_hz'], 
                        s=2, alpha=0.3, label=short_name)
        
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Time (UTC)', fontsize=10)
        plt.ylabel('Doppler Residual (Hz)', fontsize=10)
        plt.title('Doppler Residuals for All Stations', fontsize=12)

        valid_times = df['time_utc'].dropna()
        if len(valid_times) > 1:
            time_range = valid_times.max() - valid_times.min()
            if time_range.total_seconds() < 3600:
                num_ticks = 6
                time_format = '%H:%M:%S'
            elif time_range.total_seconds() < 86400:
                num_ticks = 8
                time_format = '%H:%M\n%d/%m'
            elif time_range.total_seconds() < 604800:
                num_ticks = 10
                time_format = '%d/%m\n%H:%M'
            else:
                num_ticks = 12
                time_format = '%Y-%m-%d'
            
            plt.gca().xaxis.set_major_locator(MaxNLocator(num_ticks))
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(time_format))
            plt.xticks(rotation=45, ha='right', fontsize=8)
        
        plt.legend(title='Station', fontsize=8, ncol=3, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0)
        
        filename = f'{output_dir}/all_stations_doppler_residuals.png'
        try:
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            print(f"Сохранен сводный график: {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении сводного графика: {e}")
            plt.savefig(filename, dpi=72, bbox_inches='tight')
        
        plt.close()
