import numpy as np
import plotly.graph_objects as go
from drone_simulation import DroneSimulator

def generate_stochastic_cloud():
    print("Generating Stochastic 3D Probability Cloud...")
    
    # Initialize Simulator (Just to get params and basic physics)
    sim = DroneSimulator()
    
    # Simulation Parameters
    N_TRAJECTORIES = 100  # Number of particles in the cloud
    DT = 0.1              # Time step (s)
    MAX_TIME = 60.0       # Max simulation time
    
    # Initial State (Lost Power Event at high altitude)
    init_pos = np.array([0.0, 0.0, 300.0]) # x, y, z
    init_vel = np.array([5.0, 0.0, 0.0])   # Initial velocity (e.g. was flying East)
    
    # Base Weather Conditions
    base_wind_speed = 12.0 # m/s
    base_wind_dir = 0.0    # Blowing East
    
    # Storage for plotting
    all_paths_x = []
    all_paths_y = []
    all_paths_z = []
    
    final_points_x = []
    final_points_y = []
    final_points_z = []

    print(f"Simulating {N_TRAJECTORIES} stochastic trajectories...")

    for i in range(N_TRAJECTORIES):
        # 1. Scenario Randomization (Global Uncertainty)
        # Each trajectory has a slightly different mean wind
        run_wind_speed = np.random.normal(base_wind_speed, 2.0) 
        run_wind_dir = np.random.normal(base_wind_dir, 0.1)
        
        # 2. Integration Loop (Phase 2: Uncontrolled Descent)
        # We implement a custom Euler-Maruyama integration here to add noise
        t = 0
        pos = init_pos.copy()
        vel = init_vel.copy()
        
        path_x = [pos[0]]
        path_y = [pos[1]]
        path_z = [pos[2]]
        
        while pos[2] > 0 and t < MAX_TIME:
            # --- Deterministic Physics (Mean Field) ---
            # Wind Shear Model
            altitude_factor = (max(pos[2], 1.0) / 10.0) ** 0.15
            wind_v = run_wind_speed * altitude_factor
            wind_vec = np.array([
                wind_v * np.cos(run_wind_dir),
                wind_v * np.sin(run_wind_dir),
                0.0
            ])
            
            # Relative Velocity
            v_rel = vel - wind_vec
            v_rel_mag = np.linalg.norm(v_rel)
            
            # Drag Force (Quadratic)
            if v_rel_mag > 0:
                drag = 0.5 * sim.rho * sim.cd * sim.area * v_rel_mag * (-v_rel)
            else:
                drag = np.zeros(3)
                
            # Gravity
            gravity = np.array([0.0, 0.0, -sim.g])
            
            # Total Deterministic Acceleration
            acc_det = (drag / sim.mass) + gravity
            
            # --- Stochastic Physics (Turbulence/Gusts) ---
            # Modeled as additive Gaussian noise on acceleration (Langevin dynamics)
            # Sigma represents turbulence intensity
            sigma = 3.0 
            noise = np.random.normal(0, sigma, 3) * np.sqrt(DT)
            
            # Update Velocity (Euler-Maruyama)
            vel += acc_det * DT + noise
            
            # Update Position
            pos += vel * DT
            t += DT
            
            path_x.append(pos[0])
            path_y.append(pos[1])
            path_z.append(pos[2])
        
        all_paths_x.append(path_x)
        all_paths_y.append(path_y)
        all_paths_z.append(path_z)
        
        final_points_x.append(path_x[-1])
        final_points_y.append(path_y[-1])
        final_points_z.append(path_z[-1])

    # 3. Visualization using Plotly
    fig = go.Figure()

    # Plot Trajectories (Thin lines)
    for i in range(N_TRAJECTORIES):
        fig.add_trace(go.Scatter3d(
            x=all_paths_x[i], y=all_paths_y[i], z=all_paths_z[i],
            mode='lines',
            line=dict(color='blue', width=2),
            opacity=0.3,
            showlegend=False,
            name=f'Path {i}'
        ))

    # Plot Crash Points (Red Cloud)
    fig.add_trace(go.Scatter3d(
        x=final_points_x, y=final_points_y, z=final_points_z,
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            symbol='x',
            opacity=0.8
        ),
        name='Impact Distribution'
    ))

    # Plot Start Point
    fig.add_trace(go.Scatter3d(
        x=[init_pos[0]], y=[init_pos[1]], z=[init_pos[2]],
        mode='markers',
        marker=dict(size=3, color='black', symbol='circle'),
        name='Failure Point'
    ))

    fig.update_layout(
        title="3D Probabilistic Trajectory Cloud (Uncontrolled Descent with Turbulence)",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Altitude (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_file = "probability_cloud_3d.html"
    fig.write_html(output_file)
    print(f"Visualization saved to '{output_file}'")

if __name__ == "__main__":
    generate_stochastic_cloud()
