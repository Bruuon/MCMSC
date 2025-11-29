import numpy as np
import plotly.graph_objects as go
from drone_simulation import DroneSimulator

def generate_3d_visualization():
    print("Generating 3D Trajectory Cloud...")
    
    # 1. Setup Simulation
    # Scenario: Drone is at (0,0) altitude 300m when Lost Link occurs.
    # Wind is blowing from West to East (0 rad) at 15 m/s.
    initial_pos = [0, 0, 300]
    base_wind = {'speed': 15.0, 'dir': 0.0} 
    
    sim = DroneSimulator()
    
    # We will simulate the "LAND" strategy (Recommended)
    # We run N simulations with random wind perturbations to show uncertainty
    N_SIMS = 50
    
    fig = go.Figure()
    
    print(f"Simulating {N_SIMS} trajectories...")
    
    # Store all final points to show the "Crash Zone"
    crash_x = []
    crash_y = []
    crash_z = []
    
    for i in range(N_SIMS):
        # Perturb wind to simulate gusts/uncertainty
        # Wind speed variation: +/- 3 m/s
        # Wind dir variation: +/- 10 degrees
        sim_wind = {
            'speed': np.random.normal(base_wind['speed'], 3.0),
            'dir': np.random.normal(base_wind['dir'], 0.2)
        }
        
        # Simulate
        # t_fail is irrelevant for LAND strategy (it lands immediately), 
        # but we pass a value anyway.
        final_state, t, y = sim.simulate_mission(initial_pos, 'LAND', sim_wind, t_fail=0)
        
        # Extract path
        xs = y[0, :]
        ys = y[1, :]
        zs = y[2, :]
        
        # Add trace to 3D plot
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='lines',
            line=dict(color='blue', width=2),
            opacity=0.3,
            name=f'Sim {i+1}',
            showlegend=False
        ))
        
        crash_x.append(xs[-1])
        crash_y.append(ys[-1])
        crash_z.append(zs[-1])

    # Add "Crash Zone" (Final Points)
    fig.add_trace(go.Scatter3d(
        x=crash_x, y=crash_y, z=crash_z,
        mode='markers',
        marker=dict(size=4, color='red', symbol='x'),
        name='Predicted Crash Points'
    ))

    # Add Start Point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[300],
        mode='markers',
        marker=dict(size=6, color='green', symbol='circle'),
        name='Lost Link Event'
    ))

    # Layout
    fig.update_layout(
        title="Drone Trajectory Probability Cloud (Strategy: LAND, Wind: 15m/s West->East)",
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Altitude (m)',
            aspectmode='data' # Keep 1:1:1 scale
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save
    output_file = "trajectory_cloud_3d.html"
    fig.write_html(output_file)
    print(f"3D Visualization saved to '{output_file}'. Open this file in your browser.")

if __name__ == "__main__":
    generate_3d_visualization()
