import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from drone_simulation import DroneSimulator

def generate_2d_distributions():
    # 1. Define the Scenario (One specific case from the "200 scenarios")
    # We pick a scenario that highlights the differences well.
    # Scenario: Moderate Wind, High Altitude (so drift is visible)
    scenario = {
        'wind_speed': 15.0,       # m/s (Strong wind)
        'wind_dir': np.radians(45), # Blowing North-East
        'initial_pos': [0, 0, 200], # Start at (0,0), Altitude 200m
        'origin': [3000, 3000, 0]   # Home is far away at (3000, 3000)
    }
    
    sim = DroneSimulator()
    N_RUNS = 500 # As per the user's reference image title
    
    strategies = ['LAND', 'HOVER', 'RTH']
    
    print(f"Generating 2D Probability Maps for Scenario: Wind={scenario['wind_speed']}m/s, Alt={scenario['initial_pos'][2]}m")
    
    for strategy in strategies:
        print(f"Simulating {strategy} ({N_RUNS} runs)...")
        crash_x = []
        crash_y = []
        
        for _ in range(N_RUNS):
            # Randomize parameters per run (Monte Carlo)
            # 1. Wind Uncertainty (Gusts/Variation)
            run_wind = {
                'speed': np.random.normal(scenario['wind_speed'], 2.0),
                'dir': np.random.normal(scenario['wind_dir'], 0.1)
            }
            
            # 2. Power Failure Time Uncertainty
            # Mean 5 mins (300s), max 10 mins
            t_fail = np.random.exponential(scale=300)
            if t_fail > 600: t_fail = 600
            if t_fail < 1: t_fail = 1
            
            # Simulate
            final_pos, _, _ = sim.simulate_mission(
                scenario['initial_pos'], 
                strategy, 
                run_wind, 
                t_fail, 
                origin=scenario['origin']
            )
            
            crash_x.append(final_pos[0])
            crash_y.append(final_pos[1])
            
        # --- Plotting (Mimicking crash_dist.png) ---
        plt.figure(figsize=(10, 8))
        
        # 1. Scatter points (Blue, small, semi-transparent)
        # "steelblue" matches the look well
        plt.scatter(crash_x, crash_y, c='steelblue', s=15, alpha=0.5, label='Crash Sites', zorder=2)
        
        # 2. KDE Plot (Red contours, filled)
        # We use 'fill=True' for the shaded effect and 'levels' for contours
        try:
            sns.kdeplot(x=crash_x, y=crash_y, cmap="Reds", fill=True, alpha=0.3, levels=10, zorder=1)
            sns.kdeplot(x=crash_x, y=crash_y, cmap="Reds", fill=False, linewidths=0.8, levels=10, zorder=1)
        except Exception as e:
            print(f"Warning: KDE failed for {strategy} (Data might be too concentrated): {e}")
        
        # 3. Start Point (Black dot)
        plt.scatter([scenario['initial_pos'][0]], [scenario['initial_pos'][1]], c='black', s=40, label='Start Point', zorder=3)
        
        # 4. Home Point (For RTH context)
        if strategy == 'RTH':
             plt.scatter([scenario['origin'][0]], [scenario['origin'][1]], c='green', marker='^', s=60, label='Home', zorder=3)

        # Styling
        plt.title(f'Drone Crash Location Probability Map ({N_RUNS} runs)\nStrategy: {strategy}', fontsize=14)
        plt.xlabel('X Position (m)', fontsize=12)
        plt.ylabel('Y Position (m)', fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.legend(loc='upper right')
        
        # Ensure aspect ratio is reasonable so circles look like circles
        plt.axis('equal')
        
        # Save
        filename = f'dist_{strategy}.png'
        plt.savefig(filename, dpi=300)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    generate_2d_distributions()
