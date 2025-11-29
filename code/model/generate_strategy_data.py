import numpy as np
import pandas as pd
from drone_simulation import DroneSimulator
import time

def generate_data():
    N_SCENARIOS = 200  # Number of data points for the decision tree
    N_SIMS_PER_STRATEGY = 30 # Monte Carlo runs to evaluate each strategy
    
    data = []
    
    print(f"Generating {N_SCENARIOS} scenarios...")
    
    for i in range(N_SCENARIOS):
        if i % 10 == 0:
            print(f"Processing scenario {i}/{N_SCENARIOS}...")
            
        # 1. Generate Scenario Features (The "Input" for Decision Tree)
        wind_speed = np.random.uniform(2.0, 25.0) # m/s
        wind_dir = np.random.uniform(0, 2*np.pi)
        
        # Drone state at moment of Lost Link
        dist_to_origin = np.random.uniform(100, 5000) # meters
        angle_to_origin = np.random.uniform(0, 2*np.pi)
        x0 = dist_to_origin * np.cos(angle_to_origin)
        y0 = dist_to_origin * np.sin(angle_to_origin)
        z0 = np.random.uniform(50, 300) # Altitude
        
        initial_pos = [x0, y0, z0]
        
        # Drone capabilities (could be features too, but let's keep fixed for now or vary slightly)
        drone_params = {
            'max_speed': 15.0,
            'cruise_speed': 10.0,
            'descent_speed': 3.0
        }
        sim = DroneSimulator(drone_params)
        
        # 2. Evaluate Strategies
        strategies = ['RTH', 'HOVER', 'LAND']
        scores = {}
        
        for strat in strategies:
            crash_positions = []
            for _ in range(N_SIMS_PER_STRATEGY):
                # Randomize Failure Time (The "Uncertainty")
                # Assume failure could happen anytime within next 10 mins (600s)
                # or until battery dies.
                t_fail = np.random.exponential(scale=300) # Mean 5 mins
                if t_fail > 600: t_fail = 600
                
                # Randomize Wind slightly per sim (Gusts)
                sim_wind = {
                    'speed': np.abs(np.random.normal(wind_speed, 2.0)),
                    'dir': np.random.normal(wind_dir, 0.1)
                }
                
                # Use a smaller max_step to prevent solver from getting stuck, 
                # but not too small to be slow.
                # Also, ensure t_fail is not 0 to avoid issues.
                if t_fail < 1.0: t_fail = 1.0

                try:
                    final_pos, _, _ = sim.simulate_mission(initial_pos, strat, sim_wind, t_fail)
                    crash_positions.append(final_pos[:2])
                except Exception as e:
                    # Fallback if simulation fails (e.g. stiff problem)
                    print(f"Sim failed: {e}")
                    crash_positions.append(initial_pos[:2]) # Assume crashed at start
            
            crash_positions = np.array(crash_positions)
            
            # Metric: Dispersion (Standard Deviation)
            # We want the strategy that keeps the drone in the most predictable area
            std_dev = np.std(crash_positions, axis=0)
            score = np.sum(std_dev) # Sum of std_x and std_y
            scores[strat] = score
            
        # 3. Determine Best Strategy
        best_strat = min(scores, key=scores.get)
        
        # 4. Save Data
        row = {
            'Wind_Speed': wind_speed,
            'Dist_to_Origin': dist_to_origin,
            'Altitude': z0,
            'Score_RTH': scores['RTH'],
            'Score_HOVER': scores['HOVER'],
            'Score_LAND': scores['LAND'],
            'Best_Strategy': best_strat
        }
        data.append(row)
        
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv('strategy_dataset.csv', index=False)
    print("Data generation complete. Saved to 'strategy_dataset.csv'.")
    
    # Simple Analysis
    print("\nClass Distribution:")
    print(df['Best_Strategy'].value_counts())

if __name__ == "__main__":
    generate_data()
