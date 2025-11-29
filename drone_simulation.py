import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

# --- Constants & Parameters ---
G = 9.81  # Gravity (m/s^2)
RHO = 1.225  # Air density at sea level (kg/m^3)

class DroneSimulation:
    def __init__(self, mass=2.0, area=0.1, cd=0.5):
        """
        Initialize drone parameters.
        :param mass: Mass of the drone (kg)
        :param area: Cross-sectional area (m^2)
        :param cd: Drag coefficient
        """
        self.mass = mass
        self.area = area
        self.cd = cd
        self.drag_const = 0.5 * RHO * self.cd * self.area

    def terrain_height(self, x, y):
        """
        Define the terrain height at position (x, y).
        For simplicity, let's assume a sloped terrain or flat ground.
        Here: A simple slope z = 0.1x
        """
        return 0.0  # Flat ground for now, can be changed to a function of x, y

    def dynamics(self, t, state, wind_speed):
        """
        Differential equations of motion.
        state = [x, y, z, vx, vy, vz]
        wind_speed = [wx, wy, wz]
        """
        x, y, z, vx, vy, vz = state
        wx, wy, wz = wind_speed

        # Relative velocity
        v_rel_x = vx - wx
        v_rel_y = vy - wy
        v_rel_z = vz - wz
        v_rel_mag = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)

        # Drag force magnitude: Fd = 0.5 * rho * Cd * A * v_rel^2
        # Drag force vector: Fd_vec = -Fd * (v_rel_vec / v_rel_mag)
        #                  = - (0.5 * rho * Cd * A * v_rel_mag) * v_rel_vec
        
        drag_factor = self.drag_const * v_rel_mag / self.mass

        ax = -drag_factor * v_rel_x
        ay = -drag_factor * v_rel_y
        az = -G - drag_factor * v_rel_z

        return [vx, vy, vz, ax, ay, az]

    def hit_ground_event(self, t, state, wind_speed):
        """
        Event function for solve_ivp to detect collision with ground.
        Value should be positive when flying and 0 when hitting ground.
        """
        x, y, z, _, _, _ = state
        return z - self.terrain_height(x, y)
    
    hit_ground_event.terminal = True
    hit_ground_event.direction = -1

    def simulate_single_trajectory(self, initial_state, wind_speed, t_span=(0, 300)):
        """
        Run a single simulation.
        :param initial_state: [x0, y0, z0, vx0, vy0, vz0]
        :param wind_speed: [wx, wy, wz]
        """
        # Define event wrapper to ensure attributes are visible to solve_ivp
        def event_wrapper(t, y):
            return self.hit_ground_event(t, y, wind_speed)
        event_wrapper.terminal = True
        event_wrapper.direction = -1

        sol = solve_ivp(
            fun=lambda t, y: self.dynamics(t, y, wind_speed),
            t_span=t_span,
            y0=initial_state,
            events=event_wrapper,
            rtol=1e-6, atol=1e-9
        )
        return sol

def run_monte_carlo(n_simulations=500):
    """
    Run Monte Carlo simulation with uncertain parameters.
    """
    drone = DroneSimulation()
    
    final_positions = []
    
    print(f"Running {n_simulations} Monte Carlo simulations...")
    
    for _ in range(n_simulations):
        # --- 1. Randomize Initial Conditions (Uncertainty) ---
        # Assume drone fails at roughly x=0, y=0, z=500m with some GPS error
        z0 = np.random.normal(500, 10)  # Mean 500m, std 10m
        x0 = np.random.normal(0, 5)
        y0 = np.random.normal(0, 5)
        
        # Initial velocity (e.g., cruising speed 15 m/s + random direction)
        # Assume it was flying roughly along X axis
        vx0 = np.random.normal(15, 2)
        vy0 = np.random.normal(0, 1)
        vz0 = np.random.normal(0, 1)
        
        initial_state = [x0, y0, z0, vx0, vy0, vz0]
        
        # --- 2. Randomize Wind (Uncertainty) ---
        # Wind might be strong in mountains. Mean 5 m/s, std 2 m/s, random direction
        wind_mag = np.abs(np.random.normal(5, 2))
        wind_angle = np.random.uniform(0, 2 * np.pi)
        wx = wind_mag * np.cos(wind_angle)
        wy = wind_mag * np.sin(wind_angle)
        wz = np.random.normal(0, 0.5) # Vertical wind (updrafts)
        
        wind_speed = [wx, wy, wz]
        
        # --- 3. Run Simulation ---
        sol = drone.simulate_single_trajectory(initial_state, wind_speed)
        
        if sol.status == 1: # A termination event occurred (hit ground)
            final_pos = sol.y[:, -1] # [x, y, z, vx, vy, vz]
            final_positions.append(final_pos[:2]) # Keep x, y
            
    final_positions = np.array(final_positions)

    if len(final_positions) == 0:
        print("Error: No simulations hit the ground. Check initial conditions or simulation time.")
        return
    
    # --- 4. Visualization ---
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of crash sites
    plt.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.5, s=10, label='Crash Sites')
    
    # Kernel Density Estimate (Heatmap)
    sns.kdeplot(x=final_positions[:, 0], y=final_positions[:, 1], cmap="Reds", fill=True, alpha=0.3, levels=10)
    
    plt.title(f"Drone Crash Location Probability Map ({n_simulations} runs)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.legend()
    
    # Mark the 'last known position' (approx origin)
    plt.plot(0, 0, 'ko', markersize=5, label='Last Known Position')
    
    plt.axis('equal')
    plt.savefig('crash_distribution.png')
    print("Simulation complete. Saved plot to 'crash_distribution.png'.")
    # plt.show() # Uncomment if running locally with GUI

if __name__ == "__main__":
    run_monte_carlo()
