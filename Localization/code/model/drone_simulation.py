import numpy as np
from scipy.integrate import solve_ivp

class DroneSimulator:
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.mass = params.get('mass', 2.0)       # kg
        self.area = params.get('area', 0.1)       # m^2
        self.cd = params.get('cd', 0.5)           # Drag coefficient
        self.max_speed = params.get('max_speed', 15.0) # Max air speed (m/s)
        self.cruise_speed = params.get('cruise_speed', 10.0) # RTH speed
        self.descent_speed = params.get('descent_speed', 3.0) # Landing speed
        self.g = 9.81
        self.rho = 1.225

    def get_wind(self, t, z, wind_params):
        """
        Calculate wind vector at given time and altitude.
        wind_params: {'speed': float, 'dir': float (rad)}
        """
        base_speed = wind_params.get('speed', 5.0)
        angle = wind_params.get('dir', 0.0)
        
        # Simple wind shear model: wind increases with altitude
        # v_z = v_ref * (z / z_ref)^alpha
        # Let's assume base_speed is at 10m.
        if z < 0: z = 0
        altitude_factor = (max(z, 1.0) / 10.0) ** 0.15
        current_speed = base_speed * altitude_factor
        
        wx = current_speed * np.cos(angle)
        wy = current_speed * np.sin(angle)
        wz = 0 # Assume horizontal wind for now
        return np.array([wx, wy, wz])

    def dynamics_phase1(self, t, state, strategy, wind_params, origin):
        """
        Kinematic model for Controlled Flight Phase (0 to T_fail).
        State: [x, y, z]
        Returns: [vx, vy, vz]
        """
        x, y, z = state
        wind = self.get_wind(t, z, wind_params)
        
        vx, vy, vz = 0, 0, 0
        
        if strategy == 'HOVER':
            # Strategy: Try to maintain position (Ground Speed = 0)
            # Airspeed required = -Wind
            w_mag = np.linalg.norm(wind[:2])
            
            if w_mag <= self.max_speed:
                # Can fight the wind
                vx, vy = 0, 0
            else:
                # Cannot fight the wind, drift downwind
                # Ground Speed = Wind + Max_Air_Speed_Vector (opposing wind)
                # v_ground = W - V_max * (W / |W|)
                drift_speed = w_mag - self.max_speed
                wind_dir = wind[:2] / w_mag
                vx = drift_speed * wind_dir[0]
                vy = drift_speed * wind_dir[1]
            vz = 0 # Maintain altitude

        elif strategy == 'RTH':
            # Strategy: Fly towards origin
            dx = origin[0] - x
            dy = origin[1] - y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < 1.0:
                dir_vec = np.array([0, 0])
            else:
                dir_vec = np.array([dx, dy]) / dist
            
            # We command Airspeed towards home
            # v_air = cruise_speed * dir_vec
            # v_ground = v_air + wind
            
            v_air = dir_vec * self.cruise_speed
            v_ground = v_air + wind[:2]
            
            vx, vy = v_ground[0], v_ground[1]
            vz = 0 # Maintain altitude

        elif strategy == 'LAND':
            # Strategy: Descend immediately, try to hold X/Y
            w_mag = np.linalg.norm(wind[:2])
            if w_mag <= self.max_speed:
                vx, vy = 0, 0
            else:
                drift_speed = w_mag - self.max_speed
                wind_dir = wind[:2] / w_mag
                vx = drift_speed * wind_dir[0]
                vy = drift_speed * wind_dir[1]
            vz = -self.descent_speed

        return [vx, vy, vz]

    def dynamics_phase2(self, t, state, wind_params):
        """
        Dynamic model for Uncontrolled Descent Phase (T_fail to Impact).
        State: [x, y, z, vx, vy, vz]
        Returns: [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = state
        wind = self.get_wind(t, z, wind_params)
        
        # Relative velocity: V_rel = V_drone - V_wind
        v_rel = np.array([vx, vy, vz]) - wind
        v_rel_mag = np.linalg.norm(v_rel)
        
        if v_rel_mag == 0:
            drag_force = np.zeros(3)
        else:
            drag_force = 0.5 * self.rho * self.cd * self.area * v_rel_mag * (-v_rel)
        
        ax = drag_force[0] / self.mass
        ay = drag_force[1] / self.mass
        az = -self.g + drag_force[2] / self.mass
        
        return [vx, vy, vz, ax, ay, az]

    def simulate_mission(self, initial_pos, strategy, wind_params, t_fail, origin=(0,0,0)):
        """
        Simulate the full mission: Phase 1 (Controlled) -> Failure -> Phase 2 (Crash).
        """
        # --- Phase 1: Controlled Flight ---
        # State: [x, y, z]
        t_span1 = (0, t_fail)
        y0_phase1 = initial_pos # [x, y, z]
        
        # Event to detect ground impact during Phase 1 (e.g. LAND strategy)
        def hit_ground_p1(t, y): return y[2]
        hit_ground_p1.terminal = True
        hit_ground_p1.direction = -1

        sol1 = solve_ivp(
            fun=lambda t, y: self.dynamics_phase1(t, y, strategy, wind_params, origin),
            t_span=t_span1,
            y0=y0_phase1,
            events=hit_ground_p1,
            rtol=1e-3, atol=1e-3,  # Relaxed tolerances for speed
            method='RK23' # Faster method for non-stiff problems
        )
        
        # Check if it crashed/landed during Phase 1
        if sol1.status == 1: # Hit ground
            return sol1.y[:, -1], sol1.t, sol1.y
            
        # --- Phase 2: Uncontrolled Descent ---
        # Initial state for Phase 2 is Final state of Phase 1 + Velocity
        pos_fail = sol1.y[:, -1]
        t_start_p2 = sol1.t[-1]
        
        # Calculate velocity at moment of failure
        # (Since Phase 1 is kinematic, we calculate v from dynamics function)
        vel_fail = self.dynamics_phase1(t_start_p2, pos_fail, strategy, wind_params, origin)
        
        y0_phase2 = np.concatenate([pos_fail, vel_fail]) # [x, y, z, vx, vy, vz]
        
        def hit_ground_p2(t, y): return y[2]
        hit_ground_p2.terminal = True
        hit_ground_p2.direction = -1
        
        sol2 = solve_ivp(
            fun=lambda t, y: self.dynamics_phase2(t, y, wind_params),
            t_span=(t_start_p2, t_start_p2 + 300), # Max 300s fall time
            y0=y0_phase2,
            events=hit_ground_p2,
            rtol=1e-3, atol=1e-3, # Relaxed tolerances
            method='RK23'
        )
        
        # Combine trajectories
        # Phase 1: [x, y, z] -> expand to [x, y, z, vx, vy, vz] for consistency?
        # For now, just return the final crash position
        final_state = sol2.y[:, -1] # [x, y, z, vx, vy, vz]
        
        # Concatenate time and position for full path visualization
        full_t = np.concatenate([sol1.t, sol2.t])
        
        # Pad Phase 1 states with 0 velocity for simple concatenation (visualization only needs pos)
        p1_vels = np.array([self.dynamics_phase1(t, y, strategy, wind_params, origin) for t, y in zip(sol1.t, sol1.y.T)]).T
        full_y_p1 = np.vstack([sol1.y, p1_vels])
        full_y = np.hstack([full_y_p1, sol2.y])
        
        return final_state, full_t, full_y

if __name__ == "__main__":
    # Simple test
    sim = DroneSimulator()
    wind = {'speed': 12.0, 'dir': np.pi/4} # Strong wind
    # Test RTH
    final_pos, _, _ = sim.simulate_mission([1000, 1000, 100], 'RTH', wind, t_fail=30)
    print(f"RTH Crash Pos: {final_pos[:2]}")
    
    # Test Hover
    final_pos, _, _ = sim.simulate_mission([1000, 1000, 100], 'HOVER', wind, t_fail=30)
    print(f"Hover Crash Pos: {final_pos[:2]}")
