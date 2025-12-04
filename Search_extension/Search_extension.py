# 任务四扩展：多单元协同搜索 (集成 PSO, 优化贪婪搜索, 遗传算法)

import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import cv2
import time
import os
import traceback 
import itertools 
import random
import math 
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist

# --- 占位配置和物理引擎 (假设已在实际项目中存在并可用) ---
try:
    # 模拟导入新物理模型和配置
    class Config: # 占位类
        NUM_SIMULATIONS = 5000
        WIND_A = 11.2
        WIND_K = 2.0
        WIND_DIR_MEAN = np.radians(135)
        WIND_DIR_KAPPA = 4.0
        CD_MEAN = 1.0
        CD_STD = 0.1
        AIR_DENSITY = 1.225
        MASS = 6.3
        INITIAL_SPEED_X = 10.0
        INITIAL_SPEED_Y = 0.0
        INITIAL_HEIGHT = 50.0
        GRAVITY = 9.81
        def get_terrain_z(self, x, y): 
            # 模拟一个简单的地形坡度
            return np.maximum(0, -0.1 * x - 0.05 * y + 10)
    config = Config()

    class PhysicsEngine: # 占位类
        @staticmethod
        def calculate_trajectory(m, c, v_xi, v_yi, y_initial, g):
            # 简化闭式解模拟，仅用于 P(C) 生成
            t_im = np.sqrt(2 * y_initial / g)
            x_air = v_xi * t_im
            return {"fall_time": t_im, "horizontal_distance": x_air, "t_c": np.full_like(t_im, 1000.0), "impact_before_crossover": np.full_like(t_im, True)}
    physics_engine = PhysicsEngine()

except Exception as e:
    print(f"Warning: Failed to import core physics modules. Using simple simulation. Error: {e}")
    class DummyConfig:
        NUM_SIMULATIONS = 500
        WIND_A = 10.0
        WIND_K = 2.0
        WIND_DIR_MEAN = 0.0
        WIND_DIR_KAPPA = 0.5
        CD_MEAN = 0.5
        CD_STD = 0.05
        AIR_DENSITY = 1.225
        MASS = 2.0
        INITIAL_SPEED_X = 15.0
        INITIAL_SPEED_Y = 0.0
        INITIAL_HEIGHT = 150.0
        GRAVITY = 9.81
        def get_terrain_z(self, x, y): return 0.0
    config = DummyConfig()
    
# 移除对旧版 drone_simulation.py 的依赖
class CrashProbabilityGenerator:
    """提供一个接口，用于兼容旧的 P(C) 生成逻辑"""
    @staticmethod
    def modified_generate_crash_probability(crash_positions, terrain_type, map_shape):
        """将离散的撞击点转换为概率热力图"""
        H, W = map_shape
        x_coords = np.array([p[0] for p in crash_positions])
        y_coords = np.array([p[1] for p in crash_positions])
        
        # 寻找 min/max 以进行归一化
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 避免除零
        x_range, y_range = x_max - x_min, y_max - y_min
        if x_range < 1e-3: x_range = 1.0
        if y_range < 1e-3: y_range = 1.0
        
        norm_x = (x_coords - x_min) / x_range * W
        norm_y = (y_coords - y_min) / y_range * H
        
        hist, x_edges, y_edges = np.histogram2d(norm_x, norm_y, bins=(W, H), range=[[0, W], [0, H]])
        prob_map = hist.T / np.sum(hist) # 归一化为概率
        
        prob_map = cv2.GaussianBlur(prob_map.astype(np.float32), (5, 5), 0)
        prob_map /= np.sum(prob_map)
        
        return prob_map, crash_positions 

# --- T4 地形因子定义 (保持不变) ---
TERRAIN_FACTORS = {
    'Mountain_Forest': {
        'desc': '山地森林 (高遮挡，高风险)',
        'W_factor': 0.65, 
        'K_factor': 0.70, 
        'Max_Time_Factor': 0.80,  
        'Crash_Prob_Per_Step': 0.005, 
        'Time_Increase_Factor': 1.25 
    },
    'Open_Desert': {
        'desc': '开阔沙漠 (低遮挡，强热背景)',
        'W_factor': 1.15, 
        'K_factor': 0.85, 
        'Max_Time_Factor': 0.90, 
        'Crash_Prob_Per_Step': 0.001,
        'Time_Increase_Factor': 1.05 
    },
    'Urban_Clear': {
        'desc': '城市空旷区 (基准)',
        'W_factor': 1.0,
        'K_factor': 1.0,
        'Max_Time_Factor': 1.0,
        'Crash_Prob_Per_Step': 0.0005,
        'Time_Increase_Factor': 1.0
    }
}


# --- 设备组合参数定义 (保持不变) ---
EQUIPMENT_COMBOS = {
    'A_HighEfficiency': {
        'desc': '组合A: 热成像 + 旋翼',
        'base_coverage_width': 45.0, 
        'base_k_efficiency': 0.35,  
        'color': '#00A693'
    },
    'B_HighPrecision': {
        'desc': '组合B: 旋翼 + 变焦相机',
        'base_coverage_width': 20.0, 
        'base_k_efficiency': 0.25,  
        'color': '#FF00AA'
    },
    'C_AllRound': {
        'desc': '组合C: 热成像 + 变焦 + 旋翼 (全能型)',
        'base_coverage_width': 35.0, 
        'base_k_efficiency': 0.45,  
        'color': '#007ACC'
    }
}


class SingleUnitDroneSearch:
    """
    T3 单单元搜索模型 (用于生成 P(C) 和路径规划基础)
    """
    def __init__(self, strategy='LAND', search_speed=15.0, coverage_width=30.0, 
                 grid_width=200, grid_height=150, k_efficiency=0.25, terrain='Urban_Clear'):
        self.strategy = strategy
        self.num_drones = 1
        self.search_speed = search_speed
        self.coverage_width = coverage_width
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.k_efficiency = k_efficiency
        self.terrain = terrain 
        self.probability_map, self.crash_points = self.generate_probability_distribution()
        self.search_paths = {}

    # --- P(C) 生成 (保持不变) ---
    def generate_probability_distribution(self, n_runs=config.NUM_SIMULATIONS):
        """
        使用新的 physics_engine.py 和 config.py 生成坠毁概率分布图
        """
        # 1. 生成随机输入 
        wind_speeds = config.WIND_A * np.random.weibull(config.WIND_K, n_runs)
        wind_directions = np.random.vonmises(config.WIND_DIR_MEAN, config.WIND_DIR_KAPPA, n_runs)
        cd_values = np.random.normal(config.CD_MEAN, config.CD_STD, n_runs)
        cd_values = np.maximum(0.1, cd_values)
        area_values = np.random.uniform(0.05, 0.11, n_runs)
        c_values = 0.5 * config.AIR_DENSITY * area_values * cd_values
        
        v_xi = np.full(n_runs, config.INITIAL_SPEED_X)
        v_yi = np.full(n_runs, config.INITIAL_SPEED_Y)
        
        # 2. 运行物理引擎 (初始平地假设)
        results = physics_engine.calculate_trajectory(
            m=config.MASS, c=c_values, v_xi=v_xi, v_yi=v_yi,
            y_initial=config.INITIAL_HEIGHT, g=config.GRAVITY
        )
        fall_time = results["fall_time"]
        radial_distance_air = results["horizontal_distance"]
        
        # 3. 坐标变换与风场叠加 (平地坐标)
        drone_dx_air = radial_distance_air
        wind_dx = wind_speeds * np.cos(wind_directions) * fall_time
        wind_dy = wind_speeds * np.sin(wind_directions) * fall_time
        x_flat = drone_dx_air + wind_dx
        y_flat = wind_dy 
        
        # 4. 地形碰撞修正 (Iterative Solver) - 简化 3 次迭代
        x_curr, y_curr = x_flat, y_flat
        for i in range(3):
            # 注意：config.get_terrain_z 需要 (x, y) 坐标
            z_terrain = config.get_terrain_z(x_curr, y_curr)
            effective_drop_height = config.INITIAL_HEIGHT - z_terrain
            effective_drop_height = np.maximum(0.1, effective_drop_height) # 物理约束
            
            new_results = physics_engine.calculate_trajectory(
                m=config.MASS, c=c_values, v_xi=v_xi, v_yi=v_yi,
                y_initial=effective_drop_height, g=config.GRAVITY
            )
            
            new_fall_time = new_results["fall_time"]
            new_air_dist = new_results["horizontal_distance"]
            
            new_drone_dx_air = new_air_dist
            new_wind_dx = wind_speeds * np.cos(wind_directions) * new_fall_time
            new_wind_dy = wind_speeds * np.sin(wind_directions) * new_fall_time
            
            x_curr = new_drone_dx_air + new_wind_dx
            y_curr = new_wind_dy 
            
        x_final = x_curr
        y_final = y_curr
        
        crash_positions = np.stack([x_final, y_final], axis=1).tolist()
        
        # 5. 生成概率热力图
        prob_map, _ = CrashProbabilityGenerator.modified_generate_crash_probability(
            crash_positions, self.terrain, (self.grid_height, self.grid_width)
        )
        
        return prob_map, crash_positions

    def probability_based_deployment(self, num_drones=1):
        # 保持不变
        flat_prob = self.probability_map.flatten()
        top_index = np.argmax(flat_prob)
        y = top_index // self.grid_width
        x = top_index % self.grid_width
        return np.array([[x, y]])

    def _connect_points_to_path(self, points, start_point, max_steps):
        # 保持不变
        valid_points = [(np.clip(int(p[0]), 0, self.grid_width - 1), 
                         np.clip(int(p[1]), 0, self.grid_height - 1)) for p in points]
        if not valid_points: return [start_point]
            
        path = [start_point]
        remaining_points = set(valid_points)
        current_pos = start_point
        
        # 假设一步移动的网格距离，用于路径平滑和步数控制
        move_distance = 3 
        
        while remaining_points and len(path) < max_steps:
            # 找到最近点
            nearest_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(current_pos)))
            distance = np.linalg.norm(np.array(nearest_point) - np.array(current_pos))
            steps_between = max(1, int(distance / move_distance)) 
            
            # 平滑连接
            x_values = np.linspace(current_pos[0], nearest_point[0], steps_between).astype(int)
            y_values = np.linspace(current_pos[1], nearest_point[1], steps_between).astype(int)
            
            for i in range(1, steps_between):
                new_point = (x_values[i], y_values[i])
                path.append(new_point)
                if len(path) >= max_steps: break
            
            if len(path) >= max_steps: break
            current_pos = nearest_point
            remaining_points.discard(nearest_point)
            
        while len(path) < max_steps: path.append(path[-1])
        return path[:max_steps]


    # --- 新算法 1: 优化贪婪搜索 (Greedy Optimized Search, GOS) ---
    def optimized_greedy_search(self, start_point, max_steps=100, current_search_map=None):
        """
        优化后的贪婪搜索 (GO):
        结合了局部最高概率选择和Lin & Goodrich的“全球变暖效应” (GWE)进行概率衰减。
        该算法取代了原有的贪婪和蛇形搜索。
        """
        prob_map = np.copy(current_search_map if current_search_map is not None else self.probability_map)
        path = [start_point]
        current_pos = np.array(start_point, dtype=float)
        
        move_distance = 3.0
        search_radius_grid = 5  # 局部搜索半径
        
        for _ in range(max_steps - 1):
            if np.sum(prob_map) < 1e-6: break # 搜索完毕
            
            # 1. 局部贪婪选择：在邻域内找到最佳方向
            max_local_prob = -np.inf
            local_max_point = None
            
            r_curr, c_curr = int(current_pos[1]), int(current_pos[0]) # (Y, X)
            
            r_start = max(0, r_curr - search_radius_grid)
            r_end = min(self.grid_height, r_curr + search_radius_grid + 1)
            c_start = max(0, c_curr - search_radius_grid)
            c_end = min(self.grid_width, c_curr + search_radius_grid + 1)
            
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if prob_map[r, c] > max_local_prob:
                        max_local_prob = prob_map[r, c]
                        local_max_point = np.array([c, r], dtype=float) # (X, Y)
            
            if local_max_point is None: break
                 
            target_point = local_max_point
            
            # 2. 计算朝向目标点的下一步 (平滑移动)
            direction_vector = target_point - current_pos
            distance = np.linalg.norm(direction_vector)
            
            if distance < move_distance:
                next_pos = target_point
            else:
                unit_vector = direction_vector / distance
                next_pos = current_pos + unit_vector * move_distance
                
            current_pos = next_pos
            path_point = (np.clip(int(current_pos[0]), 0, self.grid_width - 1), 
                          np.clip(int(current_pos[1]), 0, self.grid_height - 1))
            path.append(path_point)
            
            # 3. GWE 模拟: 全球变暖效应 (Global Warming Effect)
            x, y = path_point
            # 覆盖半径
            radius = max(1, int(self.coverage_width / 3 / (200/self.grid_width)))
            gwe_radius = radius + 2 
            gwe_factor = 0.7 # 70% 衰减
            
            for dx in range(-gwe_radius, gwe_radius + 1):
                for dy in range(-gwe_radius, gwe_radius + 1):
                    nx, ny = y + dy, x + dx # 注意 y/x 对应 map 的 ny/nx
                    if 0 <= nx < self.grid_height and 0 <= ny < self.grid_width:
                        distance_sq = (dx**2 + dy**2)
                        if distance_sq <= gwe_radius**2:
                            # 衰减因子：距离中心越近，衰减越多
                            decay = gwe_factor * np.exp(-distance_sq / (2 * (gwe_radius/2)**2)) 
                            prob_map[ny, nx] = np.maximum(0, prob_map[ny, nx] * (1 - decay))

        while len(path) < max_steps: path.append(path[-1])
        return path[:max_steps]
        
    
    # --- 新算法 2: 遗传算法搜索 (Genetic Algorithm Search, GA) ---
    def genetic_path_search(self, start_point, max_steps=100):
        """
        新增算法：基于遗传算法 (Genetic Algorithm, GA) 的全局路径优化。
        目标：最大化路径覆盖的累计概率。
        """
        map_h, map_w = self.probability_map.shape
        
        # 1. GA 参数
        GA_POP_SIZE = 50
        GA_GENERATIONS = 15
        GA_PATH_LENGTH = 30 # 路径分段的长度
        
        # 2. 编码：路径为一系列目标点 (Waypoint)
        def generate_chromosome(start_pos):
            """生成一个由若干随机目标点组成的染色体"""
            waypoints = []
            current_pos = np.array(start_pos)
            for _ in range(GA_PATH_LENGTH):
                # 随机生成一个相对于当前位置的偏移量，确保在地图范围内
                offset = np.random.uniform(-map_w / 4, map_w / 4, 2)
                next_pos = current_pos + offset
                next_pos = np.clip(next_pos, [0, 0], [map_w - 1, map_h - 1])
                waypoints.append(tuple(next_pos.astype(int)))
                current_pos = next_pos
            return waypoints

        # 3. 适应度函数：评估路径覆盖的概率
        def fitness_func(waypoints):
            # 将一系列目标点转换为完整的路径
            path = self._connect_points_to_path(waypoints, start_point, max_steps)
            coverage_map = self._calculate_coverage_map(path, self.coverage_width)
            
            # 适应度 = (覆盖地图 * 概率地图) 的和
            return np.sum(coverage_map * self.probability_map)

        # 4. 初始化种群
        population = [generate_chromosome(start_point) for _ in range(GA_POP_SIZE)]
        
        best_chromosome = population[0]
        max_fitness = fitness_func(best_chromosome)

        for gen in range(GA_GENERATIONS):
            fitnesses = [(fitness_func(p), p) for p in population]
            fitnesses.sort(key=lambda x: x[0], reverse=True)
            
            # 更新全局最优
            if fitnesses[0][0] > max_fitness:
                max_fitness = fitnesses[0][0]
                best_chromosome = fitnesses[0][1]
            
            # 选择 (精英保留 + 轮盘赌选择)
            elite_count = int(GA_POP_SIZE * 0.1)
            elite = [p for _, p in fitnesses[:elite_count]]
            
            # 轮盘赌选择
            fit_values = np.array([f for f, _ in fitnesses])
            if np.sum(fit_values) > 0:
                probabilities = fit_values / np.sum(fit_values)
            else:
                probabilities = np.full(GA_POP_SIZE, 1/GA_POP_SIZE)

            selected = random.choices([p for _, p in fitnesses], probabilities, k=GA_POP_SIZE - elite_count)
            
            # 交叉
            offspring = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[i+1] if i + 1 < len(selected) else selected[i]
                
                crossover_point = random.randint(1, GA_PATH_LENGTH - 1)
                child1 = p1[:crossover_point] + p2[crossover_point:]
                child2 = p2[:crossover_point] + p1[crossover_point:]
                offspring.extend([child1, child2])

            # 变异
            for child in offspring:
                if random.random() < 0.1: # 10% 变异率
                    mutation_point = random.randint(0, GA_PATH_LENGTH - 1)
                    # 变异：随机修改一个目标点
                    r_offset = np.random.uniform(-map_w / 8, map_w / 8, 2)
                    new_pos = np.array(child[mutation_point]) + r_offset
                    child[mutation_point] = tuple(np.clip(new_pos, [0, 0], [map_w - 1, map_h - 1]).astype(int))

            population = elite + offspring[:GA_POP_SIZE - elite_count]

        # 5. 转换最优染色体为最终路径
        return self._connect_points_to_path(best_chromosome, start_point, max_steps)


    # --- 现有算法 (保持不变) ---
    def pso_search_path(self, start_point, max_steps=100):
        # ... (PSO 逻辑保持不变) ...
        map_h, map_w = self.probability_map.shape
        def fitness_func(position):
            x, y = int(position[0]), int(position[1])
            return self.probability_map[y, x] if 0 <= x < map_w and 0 <= y < map_h else 0.0

        n_particles = 15
        max_iterations = 10 
        
        flat_idx = np.argmax(self.probability_map)
        gbest_y, gbest_x = np.unravel_index(flat_idx, self.probability_map.shape)
        gbest_pos = np.array([gbest_x, gbest_y], dtype=float)

        particles = {
            'pos': np.random.uniform([0, 0], [map_w - 1, map_h - 1], (n_particles, 2)),
            'vel': np.zeros((n_particles, 2)),
            'pbest_pos': np.zeros((n_particles, 2)),
            'pbest_value': np.zeros(n_particles)
        }
        particles['pbest_pos'] = np.copy(particles['pos'])
        for i in range(n_particles): particles['pbest_value'][i] = fitness_func(particles['pos'][i])

        path_points = [start_point]
        c1, c2, w = 2.0, 2.0, 0.9

        for t in range(max_iterations):
            for i in range(n_particles):
                r1, r2 = np.random.rand(2), np.random.rand(2)
                cognitive = c1 * r1 * (particles['pbest_pos'][i] - particles['pos'][i])
                social = c2 * r2 * (gbest_pos - particles['pos'][i])
                particles['vel'][i] = w * particles['vel'][i] + cognitive + social
                
                particles['pos'][i] += particles['vel'][i]
                particles['pos'][i, 0] = np.clip(particles['pos'][i, 0], 0, map_w - 1)
                particles['pos'][i, 1] = np.clip(particles['pos'][i, 1], 0, map_h - 1)

                current_value = fitness_func(particles['pos'][i])
                if current_value > particles['pbest_value'][i]:
                    particles['pbest_value'][i] = current_value
                    particles['pbest_pos'][i] = particles['pos'][i]
                    
                if current_value > fitness_func(gbest_pos): gbest_pos = particles['pos'][i]
            
            for pos in particles['pbest_pos']: path_points.append(tuple(pos.astype(int)))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)

    
    def simulate_search(self, algorithm, max_time=100, setup_time=5):
        # 保持不变
        deployment_points = self.probability_based_deployment() 
        point = deployment_points[0]
        point_tuple = (int(point[0]), int(point[1]))
        
        max_steps = int(max_time * 2) # 简化步数估算
        
        if algorithm == 'pso':
            path = self.pso_search_path(point_tuple, max_steps=max_steps)
        elif algorithm == 'greedy_opt':
            path = self.optimized_greedy_search(point_tuple, max_steps=max_steps)
        elif algorithm == 'ga_search':
            path = self.genetic_path_search(point_tuple, max_steps=max_steps)
        else:
             raise NotImplementedError(f"算法 {algorithm} 不支持")
             
        self.search_paths[algorithm] = path
        
        # 简化的 CPD 曲线，因为单机模式不涉及多机协同细节
        time_points = np.linspace(0, max_time, 50)
        probabilities_time = np.clip(time_points / max_time * 0.9, 0, 0.9) 
        
        return time_points, probabilities_time.tolist(), deployment_points, [], [], max_time

    def _calculate_coverage_map(self, path_or_paths, coverage_width):
        # 保持不变
        coverage_map = np.zeros_like(self.probability_map, dtype=float)
        # 将覆盖宽度 (米) 转换为网格距离
        radius = max(1, int(coverage_width / 3 / (200/self.grid_width))) 
        
        if isinstance(path_or_paths, dict):
            all_paths = list(itertools.chain.from_iterable(path_or_paths.values()))
        else:
            all_paths = path_or_paths

        for point in all_paths:
            x, y = point
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = x + dx, y + dy
                        distance_sq = (dx**2 + dy**2)
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            if distance_sq <= radius**2:
                                # 累积覆盖努力，中心高斯衰减
                                coverage_map[ny, nx] += np.exp(-distance_sq / (2 * (radius/2)**2))
        return coverage_map


# --- T4 核心：多单元搜索模型类 (保持不变) ---
class MultiUnitDroneSearch(SingleUnitDroneSearch):
    COLLISION_THRESHOLD = 3 
    COLLISION_PROB_BASE = 0.0001 
    
    def __init__(self, strategy='LAND', num_drones=3, terrain='Urban_Clear', 
                 combo_params=EQUIPMENT_COMBOS['C_AllRound'], max_time_base=100, **kwargs):
        
        self.max_time_base = max_time_base
        self.num_drones = num_drones
        self.terrain = terrain
        
        factor = TERRAIN_FACTORS.get(terrain, TERRAIN_FACTORS['Urban_Clear'])
        coverage_width = combo_params['base_coverage_width'] * factor['W_factor']
        k_efficiency = combo_params['base_k_efficiency'] * factor['K_factor']
        
        # 调用父类初始化，此时会调用 generate_probability_distribution
        super().__init__(strategy=strategy, coverage_width=coverage_width, 
                         k_efficiency=k_efficiency, terrain=terrain, **kwargs)
        
        self.max_time_actual = int(max_time_base * factor['Max_Time_Factor'])
        self.step_crash_prob = factor['Crash_Prob_Per_Step']
        self.time_per_step_increase = factor['Time_Increase_Factor']
        
        self.drone_positions = self.probability_based_deployment(num_drones=num_drones)
        self.all_drone_paths = {i: [] for i in range(num_drones)}
        self.drone_status = {i: True for i in range(num_drones)}
        
    def probability_based_deployment(self, num_drones=1):
        # 保持不变
        if num_drones == 1: return super().probability_based_deployment()
            
        map_h, map_w = self.probability_map.shape
        flat_prob = self.probability_map.flatten()
        
        # 使用 95% 分位数来获取高概率区域进行聚类
        threshold = np.percentile(flat_prob, 95)
        high_prob_indices = np.where(self.probability_map >= threshold)
        points_to_cluster = np.array(list(zip(high_prob_indices[1], high_prob_indices[0])))
        
        if len(points_to_cluster) < num_drones:
            # 如果高概率区域点太少，则直接取概率最高的 N 个点
            sorted_indices = np.argsort(flat_prob)[::-1]
            top_indices = sorted_indices[:num_drones]
            deployment_points = np.array([[idx % map_w, idx // map_w] for idx in top_indices])
            return deployment_points.astype(int)

        # K-Means 聚类以找到 N 个最佳起始点
        kmeans = KMeans(n_clusters=num_drones, random_state=42, n_init=10)
        kmeans.fit(points_to_cluster)
        deployment_points = kmeans.cluster_centers_
        
        return deployment_points.astype(int)
        
    def simulate_cooperative_search(self, algorithm='pso', max_time=None, setup_time=5):
        # 更新：使用新的算法代号
        if max_time is None: max_time = self.max_time_actual
            
        num_drones = self.num_drones
        deployment_points = self.drone_positions
        path_segments = {}
        
        # 简化步数估算，考虑了地形/策略的时间增加因子
        max_steps = int(max_time * 2 / self.time_per_step_increase) 
        
        # --- 路径规划：多机路径根据所选算法生成 ---
        for i in range(num_drones):
            point_tuple = tuple(deployment_points[i])
            
            # 使用新的算法分支
            if algorithm == 'pso':
                path = self.pso_search_path(point_tuple, max_steps=max_steps)
            elif algorithm == 'greedy_opt':
                # 注意：这里传递原始概率图，因为 GO 内部会进行模拟衰减
                path = self.optimized_greedy_search(point_tuple, max_steps=max_steps, current_search_map=self.probability_map)
            elif algorithm == 'ga_search':
                path = self.genetic_path_search(point_tuple, max_steps=max_steps)
            else:
                raise NotImplementedError(f"算法 {algorithm} 不支持")
                
            path_segments[i] = path
            self.all_drone_paths[i] = path 
        # -----------------------------------------------

        time_points_raw = [0.0] 
        probabilities_raw = [0.0]
        cumulative_prob = 0.0
        
        # 存储每个网格被搜索的次数 N
        searched_count_map = np.zeros_like(self.probability_map, dtype=int)
        k = self.k_efficiency
        radius = max(1, int(self.coverage_width / 3 / (200/self.grid_width)))
        
        for step in range(max_steps):
            if time_points_raw[-1] >= max_time: break
            
            new_found_prob_in_step = 0.0
            active_drones_pos = {}
            
            # A. 风险评估：单机触地/地形故障
            for drone_id in range(num_drones):
                if self.drone_status[drone_id]:
                    if random.random() < self.step_crash_prob:
                        self.drone_status[drone_id] = False
                        continue
                    
                    path_len = len(path_segments[drone_id])
                    x, y = path_segments[drone_id][step % path_len]
                    active_drones_pos[drone_id] = np.array([x, y])

            # B. 风险评估：碰撞风险 
            drones_to_remove = set()
            active_ids = list(active_drones_pos.keys())
            for i in range(len(active_ids)):
                for j in range(i + 1, len(active_ids)):
                    id1, id2 = active_ids[i], active_ids[j]
                    dist = np.linalg.norm(active_drones_pos[id1] - active_drones_pos[id2])
                    
                    if dist <= self.COLLISION_THRESHOLD:
                        # 碰撞概率随着距离减小而增加
                        if random.random() < self.COLLISION_PROB_BASE * (1 + (self.COLLISION_THRESHOLD - dist)): 
                            drones_to_remove.add(id1)
                            drones_to_remove.add(id2)

            for d_id in drones_to_remove: self.drone_status[d_id] = False

            # C. 搜索努力累积 (Detection Probability P(D|C))
            for drone_id in active_ids:
                if not self.drone_status[drone_id]: continue 
                
                x, y = active_drones_pos[drone_id].astype(int)
                
                y_start = max(0, y - radius)
                y_end = min(self.grid_height, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(self.grid_width, x + radius + 1)
                
                for ny in range(y_start, y_end):
                    for nx in range(x_start, x_end):
                        distance_sq = (nx - x)**2 + (ny - y)**2
                        if distance_sq <= radius**2:
                            P_Ci = self.probability_map[ny, nx]
                            
                            # 累积搜索次数 N (考虑搜索努力的权重，中心点权重更高)
                            weight = np.exp(-distance_sq / (2 * (radius/2)**2))
                            # 简化：直接增加搜索次数 (如果需要更精确，可以累积带权重的努力值)
                            searched_count_map[ny, nx] += 1
                            N = searched_count_map[ny, nx]
                            
                            # P(D|C) = 1 - e^(-k * N)
                            P_Di = 1 - np.exp(-k * N)
                            P_Di_old = 1 - np.exp(-k * (N - 1)) if N > 1 else 0
                                
                            # 增量概率 P(Found_in_Step) = P(C) * P(D_new | C) - P(C) * P(D_old | C)
                            P_increment = P_Ci * (P_Di - P_Di_old)
                            new_found_prob_in_step += P_increment
            
            cumulative_prob += new_found_prob_in_step
            probabilities_raw.append(cumulative_prob)
            
            time_points_raw.append(time_points_raw[-1] + self.time_per_step_increase)
            
            # 如果所有无人机都失联，搜索停止
            if not any(self.drone_status.values()):
                 final_prob = cumulative_prob
                 while time_points_raw[-1] < max_time:
                     time_points_raw.append(time_points_raw[-1] + 1.0)
                     probabilities_raw.append(final_prob)
                 break
        
        # 统一到 50 个时间点用于绘图
        fixed_time_points = np.linspace(0, max_time, 50)
        interpolated_prob_time = np.interp(fixed_time_points, time_points_raw, probabilities_raw)
        
        # 返回搜索努力地图 (用于可视化图4)
        return fixed_time_points, interpolated_prob_time.tolist(), deployment_points, probabilities_raw, searched_count_map, max_time

# --- 运行函数 (扩展支持所有算法) ---

def run_all_combinations_and_algorithms(strategies_to_test, algorithms_to_test, max_time=100, num_drones=3, terrain='Urban_Clear'):
    """运行任务四所需的策略和组合"""
    
    combo_name = 'C_AllRound'
    combo_params = EQUIPMENT_COMBOS[combo_name]
    
    all_results = {}
    base_models = {}
    
    for strategy in strategies_to_test:
        # 1. 生成基准 P(C)
        base_models[strategy] = SingleUnitDroneSearch( 
            strategy=strategy,
            coverage_width=30.0,
            grid_width=200,
            grid_height=150,
            terrain=terrain 
        )
        all_results[strategy] = {}
        
        # 2. 运行多单元模拟
        all_results[strategy][combo_name] = {}
        for algo in algorithms_to_test:
            try:
                search_model = MultiUnitDroneSearch(
                    strategy=strategy,
                    num_drones=num_drones,
                    terrain=terrain,
                    combo_params=combo_params,
                    max_time_base=max_time,
                    grid_width=base_models[strategy].grid_width,
                    grid_height=base_models[strategy].grid_height
                )
                
                # 保证 P(C) 的一致性
                search_model.probability_map = base_models[strategy].probability_map 
                
                time_points, probabilities_time, deployment_points, _, searched_count_map, current_max_time = search_model.simulate_cooperative_search(algo, max_time)
                path_data = search_model.all_drone_paths 
                
                # 使用 searched_count_map 作为覆盖图 (更准确地反映搜索努力)
                coverage_map = searched_count_map 
                
                all_results[strategy][combo_name][algo] = {
                    'time': time_points,
                    'prob_time': probabilities_time,
                    'deployment': deployment_points,
                    'max_time': current_max_time,
                    'path': path_data,
                    'coverage_map': coverage_map,
                    'num_drones': num_drones
                }
            except Exception as e:
                print(f"多单元模拟失败 ({strategy}/{terrain}/{algo}): {e}")
                # traceback.print_exc() # 调试时启用
    
    return base_models, all_results


# --- 任务四可视化函数 (N策略 x 4图结构, 路径图整合) ---

def visualize_cooperative_results(base_models, all_results, strategies_to_test, algorithms_to_test, terrain, save_dir='T4_Visualizations_MultiUnit'):
    """
    可视化函数：生成 N策略 x 4 图，路径图整合所有算法。
    """
    
    num_strategies = len(strategies_to_test)
    num_columns = 4 
    combo_name = 'C_AllRound'
    
    plt.figure(figsize=(24, 6 * num_strategies))
    
    title_terrain_desc = TERRAIN_FACTORS[terrain]['desc']
    
    # 调整算法显示名称
    ALGO_DISPLAY_NAMES = {
        'pso': 'PSO',
        'greedy_opt': 'GREEDY (优化)',
        'ga_search': '遗传算法 (GA)'
    }
    display_algos = [ALGO_DISPLAY_NAMES.get(a, a.upper()) for a in algorithms_to_test]

    plt.suptitle(f"任务四多单元协同搜索结果对比 | 地形: {title_terrain_desc} | 设备: {combo_name}\n"
                 f"对比算法: {', '.join(display_algos)}", 
                 fontsize=22, y=0.97, fontweight='bold') 
    
    # 颜色映射定义
    cmap_prob = cm.YlOrRd
    # 针对覆盖努力图，使用类似紫色/蓝色的 colormap (如 viridis_r)
    cmap_cov = cm.Purples_r 
    
    # 路径颜色和线型定义 (更新颜色以区分)
    ALGO_STYLES = {
        'pso': {'color': '#FF0000', 'linestyle': '-', 'label': f'PSO 路径 ({ALGO_DISPLAY_NAMES["pso"]})'}, # 红色实线
        'greedy_opt': {'color': '#00B050', 'linestyle': '--', 'label': f'GO 路径 ({ALGO_DISPLAY_NAMES["greedy_opt"]})'}, # 绿色虚线
        'ga_search': {'color': '#0000FF', 'linestyle': '-.', 'label': f'GA 路径 ({ALGO_DISPLAY_NAMES["ga_search"]})'} # 蓝色点划线
    }
    
    for i, strategy_key in enumerate(strategies_to_test):
        base_model = base_models[strategy_key]
        results = all_results.get(strategy_key, {}).get(combo_name, {})
        row_start_index = i * num_columns
        
        # --- 1. 累计探测概率 (CPD) 曲线 ---
        ax = plt.subplot(num_strategies, num_columns, row_start_index + 1)
        
        final_probs = {}
        
        for algo in algorithms_to_test:
            if algo in results:
                res = results[algo]
                final_prob = res['prob_time'][-1] if res['prob_time'] else 0
                final_probs[algo] = final_prob
                
                # 在 CPD 曲线上绘制所有算法
                ax.plot(res['time'], res['prob_time'], 
                        label=f"{ALGO_DISPLAY_NAMES.get(algo, algo.upper())} ({final_prob*100:.1f}%)", 
                        color=ALGO_STYLES[algo]['color'], linewidth=2)
        
        max_prob_str = f"Max: {max(final_probs.values())*100:.1f}%" if final_probs else "N/A"
        ax.set_title(f"策略: {strategy_key} - CPD曲线 ({max_prob_str})", fontsize=16)
        ax.set_xlabel("时间 (单位)")
        ax.set_ylabel("累计探测概率 (CPD)")
        ax.axhline(y=0.9, color='r', linestyle='--', linewidth=0.8, label='90%目标')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize='small')


        # --- 2. 坠毁概率分布 P(C) ---
        ax = plt.subplot(num_strategies, num_columns, row_start_index + 2)
        im = ax.imshow(base_model.probability_map, cmap=cmap_prob, origin='lower')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="概率密度")
        
        # 假设所有算法使用相同的部署点 (来自任意一个有效算法)
        res_ref = results.get(algorithms_to_test[0]) 
        if res_ref:
            deploy_points = res_ref['deployment']
            ax.scatter(deploy_points[:, 0], deploy_points[:, 1], c='lime', marker='^', s=50, edgecolors='k', label='部署点')
             
        ax.set_title(f"策略: {strategy_key} - P(C)热力图 (部署点)", fontsize=16)
        ax.set_xlabel("X (网格)")
        ax.set_ylabel("Y (网格)")
        ax.set_aspect('equal', adjustable='box')
        
        
        # --- 3. 多单元路径对比图 (核心修改) ---
        ax = plt.subplot(num_strategies, num_columns, row_start_index + 3)
        # 绘制 P(C) 作为背景
        ax.imshow(base_model.probability_map, cmap=cmap_prob, origin='lower', alpha=0.3)
        
        legend_handles = []
        added_algo_labels = set()
        
        # 确保 res_ref 已定义
        res_ref = results.get(algorithms_to_test[0]) 

        for algo in algorithms_to_test:
            if algo in results:
                res = results[algo]
                path_data = res['path']
                style = ALGO_STYLES[algo]
                
                # 绘制所有无人机的路径
                for drone_id, path in path_data.items():
                    if path:
                        path_np = np.array(path)
                        label = ALGO_DISPLAY_NAMES.get(algo, algo.upper()) if algo not in added_algo_labels else None
                        
                        # 绘制路径
                        line, = ax.plot(path_np[:, 0], path_np[:, 1], color=style['color'], 
                                        linestyle=style['linestyle'], linewidth=2.0, alpha=0.8, label=label)
                        
                        if label is not None:
                             legend_handles.append(line)
                             added_algo_labels.add(algo)

        # 绘制部署点 (只绘制一次)
        if res_ref:
            ax.scatter(res_ref['deployment'][:, 0], res_ref['deployment'][:, 1], 
                       c='lime', marker='^', s=60, edgecolors='k', zorder=10)
            # 手动添加部署点到 legend
            deploy_handle = plt.Line2D([0], [0], marker='^', color='k', 
                                      linestyle='None', markersize=6, markerfacecolor='lime', label='部署点')
            # 仅在需要时添加部署点图例
            if not any(h.get_label() == '部署点' for h in legend_handles):
                legend_handles.append(deploy_handle)


        ax.legend(handles=[h for h in legend_handles if h.get_label() is not None], loc='upper right', fontsize='small')

        ax.set_title(f"策略: {strategy_key} - 路径对比图 ({res_ref['num_drones'] if res_ref else 0}架)", fontsize=16)
        ax.set_xlabel("X (网格)")
        ax.set_ylabel("Y (网格)")
        ax.set_aspect('equal', adjustable='box')


        # --- 4. 协同覆盖热力图 ---
        ax = plt.subplot(num_strategies, num_columns, row_start_index + 4)
        
        if final_probs:
            best_algo = max(final_probs, key=final_probs.get)
            # 使用 searched_count_map 作为覆盖图
            coverage_map = results[best_algo]['coverage_map']
            
            # 绘制 P(C) 作为背景
            im_prob = ax.imshow(base_model.probability_map, cmap=cmap_prob, origin='lower', alpha=0.25)
            
            # 绘制覆盖热力图
            max_cov = np.max(coverage_map)
            norm = mcolors.Normalize(vmin=0, vmax=max_cov * 0.8) if max_cov > 0 else mcolors.Normalize(vmin=0, vmax=1)
            
            im_cov = ax.imshow(coverage_map, cmap=cmap_cov, origin='lower', alpha=0.7, norm=norm)
            
            plt.colorbar(im_cov, ax=ax, fraction=0.046, pad=0.04, label="累计搜索努力强度 (N)")
            
            ax.set_title(f"策略: {strategy_key} - 协同覆盖 ({ALGO_DISPLAY_NAMES.get(best_algo, best_algo.upper())})", fontsize=16)
            
            # 绘制部署点
            if res_ref:
                ax.scatter(res_ref['deployment'][:, 0], res_ref['deployment'][:, 1], 
                           c='lime', marker='^', s=60, edgecolors='k', zorder=10)
        else:
             ax.set_title(f"策略: {strategy_key} - 协同覆盖 (数据缺失)", fontsize=16)
        
        ax.set_xlabel("X (网格)")
        ax.set_ylabel("Y (网格)")
        ax.set_aspect('equal', adjustable='box')
            
    plt.tight_layout() 
    plt.subplots_adjust(top=0.9) 
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"T4_MultiUnit_MultiAlgo_{terrain}_Results.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    print(f"\n可视化结果已保存至: {save_path}")
    
    plt.show()
    plt.close() 


# --- 主运行块 (任务四核心) ---

if __name__ == "__main__":
    STRATEGIES_TO_TEST = ['LAND', 'HOVER', 'RTH'] 
    DEFAULT_MAX_TIME = 100 
    
    # --- 增加新的算法 ---
    # 替换 'greedy' -> 'greedy_opt', 移除 'serpentine', 新增 'ga_search'
    ALGORITHMS_TO_TEST = ['pso', 'greedy_opt', 'ga_search']
    
    TERRAINS_TO_TEST = ['Urban_Clear', 'Mountain_Forest', 'Open_Desert']
    NUM_DRONES = 3 
    
    for terrain in TERRAINS_TO_TEST:
        print("\n" + "="*60)
        print(f"=== 运行任务四：多单元协同模型 ({terrain} 地形, {NUM_DRONES}架无人机) ===")
        print(f"=== 对比算法: {', '.join([a.upper() for a in ALGORITHMS_TO_TEST])} ===")
        print("="*60)
        
        # 1. 运行多单元协同搜索 (所有算法)
        base_models, all_results = run_all_combinations_and_algorithms(
            STRATEGIES_TO_TEST, ALGORITHMS_TO_TEST, DEFAULT_MAX_TIME, num_drones=NUM_DRONES, terrain=terrain
        )
        
        # 2. 生成并保存可视化结果 (路径图整合)
        if all_results:
            visualize_cooperative_results(
                base_models, 
                all_results, 
                STRATEGIES_TO_TEST, 
                ALGORITHMS_TO_TEST,
                terrain, 
                save_dir='T4_Visualizations_MultiUnit_MultiAlgo'
            )