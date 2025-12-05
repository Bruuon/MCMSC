import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
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

# 导入任务一的模拟器，用于生成坠毁概率分布
try:
    from drone_simulation import DroneSimulator
except ImportError:
    # 占位符，防止未导入时的错误
    class DroneSimulator:
        """用于占位，防止未导入时的NameError"""
        def simulate_mission(self, initial_pos, strategy, wind, t_fail, origin):
            # 返回一个默认的坐标，避免崩溃
            return [0, 0, 0], [0, 0, 0], [0, 0, 0]


# --- 设备组合参数定义 ---
EQUIPMENT_COMBOS = {
    'A_HighEfficiency': {
        'desc': '组合A: 热成像 + 旋翼 (高效率广域搜索)',
        'coverage_width': 45.0, # 热成像视野宽，覆盖宽度大
        'k_efficiency': 0.35,  # 探测效率较高
        'color': '#00A693'
    },
    'B_HighPrecision': {
        'desc': '组合B: 旋翼 + 变焦相机 (高精度热点搜索)',
        'coverage_width': 20.0, # 变焦相机视角窄，覆盖宽度小
        'k_efficiency': 0.25,  # 探测效率中等 (需要多次确认)
        'color': '#FF00AA'
    },
    'C_AllRound': {
        'desc': '组合C: 热成像 + 变焦 + 旋翼 (全能型均衡搜索)',
        'coverage_width': 35.0, # 均衡覆盖宽度
        'k_efficiency': 0.45,  # 最佳探测效率 (系统成熟，虚警率低)
        'color': '#007ACC'
    }
}


class SingleUnitDroneSearch:
    """
    任务三核心：单单元搜索模型
    基于无人机失联前的策略，预测坠毁概率，并评估四种单单元搜索算法
    （螺旋、网格、扇形、PSO）在最小化定位时间方面的性能。
    """
    def __init__(self, strategy='LAND', search_speed=15.0, coverage_width=30.0, 
                 grid_width=200, grid_height=150, k_efficiency=0.25):
        self.strategy = strategy
        self.num_drones = 1
        self.search_speed = search_speed
        self.coverage_width = coverage_width
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.k_efficiency = k_efficiency
        
        # 1. 生成坠毁概率分布图
        self.probability_map, self.crash_points = self.generate_probability_distribution()
        
        self.search_paths = {}
        # 搜索计数图：记录每个网格单元被搜索了多少次
        self.search_count_map = np.zeros((self.grid_height, self.grid_width), dtype=int) 

    # --- 概率分布和部署点方法 (保持不变) ---
    
    def generate_probability_distribution(self, n_runs=500):
        """生成坠毁概率分布图（依赖于 drone_simulation.py 的结果）。"""
        crash_positions = []
        scenario = {
            'wind_speed': 15.0, 'wind_dir': np.radians(45), 
            'initial_pos': [0, 0, 200], 'origin': [3000, 3000, 0]
        }
        
        try:
            sim = DroneSimulator()
            
            for i in range(n_runs):
                run_wind = {
                    'speed': np.random.normal(scenario['wind_speed'], 2.0),
                    'dir': np.random.normal(scenario['wind_dir'], 0.1)
                }
                t_fail = np.clip(np.random.exponential(scale=300), 1, 600)
                final_pos, _, _ = sim.simulate_mission(
                    scenario['initial_pos'], self.strategy, run_wind, t_fail, origin=scenario['origin']
                )
                crash_positions.append(final_pos[:2])
            
            crash_positions = np.array(crash_positions)
        
        except (NameError, NotImplementedError, Exception) as e:
            # 如果模拟器不可用，使用默认分布
            return self.create_default_distribution(), []

        # 将模拟的坠毁点映射到网格并生成高斯平滑的概率图
        probability_map = np.zeros((self.grid_height, self.grid_width))
        
        if len(crash_positions) > 0:
            x_min, x_max = -1000, 4000
            y_min, y_max = -1000, 4000
            grid_x = ((crash_positions[:, 0] - x_min) / (x_max - x_min) * (self.grid_width - 1)).astype(int)
            grid_y = ((crash_positions[:, 1] - y_min) / (y_max - y_min) * (self.grid_height - 1)).astype(int)
            grid_x = np.clip(grid_x, 0, self.grid_width-1)
            grid_y = np.clip(grid_y, 0, self.grid_height-1)
            crash_points = list(zip(grid_x, grid_y))
            unique_crash_points = list(set(crash_points))
            
            for x, y in unique_crash_points:
                xx, yy = np.meshgrid(np.arange(self.grid_width), np.arange(self.grid_height))
                distance_sq = (xx - x)**2 + (yy - y)**2
                probability_map += np.exp(-distance_sq / (2 * (8**2)))
        else:
            probability_map = self.create_default_distribution()
            crash_points = []
        
        probability_map = cv2.GaussianBlur(probability_map, (5, 5), 1.5)
        if np.sum(probability_map) > 0:
            probability_map = probability_map / np.sum(probability_map)
        
        return probability_map, crash_points

    def create_default_distribution(self):
        """创建默认的概率分布 (在模拟器不可用时使用)"""
        x = np.linspace(0, 10, self.grid_width)
        y = np.linspace(0, 10, self.grid_height)
        X, Y = np.meshgrid(x, y)
        probability_map = np.zeros((self.grid_height, self.grid_width))
        hotspots = [
            (0.4, 3, 3, 1, 1),
            (0.3, 7, 2, 0.8, 0.8),
            (0.25, 5, 5, 1.5, 1.5)
        ]
        for weight, cx, cy, sx, sy in hotspots:
            gaussian = np.exp(-((X - cx)**2/(2*sx**2) + (Y - cy)**2/(2*sy**2)))
            probability_map += weight * gaussian
        probability_map += 0.02 * np.random.random(probability_map.shape)
        return probability_map / np.sum(probability_map)

    def probability_based_deployment(self):
        """基于概率分布推荐设备的初始部署点 (网格中心)"""
        flat_prob = self.probability_map.flatten()
        top_index = np.argmax(flat_prob)
        y = top_index // self.grid_width
        x = top_index % self.grid_width
        deployment_points = np.array([[x, y]])
        return deployment_points
    
    def _connect_points_to_path(self, points, start_point, max_steps):
        """将一组离散点连接成一个连续的搜索路径 (路径规划)"""
        valid_points = []
        for p in points:
            x = np.clip(int(p[0]), 0, self.grid_width - 1)
            y = np.clip(int(p[1]), 0, self.grid_height - 1)
            valid_points.append((x, y))
        
        if not valid_points: return [start_point]
            
        path = [start_point]
        remaining_points = set(valid_points)
        current_pos = start_point
        
        move_distance = 2.5
        
        while remaining_points and len(path) < max_steps:
            nearest_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(current_pos)))
            distance = np.linalg.norm(np.array(nearest_point) - np.array(current_pos))
            steps_between = int(distance / move_distance) + 1 
            
            x_values = np.linspace(current_pos[0], nearest_point[0], steps_between).astype(int)
            y_values = np.linspace(current_pos[1], nearest_point[1], steps_between).astype(int)
            
            for i in range(1, steps_between):
                new_point = (x_values[i], y_values[i])
                path.append(new_point)
                if len(path) >= max_steps: break
            
            if len(path) >= max_steps: break
                
            current_pos = nearest_point
            remaining_points.discard(nearest_point)
            
        while len(path) < max_steps:
            path.append(path[-1])
            
        return path[:max_steps]

    # --- 四种搜索算法路径生成 ---
    # (spiral, grid, sector 保持不变)
    def spiral_search_path(self, start_point, max_steps=100):
        """螺旋搜索模式 (Spiral Search)"""
        path = [start_point]
        center_x, center_y = start_point
        max_radius = min(center_x, self.grid_width-center_x, center_y, self.grid_height-center_y, 40)
        t_max = 8 * np.pi 
        t_steps = max_steps - 1
        a = 1.0 
        b = max_radius / t_max 
        
        for t in np.linspace(0, t_max, t_steps):
            radius = a + b * t
            new_x = int(center_x + radius * np.cos(t))
            new_y = int(center_y + radius * np.sin(t))
            new_point = (new_x, new_y)
            
            if (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and
                len(path) < max_steps):
                if new_point != path[-1]: path.append(new_point)
            
            if len(path) >= max_steps: break
            
        return self._connect_points_to_path(path, start_point, max_steps)

    def grid_search_path(self, start_point, max_steps=100):
        """网格搜索模式 (Grid Search)"""
        path = [start_point]
        center_x, center_y = start_point
        strip_width = max(5, int(self.coverage_width * 0.8 / (200/self.grid_width)))
        max_dist = 60
        
        x_start = max(0, int(center_x - max_dist))
        x_end = min(self.grid_width, int(center_x + max_dist))
        y_start = max(0, int(center_y - max_dist))
        y_end = min(self.grid_height, int(center_y + max_dist))

        x_grid = np.arange(x_start, x_end, 1)
        y_strips = np.arange(y_start, y_end, strip_width)
        path_points = []
        
        for i, y in enumerate(y_strips):
            y_clamped = np.clip(y, 0, self.grid_height - 1)
            
            if i % 2 == 0:
                for x in x_grid: path_points.append((x, y_clamped))
            else:
                for x in x_grid[::-1]: path_points.append((x, y_clamped))
            
            if i < len(y_strips) - 1:
                next_y_clamped = np.clip(y_strips[i+1], 0, self.grid_height - 1)
                current_x = path_points[-1][0]
                y_step_interpolated = np.linspace(path_points[-1][1], next_y_clamped, int(strip_width/2)).astype(int)
                for y_val in y_step_interpolated: path_points.append((current_x, y_val))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)

    def sector_search_path(self, start_point, max_steps=100):
        """扇形搜索模式 (Sector Search)"""
        path = [start_point]
        center_x, center_y = start_point
        max_dist = 50.0 
        num_sectors = 6
        t_max = 10.0
        path_points = []
        
        for sector_index in range(num_sectors):
            start_angle = sector_index * (2 * np.pi / num_sectors)
            end_angle = (sector_index + 1) * (2 * np.pi / num_sectors)
            
            for t in np.linspace(0, t_max, int(max_steps / num_sectors)):
                radius = (t / t_max) * max_dist
                angle = start_angle + (end_angle - start_angle) * (t / t_max) + np.sin(t*3) * 0.1
                new_x = center_x + radius * np.cos(angle)
                new_y = center_y + radius * np.sin(angle)
                path_points.append((new_x, new_y))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)

    def pso_search_path(self, start_point, max_steps=100, pso_guidance_map=None):
        """
        粒子群优化 (PSO) 路径，使用当前的**后验概率图**进行引导 (递归贝叶斯)。
        pso_guidance_map: 上一搜索段的后验概率图，作为当前段的先验指导。
        """
        # 如果未提供后验概率图，则使用初始静态概率图
        if pso_guidance_map is None:
            pso_guidance_map = self.probability_map
            
        map_h, map_w = pso_guidance_map.shape
        def fitness_func(position):
            x, y = int(position[0]), int(position[1])
            # 适应度函数：最大化 PSO 粒子位置的当前后验概率（即当前 EPOS）
            if 0 <= x < map_w and 0 <= y < map_h: return pso_guidance_map[y, x]
            return 0.0

        n_particles = 15
        max_iterations = 10 
        
        flat_idx = np.argmax(pso_guidance_map)
        gbest_y, gbest_x = np.unravel_index(flat_idx, pso_guidance_map.shape)
        gbest_pos = np.array([gbest_x, gbest_y], dtype=float)

        particles = {
            'pos': np.random.uniform([0, 0], [map_w - 1, map_h - 1], (n_particles, 2)),
            'vel': np.zeros((n_particles, 2)),
            'pbest_pos': np.zeros((n_particles, 2)),
            'pbest_value': np.zeros(n_particles)
        }
        particles['pbest_pos'] = np.copy(particles['pos'])
        for i in range(n_particles):
            particles['pbest_value'][i] = fitness_func(particles['pos'][i])

        path_points = [start_point]
        c1, c2, w = 2.0, 2.0, 0.9

        for t in range(max_iterations):
            for i in range(n_particles):
                r1 = np.random.rand(2)
                r2 = np.random.rand(2)

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
                    
                # 重新计算全局最优位置
                current_gbest_val = fitness_func(gbest_pos)
                if current_value > current_gbest_val:
                    gbest_pos = particles['pos'][i]
            
            for pos in particles['pbest_pos']:
                path_points.append(tuple(pos.astype(int)))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)


    # --- 核心方法 (贝叶斯更新路径搜索模拟) ---
    
    def simulate_search_bayesian(self, algorithm, max_time=100, setup_time=5, bayesian_update=False):
        """
        贝叶斯搜索模拟：计算累计探测概率 (CPD) 随时间的变化。
        如果 bayesian_update=True，则 PSO 算法将利用递归贝叶斯更新进行动态路径规划。
        """
        deployment_points = self.probability_based_deployment() 
        point = deployment_points[0]
        point_tuple = (int(point[0]), int(point[1]))
        
        # 1. 初始化 (递归贝叶斯框架)
        # pso_guidance_map: 作为下一规划段的先验 (P_k|k-1)，初始为原始先验 (P_0)
        pso_guidance_map = np.copy(self.probability_map) 
        self.search_count_map = np.zeros_like(pso_guidance_map, dtype=int) 
        radius = max(1, int(self.coverage_width / 3 / (200/self.grid_width)))
        k = self.k_efficiency 
        
        # 2. 路径函数映射
        if algorithm == 'spiral': path_func = self.spiral_search_path
        elif algorithm == 'grid': path_func = self.grid_search_path
        elif algorithm == 'sector': path_func = self.sector_search_path
        elif algorithm == 'pso': path_func = self.pso_search_path
        else: raise ValueError(f"不支持的算法: {algorithm}")
            
        
        # 路径分段：只有 PSO + 贝叶斯更新时进行分段重规划
        num_segments = 1 if algorithm != 'pso' or not bayesian_update else 5
        steps_per_segment = (max_time - setup_time) // num_segments
        if steps_per_segment <= 0: steps_per_segment = max_time - setup_time
        
        full_path = [point_tuple] 
        cumulative_prob_history = [0.0] 
        total_time_history = [0.0]
        prob_raw_history = [0.0]
        weight_raw_history = [0.0]
        current_cumulative_prob = 0.0
        
        # 3. 循环分段模拟
        for segment in range(num_segments):
            
            # 3.1 路径规划 (预测步)
            if algorithm == 'pso' and bayesian_update:
                current_pos = full_path[-1]
                # PSO 使用上一段的后验概率图 pso_guidance_map 作为指导
                path_segment = self.pso_search_path(current_pos, steps_per_segment, pso_guidance_map=pso_guidance_map)
            else:
                # 非自适应算法只规划一次全路径
                if segment == 0:
                    path_segment = path_func(point_tuple, max_steps=(max_time-setup_time) + 1)
                else:
                    path_segment = [] 
            
            if len(path_segment) <= 1 and segment > 0: break

            if segment > 0 and path_segment: 
                path_segment = path_segment[1:] # 移除起点（与上一段的终点重复）

            # 3.2 模拟搜索 (更新步)
            for step_point in path_segment:
                if len(full_path) >= (max_time - setup_time) + 1: break 
                
                x, y = step_point
                
                new_found_prob_in_step = 0.0
                new_searched_prob_weight = 0.0
                
                # 确定当前搜索范围
                y_start = max(0, y - radius)
                y_end = min(self.grid_height, y + radius + 1)
                x_start = max(0, x - radius)
                x_end = min(self.grid_width, x + radius + 1)
                
                # 3.2.1 更新 CPD 和搜索计数
                for ny in range(y_start, y_end):
                    for nx in range(x_start, x_end):
                        distance_sq = (nx - x)**2 + (ny - y)**2
                        
                        if distance_sq <= radius**2:
                            
                            # P_Ci: 当前位置 (ny, nx) 的最新信念概率 (上一阶段的后验)
                            P_Ci = pso_guidance_map[ny, nx] 
                            N_old = self.search_count_map[ny, nx]
                            
                            if N_old == 0:
                                new_searched_prob_weight += self.probability_map[ny, nx] # 初始权重用原始先验
                            
                            self.search_count_map[ny, nx] += 1
                            N_new = self.search_count_map[ny, nx]
                            
                            # P(D | C_i): 探测概率
                            P_Di_new = 1 - np.exp(-k * N_new)
                            P_Di_old = 1 - np.exp(-k * N_old) if N_old > 0 else 0
                                
                            # P_increment: 增量发现概率 = P(C_i) * [P(D_new | C_i) - P(D_old | C_i)]
                            P_increment = P_Ci * (P_Di_new - P_Di_old)
                            new_found_prob_in_step += P_increment
                
                current_cumulative_prob += new_found_prob_in_step
                
                full_path.append(step_point)
                
                # 3.2.2 贝叶斯更新 (仅在启用时)：计算新的后验概率图 (假设未找到)
                if bayesian_update:
                    # 贝叶斯公式 for 静态目标: P(C_i | NF) = P(NF | C_i) * P(C_i)_original / P(NF)_total
                    
                    # 1. 计算总的发现概率 P(D_total) 和总的未发现概率 P(NF)_total
                    P_D_total = 0.0
                    for row in range(self.grid_height):
                        for col in range(self.grid_width):
                            # P(C_i)_original: 原始先验概率
                            P_Ci_prior_original = self.probability_map[row, col]
                            N = self.search_count_map[row, col]
                            P_Di = 1 - np.exp(-k * N) 
                            P_D_total += P_Ci_prior_original * P_Di
                            
                    P_NF_total = 1.0 - P_D_total
                    new_posterior_map = np.zeros_like(self.probability_map)
                    
                    if P_NF_total > 1e-9: 
                        for row in range(self.grid_height):
                            for col in range(self.grid_width):
                                P_Ci_prior_original = self.probability_map[row, col]
                                N = self.search_count_map[row, col]
                                P_Di = 1 - np.exp(-k * N)
                                
                                P_NF_given_Ci = 1.0 - P_Di
                                
                                # 计算新的后验 P(C_i | NF)
                                new_posterior_map[row, col] = (P_NF_given_Ci * P_Ci_prior_original) / P_NF_total
                        
                        # 更新指导图，用于下一次 PSO 规划 (递归)
                        pso_guidance_map = new_posterior_map
                    else:
                        pass
                
                # 3.2.3 记录历史数据
                time_per_step = (max_time - setup_time) / ((max_time - setup_time) if (max_time - setup_time) > 0 else 1)
                total_time_history.append(total_time_history[-1] + time_per_step)
                cumulative_prob_history.append(current_cumulative_prob)
                prob_raw_history.append(current_cumulative_prob)
                weight_raw_history.append(weight_raw_history[-1] + new_searched_prob_weight)
                
            if algorithm != 'pso' or not bayesian_update: break 
        
        # 4. 数据处理和返回
        self.search_paths[algorithm] = full_path
        
        time_points = total_time_history
        prob_points = cumulative_prob_history
        
        fixed_time_points = np.linspace(0, max_time, 50)
        interpolated_prob_time = np.interp(fixed_time_points, time_points, prob_points).tolist()
        
        return fixed_time_points, interpolated_prob_time, deployment_points, prob_raw_history, weight_raw_history, max_time

    def _calculate_coverage_map(self, path, coverage_width):
        """计算给定路径的覆盖热力图 (用于可视化搜索效率)"""
        coverage_map = np.zeros_like(self.probability_map, dtype=float)
        # 确保使用传入的 coverage_width
        radius = max(1, int(coverage_width / 3 / (200/self.grid_width))) 
        
        for point in path:
            x, y = point
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = x + dx, y + dy
                        distance_sq = (dx**2 + dy**2)
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            if distance_sq <= radius**2:
                                coverage_map[ny, nx] += np.exp(-distance_sq / (2 * (radius/2)**2))
        return coverage_map


# --- 运行函数：run_all_combinations_and_algorithms (修改为调用新的模拟函数) ---

def run_all_combinations_and_algorithms(strategies_to_test, max_time=100):
    """运行所有策略、所有设备组合和所有算法并收集结果"""
    algorithms = ['spiral', 'grid', 'sector', 'pso'] 
    all_results = {}
    base_models = {}
    
    for strategy in strategies_to_test:
        base_models[strategy] = SingleUnitDroneSearch(
            strategy=strategy,
            coverage_width=30.0,
            grid_width=200,
            grid_height=150
        )
        all_results[strategy] = {}
        
    for strategy in strategies_to_test:
        print(f"\n--- 正在收集 {strategy} 策略数据 ---")
        base_model = base_models[strategy]
        
        for combo_name, combo_params in EQUIPMENT_COMBOS.items():
            
            search_model = SingleUnitDroneSearch(
                strategy=strategy,
                search_speed=15.0,
                coverage_width=combo_params['coverage_width'],
                grid_width=base_model.grid_width,
                grid_height=base_model.grid_height,
                k_efficiency=combo_params['k_efficiency']
            )
            search_model.probability_map = base_model.probability_map
            search_model.crash_points = base_model.crash_points
            
            all_results[strategy][combo_name] = {}
            
            for algo in algorithms:
                try:
                    # 关键修改：调用新的 simulate_search_bayesian 函数
                    # 只有 PSO 算法启用 bayesian_update=True，实现动态路径规划
                    is_pso_bayesian = (algo == 'pso')
                    time_points, probabilities_time, deployment_points, prob_points_raw, weight_points_raw, current_max_time = search_model.simulate_search_bayesian(
                        algo, 
                        max_time, 
                        bayesian_update=is_pso_bayesian
                    )
                    path_data = search_model.search_paths.get(algo, [])
                    
                    all_results[strategy][combo_name][algo] = {
                        'time': time_points,
                        'prob_time': probabilities_time,
                        'deployment': deployment_points,
                        'prob_raw': prob_points_raw, 
                        'weight_raw': weight_points_raw, 
                        'max_time': current_max_time,
                        'path': path_data
                    }
                except ValueError as e:
                    print(f"警告: {e}，跳过 {combo_name} / {algo}。")
                except Exception as e:
                    print(f"错误: {combo_name} / {algo} 运行失败。")
                    traceback.print_exc()
    
    return base_models, all_results


# --- 性能打印函数 (更新 PSO 描述) ---

def print_performance_metrics(all_results, strategy_key, combos_order, algorithms_order):
    """打印指定策略下，所有组合和算法的性能指标"""
    print(f"\n{'='*20} 性能指标 ({strategy_key} 策略) {'='*20}")
    
    results = all_results[strategy_key]
    
    for combo_name in combos_order:
        combo_results = results[combo_name]
        print(f"\n--- 设备组合: {EQUIPMENT_COMBOS[combo_name]['desc']} ---")
        
        for algo in algorithms_order:
            if algo not in combo_results: continue
            
            res = combo_results[algo]
            time_points = res['time']
            probabilities = res['prob_time'] 
            max_time = res['max_time']
                
            algo_name = algo.upper()
            if algo == 'pso':
                 algo_name += ' (递归贝叶斯优化)'

            print(f"  > 搜索模式: {algo_name}")
            
            target_probabilities = [0.5, 0.75, 0.9]
            
            for target in target_probabilities:
                idx = np.argmax(np.array(probabilities) >= target)
                if idx > 0 and probabilities[idx] >= target:
                    print(f"    达到{target*100:.0f}%探测概率所需时间 (TTF): {time_points[idx]:.2f} 单位时间")
                else:
                    max_prob = max(probabilities) if probabilities else 0
                    print(f"    无法在 {max_time} 单位时间内达到{target*100:.0f}%概率 (最高: {max_prob*100:.1f}%)")
            
            if len(probabilities) > 1:
                final_prob = probabilities[-1]
                search_time = time_points[-1]
                print(f"    最终累计探测概率 (CPD, T={search_time:.0f}): {final_prob*100:.1f}%")


# --- 可视化函数 (更新 PSO 描述) ---

def visualize_3x4_results(base_models, all_results, strategy_key, max_time=100):
    """
    可视化函数：生成 3x4 图。
    每行 (3行) 代表一个设备组合的**完整分析**。
    每列 (4列) 代表四种分析维度。
    """
    print(f"\n开始生成任务三：设备组合 ({strategy_key} 策略) 综合分析图 ...")

    combos_order = list(EQUIPMENT_COMBOS.keys())
    algorithms_order = ['spiral', 'grid', 'sector', 'pso']
    
    analysis_titles = [
        'A. 搜索路径和覆盖', 
        'B. CPD 随时间变化', 
        'C. 搜索效率对比 (CPD vs. 覆盖权重)', 
        'D. 累计搜索覆盖强度'
    ]
    
    fig, axes = plt.subplots(len(combos_order), 4, figsize=(22, 18)) 
    fig.suptitle(f'任务三：单单元搜索性能分析 - 设备组合对比 ({strategy_key} 策略)', 
                 fontsize=22, fontweight='bold', y=1.02)
    
    model = base_models[strategy_key]
    probability_map = model.probability_map
    deployment_point = model.probability_based_deployment()[0]
    
    path_colors = {
        'spiral': '#00A693', 'grid': '#808080', 'sector': '#FF00AA', 'pso': '#007ACC',
    }
    path_linestyles = {
        'spiral': '-', 'grid': '--', 'sector': ':', 'pso': '-.', 
    }

    for row_idx, combo_name in enumerate(combos_order):
        combo_results = all_results[strategy_key][combo_name]
        combo_params = EQUIPMENT_COMBOS[combo_name]
        
        temp_model = SingleUnitDroneSearch(
            strategy=strategy_key,
            coverage_width=combo_params['coverage_width'],
            grid_width=model.grid_width,
            grid_height=model.grid_height,
            k_efficiency=combo_params['k_efficiency']
        )
        temp_model.probability_map = probability_map
        
        row_label = f'{combo_name.split("_")[0]}: {combo_params["desc"]}\n(W={combo_params["coverage_width"]:.0f}m, k={combo_params["k_efficiency"]})'
        axes[row_idx, 0].text(-0.35, 0.5, row_label, 
                              transform=axes[row_idx, 0].transAxes, 
                              fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
        
        if row_idx == 0:
            for col_idx, title in enumerate(analysis_titles):
                axes[row_idx, col_idx].set_title(title, fontsize=16)

        
        # --- 列 0: 路径 ---
        ax0 = axes[row_idx, 0]
        im0 = ax0.imshow(probability_map, cmap='hot_r', origin='lower', extent=[0, model.grid_width, 0, model.grid_height])
        ax0.plot(deployment_point[0], deployment_point[1], 'D', color='lime', markersize=12, markeredgecolor='black', markeredgewidth=1.5, label='部署点', zorder=5)
        
        for algo in algorithms_order:
            path = combo_results[algo]['path']
            algo_label = f'{algo.capitalize()}'
            if algo == 'pso': algo_label += ' (Bayes-Opt)'

            if len(path) > 0:
                path_array = np.array(path)
                ax0.plot(path_array[:, 0], path_array[:, 1], color=path_colors[algo], linestyle=path_linestyles[algo], linewidth=2.5, alpha=0.9, label=algo_label, zorder=4)
        
        ax0.legend(loc='upper left', fontsize=8, frameon=True, fancybox=True, shadow=True)
        ax0.set_xlabel('X坐标 (网格单位)')
        ax0.set_ylabel('Y坐标 (网格单位)')
        if row_idx == 0:
            fig.colorbar(im0, ax=ax0, orientation='vertical', shrink=0.85, aspect=20, pad=0.02, label='坠毁概率密度 P(C)')


        # --- 列 1: CPD-时间 ---
        ax1 = axes[row_idx, 1]
        for algo in algorithms_order:
            time_points = combo_results[algo]['time']
            probabilities = combo_results[algo]['prob_time']
            algo_label = f'{algo.capitalize()}'
            if algo == 'pso': algo_label += ' (Bayes-Opt)'
            ax1.plot(time_points, probabilities, color=path_colors[algo], linestyle=path_linestyles[algo], linewidth=3.0, alpha=0.9, label=algo_label)
        
        ax1.set_xlabel(f'搜索时间 (单位)')
        ax1.set_ylabel('累计探测概率 (CPD)')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, max_time)
        ax1.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True, shadow=True)


        # --- 列 2: CPD vs. P_searched ---
        ax2 = axes[row_idx, 2]
        for algo in algorithms_order:
            prob_raw = combo_results[algo]['prob_raw']
            weight_raw = combo_results[algo]['weight_raw']
            algo_label = f'{algo.capitalize()}'
            if algo == 'pso': algo_label += ' (Bayes-Opt)'
            ax2.plot(weight_raw, prob_raw, color=path_colors[algo], linestyle=path_linestyles[algo], linewidth=3.0, alpha=0.9, label=algo_label)

        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='理想效率')
        
        ax2.set_xlabel('累积搜索权重 (P_searched)')
        ax2.set_ylabel('累计探测概率 (CPD - P_found)')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, 1.05)
        ax2.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True, shadow=True)


        # --- 列 3: 综合搜索覆盖热力图 ---
        ax3 = axes[row_idx, 3]
        total_coverage_sum = np.zeros_like(probability_map, dtype=float)
        
        for algo in algorithms_order:
            path = combo_results[algo]['path']
            total_coverage_sum += temp_model._calculate_coverage_map(path, combo_params['coverage_width'])

        coverage_map_total = total_coverage_sum
        
        max_coverage = np.max(coverage_map_total)
        if max_coverage > 0: coverage_map_normalized = coverage_map_total / max_coverage
        else: coverage_map_normalized = coverage_map_total
            
        im3 = ax3.imshow(coverage_map_normalized, cmap='viridis', origin='lower', extent=[0, model.grid_width, 0, model.grid_height])
        ax3.plot(deployment_point[0], deployment_point[1], 'D', color='lime', markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        
        ax3.set_xlabel('X坐标 (网格单位)')
        ax3.set_ylabel('Y坐标 (网格单位)')
        
        fig.colorbar(im3, ax=ax3, orientation='vertical', shrink=0.85, aspect=20, pad=0.02, label=f'累计搜索覆盖强度')


    plt.tight_layout(rect=[0, 0, 1, 0.98]) 
    # 保存文件名包含策略名称
    plt.savefig(f'search_3x4_analysis_recursive_bayesian_pso_{strategy_key}_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# 运行修正后的模型
if __name__ == "__main__":
    STRATEGIES_TO_TEST = ['LAND', 'HOVER', 'RTH'] 
    DEFAULT_MAX_TIME = 100 
    
    base_models, all_results = run_all_combinations_and_algorithms(STRATEGIES_TO_TEST, DEFAULT_MAX_TIME)

    if all_results:
        combos_order = list(EQUIPMENT_COMBOS.keys())
        algorithms_order = ['spiral', 'grid', 'sector', 'pso']
        
        for strategy_key in STRATEGIES_TO_TEST:
            if strategy_key in all_results:
                print(f"\n{'#'*50}")
                print(f"### 正在处理策略: {strategy_key} ###")
                print(f"{'#'*50}")
                
                print_performance_metrics(all_results, strategy_key, combos_order, algorithms_order)
                
                try:
                    visualize_3x4_results(base_models, all_results, strategy_key, max_time=DEFAULT_MAX_TIME)
                except Exception as e:
                    print(f"策略 {strategy_key} 的 3x4 合并可视化出错: {e}")
                    traceback.print_exc()
            else:
                 print(f"警告: 策略 {strategy_key} 未能生成结果。")
    else:
        print("\n未能运行任何策略。请检查代码和环境设置。")