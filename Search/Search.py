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


class SingleUnitDroneSearch:
    """
    任务三核心：单单元搜索模型
    基于无人机失联前的策略，预测坠毁概率，并评估四种单单元搜索算法
    （螺旋、网格、扇形、PSO）在最小化定位时间方面的性能。
    """
    def __init__(self, strategy='LAND', search_speed=15.0, coverage_width=30.0, 
                 grid_width=200, grid_height=150):
        self.strategy = strategy # 故障前的无人机策略 (LAND/HOVER/RTH)
        self.num_drones = 1 # 单一搜索单元模型
        self.search_speed = search_speed # 搜索无人机的速度 (m/s)
        self.coverage_width = coverage_width # 搜索单元的有效覆盖宽度 (m)
        self.grid_width = grid_width # 概率图的网格宽度
        self.grid_height = grid_height # 概率图的网格高度
        
        # 1. 生成坠毁概率分布图
        self.probability_map, self.crash_points = self.generate_probability_distribution()
        
        print(f"[{self.strategy}] 概率图尺寸: {self.grid_width} x {self.grid_height}")
        self.search_paths = {} # 存储不同算法生成的路径
        
        self.searched_flag_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)

    # --- 概率分布和部署点方法 ---
    
    def generate_probability_distribution(self, n_runs=500):
        """
        生成坠毁概率分布图（依赖于 drone_simulation.py 的结果）。
        通过蒙特卡洛模拟和高斯平滑获得最终概率密度图。
        """
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
                # 故障时间 T_fail 建模
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
                # 使用高斯核平滑点 (标准差为 8)
                probability_map += np.exp(-distance_sq / (2 * (8**2)))
        else:
            probability_map = self.create_default_distribution()
            crash_points = []
        
        # 进一步平滑和归一化
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
        # 对于单单元搜索，部署点通常是概率最高的点
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
        
        if not valid_points:
            return [start_point]
            
        path = [start_point]
        remaining_points = set(valid_points)
        current_pos = start_point
        
        move_distance = 2.5 # 每一步的最小移动距离
        
        while remaining_points and len(path) < max_steps:
            # 找到最近点作为下一个目标
            nearest_point = min(remaining_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(current_pos)))
            
            distance = np.linalg.norm(np.array(nearest_point) - np.array(current_pos))
            steps_between = int(distance / move_distance) + 1 
            
            # 插值以创建连续路径
            x_values = np.linspace(current_pos[0], nearest_point[0], steps_between).astype(int)
            y_values = np.linspace(current_pos[1], nearest_point[1], steps_between).astype(int)
            
            for i in range(1, steps_between):
                new_point = (x_values[i], y_values[i])
                path.append(new_point)
                if len(path) >= max_steps:
                    break
            
            if len(path) >= max_steps:
                break
                
            current_pos = nearest_point
            remaining_points.discard(nearest_point)
            
        while len(path) < max_steps:
            path.append(path[-1])
            
        return path[:max_steps]

    # --- 四种搜索算法路径生成 ---

    def spiral_search_path(self, start_point, max_steps=100):
        """螺旋搜索模式 (Spiral Search)：适用于热点集中、但略有不确定的区域"""
        path = [start_point]
        center_x, center_y = start_point
        max_radius = min(center_x, self.grid_width-center_x, center_y, self.grid_height-center_y, 40)
        t_max = 8 * np.pi 
        t_steps = max_steps - 1
        a = 1.0 
        b = max_radius / t_max 
        
        for t in np.linspace(0, t_max, t_steps):
            # 阿基米德螺线公式
            radius = a + b * t
            new_x = int(center_x + radius * np.cos(t))
            new_y = int(center_y + radius * np.sin(t))
            new_point = (new_x, new_y)
            
            if (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height and
                len(path) < max_steps):
                if new_point != path[-1]:
                    path.append(new_point)
            
            if len(path) >= max_steps: break
            
        return self._connect_points_to_path(path, start_point, max_steps)

    def grid_search_path(self, start_point, max_steps=100):
        """网格搜索模式 (Grid Search)：适用于广域、均匀覆盖区域"""
        path = [start_point]
        center_x, center_y = start_point
        # 基于覆盖宽度确定条带宽度
        strip_width = max(5, int(self.coverage_width * 0.8 / (200/self.grid_width)))
        max_dist = 60 # 搜索范围距离
        
        x_start = max(0, int(center_x - max_dist))
        x_end = min(self.grid_width, int(center_x + max_dist))
        y_start = max(0, int(center_y - max_dist))
        y_end = min(self.grid_height, int(center_y + max_dist))

        x_grid = np.arange(x_start, x_end, 1)
        y_strips = np.arange(y_start, y_end, strip_width)
        
        path_points = []
        
        for i, y in enumerate(y_strips):
            y_clamped = np.clip(y, 0, self.grid_height - 1)
            
            if i % 2 == 0: # 奇数条带向右扫
                for x in x_grid:
                    path_points.append((x, y_clamped))
            else: # 偶数条带向左扫 (S型)
                for x in x_grid[::-1]:
                    path_points.append((x, y_clamped))
            
            # 连接两个条带
            if i < len(y_strips) - 1:
                next_y_clamped = np.clip(y_strips[i+1], 0, self.grid_height - 1)
                
                current_x = path_points[-1][0]
                y_step_interpolated = np.linspace(path_points[-1][1], next_y_clamped, 
                                                  int(strip_width/2)).astype(int)
                for y_val in y_step_interpolated:
                     path_points.append((current_x, y_val))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)

    def sector_search_path(self, start_point, max_steps=100):
        """扇形搜索模式 (Sector Search)：适用于快速覆盖以原点为中心的区域"""
        path = [start_point]
        center_x, center_y = start_point
        
        max_dist = 50.0 
        num_sectors = 6 # 分为 6 个扇区
        t_max = 10.0
        
        path_points = []
        
        for sector_index in range(num_sectors):
            start_angle = sector_index * (2 * np.pi / num_sectors)
            end_angle = (sector_index + 1) * (2 * np.pi / num_sectors)
            
            for t in np.linspace(0, t_max, int(max_steps / num_sectors)):
                # 径向扩展，同时角度扫描
                radius = (t / t_max) * max_dist
                angle = start_angle + (end_angle - start_angle) * (t / t_max) + np.sin(t*3) * 0.1
                
                new_x = center_x + radius * np.cos(angle)
                new_y = center_y + radius * np.sin(angle)
                
                path_points.append((new_x, new_y))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)

    def pso_search_path(self, start_point, max_steps=100):
        """粒子群优化 (PSO) 路径：集中在概率最高的区域，追求高效率"""
        map_h, map_w = self.probability_map.shape
        def fitness_func(position):
            """适应度函数：坠毁概率密度"""
            x, y = int(position[0]), int(position[1])
            if 0 <= x < map_w and 0 <= y < map_h:
                return self.probability_map[y, x]
            return 0.0

        n_particles = 15
        max_iterations = 10 
        
        # 全局最优解初始化为概率图的最高点
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
        for i in range(n_particles):
            particles['pbest_value'][i] = fitness_func(particles['pos'][i])

        path_points = [start_point]
        c1, c2, w = 2.0, 2.0, 0.9 # 惯性权重、个体学习因子、社会学习因子

        for t in range(max_iterations):
            for i in range(n_particles):
                r1 = np.random.rand(2)
                r2 = np.random.rand(2)

                # 速度更新公式 (V_new = w*V_old + c1*r1*(pbest-X) + c2*r2*(gbest-X))
                cognitive = c1 * r1 * (particles['pbest_pos'][i] - particles['pos'][i])
                social = c2 * r2 * (gbest_pos - particles['pos'][i])
                particles['vel'][i] = w * particles['vel'][i] + cognitive + social
                
                particles['pos'][i] += particles['vel'][i]
                
                # 边界限制
                particles['pos'][i, 0] = np.clip(particles['pos'][i, 0], 0, map_w - 1)
                particles['pos'][i, 1] = np.clip(particles['pos'][i, 1], 0, map_h - 1)

                current_value = fitness_func(particles['pos'][i])
                if current_value > particles['pbest_value'][i]:
                    particles['pbest_value'][i] = current_value
                    particles['pbest_pos'][i] = particles['pos'][i]
                    
                if current_value > fitness_func(gbest_pos):
                    gbest_pos = particles['pos'][i]
            
            # 记录每一次迭代的最优粒子位置，作为路径点
            for pos in particles['pbest_pos']:
                path_points.append(tuple(pos.astype(int)))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)


    # --- 核心方法 (路径依赖的 CPD - 累计探测概率模型) ---
    
    def simulate_search(self, algorithm, max_time=100, setup_time=5):
        """
        模拟单个搜索算法，计算累计探测概率 (CPD) 随时间的变化。
        
        模型公式（基于搜索努力 $N_i$）：
        $CPD = \sum P(C_i) \times P(D|C_i, N_i)$
        $P(D|C_i, N_i) = 1 - e^{-k \times N_i}$ 
        其中 $P(C_i)$ 是区域 $i$ 的坠毁概率，$N_i$ 是对区域 $i$ 的累计搜索次数。
        """
        deployment_points = self.probability_based_deployment() 
        point = deployment_points[0]
        point_tuple = (int(point[0]), int(point[1]))
        
        # 1. 生成路径
        if algorithm == 'spiral':
            path = self.spiral_search_path(point_tuple)
        elif algorithm == 'grid':
            path = self.grid_search_path(point_tuple)
        elif algorithm == 'sector':
            path = self.sector_search_path(point_tuple)
        elif algorithm == 'pso':
            path = self.pso_search_path(point_tuple)
        else:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        self.search_paths[algorithm] = path
        
        effective_path_length = len(path) - 1
        
        # 2. 计算时间步长
        if effective_path_length > 0:
            time_per_step = (max_time - setup_time) / effective_path_length
        else:
            time_per_step = max_time 
            
        # 搜索半径（网格单位）
        radius = max(1, int(self.coverage_width / 3 / (200/self.grid_width)))
        
        cumulative_prob = 0.0
        probabilities = [0.0] * effective_path_length
        
        # 累积搜索权重 P_searched：用于衡量搜索覆盖的概率质量（每个网格只算一次）
        cumulative_prob_weights = [0.0] * effective_path_length 
        current_searched_prob_sum = 0.0
        
        # 追踪每个网格被搜索的次数 (搜索努力 $N_i$)
        searched_count_map = np.zeros_like(self.probability_map, dtype=int)
        
        # 搜索效率参数 k (探测概率 P(D) 的陡峭度)
        k = 0.25 

        # 3. 路径遍历和 CPD 累积
        for step in range(effective_path_length):
            x, y = path[step]
            
            new_searched_prob_weight = 0.0 # 这一步新覆盖的概率权重
            new_found_prob_in_step = 0.0 # 这一步新贡献的探测概率
            
            # 确定当前搜索单元的覆盖区域
            y_start = max(0, y - radius)
            y_end = min(self.grid_height, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(self.grid_width, x + radius + 1)
            
            for ny in range(y_start, y_end):
                for nx in range(x_start, x_end):
                    distance_sq = (nx - x)**2 + (ny - y)**2
                    
                    if distance_sq <= radius**2:
                        
                        P_Ci = self.probability_map[ny, nx] # 区域坠毁概率
                        
                        # A. 计算累积搜索权重 (P_searched)
                        # 仅当网格首次被覆盖时，其 P(C) 才计入 P_searched
                        if searched_count_map[ny, nx] == 0:
                            new_searched_prob_weight += P_Ci
                        
                        # B. 增加搜索努力 $N_i$
                        searched_count_map[ny, nx] += 1
                        
                        # C. 计算探测概率 P(D) 的增量
                        N = searched_count_map[ny, nx]
                        P_Di = 1 - np.exp(-k * N)
                        
                        P_Di_old = 1 - np.exp(-k * (N - 1)) if N > 1 else 0
                            
                        # 探测概率的增量 $P_{increment} = P(C_i) \times (P(D|N_{new}) - P(D|N_{old}))$
                        P_increment = P_Ci * (P_Di - P_Di_old)
                        
                        new_found_prob_in_step += P_increment
            
            # 更新累积量
            cumulative_prob += new_found_prob_in_step
            current_searched_prob_sum += new_searched_prob_weight # P_searched 只加新的
            
            probabilities[step] = cumulative_prob
            cumulative_prob_weights[step] = current_searched_prob_sum # P_searched 随时间增长

        # 4. 封装结果并插值到固定时间轴
        time_points = [0.0] * len(path)
        prob_points = [0.0] * len(path)
        weight_points = [0.0] * len(path)
        
        time_points[0] = 0
        prob_points[0] = 0.0
        weight_points[0] = 0.0 
        
        for step in range(effective_path_length):
            time_points[step + 1] = setup_time + (step + 1) * time_per_step
            prob_points[step + 1] = probabilities[step] # 累计探测概率
            weight_points[step + 1] = cumulative_prob_weights[step] # 累计搜索权重
            
        fixed_time_points = np.linspace(0, max_time, 50)
        interpolated_prob_time = np.interp(fixed_time_points, time_points, prob_points)
        
        # 返回 max_time 用于下游函数
        return fixed_time_points, interpolated_prob_time.tolist(), deployment_points, prob_points, weight_points, max_time

    def run_all_algorithms(self, max_time=100):
        """运行所有算法并收集结果"""
        algorithms = ['spiral', 'grid', 'sector', 'pso'] 
        results = {}
        
        for algo in algorithms:
            # 返回结果中：prob_raw 是 P_found (CPD), weight_raw 是 P_searched
            time_points, probabilities_time, deployment_points, prob_points_raw, weight_points_raw, current_max_time = self.simulate_search(algo, max_time)
            results[algo] = {
                'time': time_points,
                'prob_time': probabilities_time, # 50个插值点
                'deployment': deployment_points,
                'prob_raw': prob_points_raw, # 原始路径点
                'weight_raw': weight_points_raw, # P_searched (累积搜索结果)
                'max_time': current_max_time # 将 max_time 存储在结果中
            }
        return results

    def _calculate_coverage_map(self, path):
        """计算给定路径的覆盖热力图 (用于可视化搜索效率)"""
        coverage_map = np.zeros_like(self.probability_map, dtype=float)
        radius = max(1, int(self.coverage_width / 3 / (200/self.grid_width))) 
        
        for point in path:
            x, y = point
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = x + dx, y + dy
                        distance_sq = (dx**2 + dy**2)
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            if distance_sq <= radius**2:
                                # 覆盖强度随距离中心点的距离呈高斯衰减
                                coverage_map[ny, nx] += np.exp(-distance_sq / (2 * (radius/2)**2))
        return coverage_map


# --- 性能打印和可视化函数 (修复 max_time 引用) ---

def print_performance_metrics(all_results, strategies_order):
    """打印四种搜索算法在不同策略下的关键性能指标"""
    print(f"\n{'='*20} 任务三：单单元搜索模型性能指标 (基于路径CPD) {'='*20}")
    
    algorithms_order = ['spiral', 'grid', 'sector', 'pso']
    
    for strategy in strategies_order:
        results = all_results[strategy]
        
        # FIX: 从结果中安全地获取 max_time
        max_time = 100 # 默认值，如果结果中找不到
        for algo in algorithms_order:
            if algo in results and 'max_time' in results[algo]:
                max_time = results[algo]['max_time']
                break
                
        print(f"\n--- 坠毁前策略: {strategy} ---")
        
        for algo in algorithms_order:
            if algo not in results: continue
            
            time_points = results[algo]['time']
            # 使用时间轴上的概率结果进行指标分析
            probabilities = results[algo]['prob_time'] 
            
            print(f"  > 搜索模式: {algo.upper()} (用于 {strategy} 策略的搜索)")
            
            target_probabilities = [0.25, 0.5, 0.75, 0.9]
            
            for target in target_probabilities:
                idx = np.argmax(np.array(probabilities) >= target)
                if idx > 0 and probabilities[idx] >= target:
                    print(f"    达到{target*100:.0f}%探测概率所需时间 (TTF): {time_points[idx]:.2f} 单位时间")
                else:
                    max_prob = max(probabilities) if probabilities else 0
                    # FIX: max_time 现在已定义
                    print(f"    无法在 {max_time} 单位时间内达到{target*100:.0f}%概率 (最高: {max_prob*100:.1f}%)")
            
            if len(probabilities) > 1:
                final_prob = probabilities[-1]
                search_time = time_points[-1]
                efficiency = final_prob / search_time if search_time > 0 else 0
                print(f"    最终累计探测概率 (CPD, T={search_time:.0f}): {final_prob*100:.1f}%")
                print(f"    平均搜索效率 (CPD/时间): {efficiency:.4f}")

def visualize_combined_results(all_results, models, max_time=100):
    """
    可视化函数：生成 3 张图，每张图对应一个策略。
    图表内容包括：概率分布与路径、CPD-时间、CPD-P_searched、综合覆盖热力图。
    """
    print("\n开始生成任务三：单单元搜索模式综合分析图...")

    strategies_order = ['LAND', 'HOVER', 'RTH']
    algorithms_order = ['spiral', 'grid', 'sector', 'pso']
    
    colors = {
        'spiral': '#00A693',  
        'grid':   '#808080',  
        'sector': '#FF00AA',  
        'pso':    '#007ACC',  
    }
    
    linestyles = {
        'spiral': '-', 'grid': '--', 'sector': ':', 
        'pso': '-.', 
    }
    
    default_linewidth = 3.5 
    path_outline_color = 'white'
    path_outline_width = 1.0 
    
    
    for i, strategy in enumerate(strategies_order):
        model = models[strategy]
        results = all_results[strategy]
        deployment_point = results[algorithms_order[0]]['deployment'][0] if algorithms_order[0] in results else (0, 0)

        # 针对 RTH 策略，将图例移动到左下角，以避免遮挡右上方的路径
        if strategy == 'RTH':
            legend_loc = 'lower left'
        else:
            legend_loc = 'upper right'

        # 从结果中获取 max_time，确保与 CPD 图一致
        current_max_time = max_time
        if algorithms_order[0] in results and 'max_time' in results[algorithms_order[0]]:
             current_max_time = results[algorithms_order[0]]['max_time']

        # 创建一个 2x2 的图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12)) 
        fig.suptitle(f'任务三：单单元搜索模式性能分析 ({strategy} 策略)', 
                     fontsize=18, fontweight='bold', y=1.02)
        
        # 将 axes 展平方便索引
        ax_flat = axes.flatten()

        # --------------------------------------------------
        # --- 子图 (0, 0): 概率分布和搜索路径 ---
        # --------------------------------------------------
        ax0 = ax_flat[0]
        ax0.set_title(f'坠毁概率分布和四种搜索路径 ({strategy} 模式)', fontsize=14)
        im0 = ax0.imshow(model.probability_map, cmap='hot_r', origin='lower', 
                         extent=[0, model.grid_width, 0, model.grid_height])
        
        if len(model.crash_points) > 0:
            crash_x, crash_y = zip(*list(set(model.crash_points)))
            ax0.scatter(crash_x, crash_y, c='deepskyblue', s=30, alpha=0.8, marker='X', 
                        edgecolor='white', linewidth=0.8, label='蒙特卡洛坠毁点', zorder=3)
        
        ax0.plot(deployment_point[0], deployment_point[1], 'D', color='lime', 
                 markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                 label='最佳初始部署点', zorder=5) # 文本修改
        
        for algo in algorithms_order:
            path = model.search_paths.get(algo, [])
            if len(path) > 0:
                path_array = np.array(path)
                
                # 绘制路径外描边
                ax0.plot(path_array[:, 0], path_array[:, 1], 
                         color=path_outline_color, linestyle=linestyles[algo], 
                         linewidth=default_linewidth + path_outline_width, alpha=1.0, zorder=3.5) 
                
                # 绘制路径主体
                ax0.plot(path_array[:, 0], path_array[:, 1], 
                         color=colors[algo], linestyle=linestyles[algo], 
                         linewidth=default_linewidth, alpha=0.9, label=f'{algo.capitalize()} 搜索路径', zorder=4) # 文本修改
        
        handles, labels = ax0.get_legend_handles_labels()
        unique_labels = {}
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels[l] = h
        
        # ******** 调整图例位置的修改点 ********
        ax0.legend(unique_labels.values(), unique_labels.keys(), loc=legend_loc, fontsize=8, frameon=True, fancybox=True, shadow=True)
        # ***********************************
        
        ax0.set_xlabel('X坐标 (网格单位)', fontsize=10)
        ax0.set_ylabel('Y坐标 (网格单位)', fontsize=10)
        
        cbar0 = fig.colorbar(im0, ax=ax0, orientation='vertical', 
                             shrink=0.85, aspect=20, pad=0.02, label='坠毁概率密度 (P(C))') # 文本修改

        # --------------------------------------------------
        # --- 子图 (0, 1): 搜索概率随时间变化 (CPD) ---
        # --------------------------------------------------
        ax1 = ax_flat[1]
        ax1.set_title('累计探测概率 (CPD) 随时间变化曲线', fontsize=14) # 文本修改
        
        for algo in algorithms_order:
            time_points = results[algo]['time']
            probabilities = results[algo]['prob_time']
            
            ax1.plot(time_points, probabilities, 
                     color=colors[algo], linestyle=linestyles[algo], 
                     linewidth=3.5, alpha=0.9, label=f'{algo.capitalize()} 模式') # 文本修改
            
            targets = [0.5, 0.9]
            for target in targets:
                idx = np.argmax(np.array(probabilities) >= target)
                if idx > 0 and probabilities[idx] >= target:
                    ax1.plot(time_points[idx], probabilities[idx], 'o', 
                             color=colors[algo], markersize=7, markeredgecolor='black', markeredgewidth=0.8)
            
        ax1.set_xlabel('搜索时间 (单位)', fontsize=10) # 文本修改
        ax1.set_ylabel('累计探测概率 (CPD)', fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, current_max_time)
        ax1.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True, shadow=True)


        # -------------------------------------------------------------
        # --- 子图 (1, 0): 探测概率 vs. 累积搜索权重 (效率曲线) ---
        # -------------------------------------------------------------
        ax2 = ax_flat[2]
        ax2.set_title('搜索效率对比 (CPD vs. 覆盖权重)', fontsize=14) # 文本修改
        
        for algo in algorithms_order:
            # P_found (CPD) vs. P_searched (累积搜索权重)
            prob_raw = results[algo]['prob_raw']
            weight_raw = results[algo]['weight_raw']
            
            ax2.plot(weight_raw, prob_raw, 
                     color=colors[algo], linestyle=linestyles[algo], 
                     linewidth=3.5, alpha=0.9, label=f'{algo.capitalize()} 效率') # 文本修改
            
            targets = [0.5, 0.9]
            for target in targets:
                idx = np.argmax(np.array(prob_raw) >= target)
                if idx > 0 and prob_raw[idx] >= target:
                    ax2.plot(weight_raw[idx], prob_raw[idx], 's', 
                             color=colors[algo], markersize=7, markeredgecolor='black', markeredgewidth=0.8)

        # 效率分析：理想曲线是 y=x，斜率越高越好
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='理想效率 ($P_{found} = P_{searched}$)')
        
        ax2.set_xlabel('累积搜索权重 (P_searched)', fontsize=10) # 文本修改
        ax2.set_ylabel('累计探测概率 (CPD - P_found)', fontsize=10) # 文本修改
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, 1.05)
        ax2.legend(loc='lower right', fontsize=8, frameon=True, fancybox=True, shadow=True)

        # --------------------------------------------------
        # --- 子图 (1, 1): 综合搜索覆盖热力图 ---
        # --------------------------------------------------
        ax3 = ax_flat[3]
        ax3.set_title('四种模式叠加的累计搜索覆盖强度', fontsize=14) # 文本修改
        
        total_coverage_sum = np.zeros_like(model.probability_map, dtype=float)
        for algo in algorithms_order:
            path = model.search_paths.get(algo, [])
            # 这里的覆盖图是四种算法的叠加，用于展示其整体搜索范围
            total_coverage_sum += model._calculate_coverage_map(path)

        coverage_map_total = total_coverage_sum
        
        max_coverage = np.max(coverage_map_total)
        if max_coverage > 0:
            coverage_map_normalized = coverage_map_total / max_coverage
        else:
            coverage_map_normalized = coverage_map_total
            
        im3 = ax3.imshow(coverage_map_normalized, cmap='viridis', origin='lower',
                         extent=[0, model.grid_width, 0, model.grid_height])
        
        ax3.plot(deployment_point[0], deployment_point[1], 'D', color='lime', 
                 markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        if len(model.crash_points) > 0:
            crash_x, crash_y = zip(*list(set(model.crash_points)))
            ax3.scatter(crash_x, crash_y, c='deepskyblue', s=30, alpha=0.8, marker='X', 
                        edgecolor='white', linewidth=0.8, zorder=4)
        
        ax3.set_xlabel('X坐标 (网格单位)', fontsize=10)
        ax3.set_ylabel('Y坐标 (网格单位)', fontsize=10)

        cbar3 = fig.colorbar(im3, ax=ax3, orientation='vertical', 
                             shrink=0.85, aspect=20, pad=0.02, label='累计搜索覆盖强度 (归一化)')

        plt.tight_layout(rect=[0, 0, 1, 0.98]) 
        # 规范图片命名
        plt.savefig(f'search_analysis_{strategy}_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

# 运行修正后的模型
if __name__ == "__main__":
    strategies_to_test = ['LAND', 'HOVER', 'RTH']
    all_results = {}
    models = {}
    
    # 设定默认的最大搜索时间，与 simulate_search 中的默认值保持一致
    DEFAULT_MAX_TIME = 100 
        
    for strategy in strategies_to_test:
        print(f"\n{'='*60}")
        print(f"正在收集 {strategy} 策略数据 (4 种搜索模式性能分析)...")
        
        search_model = SingleUnitDroneSearch(
            strategy=strategy,
            search_speed=15.0,
            coverage_width=30.0,
            grid_width=200,
            grid_height=150
        )
        models[strategy] = search_model
        
        try:
            # 传入 DEFAULT_MAX_TIME
            results = search_model.run_all_algorithms(max_time=DEFAULT_MAX_TIME)
            all_results[strategy] = results
        except ValueError as e:
            print(f"警告: {e}，跳过 {strategy} 策略。")
        except Exception as e:
            print(f"错误: {strategy} 策略运行失败。")
            traceback.print_exc()

    if all_results:
        strategies_ran = list(all_results.keys())
        print_performance_metrics(all_results, strategies_ran)
        
        try:
            visualize_combined_results(all_results, models, max_time=DEFAULT_MAX_TIME)
        except Exception as e:
            print(f"合并可视化出错: {e}")
            traceback.print_exc()
    else:
        print("\n未能运行任何策略。请检查代码和环境设置。")