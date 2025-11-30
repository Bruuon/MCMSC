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

# 假设 drone_simulation.py 存在
try:
    from drone_simulation import DroneSimulator
except ImportError:
    class DroneSimulator:
        """用于占位，防止未导入时的NameError"""
        def simulate_mission(self, initial_pos, strategy, wind, t_fail, origin):
            # 返回一个默认的坐标，避免崩溃
            return [0, 0, 0], [0, 0, 0], [0, 0, 0]


class SingleUnitDroneSearch:
    # 任务三：搜索模型 - 单个搜索单元
    def __init__(self, strategy='LAND', search_speed=15.0, coverage_width=30.0, 
                 grid_width=200, grid_height=150):
        self.strategy = strategy # 故障前的无人机策略 (LAND/HOVER/RTH)
        self.num_drones = 1 # 单一搜索单元模型
        self.search_speed = search_speed # 搜索无人机的速度 (m/s)
        self.coverage_width = coverage_width # 搜索单元的有效覆盖宽度 (m)
        self.grid_width = grid_width # 概率图的网格宽度
        self.grid_height = grid_height # 概率图的网格高度
        
        # 调用任务一定位模型生成坠毁概率分布
        self.probability_map, self.crash_points = self.generate_probability_distribution()
        
        print(f"[{self.strategy}] 概率图尺寸: {self.grid_width} x {self.grid_height}")
        self.search_paths = {} # 存储不同算法生成的路径
        
        self.searched_flag_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)

    def generate_probability_distribution(self, n_runs=500):
        """
        生成坠毁概率分布图。
        （依赖于 drone_simulation.py 中任务一定位模型的结果）
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
    
    # --- 路径连接辅助函数 (保持代码不变) ---
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
                if len(path) >= max_steps:
                    break
            
            if len(path) >= max_steps:
                break
                
            current_pos = nearest_point
            remaining_points.discard(nearest_point)
            
        while len(path) < max_steps:
            path.append(path[-1])
            
        return path[:max_steps]


    # --- 4 种搜索路径算法实现 (用于搜索模式推荐) ---
    
    def spiral_search_path(self, start_point, max_steps=100):
        """螺旋搜索模式：适用于热点集中、但略有不确定的区域"""
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
                if new_point != path[-1]:
                    path.append(new_point)
            
            if len(path) >= max_steps: break
            
        return self._connect_points_to_path(path, start_point, max_steps)

    def grid_search_path(self, start_point, max_steps=100):
        """网格搜索模式：适用于广域、均匀覆盖区域"""
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
                for x in x_grid:
                    path_points.append((x, y_clamped))
            else: 
                for x in x_grid[::-1]:
                    path_points.append((x, y_clamped))
            
            if i < len(y_strips) - 1:
                next_y_clamped = np.clip(y_strips[i+1], 0, self.grid_height - 1)
                
                current_x = path_points[-1][0]
                y_step_interpolated = np.linspace(path_points[-1][1], next_y_clamped, 
                                                  int(strip_width/2)).astype(int)
                for y_val in y_step_interpolated:
                     path_points.append((current_x, y_val))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)

    def sector_search_path(self, start_point, max_steps=100):
        """扇形搜索模式：适用于快速覆盖以原点为中心的区域"""
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

    def pso_search_path(self, start_point, max_steps=100):
        """粒子群优化 (PSO) 路径：集中在概率最高的区域，追求高效率"""
        map_h, map_w = self.probability_map.shape
        def fitness_func(position):
            x, y = int(position[0]), int(position[1])
            if 0 <= x < map_w and 0 <= y < map_h:
                return self.probability_map[y, x]
            return 0.0

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
                    
                if current_value > fitness_func(gbest_pos):
                    gbest_pos = particles['pos'][i]
            
            for pos in particles['pbest_pos']:
                path_points.append(tuple(pos.astype(int)))
        
        return self._connect_points_to_path(path_points, start_point, max_steps)


    # --- 核心方法 (路径依赖的 CPD - 探测概率模型) ---
    
    def simulate_search(self, algorithm, max_time=100, setup_time=5):
        """
        模拟单个搜索算法，计算作为时间和累积搜索结果的函数的探测概率。
        (CPD: Cumulative Probability of Detection)
        """
        deployment_points = self.probability_based_deployment() 
        point = deployment_points[0]
        point_tuple = (int(point[0]), int(point[1]))
        
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
        
        if effective_path_length > 0:
            # 计算每一步耗时，以满足 max_time 的总搜索时间
            time_per_step = (max_time - setup_time) / effective_path_length
        else:
            time_per_step = max_time 
            
        # 计算搜索覆盖半径 (基于 coverage_width)
        radius = max(1, int(self.coverage_width / 3 / (200/self.grid_width)))
        
        cumulative_prob = 0.0
        probabilities = [0.0] * effective_path_length
        current_searched_flag_map = np.zeros_like(self.probability_map, dtype=bool)

        for step in range(effective_path_length):
            x, y = path[step]
            new_prob_found_in_step = 0.0
            
            # 搜索区域：以当前点 (x, y) 为中心，半径为 radius 的圆
            y_start = max(0, y - radius)
            y_end = min(self.grid_height, y + radius + 1)
            x_start = max(0, x - radius)
            x_end = min(self.grid_width, x + radius + 1)
            
            for ny in range(y_start, y_end):
                for nx in range(x_start, x_end):
                    distance_sq = (nx - x)**2 + (ny - y)**2
                    # 仅搜索未被覆盖且在有效搜索半径内的网格
                    if distance_sq <= radius**2 and not current_searched_flag_map[ny, nx]:
                        # 累加网格点的坠毁概率
                        new_prob_found_in_step += self.probability_map[ny, nx]
                        current_searched_flag_map[ny, nx] = True
            
            cumulative_prob += new_prob_found_in_step
            probabilities[step] = cumulative_prob

        # 时间点和概率点用于绘制 CPD 曲线
        time_points = [0.0] * len(path)
        prob_points = [0.0] * len(path)
        
        time_points[0] = 0
        prob_points[0] = 0.0
        
        for step in range(effective_path_length):
            time_points[step + 1] = setup_time + (step + 1) * time_per_step
            prob_points[step + 1] = probabilities[step] 
            
        # 插值以获得平滑的 CPD 曲线
        fixed_time_points = np.linspace(0, max_time, 50)
        interpolated_prob = np.interp(fixed_time_points, time_points, prob_points)
        
        return fixed_time_points, interpolated_prob.tolist(), deployment_points

    def run_all_algorithms(self, max_time=100):
        """运行所有算法并收集结果"""
        algorithms = ['spiral', 'grid', 'sector', 'pso'] 
        results = {}
        
        for algo in algorithms:
            time_points, probabilities, deployment_points = self.simulate_search(algo, max_time)
            results[algo] = {
                'time': time_points,
                'prob': probabilities,
                'deployment': deployment_points
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


# --- 性能打印和可视化函数 (修改文本内容) ---

def print_performance_metrics(all_results, strategies_order):
    """打印性能指标，评估搜索效率"""
    
    print(f"\n{'='*20} 任务三：搜索模型性能指标 (路径依赖CPD) {'='*20}")
    
    algorithms_order = ['spiral', 'grid', 'sector', 'pso']
    
    for strategy in strategies_order:
        results = all_results[strategy]
        print(f"\n--- 坠毁前策略: {strategy} ---")
        
        for algo in algorithms_order:
            if algo not in results: continue
            
            time_points = results[algo]['time']
            probabilities = results[algo]['prob']
            
            print(f"  > 搜索模式: {algo.upper()}")
            
            target_probabilities = [0.25, 0.5, 0.75, 0.9]
            
            for target in target_probabilities:
                idx = np.argmax(np.array(probabilities) >= target)
                if idx > 0 and probabilities[idx] >= target:
                    print(f"    达到{target*100:.0f}%探测概率所需时间: {time_points[idx]:.2f} 单位")
                else:
                    max_prob = max(probabilities) if probabilities else 0
                    print(f"    无法达到{target*100:.0f}%概率 (最高: {max_prob*100:.1f}%)")
            
            if len(probabilities) > 1:
                final_prob = probabilities[-1]
                search_time = time_points[-1]
                efficiency = final_prob / search_time if search_time > 0 else 0
                print(f"    最终累积探测概率 (CPD): {final_prob*100:.1f}%")
                print(f"    搜索效率: {efficiency:.4f} 概率/单位时间")

def visualize_combined_results(all_results, models, max_time=100):
    """
    可视化结果：比较不同坠毁策略下的搜索模式性能。
    注意：仅调整路径颜色，**路径宽度和所有代码逻辑保持不变**。
    """
    print("\n开始生成任务三：搜索模型综合可视化结果...")

    fig, axes = plt.subplots(3, 3, figsize=(24, 18)) 
    
    # 修改总标题
    fig.suptitle('任务三：搜索模型 - 无人机搜索模式性能对比 (基于坠毁前策略)', fontsize=24, fontweight='bold', y=1.01)
    
    algorithms_order = ['spiral', 'grid', 'sector', 'pso']
    
    # --- 颜色方案优化 (仅修改颜色定义，不改动代码逻辑) ---
    colors = {
        'spiral': '#00A693',  # 鲜艳的青绿色 (突出)
        'grid':   '#808080',  # 中灰色 (不突出)
        'sector': '#FF00AA',  # 亮粉色/洋红色 (突出)
        'pso':    '#007ACC',  # 亮蓝色 (突出)
    }
    
    linestyles = {
        'spiral': '-', 'grid': '--', 'sector': ':', 
        'pso': '-.', 
    }
    
    # 路径宽度和描边保持不变
    default_linewidth = 3.5 
    path_outline_color = 'white'
    path_outline_width = 1.0 
    # -----------------------------------------------------
    
    strategies_order = ['LAND', 'HOVER', 'RTH']
    # 修改列标题
    column_titles = ['任务一：坠毁概率分布与搜索路径', '任务三：探测概率随时间变化 (CPD)', '任务三：综合搜索覆盖热力图']
    
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(f'{title}', fontsize=16, fontweight='bold')

    for i, strategy in enumerate(strategies_order):
        model = models[strategy]
        results = all_results[strategy]
        deployment_point = results[algorithms_order[0]]['deployment'][0] if algorithms_order[0] in results else (0, 0)
        
        # 修改行标签
        axes[i, 0].text(-0.2, 0.5, f'坠毁前策略:\n{strategy}', 
                        transform=axes[i, 0].transAxes, 
                        fontsize=18, fontweight='bold', ha='right', va='center')
        
        # --- 子图 (i, 0): 概率分布和搜索路径 ---
        ax0 = axes[i, 0]
        im0 = ax0.imshow(model.probability_map, cmap='hot_r', origin='lower', 
                         extent=[0, model.grid_width, 0, model.grid_height])
        
        if len(model.crash_points) > 0:
            unique_crash_points = list(set(model.crash_points))
            crash_x, crash_y = zip(*unique_crash_points)
            ax0.scatter(crash_x, crash_y, c='deepskyblue', s=30, alpha=0.8, marker='X', 
                        edgecolor='white', linewidth=0.8, label='模拟坠毁点', zorder=3)
        
        ax0.plot(deployment_point[0], deployment_point[1], 'D', color='lime', 
                 markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                 label='推荐部署点', zorder=5)
        
        total_coverage_sum = np.zeros_like(model.probability_map, dtype=float)
        
        # 绘制搜索路径
        for algo in algorithms_order:
            path = model.search_paths.get(algo, [])
            if len(path) > 0:
                path_array = np.array(path)
                
                # 绘制白色描边 (底层)
                ax0.plot(path_array[:, 0], path_array[:, 1], 
                         color=path_outline_color, linestyle=linestyles[algo], 
                         linewidth=default_linewidth + path_outline_width, alpha=1.0, zorder=3.5) 
                
                # 绘制路径颜色 (上层)
                ax0.plot(path_array[:, 0], path_array[:, 1], 
                         color=colors[algo], linestyle=linestyles[algo], 
                         linewidth=default_linewidth, alpha=0.9, label=f'{algo.capitalize()}搜索模式', zorder=4)
                
                coverage_map_single = model._calculate_coverage_map(path)
                total_coverage_sum += coverage_map_single


        if i == 0:
            handles, labels = ax0.get_legend_handles_labels()
            unique_labels = {}
            for h, l in zip(handles, labels):
                if l not in unique_labels:
                    unique_labels[l] = h
            ax0.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        ax0.set_xlabel('X坐标 (网格单位)', fontsize=10)
        ax0.set_ylabel('Y坐标 (网格单位)', fontsize=10)
        ax0.tick_params(axis='both', which='major', labelsize=9)


        # --- 子图 (i, 1): 搜索概率随时间变化 (CPD) ---
        ax1 = axes[i, 1]
        
        for algo in algorithms_order:
            time_points = results[algo]['time']
            probabilities = results[algo]['prob']
            
            ax1.plot(time_points, probabilities, 
                     color=colors[algo], linestyle=linestyles[algo], 
                     linewidth=3.5, alpha=0.9, label=f'{algo.capitalize()} CPD')
            
            # 标记关键时间点 (50%和90%)
            targets = [0.5, 0.9]
            for target in targets:
                idx = np.argmax(np.array(probabilities) >= target)
                if idx > 0 and probabilities[idx] >= target:
                    ax1.plot(time_points[idx], probabilities[idx], 'o', 
                             color=colors[algo], markersize=7, markeredgecolor='black', markeredgewidth=0.8)
            
        ax1.set_xlabel('时间 (单位)', fontsize=10)
        ax1.set_ylabel('累积探测概率 (CPD)', fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, max_time)
        ax1.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax1.tick_params(axis='both', which='major', labelsize=9)

        # --- 子图 (i, 2): 综合搜索覆盖热力图 ---
        ax2 = axes[i, 2]
        
        coverage_map_total = total_coverage_sum
        
        max_coverage = np.max(coverage_map_total)
        if max_coverage > 0:
            coverage_map_normalized = coverage_map_total / max_coverage
        else:
            coverage_map_normalized = coverage_map_total
            
        # 使用 'viridis' 颜色图
        im2 = ax2.imshow(coverage_map_normalized, cmap='viridis', origin='lower',
                         extent=[0, model.grid_width, 0, model.grid_height])
        
        ax2.plot(deployment_point[0], deployment_point[1], 'D', color='lime', 
                 markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        if len(model.crash_points) > 0:
            ax2.scatter(crash_x, crash_y, c='deepskyblue', s=30, alpha=0.8, marker='X', 
                        edgecolor='white', linewidth=0.8, zorder=4)
        
        ax2.set_xlabel('X坐标 (网格单位)', fontsize=10)
        ax2.set_ylabel('Y坐标 (网格单位)', fontsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=9)

    # 绘制色带
    for i in range(3):
        # 概率分布色带 (左侧)
        cbar0 = fig.colorbar(axes[i, 0].images[0], ax=axes[i, 0], orientation='vertical', 
                             shrink=0.9, aspect=20, pad=0.02, label='坠毁概率密度')
        cbar0.ax.tick_params(labelsize=8)

        # 覆盖热力图色带 (右侧)
        cbar2 = fig.colorbar(axes[i, 2].images[0], ax=axes[i, 2], orientation='vertical', 
                             shrink=0.9, aspect=20, pad=0.02, label='搜索覆盖强度')
        cbar2.ax.tick_params(labelsize=8)


    plt.tight_layout(rect=[0.05, 0.0, 1, 0.98]) 
    plt.savefig('search_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 运行修正后的模型
if __name__ == "__main__":
    strategies_to_test = ['LAND', 'HOVER', 'RTH']
    all_results = {}
    models = {}
        
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
            results = search_model.run_all_algorithms(max_time=100)
            all_results[strategy] = results
        except ValueError as e:
            print(f"警告: {e}，跳过 {strategy} 策略。")

    if all_results:
        strategies_ran = list(all_results.keys())
        print_performance_metrics(all_results, strategies_ran)
        
        try:
            visualize_combined_results(all_results, models, max_time=100)
        except Exception as e:
            print(f"合并可视化出错: {e}")
            traceback.print_exc()
    else:
        print("\n未能运行任何策略。请检查代码和环境设置。")