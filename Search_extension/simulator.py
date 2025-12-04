# simulator.py
# 模拟器 (The Runner)
# 执行蒙特卡洛模拟并可视化结果

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
import physics_engine
import os

def run_simulation():
    print(f"Starting Monte Carlo Simulation with {config.NUM_SIMULATIONS} particles...")
    
    # 1. 生成随机输入 (Stochastic Inputs)
    # -------------------------------------------------
    # A. 风速 (Wind Speed) - Weibull Distribution
    # numpy.random.weibull 生成的是标准 Weibull (scale=1), 需要乘以尺度参数 A
    wind_speeds = config.WIND_A * np.random.weibull(config.WIND_K, config.NUM_SIMULATIONS)
    
    # B. 风向 (Wind Direction) - Von Mises Distribution (模拟主导风向)
    # 论文指出：落点分布通常不是圆形的，而是受风向影响的非对称分布
    wind_directions = np.random.vonmises(config.WIND_DIR_MEAN, config.WIND_DIR_KAPPA, config.NUM_SIMULATIONS)
    
    # C. 阻力系数 (Drag Coefficient) - Normal Distribution
    # 模拟无人机翻滚时的姿态不确定性
    cd_values = np.random.normal(config.CD_MEAN, config.CD_STD, config.NUM_SIMULATIONS)
    cd_values = np.maximum(0.1, cd_values) # 物理约束: Cd > 0
    
    # D. 迎风面积 (Area) - 也可以随机化，这里暂时取定值或均匀分布
    # 假设面积在 0.05 到 0.11 之间均匀分布 (参考 config.AREA = 0.08)
    area_values = np.random.uniform(0.05, 0.11, config.NUM_SIMULATIONS)
    
    # E. 初始速度 (Initial Velocities) - 也可以添加随机扰动
    # 这里暂时使用定值
    v_xi = np.full(config.NUM_SIMULATIONS, config.INITIAL_SPEED_X)
    v_yi = np.full(config.NUM_SIMULATIONS, config.INITIAL_SPEED_Y)
    
    # 2. 运行物理引擎 (Run Physics Engine)
    # -------------------------------------------------
    # 计算阻力常数 c = 0.5 * rho * A * Cd
    c_values = 0.5 * config.AIR_DENSITY * area_values * cd_values
    
    # 调用闭式解核心
    results = physics_engine.calculate_trajectory(
        m=config.MASS,
        c=c_values,
        v_xi=v_xi,
        v_yi=v_yi,
        y_initial=config.INITIAL_HEIGHT,
        g=config.GRAVITY
    )
    
    fall_time = results["fall_time"]
    radial_distance_air = results["horizontal_distance"]
    
    # 3. 坐标变换与风场叠加 (Coordinate Transformation)
    # -------------------------------------------------
    # Step 1: Stochastic Generation (Done above in Section 1)
    # Step 2: Physics Engine Calculation (Done above in Section 2)
    #   - 'radial_distance_air' is the distance the drone travels relative to the air mass.
    
    # Step 3: Vector Addition (Drone Vector + Wind Drift Vector)
    #   - 假设无人机初始航向为 X 轴正方向 (East, 0 rad)
    #   - 矢量加法原理: P_ground = P_air + V_wind * t
    #   - 这里的逻辑是正确的: 我们先计算无人机相对于空气的位移 (drone_dx_air)，
    #     然后加上空气本身相对于地面的位移 (wind_dx, wind_dy)。
    
    # 1. 无人机在空气中的位移 (相对于气团)
    drone_dx_air = radial_distance_air  # 沿航向 (X轴)
    drone_dy_air = np.zeros_like(radial_distance_air)
    
    # 2. 风的漂移位移 (Wind Drift)
    # 风向 wind_directions 是风吹去的方向 (Vector direction).
    wind_dx = wind_speeds * np.cos(wind_directions) * fall_time
    wind_dy = wind_speeds * np.sin(wind_directions) * fall_time
    
    # 3. 最终地面坐标 (Ground Coordinates) - Flat Earth Assumption
    x_flat = drone_dx_air + wind_dx
    y_flat = drone_dy_air + wind_dy
    
    # Step 4: Terrain Collision Correction (Iterative Solver)
    # -------------------------------------------------
    # 解决 "Ghost Drone" 问题：无人机不能穿过山体到达 z=0
    # 使用迭代法解决循环依赖: Impact(x,y) <-> Terrain(z) <-> FlightTime(t)
    
    # 初始猜测: 假设地形高度为 0 (Flat Earth)
    x_curr = x_flat
    y_curr = y_flat
    
    # 迭代参数
    MAX_ITER = 3  # 3次迭代通常足够收敛
    
    print("Refining impact points with iterative terrain collision solver...")
    
    for i in range(MAX_ITER):
        # 1. 获取当前猜测点的地形高度
        z_terrain = config.get_terrain_z(x_curr, y_curr)
        
        # 2. 计算有效下落高度 (Effective Drop Height)
        # 目标是落到 z_terrain，所以总下落距离是 H_initial - z_terrain
        # 如果 z_terrain > H_initial，说明起飞点就在地下或者撞在起飞点，设为极小值
        effective_drop_height = config.INITIAL_HEIGHT - z_terrain
        
        # 物理约束: 下落高度不能为负 (不能向上飞去撞地)
        # 如果 effective_drop_height <= 0, 说明撞在起飞点上方(山峰)，
        # 这种情况下距离应极短。设为 0.1m 以避免除零错误。
        effective_drop_height = np.maximum(0.1, effective_drop_height)
        
        # 3. 重新计算物理轨迹 (Recalculate Physics)
        # 使用新的下落高度调用物理引擎
        # 注意: 我们只关心下落阶段的变化，假设初始速度不变
        # physics_engine.calculate_trajectory 的 y_initial 参数实际上就是"总下落高度"
        # (假设最终落到 y=0 的相对坐标系中)
        new_results = physics_engine.calculate_trajectory(
            m=config.MASS,
            c=c_values,
            v_xi=v_xi,
            v_yi=v_yi,
            y_initial=effective_drop_height, # 传入相对高度
            g=config.GRAVITY
        )
        
        new_fall_time = new_results["fall_time"]
        new_air_dist = new_results["horizontal_distance"]
        
        # 4. 更新地面坐标 (Update Coordinates)
        # P_ground = P_air(new_t) + V_wind * new_t
        
        # 空气位移
        new_drone_dx_air = new_air_dist # 沿航向
        
        # 风漂移
        new_wind_dx = wind_speeds * np.cos(wind_directions) * new_fall_time
        new_wind_dy = wind_speeds * np.sin(wind_directions) * new_fall_time
        
        # 更新坐标
        x_curr = new_drone_dx_air + new_wind_dx
        y_curr = 0 + new_wind_dy # drone_dy_air is 0
        
    # 迭代结束，使用最终结果
    x_final = x_curr
    y_final = y_curr
    z_final = config.get_terrain_z(x_final, y_final)
    fall_time = new_fall_time
    radial_distance_air = new_air_dist
    
    # 4. 导出数据 (Export Data)
    # -------------------------------------------------
    import pandas as pd
    df_export = pd.DataFrame({
        'x_impact': x_final,
        'y_impact': y_final,
        'z_impact': z_final,
        'fall_time': fall_time,
        'travel_distance_flat': radial_distance_air
    })
    
    # 输出目录: Localization/outputs/
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, 'simulation_results.csv')
    
    df_export.to_csv(csv_filename, index=False)
    print(f"Simulation data exported to '{csv_filename}'")

    # 打印统计信息 (提前打印，以便调试)
    print("-" * 30)
    print(f"Statistics:")
    print(f"Mean Fall Time: {np.mean(fall_time):.2f} s")
    print(f"Mean Travel Distance (Air): {np.mean(radial_distance_air):.2f} m")
    print(f"Mean Impact Location: X={np.mean(x_final):.2f}, Y={np.mean(y_final):.2f}")
    print(f"Standard Deviation: X={np.std(x_final):.2f}, Y={np.std(y_final):.2f}")
    print(f"Z Impact Range: Min={np.min(z_final):.2f}, Max={np.max(z_final):.2f}")
    print("-" * 30)

    # 5. 可视化 (Visualization)
    # -------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    # --- 新增：地形绘制 (Terrain Rendering) ---
    # 定义网格范围 (根据落点分布自动调整，并留出余量)
    # 限制绘图范围，避免因极端离群点导致地图过大
    x_mean, x_std = np.mean(x_final), np.std(x_final)
    y_mean, y_std = np.mean(y_final), np.std(y_final)
    x_min, x_max = x_mean - 4*x_std, x_mean + 4*x_std
    y_min, y_max = y_mean - 4*y_std, y_mean + 4*y_std
    
    # 生成高分辨率网格 (High-Resolution Grid for Visualization)
    # 解决 "Minecraft" 像素化问题，使用 1米/像素 的分辨率
    grid_res = 1.0  # 1 meter per pixel
    grid_x = np.arange(x_min, x_max, grid_res)
    grid_y = np.arange(y_min, y_max, grid_res)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    
    print(f"Generating terrain mesh for visualization ({X_grid.shape[0]}x{X_grid.shape[1]} pixels)...")
    
    # 计算地形高度 (使用 config 中的函数，现在支持双线性插值)
    Z_terrain = config.get_terrain_z(X_grid, Y_grid)
    
    # 绘制地形等高线 (Background)
    # 使用 'terrain' colormap 模拟山地
    terrain_contour = plt.contourf(X_grid, Y_grid, Z_terrain, levels=20, cmap='terrain', alpha=0.3)
    plt.colorbar(terrain_contour, label='Terrain Elevation (Relative to Origin, m)')
    
    # 添加地形等高线线条
    plt.contour(X_grid, Y_grid, Z_terrain, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    # ----------------------------------------
    
    # A. 散点图 (Scatter Plot)
    # 过滤绘图范围内的点
    mask = (x_final >= x_min) & (x_final <= x_max) & (y_final >= y_min) & (y_final <= y_max)
    plt.scatter(x_final[mask], y_final[mask], alpha=0.05, s=1, color='blue', label='Impact Points')
    
    # B. 核密度估计 (KDE Heatmap)
    # 使用 seaborn 绘制等高线热力图 (使用 fill=False 加快速度，或者减少 levels)
    try:
        sns.kdeplot(x=x_final[mask], y=y_final[mask], cmap="Reds", fill=True, alpha=0.5, levels=10, thresh=0.05)
    except Exception as e:
        print(f"Warning: KDE plot failed ({e}), skipping heatmap.")
    
    # 标记原点 (Event Location)
    plt.plot(0, 0, 'k+', markersize=15, markeredgewidth=3, label='Event Location (0,0)')
    
    # --- 新增：速度矢量可视化 (Velocity Vectors) ---
    # 1. 初始无人机速度矢量 (Initial Drone Velocity)
    # 方向: 0度 (East), 长度: INITIAL_SPEED_X
    # 放大系数: 5.0 (为了在地图上可见)
    scale_vec = 5.0
    plt.arrow(0, 0, config.INITIAL_SPEED_X * scale_vec, 0, 
              color='blue', width=0.5, head_width=2, length_includes_head=True, 
              label=f'Drone Velocity ({config.INITIAL_SPEED_X} m/s)')
    
    # 2. 平均风速矢量 (Mean Wind Velocity)
    # 方向: config.WIND_DIR_MEAN, 长度: 平均风速
    # Weibull Mean approx = A * Gamma(1 + 1/k)
    # 这里直接用模拟生成的风速均值
    mean_wind_speed = np.mean(wind_speeds)
    wind_vec_x = mean_wind_speed * np.cos(config.WIND_DIR_MEAN) * scale_vec
    wind_vec_y = mean_wind_speed * np.sin(config.WIND_DIR_MEAN) * scale_vec
    
    plt.arrow(0, 0, wind_vec_x, wind_vec_y, 
              color='red', width=0.5, head_width=2, length_includes_head=True, 
              label=f'Mean Wind ({mean_wind_speed:.1f} m/s)')
    # ---------------------------------------------
    
    # 标记平均落点
    mean_x = np.mean(x_final)
    mean_y = np.mean(y_final)
    plt.plot(mean_x, mean_y, 'gx', markersize=12, markeredgewidth=3, label=f'Mean Impact ({mean_x:.1f}, {mean_y:.1f})')
    
    plt.title(f"UAV Ground Impact Probability Map (N={config.NUM_SIMULATIONS})\n"
              f"Height={config.INITIAL_HEIGHT}m, Speed={config.INITIAL_SPEED_X}m/s, Wind=Weibull(k={config.WIND_K}, A={config.WIND_A})")
    plt.xlabel("Distance East (m)")
    plt.ylabel("Distance North (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # 保存图片
    # 输出目录: Localization/images/
    image_dir = os.path.join(os.path.dirname(__file__), "..", "..", "images")
    os.makedirs(image_dir, exist_ok=True)
    
    output_filename = os.path.join(image_dir, "impact_heatmap.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Simulation complete. Heatmap saved to {output_filename}")


if __name__ == "__main__":
    run_simulation()
