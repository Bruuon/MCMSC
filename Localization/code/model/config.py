# config.py
# 参数仓库 (Parameter Warehouse)
# 集中管理所有物理常数、环境参数和模拟设置

import os
import numpy as np
from scipy.ndimage import map_coordinates

# ==========================================
# 1. 环境参数 (Environment - Alpine Region)
# ==========================================
# 假设目标区域为复杂山地地形
WIND_K = 2.0          # Weibull 形状参数 (Rayleigh distribution assumption)
WIND_A = 11.2         # Weibull 尺度参数 (对应平均风速 ~9.92 m/s)
GRAVITY = 9.81        # 重力加速度 (m/s^2)
AIR_DENSITY = 1.225   # 空气密度 (kg/m^3), 标准海平面

# 新增：风向参数 (Wind Direction)
# 模拟真实场景：风不是乱吹的，而是有主导方向的
WIND_DIR_MEAN = np.radians(135)  # 主导风向：东南风 (135度)
WIND_DIR_KAPPA = 4.0             # Von Mises 分布的集中度参数 (值越大越集中，0为均匀分布)

# ==========================================
# 2. 地形参数 (Real SRTM Data)
# ==========================================
# DEM 文件路径配置
DEM_FILE_PATH = os.path.join(os.path.dirname(__file__), "Localization", "data", "N46E008.hgt")
ORIGIN_LAT = 46.64    # 无人机事件中心纬度
ORIGIN_LON = 8.20     # 无人机事件中心经度

# 全局变量存储 DEM 数据
DEM_DATA = None
DEM_SIZE = 0
DEM_ORIGIN_ELEVATION = 0

def load_dem_data():
    """
    加载 SRTM HGT 地形数据文件。
    """
    global DEM_DATA, DEM_SIZE, DEM_ORIGIN_ELEVATION
    
    if not os.path.exists(DEM_FILE_PATH):
        print(f"Error: DEM file not found at {DEM_FILE_PATH}")
        print("Please ensure 'N46E008.hgt' is placed in 'Localization/data/'")
        # Fallback to synthetic if file missing (optional, but better to fail loud as requested)
        raise FileNotFoundError(f"DEM file missing: {DEM_FILE_PATH}")

    # HGT 文件是 big-endian 16-bit signed integers
    file_size = os.path.getsize(DEM_FILE_PATH)
    dim = int(np.sqrt(file_size / 2))
    
    if dim * dim * 2 != file_size:
        raise ValueError(f"File size {file_size} does not match standard HGT dimensions (1201 or 3601).")
        
    print(f"Loading DEM data from {DEM_FILE_PATH} (Grid: {dim}x{dim})...")
    with open(DEM_FILE_PATH, 'rb') as f:
        DEM_DATA = np.fromfile(f, dtype='>i2').reshape((dim, dim))
    
    DEM_SIZE = dim
    
    # 计算原点海拔 (用于相对高度计算)
    # HGT 覆盖 N46-N47, E008-E009
    # 像素索引计算:
    # lat_index = (1 - (lat - floor(lat))) * (dim - 1)
    # lon_index = (lon - floor(lon)) * (dim - 1)
    
    lat_idx = int((1 - (ORIGIN_LAT - int(ORIGIN_LAT))) * (DEM_SIZE - 1))
    lon_idx = int((ORIGIN_LON - int(ORIGIN_LON)) * (DEM_SIZE - 1))
    
    DEM_ORIGIN_ELEVATION = DEM_DATA[lat_idx, lon_idx]
    print(f"Origin Elevation at ({ORIGIN_LAT}, {ORIGIN_LON}): {DEM_ORIGIN_ELEVATION} m")

def get_terrain_z(x, y):
    """
    获取相对地形高度 (相对于原点海拔)。
    x, y: 距离原点的米数 (East, North)
    """
    if DEM_DATA is None:
        load_dem_data()
        
    # 坐标转换: Meters -> Lat/Lon offsets
    # 1 deg Lat ~= 111132 m
    # 1 deg Lon ~= 111132 * cos(lat) m
    meters_per_lat = 111132.0
    meters_per_lon = 111132.0 * np.cos(np.radians(ORIGIN_LAT))
    
    d_lat = y / meters_per_lat
    d_lon = x / meters_per_lon
    
    target_lat = ORIGIN_LAT + d_lat
    target_lon = ORIGIN_LON + d_lon
    
    # 转换为 HGT 索引
    # N46E008 covers Lat [46, 47], Lon [8, 9]
    # Row 0 is North (Lat 47), Row 3600 is South (Lat 46)
    # Col 0 is West (Lon 8), Col 3600 is East (Lon 9)
    
    row = (1 - (target_lat - 46.0)) * (DEM_SIZE - 1)
    col = (target_lon - 8.0) * (DEM_SIZE - 1)
    
    # 使用 scipy.ndimage.map_coordinates 进行双线性插值 (Bilinear Interpolation)
    # order=1 对应双线性插值
    # mode='nearest' 处理边界情况
    
    if np.isscalar(row):
        coords = np.array([[row], [col]])
        elevations = map_coordinates(DEM_DATA, coords, order=1, mode='nearest')[0]
    else:
        # 确保输入是扁平化的，以便 map_coordinates 处理
        coords = np.array([row.ravel(), col.ravel()])
        elevations = map_coordinates(DEM_DATA, coords, order=1, mode='nearest')
        # 恢复原始形状
        elevations = elevations.reshape(row.shape)
    
    # 返回相对高度 (Relative Elevation)
    # z > 0: 比起飞点高 (山峰)
    # z < 0: 比起飞点低 (山谷)
    return elevations - DEM_ORIGIN_ELEVATION

# ==========================================
# 3. 无人机参数 (Drone - DJI Matrice 300 RTK)
# ==========================================
MASS = 6.3            # 质量 (kg)
AREA = 0.08           # 迎风面积 (m^2), 翻滚状态下的估算平均值
CD_MEAN = 1.0         # 阻力系数均值 (Tumbling state)
CD_STD = 0.1          # 阻力系数标准差 (模拟姿态不确定性)

# ==========================================
# 0. 模拟设置 (Simulation Settings)
# ==========================================
NUM_SIMULATIONS = 50000  # 蒙特卡洛模拟次数 (降低以提高可视化速度)
INITIAL_HEIGHT = 50.0    # 初始高度 (m)
INITIAL_SPEED_X = 10.0   # 初始水平速度 (m/s)
INITIAL_SPEED_Y = 0.0    # 初始垂直速度 (m/s)

def generate_valley_terrain(x_grid, y_grid):
    """
    生成合成峡谷地形高度 (Synthetic Valley Terrain)
    
    参数:
        x_grid, y_grid: 网格坐标 (numpy meshgrid)
        
    返回:
        z_terrain: 地形高度网格
    """
    # U-shaped Valley along X-axis
    # Z = 0 if |y| < width/2 else steepness * (|y| - width/2)^2
    abs_y = np.abs(y_grid)
    z_terrain = np.where(
        abs_y < VALLEY_FLOOR_WIDTH / 2,
        0,
        MOUNTAIN_STEEPNESS * (abs_y - VALLEY_FLOOR_WIDTH / 2)**2
    )
    
    # 限制最大高度，避免无限高
    z_terrain = np.minimum(z_terrain, MOUNTAIN_HEIGHT_MAX)
    
    return z_terrain
