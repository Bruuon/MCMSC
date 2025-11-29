# 无人机坠落轨迹模拟程序使用指南

## 1. 程序解决的问题

本程序主要解决题目中 **“定位 (Positioning)”** 部分的任务：

> **题目要求**：开发一个模型，预测无人机随时间变化的位置。与这些预测相关的不确定性因素是什么？

**本程序的解决方案**：
1.  **物理建模**：使用微分方程（ODE）描述无人机在重力、空气阻力和风力作用下的运动轨迹。
2.  **不确定性量化**：通过 **蒙特卡洛模拟 (Monte Carlo Simulation)**，多次随机生成初始条件和环境参数，模拟出成百上千种可能的坠落路径。
3.  **结果输出**：生成坠落点的概率分布图（热力图），直观展示无人机最可能掉落的区域，为后续的搜索策略提供数据支持。

---

## 2. 程序运行流程图

以下流程图展示了 `drone_simulation.py` 的核心逻辑：

```mermaid
graph LR
    B["初始化无人机物理参数<br>(质量, 迎风面积, 阻力系数)"] --> C["设置模拟次数 N"]
    C --> D{循环 N 次}
    
    D -->|每次迭代| E["随机生成初始状态<br>(位置误差, 初始速度)"]
    E --> F["随机生成环境参数<br>(风速, 风向, 垂直气流)"]
    
    F --> G["求解微分方程 "]
    G --> H{检测是否撞地}
    
    H -->|否| G
    H -->|是| I["记录撞击点坐标"]
    
    I --> J{达到 N 次?}
    J -->|否| D
    J -->|是| K[汇总撞击点数据]
    
    K --> L[绘制散点图与热力图]
```

---

## 3. 数学模型详解 (Mathematical Model)

本程序的核心基于 **牛顿第二定律** 的刚体动力学方程，并结合 **空气动力学阻力**。以下是详细的公式推导，可直接用于论文的 "Model Development" 章节。

### 3.1 微分方程部分 (The Differential Equations)

#### 物理原理
无人机的运动由合外力决定：
$$ \mathbf{F}_{net} = m \mathbf{a} = m \frac{d\mathbf{v}}{dt} $$

作用在无人机上的主要力有两个：
1.  **重力 (Gravity, $\mathbf{G}$)**：始终垂直向下。
    $$ \mathbf{G} = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix} $$
2.  **空气阻力 (Aerodynamic Drag, $\mathbf{D}$)**：方向与无人机相对于空气的运动方向相反。

#### 空气阻力公式
我们采用经典的二次阻力方程。关键在于**相对速度 ($\mathbf{v}_{rel}$)**，即无人机速度减去风速：
$$ \mathbf{v}_{rel} = \mathbf{v}_{drone} - \mathbf{v}_{wind} $$

阻力向量的方向与 $\mathbf{v}_{rel}$ 相反：
$$ \mathbf{D} = - \frac{1}{2} \rho C_d A \|\mathbf{v}_{rel}\| \mathbf{v}_{rel} $$
*   $\rho$: 空气密度 (1.225 kg/m³)
*   $C_d$: 阻力系数 (无量纲)
*   $A$: 迎风面积 (m²)

#### 最终的微分方程组 (ODE System)
将上述力代入 $m\frac{d\mathbf{v}}{dt} = \mathbf{G} + \mathbf{D}$，并分解到 $x, y, z$ 三个轴上，得到代码中 `dynamics` 函数使用的方程组：

令 $\mathbf{v}_{rel} = [v_x - W_x, v_y - W_y, v_z - W_z]^T$，且 $V_{mag} = \|\mathbf{v}_{rel}\|$。

$$
\begin{cases}
\frac{dx}{dt} = v_x \\
\frac{dy}{dt} = v_y \\
\frac{dz}{dt} = v_z \\
\frac{dv_x}{dt} = -\frac{\rho C_d A}{2m} V_{mag} (v_x - W_x) \\
\frac{dv_y}{dt} = -\frac{\rho C_d A}{2m} V_{mag} (v_y - W_y) \\
\frac{dv_z}{dt} = -g -\frac{\rho C_d A}{2m} V_{mag} (v_z - W_z)
\end{cases}
$$

### 3.2 蒙特卡洛模拟部分 (Monte Carlo Simulation)

微分方程描述了“给定初始条件下的轨迹”，而蒙特卡洛模拟解决了“初始条件未知”的问题。我们在代码中使用 **正态分布 (Normal Distribution)** 来描述不确定性。

#### 初始位置的不确定性
假设 GPS 数据有误差，误差服从正态分布：
$$ x_0 \sim \mathcal{N}(\mu_{x}, \sigma_{pos}^2), \quad y_0 \sim \mathcal{N}(\mu_{y}, \sigma_{pos}^2), \quad z_0 \sim \mathcal{N}(\mu_{z}, \sigma_{alt}^2) $$
*   代码对应：`np.random.normal(0, 5)` (均值0，标准差5米)

#### 初始速度的不确定性
假设无人机失控时的速度在巡航速度附近波动：
$$ v_{x0} \sim \mathcal{N}(v_{cruise}, \sigma_{vel}^2) $$
*   代码对应：`np.random.normal(15, 2)` (均值15m/s，标准差2m/s)

#### 环境风场的不确定性
假设风速大小服从正态分布（取绝对值），风向在 $0$ 到 $2\pi$ 之间均匀分布：
$$ \|\mathbf{W}_{xy}\| \sim |\mathcal{N}(\mu_{wind}, \sigma_{wind}^2)| $$
$$ \theta_{wind} \sim \mathcal{U}(0, 2\pi) $$
$$ W_x = \|\mathbf{W}_{xy}\| \cos(\theta_{wind}), \quad W_y = \|\mathbf{W}_{xy}\| \sin(\theta_{wind}) $$
*   代码对应：`wind_mag = np.abs(np.random.normal(5, 2))`

---

## 4. 如何修改参数以匹配特定场景

你可以通过修改 `drone_simulation.py` 中的特定代码行来模拟不同的场景。

### 3.1 修改无人机硬件参数
如果你知道无人机的具体型号（例如大疆 Mavic 或大型固定翼），可以在 `DroneSimulation` 类初始化时修改。

**代码位置**：`class DroneSimulation` 的 `__init__` 方法。

```python
# 原始代码
def __init__(self, mass=2.0, area=0.1, cd=0.5):

# 修改示例：模拟更重、更大的无人机
# mass: 质量 (kg)
# area: 迎风面积 (m^2)
# cd: 空气阻力系数 (流线型越低，方形越高)
def __init__(self, mass=5.0, area=0.2, cd=0.6): 
```

### 3.2 修改初始飞行状态
修改无人机失联时的状态。例如，它是悬停时掉落，还是高速飞行时掉落？

**代码位置**：`run_monte_carlo` 函数中的 `Step 1` 部分。

```python
# --- 修改高度 ---
# 原始：500米高空
z0 = np.random.normal(500, 10) 
# 修改：低空飞行 (100米)
z0 = np.random.normal(100, 5)

# --- 修改速度 ---
# 原始：巡航速度 15m/s
vx0 = np.random.normal(15, 2)
# 修改：悬停失控 (速度接近 0)
vx0 = np.random.normal(0, 0.5)
vy0 = np.random.normal(0, 0.5)
```

### 3.3 修改环境风场
山区的天气多变，风是影响落点分布最大的因素。

**代码位置**：`run_monte_carlo` 函数中的 `Step 2` 部分。

```python
# --- 修改风速 ---
# 原始：中等风速 (均值 5m/s)
wind_mag = np.abs(np.random.normal(5, 2))
# 修改：强风暴 (均值 15m/s，波动大)
wind_mag = np.abs(np.random.normal(15, 5))

# --- 修改风向 ---
# 原始：随机风向 (0 到 2pi)
wind_angle = np.random.uniform(0, 2 * np.pi)
# 修改：固定吹向东方 (风向 0度，只有小幅波动)
wind_angle = np.random.normal(0, 0.1) 
```

### 3.4 修改地形 (进阶)
目前的模型假设地面是平的 ($z=0$)。你可以加入山坡。

**代码位置**：`terrain_height` 方法。

```python
# 原始：平地
return 0.0

# 修改：简单的上坡 (坡度 10%)
return 0.1 * x 

# 修改：起伏的山地 (使用正弦函数模拟)
return 50 * np.sin(x / 100) + 50 * np.cos(y / 100)
```

---

## 5. 附加问题解答 (Additional Questions)

以下内容针对题目中关于不确定性、数据传输和设备需求的三个小问题，可用于论文的 **"Uncertainty Analysis"** 和 **"Data Strategy"** 章节。

### 5.1 与预测相关的不确定性因素是什么？
不确定性来源主要分为三类：

1.  **环境不确定性 (Environmental Uncertainty)** —— *影响最大*
    *   **风场 (Wind Field)**：山区的风速和风向瞬息万变（阵风、垂直气流）。模型中假设的“平均风速”无法捕捉瞬间强风，会极大改变坠落轨迹。
    *   **地形数据 (Terrain Data)**：地图高程数据（DEM）可能有误差，且未计入植被（树木）高度，可能导致无人机比预期更早“挂”在树上。

2.  **初始状态不确定性 (Initial State Uncertainty)**
    *   **故障时刻 (Time of Failure)**：控制中心通常只知道“最后一次收到信号”的时间，但无人机可能在此后又飞行了数秒。
    *   **姿态 (Attitude)**：无人机是平稳滑翔（阻力小，飞得远）还是翻滚下坠（阻力大，掉得近）？这直接决定了阻力系数 $C_d$。

3.  **模型误差 (Model Error)**
    *   将无人机简化为质点，忽略了空气动力学外形（升力系数 $C_l$）。若无人机有残存升力，滑翔距离会更远。

### 5.2 无人机可发送哪些信息以减少不确定性？
为了缩小预测范围（即减小蒙特卡洛模拟的方差），建议发送以下“黑匣子”数据：

1.  **高频位置与速度 (High-Frequency GPS & Velocity)**
    *   *内容*：经纬度、高度、三维地速向量。
    *   *作用*：消除“初始位置”和“初始速度”的不确定性。

2.  **姿态角 (Attitude/Orientation)**
    *   *内容*：俯仰角 (Pitch)、滚转角 (Roll)、偏航角 (Yaw)。
    *   *作用*：判断无人机是否翻滚。若姿态稳定，可假设其在滑翔，预测范围将大大缩小。

3.  **本机风速估算 (Local Wind Estimation)**
    *   *内容*：空速 (Airspeed) 与地速 (Ground Speed) 的差值。
    *   *作用*：直接告知模型“坠落点的风速”，消除最大的环境不确定性。

4.  **电池与电机状态 (System Health)**
    *   *内容*：电压、电机转速 (RPM)。
    *   *作用*：判断是“动力完全丧失”（自由落体）还是“失去控制但有动力”（随机游走），从而选择正确的预测模型。

### 5.3 无人机需要什么样的设备才能做到这一点？
为获取上述信息，无人机需搭载（或利用现有）传感器：

1.  **IMU (惯性测量单元)**：
    *   *用途*：测量加速度和角速度，计算姿态角。这是所有无人机的标配。

2.  **GNSS 模块 (全球导航卫星系统)**：
    *   *用途*：获取精确经纬度和高度。建议支持 RTK (实时动态差分) 以获厘米级精度。

3.  **空速管 (Pitot Tube) 或 超声波风速计**：
    *   *用途*：**关键额外设备**。普通无人机仅有 GPS 测地速，无法直接测风速。加装空速管可测相对空气速度，反推风速。

4.  **独立供电的“黑匣子”发射机**：
    *   *用途*：当主电池故障或断电时，利用小型备用电池和低功耗长距离通讯模块（如 LoRa 或卫星信标），在坠毁前发送关键数据包。

---

## 6. 总结

这份代码是你论文 **Model Development** 章节的核心实现。
*   **输入**：无人机参数、最后已知状态、环境预估。
*   **核心机制**：牛顿第二定律 + 蒙特卡洛随机采样。
*   **输出**：搜索区域的热力图。

在论文中，你可以展示不同参数设置下的多张热力图（例如“晴天 vs 暴风雨”），以此来回答题目中关于“不确定性因素”的讨论。
