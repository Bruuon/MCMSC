# 算法流程参考库 (Algorithm Reference Library)

本文档用于汇总本项目所使用的核心算法流程、数学公式及逻辑框架。旨在作为 AI 助手的长期记忆库，减少重复检索上下文的开销。

## 1. 弹道下降模型 (Ballistic Descent Model)
*   **来源**: *Ground impact probability distribution for small unmanned aircraft in ballistic descent* (La Cour-Harbo, 2020)
*   **类型**: 机理模型 / 闭式解 (Physics-based / Closed-form Solution)
*   **核心优势**: 计算速度极快，适合蒙特卡洛模拟；考虑了二阶阻力。

### 1.1 符号定义 (Nomenclature)

| 符号 | 定义 | 说明 |
| :--- | :--- | :--- |
| $m$ | Aircraft Mass | 无人机质量 (kg) |
| $c$ | Drag Constant | 阻力常数, $c = \frac{1}{2}\rho A C_D$ |
| $g$ | Gravity | 重力加速度 (通常取 9.8 m/s²) |
| $v_{x,i}, v_{y,i}$ | Initial Velocities | 初始水平/垂直速度 (向上为负, 向下为正) |
| $\Gamma$ | Terminal Velocity | 终端速度, $\Gamma = \sqrt{mg/c}$ |
| $\gamma$ | Inverse Terminal Vel. | $\gamma = 1/\Gamma = \sqrt{c/mg}$ |
| $H_u, H_d$ | Phase Constants | 垂直运动相位常数 (Up/Down) |
| $G_u, G_d$ | Log Constants | 对数辅助常数 |
| $t_c$ | Crossover Time | 水平/垂直阻力主导权的切换时刻 ($v_x = v_y$) |

### 1.2 核心物理模型 (Core Physics)

**半解耦二阶阻力模型 (Semi-decoupled Quadratic Drag Model)**:
为了获得闭式解，作者假设垂直运动不受水平速度影响，而水平运动受垂直速度影响。

1.  **水平运动 (Horizontal)**:
    $$ m\dot{v}_x = -c \max(v_x, v_y) v_x \quad \dots(2) $$
    *   当 $v_x > v_y$ 时，阻力主要由 $v_x$ 决定。
    *   当 $v_x \le v_y$ 时，阻力主要由 $v_y$ 决定。

2.  **垂直运动 (Vertical)**:
    $$ m\dot{v}_y = mg - c|v_y|v_y \quad \dots(3) $$
    *   完全解耦，标准自由落体阻力方程。

### 1.3 统一计算流程 (Unified Algorithm Flow)

基于论文 **Section C. Actual calculations**，引入归一化变量 (Hat variables) 以统一处理初始速度向上 ($v_{y,i} < 0$) 或向下 ($v_{y,i} \ge 0$) 的情况。

#### Step 1: 初始化辅助变量 (Initialization)
$$ \hat{v}_{y,i} = \max(0, v_{y,i}) $$
$$ \hat{H}_d = \text{arctanh}(\hat{v}_{y,i}\gamma), \quad \hat{G}_d = \ln \cosh \hat{H}_d $$
$$ \hat{H}_u = \arctan(\hat{v}_{y,i}\gamma), \quad \hat{G}_u = \ln \cos \hat{H}_u $$

#### Step 2: 计算上升阶段 (Ascent Phase)
如果 $v_{y,i} < 0$ (向上)，则存在上升段；否则为 0。
$$ \hat{t}_{top} = -\frac{1}{g\gamma} \arctan(\gamma \min(0, v_{y,i})) \quad \dots(21) $$
$$ x_1 = \frac{m}{c} \ln(1 + v_{x,i} c \hat{t}_{top} / m) \quad (\text{Horizontal dist to top}) $$
$$ y_{top} = -\frac{m}{2c} \ln(1 + (\gamma \min(0, v_{y,i}))^2) \quad (\text{Altitude gained}) $$
*更新初始状态*:
$$ v_{x,top} = \frac{m v_{x,i}}{m + v_{x,i} c \hat{t}_{top}} $$
$$ y_{total} = y_{initial} - y_{top} \quad (\text{Total drop height}) $$

#### Step 3: 计算下落时间 (Descent Time)
从最高点 (或初始点) 下落到地面的时间。
$$ \hat{t}_{drop} = \frac{1}{g\gamma} \left[ \text{arccosh}\left( \exp\left(\frac{c y_{total}}{m} + \hat{G}_d\right) \right) - \hat{H}_d \right] \quad \dots(13/22) $$
$$ \hat{t}_{im} = \hat{t}_{top} + \hat{t}_{drop} \quad (\text{Total impact time}) $$

#### Step 4: 计算切换时刻 $t_c$ (Crossover Time)
判断何时 $v_x(t)$ 降至与 $v_y(t)$ 相等。使用连分式近似公式 (14)：
$$ \hat{t}_c = \hat{t}_{top} + \frac{m(g\hat{t}_{drop} - \Gamma \hat{H}_d + v_{x,top}(1 + (\hat{H}_d - g\gamma \hat{t}_{drop})^2))}{mg + c v_{x,top}(g\hat{t}_{drop} - \Gamma \hat{H}_d)} $$
*(注：原论文公式(14)中的 $t_{top}$ 在此处对应 $\hat{t}_{drop}$ 的时间轴起点，需根据具体实现微调，通常直接计算相对于下落开始的时间)*

#### Step 5: 计算水平总距离 (Total Horizontal Distance)
根据总时间 $\hat{t}_{im}$ 与切换时间 $\hat{t}_c$ 的关系分段计算。

**Case A: $\hat{t}_{im} \le \hat{t}_c$ (全程水平速度主导)**
$$ x_{final} = x_1 + \frac{m}{c} \ln\left(1 + \frac{c v_{x,top}}{m} (\hat{t}_{im} - \hat{t}_{top})\right) $$

**Case B: $\hat{t}_{im} > \hat{t}_c$ (发生切换)**
1.  **阶段 2 距离 ($x_2$)**: 从 $t_{top}$ 到 $t_c$
    $$ x_2 = \frac{m}{c} \ln\left(1 + \frac{c v_{x,top}}{m} (\hat{t}_c - \hat{t}_{top})\right) $$
2.  **中间速度**:
    $$ v_{x,c} = v_x(\hat{t}_c), \quad v_{y,c} = v_y(\hat{t}_c - \hat{t}_{top}) $$
    $$ H_c = \text{arctanh}(v_{y,c}\gamma), \quad G_c = \ln \cosh H_c $$
3.  **阶段 3 距离 ($x_3$)**: 从 $t_c$ 到 $t_{im}$
    $$ x_3 = \frac{v_{x,c} e^{G_c}}{g\gamma} \left[ \arctan(\sinh(g\gamma(\hat{t}_{im} - \hat{t}_c) + H_c)) - \arcsin(v_{y,c}\gamma) \right] \quad \dots(18/23) $$
4.  **总距离**:
    $$ x_{final} = x_1 + x_2 + x_3 $$

---

---

### 1.4 理论验证基准 (Theoretical Validation Benchmark)

为了验证蒙特卡洛模拟的准确性，可以使用论文 **Section III** 提供的理论分布作为基准。

#### A. 验证场景设置 (Validation Setup)
*   **假设**: 阻力系数 $C_D$ 和初始速度 $v_{x,i}, v_{y,i}$ 服从正态分布 (Normal Distribution)。
*   **目标**: 验证模拟生成的落点分布是否符合理论上的 **偏态正态分布 (Skewed Normal)** 或 **对数正态分布 (Log Normal)**。

#### B. 理论分布公式 (Theoretical PDFs)
论文指出，落点距离 $x$ 的概率密度函数 (PDF) 近似于以下两种分布：

1.  **偏态正态分布 (Skewed Normal Distribution)**:
    $$ f(x) = \frac{2C}{\omega} \phi\left(\frac{x-\xi}{\omega}\right) \Phi\left(\alpha \frac{x-\xi}{\omega}\right) \quad \dots(27) $$
    *   $\phi$: 标准正态 PDF
    *   $\Phi$: 标准正态 CDF
    *   $\xi$: 位置参数 (Location)
    *   $\omega$: 尺度参数 (Scale)
    *   $\alpha$: 偏度参数 (Shape/Skewness)

2.  **对数正态分布 (Log Normal Distribution)**:
    $$ f(x) = \frac{C}{x\sigma\sqrt{2\pi}} \exp\left(\frac{-(\ln x - \mu)^2}{2\sigma^2}\right) \quad \dots(28) $$
    *   $\mu, \sigma$: 对数尺度的均值和标准差

#### C. 验证参数参考 (Reference Parameters)
可使用论文 **Table I** 中的参数进行单元测试：

| 参数 | Phantom 4 (参考值) | Talon (参考值) |
| :--- | :--- | :--- |
| Mass ($m$) | 1.4 kg | 3.75 kg |
| Front Area ($A$) | 0.02 $m^2$ | 0.1 $m^2$ |
| Drag Coeff ($C_D$) | $N(0.7, 0.2)$ | $N(0.9, 0.2)$ |
| Init $v_x$ | $N(10, 3)$ m/s | $N(18, 3)$ m/s |
| Init $v_y$ | $N(0, 2)$ m/s | $N(0, 4)$ m/s |

---

## 2. (预留) 其他算法
*(待补充)*
