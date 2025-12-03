基于改进 TOPSIS-博弈论组合赋权的无人机搜救设备选型分析
1. 项目背景 (Introduction)
在复杂的搜救（SAR）任务中，选择合适的无人机设备是决定救援效率与成功率的关键。决策过程面临着多重矛盾：我们需要设备具备极高的精确度和环境适应性，同时受限于预算，必须考虑购买与维护成本。
本文提出了一种混合多准则决策（MCDM）模型。该模型结合了层次分析法（AHP）的主观经验与熵权法（Entropy）的客观数据，通过博弈论（Game Theory）进行权重融合，并引入特定成本约束（Cost Constraint），最终利用 TOPSIS 方法对六种备选方案进行科学排序。
2. 评价对象与指标体系 (Candidates & Criteria)
2.1 备选方案 (Candidates)
根据市场调研与技术参数，选取以下六种典型方案：
S1: 多旋翼无人机 (如 Mavic 3 Pro): 成本低，可用性高，但续航和范围有限。
S2: 固定翼无人机: 航程远，速度快，但部署困难，且价格较高。
S3: 垂直起降固定翼 (VTOL): 结合了多旋翼的灵活性与固定翼的长航时，性能最强，但价格昂贵。
S4: 单旋翼无人机: 载重大，但维护成本极高，操作复杂。
S5: 多传感器结合系统: 精度极高，但依赖平台，自身机动性差。
S6: GPS定位系统: 纯辅助工具，成本几乎为零，但功能单一，精度有限。
2.2 评价指标体系 (Criteria System)
基于成本-效益分析（CBA）原则，构建包含 9 项指标的评价体系：
维度	编号	指标名称	类型	选取依据
成本 (Cost)	C1	维护成本	成本型	长期运营支出。
C2	购买成本	成本型	初始资金门槛（仅采用客观权重）。
效果 (Effectiveness)	C3	操作难度	成本型	高压环境下的认知负荷影响。
C4	部署难度	成本型	影响“黄金救援时间”的响应速度。
C5	现实可用性	效益型	设备在野外环境的实战易用程度。
C6	环境适应性	效益型	抗风雨及复杂地形能力。
C7	完成时间	效益型	任务效率（正向化处理后）。
C8	搜索精确度	效益型	识别目标与背景噪音的能力。
C9	搜索范围	效益型	单次任务覆盖面积。
3. 评价模型与算法 (Methodology)
3.1 数据标准化
为消除量纲影响，采用极差法进行标准化处理。
效益型指标 (越大越好): 
y
i
j
=
x
i
j
−
min
⁡
(
x
j
)
max
⁡
(
x
j
)
−
min
⁡
(
x
j
)
y 
ij
​
 = 
max(x 
j
​
 )−min(x 
j
​
 )
x 
ij
​
 −min(x 
j
​
 )
​
 
成本型指标 (越小越好): 
y
i
j
=
max
⁡
(
x
j
)
−
x
i
j
max
⁡
(
x
j
)
−
min
⁡
(
x
j
)
y 
ij
​
 = 
max(x 
j
​
 )−min(x 
j
​
 )
max(x 
j
​
 )−x 
ij
​
 
​
 
3.2 组合赋权策略
本模型采用“主客结合”的赋权方式，并引入硬性约束。
熵权法 (Objective Weight):
利用数据的离散程度计算权重。由于 C2 购买成本 数据差异巨大（0元至25万元），熵权法会自动赋予其较高的权重，反映客观现实。
层次分析法 (Subjective Weight):
基于专家经验构建判断矩阵。
特殊处理： 为保证价格因素的客观性，强制设定 C2 购买成本的主观权重为 0。
博弈论组合与约束修正 (Game Theory with Constraints):
通过纳什均衡寻找最优线性组合系数 
α
1
α 
1
​
 
 和 
α
2
α 
2
​
 
。
约束条件： 强制要求 成本类指标 (C1+C2) 的总权重落在 [20%, 25%] 区间内。
算法逻辑： 若初步计算的成本权重偏离该区间，程序将其强制调整至 22.5%（区间中值），并按比例缩放其他指标权重，确保总和为 1。
3.3 TOPSIS 综合评价
构建加权矩阵： 
Z
=
Y
×
W
f
i
n
a
l
Z=Y×W 
final
​
 
确定理想解：
正理想解 
Z
+
Z 
+
 
：各列最大值向量。
负理想解 
Z
−
Z 
−
 
：各列最小值向量。
计算欧氏距离：
D
i
+
=
∑
(
z
i
j
−
z
j
+
)
2
,
D
i
−
=
∑
(
z
i
j
−
z
j
−
)
2
D 
i
+
​
 = 
∑(z 
ij
​
 −z 
j
+
​
 ) 
2
 
​
 ,D 
i
−
​
 = 
∑(z 
ij
​
 −z 
j
−
​
 ) 
2
 
​
 
计算贴近度 (得分):
C
i
=
D
i
−
D
i
+
+
D
i
−
C 
i
​
 = 
D 
i
+
​
 +D 
i
−
​
 
D 
i
−
​
 
​
 
4. 结果分析 (Results)
4.1 权重分布分析
经过约束修正后，最终权重呈现如下特征：
成本效率 (22.5%)： 这一比例既保证了选型不脱离经济实际，又避免了因过分关注价格而选择了低性能设备（如单纯选择免费的GPS）。
搜救效果 (77.5%)： 搜索范围 (C9)、精确度 (C8) 和 环境适应性 (C6) 占据了主导地位。
4.2 排名讨论
第一名：S3 垂直起降固定翼 (VTOL) - 得分最高
分析： 尽管 S3 价格昂贵（购买与维护成本高），但在 20%-25% 的成本权重约束下，其在“效果”端的统治力（范围广、精度高、适应性强）完全弥补了费用的劣势。它是追求任务成功率的最佳选择。
第二名：S1 多旋翼 (Mavic 3) - 性价比之选
分析： S1 在“现实可用性”上得分极高，且成本极低。虽然搜索范围不如固定翼，但其综合得分非常稳健，适合预算有限的常规搜救。
排名下滑：S6 (GPS系统)
分析： 在未加约束的模型中，S6 常因“零成本”而排名靠前。但在本模型中，由于效果权重占比高达 77.5%，S6 在精度和功能上的巨大短板导致其排名大幅下滑。
5. Python 代码实现
以下是实现上述逻辑的完整 Python 代码。代码已包含数据录入、权重计算（含约束算法）及图表生成。
code
Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 环境设置 ---
def set_style():
    sns.set_theme(style="whitegrid")
    # 设置英文字体以确保兼容性
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
set_style()

# --- 1. 数据初始化 ---
schemes = [
    'S1: Multi-rotor', 
    'S2: Fixed-wing', 
    'S3: VTOL Fixed-wing', 
    'S4: Single-rotor', 
    'S5: Multi-sensor', 
    'S6: GPS System'
]

# 评价指标定义
criteria = [
    'C1 Maint. Cost',       # 0: 成本型
    'C2 Purch. Cost',       # 0: 成本型 (仅客观权重)
    'C3 Op. Difficulty',    # 0: 成本型
    'C4 Dep. Difficulty',   # 0: 成本型
    'C5 Usability',         # 1: 效益型
    'C6 Adaptability',      # 1: 效益型
    'C7 Comp. Time Score',  # 1: 效益型 (值越高越快)
    'C8 Precision',         # 1: 效益型
    'C9 Search Range'       # 1: 效益型
]

# 指标类型: 0=成本, 1=效益
types = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

# 原始数据矩阵 (基于严格的层次对比关系构建)
raw_data = np.array([
    # C1   C2      C3  C4  C5  C6  C7  C8  C9
    [15,   10000,  4,  2,  10, 8,  6,  2,  5],   # S1
    [50,   80000,  8,  10, 1,  1,  5,  6,  10],  # S2
    [70,   150000, 6,  5,  2,  10, 10, 10, 9],   # S3
    [100,  250000, 10, 8,  3,  7,  8,  8,  7],   # S4
    [30,   40000,  2,  2,  6,  6,  2,  4,  1],   # S5
    [1,    0,      1,  2,  9,  5,  1,  1,  3]    # S6
])

# --- 2. 数据标准化 ---
def normalize(data, types):
    norm = np.zeros_like(data, dtype=float)
    rows, cols = data.shape
    for j in range(cols):
        c_min = np.min(data[:, j])
        c_max = np.max(data[:, j])
        if c_max == c_min:
            norm[:, j] = 1.0
        elif types[j] == 1: # 效益型
            norm[:, j] = (data[:, j] - c_min) / (c_max - c_min)
        else: # 成本型
            norm[:, j] = (c_max - data[:, j]) / (c_max - c_min)
    return norm

Y = normalize(raw_data, types)
Y_entropy = np.where(Y <= 0, 1e-5, Y) # 避免 log(0)

# --- 3. 权重计算模块 ---

# A. 熵权法 (客观)
def get_entropy_weights(matrix):
    sum_cols = np.sum(matrix, axis=0)
    P = matrix / sum_cols
    k = 1 / np.log(len(matrix))
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - E
    w = d / np.sum(d)
    return w

W_entropy = get_entropy_weights(Y_entropy)

# B. 层次分析法 AHP (主观)
ahp_matrix = np.array([
    [1, 1, 1/3, 1/3, 1/4, 1/5, 1/6, 1/7, 1/7], # C1
    [1, 1, 1/3, 1/3, 1/4, 1/5, 1/6, 1/7, 1/7], # C2
    [3, 3, 1, 1, 1/2, 1/3, 1/4, 1/5, 1/5],     # C3
    [3, 3, 1, 1, 1/2, 1/3, 1/4, 1/5, 1/5],     # C4
    [4, 4, 2, 2, 1, 1/2, 1/3, 1/4, 1/4],       # C5
    [5, 5, 3, 3, 2, 1, 1/2, 1/3, 1/3],         # C6
    [6, 6, 4, 4, 3, 2, 1, 1/2, 1/2],           # C7
    [7, 7, 5, 5, 4, 3, 2, 1, 1],               # C8
    [7, 7, 5, 5, 4, 3, 2, 1, 1]                # C9
])

def get_ahp_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eigvals)
    w = np.real(eigvecs[:, max_idx])
    w = w / np.sum(w)
    return w

W_ahp_raw = get_ahp_weights(ahp_matrix)
W_ahp = W_ahp_raw.copy()
W_ahp[1] = 0 # 强制购买成本的主观权重为0
W_ahp = W_ahp / np.sum(W_ahp)

# C. 博弈论组合赋权 (带约束: 成本权重 20%-25%)
def game_theory_constrained(w1, w2):
    # 求解纳什均衡系数
    a11 = np.dot(w1, w1); a12 = np.dot(w1, w2)
    a21 = np.dot(w2, w1); a22 = np.dot(w2, w2)
    A = np.array([[a11, a12], [a21, a22]])
    B = np.array([np.dot(w1, w1), np.dot(w2, w2)])
    try:
        alpha = np.linalg.solve(A, B)
        alpha_star = alpha / np.sum(alpha)
    except:
        alpha_star = [0.5, 0.5]
    
    w_combined = alpha_star[0] * w1 + alpha_star[1] * w2
    
    # --- 约束执行模块 ---
    cost_indices = [0, 1]
    other_indices = [2, 3, 4, 5, 6, 7, 8]
    current_cost_sum = np.sum(w_combined[cost_indices])
    
    # 目标修正值
    target_val = current_cost_sum
    if current_cost_sum < 0.20 or current_cost_sum > 0.25:
        target_val = 0.225 # 强制调整至区间中值
        
    if target_val != current_cost_sum:
        # 成本权重缩放
        w_combined[cost_indices] *= (target_val / current_cost_sum)
        # 效果权重缩放 (确保总和为1)
        w_combined[other_indices] *= ((1.0 - target_val) / np.sum(w_combined[other_indices]))
        
    return w_combined, alpha_star, target_val

W_final, alphas, final_cost_sum = game_theory_constrained(W_entropy, W_ahp)

# --- 4. TOPSIS 计算 ---
Z = Y * W_final
Z_plus = np.max(Z, axis=0)
Z_minus = np.min(Z, axis=0)
D_plus = np.sqrt(np.sum((Z - Z_plus)**2, axis=1))
D_minus = np.sqrt(np.sum((Z - Z_minus)**2, axis=1))
C_score = D_minus / (D_plus + D_minus)
rank_indices = np.argsort(C_score)[::-1]

# --- 5. 结果可视化 ---
# 计算分类得分用于绘图
idx_cost = [0, 1]
idx_effect = [2, 3, 4, 5, 6, 7, 8]
score_cost = np.sum(Y[:, idx_cost] * W_final[idx_cost], axis=1) * 100
score_effect = np.sum(Y[:, idx_effect] * W_final[idx_effect], axis=1) * 100

fig = plt.figure(figsize=(16, 12))

# 子图1: TOPSIS 距离
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(schemes))
width = 0.35
ax1.bar(x - width/2, D_plus, width, label='Dist to Ideal (D+)', color='#e74c3c', alpha=0.7)
ax1.bar(x + width/2, D_minus, width, label='Dist to Anti-Ideal (D-)', color='#2ecc71', alpha=0.7)
ax1.set_xticks(x); ax1.set_xticklabels([s.split(':')[0] for s in schemes])
ax1.set_title("TOPSIS Distance Analysis (Higher D- is Better)")
ax1.legend()

# 子图2: 权重雷达图
ax2 = fig.add_subplot(2, 2, 2, polar=True)
angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
w_f_plot = np.concatenate((W_final, [W_final[0]]))
ax2.plot(angles, w_f_plot, 'o-', linewidth=2, label='Final Weight', color='purple')
ax2.fill(angles, w_f_plot, alpha=0.25, color='purple')
ax2.set_thetagrids(angles[:-1] * 180/np.pi, [c.split(' ')[0] for c in criteria])
ax2.set_title(f"Weight Distribution\n(Cost Total: {final_cost_sum*100:.1f}%)", y=1.08)

# 子图3: 成本与效果堆叠图
ax3 = fig.add_subplot(2, 1, 2)
sorted_indices = rank_indices
sorted_schemes = [schemes[i].split(':')[0] for i in sorted_indices]
sorted_cost = score_cost[sorted_indices]
sorted_effect = score_effect[sorted_indices]

b1 = ax3.bar(sorted_schemes, sorted_effect, label='Effectiveness Score', color='#3498db', alpha=0.9)
b2 = ax3.bar(sorted_schemes, sorted_cost, bottom=sorted_effect, label='Cost Efficiency Score', color='#f1c40f', alpha=0.9)

for r1, r2 in zip(b1, b2):
    h1 = r1.get_height(); h2 = r2.get_height()
    ax3.text(r1.get_x() + r1.get_width()/2, h1/2, f'{h1:.1f}', ha='center', va='center', color='white', fontweight='bold')
    if h2 > 1: ax3.text(r2.get_x() + r2.get_width()/2, h1 + h2/2, f'{h2:.1f}', ha='center', va='center', fontweight='bold')
    ax3.text(r2.get_x() + r2.get_width()/2, h1+h2+0.5, f'{h1+h2:.1f}', ha='center', va='bottom', fontweight='bold')

ax3.set_title("Final Evaluation: Effectiveness vs. Cost Efficiency")
ax3.legend()

plt.tight_layout()
plt.savefig('drone_selection_final.png')
print("分析完成，图表已生成。")
plt.show()
6. 结论 (Conclusion)
本研究构建了一套适用于复杂搜救环境的无人机选型评价体系。通过引入成本权重约束机制（20%-25%），成功平衡了“高性能”与“低成本”之间的矛盾。实证分析表明，S3 垂直起降固定翼无人机凭借其卓越的综合性能成为首选方案，而 S1 多旋翼无人机 则展现了极佳的性价比。该模型为应急管理部门的设备采购提供了科学的量化依据。
