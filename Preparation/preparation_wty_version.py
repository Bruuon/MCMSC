import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置绘图风格和字体（防止中文乱码，如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==============================================================================
# 工具函数 1: 通用表格绘图 (已增强鲁棒性)
# ==============================================================================
def save_df_to_img(df, title, filename, fontsize=10, col_format=None, notes=None):
    fig, ax = plt.subplots(figsize=(12, min(0.6 * len(df) + 2, 10))) # 动态调整高度
    ax.axis('tight')
    ax.axis('off')

    # --- 数据格式化处理 ---
    data = df.copy()
    for col in data.columns:
        # 判断列是否为数值类型
        is_numeric = pd.api.types.is_numeric_dtype(data[col])
        
        if is_numeric:
            if col_format:
                # 如果指定了通用格式，且该列是数值，则应用
                data[col] = data[col].map(col_format)
            elif data[col].dtype in [np.float64, np.float32]:
                # 默认浮点数格式
                data[col] = data[col].map('{:.4f}'.format)
        
        # 处理整数列（如Rank），转为字符串以去除小数点
        if data[col].dtype == np.int64 or data[col].dtype == np.int32:
             data[col] = data[col].astype(str)

    # 重置索引以便放入表格
    if data.index.name is None:
        data.index.name = 'Index'
    data = data.reset_index()
    
    cell_text = data.values
    col_labels = list(data.columns) # 此时包含原来的 index

    table = ax.table(cellText=cell_text,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 0.85]) 

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.5)

    # 美化表头
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # 深蓝色表头
        elif i > 0 and j == 0:
            cell.set_text_props(weight='bold') # 第一列（索引）加粗
            cell.set_facecolor('#f0f0f0')

    plt.title(title, fontsize=14, pad=15, weight='bold')

    if notes:
        plt.figtext(0.5, 0.02, notes, wrap=True, horizontalalignment='center', fontsize=10, color='darkred')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[图表保存] {filename}")

# ==============================================================================
# 工具函数 2: 最终排名可视化 (新增)
# ==============================================================================
def save_rank_chart(results_df, filename):
    plt.figure(figsize=(10, 6))
    
    # 按照分数排序
    plot_data = results_df.sort_values('Score', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
    bars = plt.barh(plot_data['Equipment'], plot_data['Score'], color=colors)
    
    plt.xlabel('TOPSIS Score ($C_i$)', fontsize=12)
    plt.title('Final Equipment Evaluation Ranking', fontsize=14, weight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 在柱状图旁标注数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[图表保存] {filename}")

# ==============================================================================
# 第一部分：数据初始化
# ==============================================================================
print("\n--- 1. 初始化数据 ---")
equipment_list = [
    'Fixed-Wing', 'Thermal UAV', 'Relay Drone', 'Rotary-Wing', 'LiDAR UAV', 
    'RF Tracker', 'Acoustic', 'UGV', 'Zoom Cam', 'Power Stn'
]
# 准则：可用性, 运行成本, 搜索效果, 操作难度, 环境适应性, 采购成本
criteria = ['Availability', 'Op Cost', 'Search Eff', 'Difficulty', 'Adaptability', 'Purch Cost']

# 原始数据矩阵 (Benefit型越高越好, Cost型越低越好)
raw_data = np.array([
    [6,  12000, 9, 8, 7, 80000], 
    [8,  5000, 10, 5, 6, 30000], 
    [7,  3000, 5, 4, 6, 10000], 
    [9,  2000, 7, 3, 8, 15000], 
    [4,  25000, 9, 9, 5, 120000],
    [6,  1500, 6, 4, 6, 5000],   
    [7,  800, 4, 3, 5, 2000],   
    [5,  6000, 6, 6, 9, 40000],  
    [9,  1000, 6, 2, 7, 8000],   
    [10, 200, 1, 1, 8, 1000],   
])

df_raw = pd.DataFrame(raw_data, index=equipment_list, columns=criteria)
print("原始数据预览：")
print(df_raw.head())

# ==============================================================================
# 第二部分：AHP主观权重
# ==============================================================================
print("\n--- 2. AHP 主观权重计算 ---")
ahp_matrix = np.array([
    [1,   2,   1/3, 3,   1/2, 4],    
    [1/2, 1,   1/4, 2,   1/3, 2],    
    [3,   4,   1,   5,   2,   7],    
    [1/3, 1/2, 1/5, 1,   1/4, 3],    
    [2,   3,   1/2, 4,   1,   5],    
    [1/4, 1/2, 1/7, 1/3, 1/5, 1]     
])

eig_vals, eig_vecs = np.linalg.eig(ahp_matrix)
max_eig_val = np.max(eig_vals).real
max_eig_vec = eig_vecs[:, np.argmax(eig_vals)].real
w_ahp = max_eig_vec / np.sum(max_eig_vec) # 归一化特征向量

# 一致性检验
n = 6 
RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}
RI = RI_dict.get(n, 1.41)
CI = (max_eig_val - n) / (n - 1)
CR = CI / RI

print(f"AHP 权重: {np.round(w_ahp, 4)}")
print(f"一致性比率 CR = {CR:.4f} " + ("(通过)" if CR < 0.1 else "(不通过)"))

# ==============================================================================
# 第三部分：熵权法客观权重
# ==============================================================================
print("\n--- 3. 熵权法客观权重计算 ---")
# 正向指标 (Availability, Search Effect, Adaptability) -> 索引 0, 2, 4
# 负向指标 (Costs, Difficulty) -> 索引 1, 3, 5
positive_indices = [0, 2, 4] 
negative_indices = [1, 3, 5] 
norm_matrix = np.zeros_like(raw_data, dtype=float)

# Min-Max 归一化用于熵权计算
for i in range(n):
    col = raw_data[:, i]
    if i in positive_indices:
        norm_matrix[:, i] = (col - col.min()) / (col.max() - col.min())
    else:
        norm_matrix[:, i] = (col.max() - col) / (col.max() - col.min())

# 平移避免 log(0)
norm_matrix_shifted = norm_matrix + 1e-6
P = norm_matrix_shifted / norm_matrix_shifted.sum(axis=0)
k = 1 / np.log(len(equipment_list))
E = -k * np.sum(P * np.log(P), axis=0)
d = 1 - E
w_entropy = d / d.sum()

print(f"熵权权重: {np.round(w_entropy, 4)}")

# ==============================================================================
# 第四部分：组合权重
# ==============================================================================
print("\n--- 4. 组合权重计算 ---")
alpha, beta = 0.5, 0.5 # 偏好系数
w_combined = alpha * w_ahp + beta * w_entropy
w_combined = w_combined / w_combined.sum() # 再次归一化确保和为1

weights_df = pd.DataFrame([w_ahp, w_entropy, w_combined],
                          index=['AHP (Subjective)', 'Entropy (Objective)', 'Combined Weight'],
                          columns=criteria)

# 保存权重对比图片
save_df_to_img(weights_df, 
               "Weight Analysis: AHP vs Entropy vs Combined", 
               "01_Weights_Analysis.png",
               notes=f"Weighting Strategy: {alpha} * AHP + {beta} * Entropy | AHP CR={CR:.4f}")

# ==============================================================================
# 第五部分：TOPSIS 综合评价
# ==============================================================================
print("\n--- 5. TOPSIS 核心计算 ---")

# 1. 向量归一化 (Vector Normalization, 标准TOPSIS做法)
denom = np.sqrt(np.sum(raw_data**2, axis=0))
norm_topsis = raw_data / denom

# 2. 构建加权归一化矩阵 Z
Z = norm_topsis * w_combined

# 3. 确定正负理想解 Z+ 和 Z-
# 注意：此时已经是同向化（如果是的话）或者需要根据原始指标性质判断
# 在向量归一化中，通常需要保留原始的正负向属性来取最大/最小值
Z_plus = np.zeros(n)
Z_minus = np.zeros(n)

for i in range(n):
    if i in positive_indices: 
        # 效益型：越大越好
        Z_plus[i] = Z[:, i].max()
        Z_minus[i] = Z[:, i].min()
    else: 
        # 成本型：越小越好 (注意：因为Z是基于原始数据的向量归一化，数值越大代表原值越大)
        # 成本型指标，在加权矩阵Z中，理想解应该是最小值，负理想解是最大值
        Z_plus[i] = Z[:, i].min() 
        Z_minus[i] = Z[:, i].max()

ideal_df = pd.DataFrame([Z_plus, Z_minus], 
                        index=['Positive Ideal ($Z^+$)', 'Negative Ideal ($Z^-$)'], 
                        columns=criteria)

# 保存理想解图片
save_df_to_img(ideal_df, 
               "TOPSIS Ideal Solutions ($Z^+$ & $Z^-$)", 
               "02_TOPSIS_Ideal_Solutions.png",
               fontsize=11,
               col_format='{:.6f}'.format)

# 4. 计算欧氏距离 D+ 和 D-
D_plus = np.sqrt(np.sum((Z - Z_plus)**2, axis=1))
D_minus = np.sqrt(np.sum((Z - Z_minus)**2, axis=1))

# 5. 计算相对贴近度 C_i (TOPSIS Score)
# 处理除零保护 (极小概率 D_plus + D_minus = 0)
epsilon = 1e-8
C = D_minus / (D_plus + D_minus + epsilon)

# 6. 生成最终排名
results = pd.DataFrame({
    "Equipment": equipment_list,
    "Dist to Best ($D^+$)": D_plus,
    "Dist to Worst ($D^-$)": D_minus,
    "Score": C
})
results["Rank"] = results["Score"].rank(ascending=False).astype(int)
results_sorted = results.sort_values("Rank")

print("\n最终排名前五：")
print(results_sorted.head())

# 保存最终排名表格 (注意：Score列和Dist列是浮点数，Rank是整数)
save_df_to_img(results_sorted, 
               "Final Equipment Evaluation Ranking", 
               "03_Final_Table.png",
               fontsize=11,
               col_format='{:.4f}'.format) # 会自动跳过非数值列

# 保存可视化统计图
save_rank_chart(results_sorted, "04_Final_Ranking_Chart.png")

print("\n[完成] 所有分析已结束，共生成 4 张分析图表。")