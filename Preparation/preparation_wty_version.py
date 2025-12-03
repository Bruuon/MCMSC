import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Settings ---
def set_style():
    sns.set_theme(style="whitegrid")
    # Standard English Fonts
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Calibri', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 
    
set_style()

# --- 1. Data Initialization (Based on NEWEST Rankings) ---
schemes = [
    'S1: Multi-rotor', 
    'S2: Fixed-wing', 
    'S3: VTOL Fixed-wing', 
    'S4: Single-rotor', 
    'S5: Multi-sensor', 
    'S6: GPS System'
]

# Criteria Definition
# Cost Group: C1, C2
# Effect Group: C3 - C9
criteria = [
    'C1 Maint. Cost',       # 0: Cost (Lower is better)
    'C2 Purch. Cost',       # 0: Cost (Objective only)
    'C3 Op. Difficulty',    # 0: Cost (Lower is better)
    'C4 Dep. Difficulty',   # 0: Cost (Lower is better)
    'C5 Usability',         # 1: Benefit (Higher is better)
    'C6 Adaptability',      # 1: Benefit
    'C7 Comp. Time Score',  # 1: Benefit (Based on ranking: S3 is best)
    'C8 Precision',         # 1: Benefit
    'C9 Search Range'       # 1: Benefit
]

# Types: 0=Cost, 1=Benefit
types = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

# Constructing Data based on strict comparisons:
# ">>" implies large gap, ">" implies small gap
# Cost/Diff: Higher value = Worse. Benefit: Higher value = Better.

raw_data = np.array([
    # C1(Maint)  C2(Purch)   C3(Op)   C4(Dep)      C5(Use)  C6(Adapt) C7(Time) C8(Prec) C9(Range)
    # Costs: S4>>S3>S2>S5>S1>S6
    # Range: S2>S3>S4>S1>S6>S5
    
    [15,         10000,      4,       2,           10,      8,        6,       2,       5],   # S1
    [50,         80000,      8,       10,          1,       1,        5,       6,       10],  # S2
    [70,         150000,     6,       5,           2,       10,       10,      10,      9],   # S3
    [100,        250000,     10,      8,           3,       7,        8,       8,       7],   # S4
    [30,         40000,      2,       2,           6,       6,        2,       4,       1],   # S5
    [1,          0,          1,       2,           9,       5,        1,       1,       3]    # S6
])

# --- 2. Normalization ---
def normalize(data, types):
    norm = np.zeros_like(data, dtype=float)
    rows, cols = data.shape
    for j in range(cols):
        c_min = np.min(data[:, j])
        c_max = np.max(data[:, j])
        if c_max == c_min:
            norm[:, j] = 1.0
        elif types[j] == 1: # Benefit
            norm[:, j] = (data[:, j] - c_min) / (c_max - c_min)
        else: # Cost
            norm[:, j] = (c_max - data[:, j]) / (c_max - c_min)
    return norm

Y = normalize(raw_data, types)
Y_entropy = np.where(Y <= 0, 1e-5, Y) # Avoid log(0)

# --- 3. Weight Calculation ---

# A. Entropy (Objective)
def get_entropy_weights(matrix):
    sum_cols = np.sum(matrix, axis=0)
    P = matrix / sum_cols
    k = 1 / np.log(len(matrix))
    E = -k * np.sum(P * np.log(P), axis=0)
    d = 1 - E
    w = d / np.sum(d)
    return w

W_entropy = get_entropy_weights(Y_entropy)

# B. AHP (Subjective)
# Prioritizing Performance (Range, Precision, Time) over Cost/Difficulty
ahp_matrix = np.array([
    [1,   1,   1/3, 1/3, 1/4, 1/5, 1/6, 1/7, 1/7], # C1
    [1,   1,   1/3, 1/3, 1/4, 1/5, 1/6, 1/7, 1/7], # C2 (Placeholder)
    [3,   3,   1,   1,   1/2, 1/3, 1/4, 1/5, 1/5], # C3
    [3,   3,   1,   1,   1/2, 1/3, 1/4, 1/5, 1/5], # C4
    [4,   4,   2,   2,   1,   1/2, 1/3, 1/4, 1/4], # C5
    [5,   5,   3,   3,   2,   1,   1/2, 1/3, 1/3], # C6
    [6,   6,   4,   4,   3,   2,   1,   1/2, 1/2], # C7
    [7,   7,   5,   5,   4,   3,   2,   1,   1],   # C8
    [7,   7,   5,   5,   4,   3,   2,   1,   1]    # C9
])

def get_ahp_weights(matrix):
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_idx = np.argmax(eigvals)
    w = np.real(eigvecs[:, max_idx])
    w = w / np.sum(w)
    return w

W_ahp_raw = get_ahp_weights(ahp_matrix)
W_ahp = W_ahp_raw.copy()
W_ahp[1] = 0 # Force Purchase Cost AHP to 0
W_ahp = W_ahp / np.sum(W_ahp)

# C. Game Theory with STRICT Constraint (20% - 25%)
def game_theory_constrained(w1, w2):
    # 1. Solve Game Theory
    a11 = np.dot(w1, w1)
    a12 = np.dot(w1, w2)
    a21 = np.dot(w2, w1)
    a22 = np.dot(w2, w2)
    A = np.array([[a11, a12], [a21, a22]])
    B = np.array([np.dot(w1, w1), np.dot(w2, w2)])
    
    try:
        alpha = np.linalg.solve(A, B)
        alpha_star = alpha / np.sum(alpha)
    except:
        alpha_star = [0.5, 0.5]
    
    w_combined = alpha_star[0] * w1 + alpha_star[1] * w2
    
    # 2. ENFORCE CONSTRAINT: Cost Weight Sum [0.20, 0.25]
    cost_indices = [0, 1]
    other_indices = [2, 3, 4, 5, 6, 7, 8]
    
    current_cost_sum = np.sum(w_combined[cost_indices])
    target_min = 0.20
    target_max = 0.25
    
    print(f"Initial Cost Weight Sum: {current_cost_sum:.4f}")
    
    target_val = current_cost_sum
    # Force value to midpoint (0.225) if out of bounds
    if current_cost_sum < target_min:
        target_val = 0.225
    elif current_cost_sum > target_max:
        target_val = 0.225
        
    if target_val != current_cost_sum:
        # Scale Cost Weights
        scale_factor_cost = target_val / current_cost_sum
        w_combined[cost_indices] *= scale_factor_cost
        
        # Scale Other Weights
        remaining_weight = 1.0 - target_val
        current_other_sum = np.sum(w_combined[other_indices])
        scale_factor_other = remaining_weight / current_other_sum
        w_combined[other_indices] *= scale_factor_other
        
    return w_combined, alpha_star, target_val

W_final, alphas, final_cost_sum = game_theory_constrained(W_entropy, W_ahp)

# --- 4. TOPSIS ---
Z = Y * W_final
Z_plus = np.max(Z, axis=0)
Z_minus = np.min(Z, axis=0)
D_plus = np.sqrt(np.sum((Z - Z_plus)**2, axis=1))
D_minus = np.sqrt(np.sum((Z - Z_minus)**2, axis=1))
C_score = D_minus / (D_plus + D_minus)
rank_indices = np.argsort(C_score)[::-1] 

# --- 5. Scores for Visualization ---
idx_cost = [0, 1]              
idx_effect = [2, 3, 4, 5, 6, 7, 8]
# Scale to 100 for display
# Cost score: Higher = Cheaper (Better)
score_cost = np.sum(Y[:, idx_cost] * W_final[idx_cost], axis=1) * 100
# Effect score: Higher = Better Performance
score_effect = np.sum(Y[:, idx_effect] * W_final[idx_effect], axis=1) * 100

# --- 6. Output & Visualization ---
print("="*30)
print(f"Final Cost Weight Sum: {final_cost_sum:.4f} (Target: 20%-25%)")
print(f"  > Maint. Cost: {W_final[0]:.4f}")
print(f"  > Purch. Cost: {W_final[1]:.4f}")
print("="*30)
print("TOPSIS Ranking:")
for i, idx in enumerate(rank_indices):
    print(f"Rank {i+1}: {schemes[idx]} (Score: {C_score[idx]:.4f})")

# Plotting
fig = plt.figure(figsize=(16, 12))

# 1. TOPSIS Distance
ax1 = fig.add_subplot(2, 2, 1)
x_labels = [schemes[i].split(':')[0] for i in range(len(schemes))]
x = np.arange(len(schemes))
width = 0.35
ax1.bar(x - width/2, D_plus, width, label='Dist to Ideal (D+)', color='#e74c3c', alpha=0.8)
ax1.bar(x + width/2, D_minus, width, label='Dist to Anti-Ideal (D-)', color='#2ecc71', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.set_title("TOPSIS Distance Analysis (Higher D- is Better)", fontsize=12)
ax1.legend()

# 2. Weights Radar
ax2 = fig.add_subplot(2, 2, 2, polar=True)
angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
w_e_plot = np.concatenate((W_entropy, [W_entropy[0]]))
w_a_plot = np.concatenate((W_ahp, [W_ahp[0]]))
w_f_plot = np.concatenate((W_final, [W_final[0]]))

ax2.plot(angles, w_e_plot, '--', linewidth=1, label='Entropy (Obj)', color='blue', alpha=0.6)
ax2.plot(angles, w_a_plot, '--', linewidth=1, label='AHP (Subj)', color='orange', alpha=0.6)
ax2.fill(angles, w_f_plot, alpha=0.25, color='purple')
ax2.plot(angles, w_f_plot, 'o-', linewidth=2, label='Final (20%-25% Cost)', color='purple')
radar_labels = [c.split(' ')[0] for c in criteria]
ax2.set_thetagrids(angles[:-1] * 180/np.pi, radar_labels, fontsize=9)
ax2.set_title(f"Weight Distribution\n(Cost Total: {final_cost_sum*100:.1f}%)", y=1.08)
ax2.legend(loc='lower right', bbox_to_anchor=(1.3, 0))

# 3. Cost vs Effect Stacked Bar
ax3 = fig.add_subplot(2, 1, 2)
sorted_indices = rank_indices
sorted_schemes = [schemes[i].split(':')[0] for i in sorted_indices]
sorted_cost_score = score_cost[sorted_indices]
sorted_effect_score = score_effect[sorted_indices]

b1 = ax3.bar(sorted_schemes, sorted_effect_score, label='Effectiveness Score', color='#3498db', alpha=0.9)
b2 = ax3.bar(sorted_schemes, sorted_cost_score, bottom=sorted_effect_score, label='Cost Efficiency Score', color='#f1c40f', alpha=0.9)

for rect1, rect2 in zip(b1, b2):
    h1 = rect1.get_height()
    h2 = rect2.get_height()
    total = h1 + h2
    # Hide label if segment is too small
    if h1 > 1:
        ax3.text(rect1.get_x() + rect1.get_width()/2, h1/2, f'{h1:.1f}', ha='center', va='center', color='white', fontweight='bold')
    if h2 > 1:
        ax3.text(rect2.get_x() + rect2.get_width()/2, h1 + h2/2, f'{h2:.1f}', ha='center', va='center', color='black', fontweight='bold')
    ax3.text(rect2.get_x() + rect2.get_width()/2, total+0.5, f'{total:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_title("Final Evaluation: Effectiveness vs. Cost Efficiency", fontsize=14)
ax3.set_ylabel("Weighted Score")
ax3.legend()

plt.tight_layout()
plt.savefig('drone_selection_20_25.png', dpi=300)
plt.show()
