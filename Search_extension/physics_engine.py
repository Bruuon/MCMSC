# physics_engine.py
# 物理引擎 (The Core)
# 实现 La Cour-Harbo (2020) 的闭式解公式
# 使用 NumPy 进行向量化计算，支持大规模并发模拟

import numpy as np

def calculate_trajectory(m, c, v_xi, v_yi, y_initial, g):
    """
    计算无人机的弹道下降轨迹参数。
    
    参数:
        m (float or array): 质量 (kg)
        c (float or array): 阻力常数
        v_xi (float or array): 初始水平速度 (m/s)
        v_yi (float or array): 初始垂直速度 (m/s, 向上为负)
        y_initial (float): 初始高度 (m)
        g (float): 重力加速度
        
    返回:
        dict: 包含总时间、水平距离等关键结果
    """
    
    # 1. 基础参数计算
    Gamma = np.sqrt(m * g / c)      # 终端速度
    gamma = 1.0 / Gamma             # 终端速度倒数
    
    # 2. 归一化变量 (Hat variables) - 统一处理上升和下落
    # Eq (21) logic: handle v_yi < 0 (upward) and v_yi >= 0 (downward)
    v_yi_hat = np.maximum(0, v_yi)  # 如果向上(<0)，则设为0
    
    # 计算相位常数 H_d, G_d
    # H_d = arctanh(v_yi_hat * gamma)
    # 注意：如果 v_yi_hat >= Gamma，arctanh 会出错。但在物理上 v_yi 不会超过终端速度(除非有推力)，
    # 这里假设 v_yi < Gamma。为防数值误差，clip 一下。
    val_for_arctanh = np.clip(v_yi_hat * gamma, -0.999999, 0.999999)
    H_d_hat = np.arctanh(val_for_arctanh)
    G_d_hat = np.log(np.cosh(H_d_hat))
    
    # 3. 计算上升阶段 (Ascent Phase)
    # Eq (21): t_top
    # 如果 v_yi < 0, min(0, v_yi) = v_yi. 否则为 0.
    min_0_vyi = np.minimum(0, v_yi)
    t_top_hat = - (1.0 / (g * gamma)) * np.arctan(gamma * min_0_vyi)
    
    # 上升高度 y_top
    # Eq in Step 2 of Library
    y_top = - (m / (2 * c)) * np.log(1 + (gamma * min_0_vyi)**2)
    
    # 上升段水平距离 x1
    x1 = (m / c) * np.log(1 + v_xi * c * t_top_hat / m)
    
    # 更新状态到最高点 (或初始点)
    v_x_top = (m * v_xi) / (m + v_xi * c * t_top_hat)
    y_total = y_initial - y_top  # 总下落高度 (注意 y_top 通常是负值? 不，公式里 y_top 是高度增量。
                                 # 让我们检查公式。Eq 9: yu(t) = ...
                                 # Library Step 2: y_top = - ... ln(...)
                                 # ln(1+...) > 0. So y_top is negative? 
                                 # Wait, if v_yi < 0 (upward), we gain altitude.
                                 # Standard physics: y increases.
                                 # Paper defines y axis downwards? 
                                 # "Ground at y=60...". Usually y=0 is ground.
                                 # Let's assume standard: y is height. v_y < 0 is upward?
                                 # Paper: "upwards motion where v_y(t) < 0". Yes.
                                 # So if v_y < 0, height increases.
                                 # The formula for y_top in Library has a minus sign.
                                 # ln(1+positive) is positive. So y_top is negative value?
                                 # If y axis points down (depth), then negative y_top means "up".
                                 # Let's assume y_initial is positive (e.g. 150m) and ground is 0?
                                 # Or y axis points down?
                                 # Paper Fig 1: "Ground at y=60". Trajectory starts at y=0?
                                 # "Note that the y axis is 'reversed' in relation to v_y(t)."
                                 # Let's stick to the Library's "Total drop height" logic.
                                 # y_total = y_initial + (altitude gained).
                                 # If y_top formula gives a negative value for altitude gain, then we subtract it?
                                 # Let's look at Eq 21 in paper: y_top = - (m/2c) ln(...)
                                 # This y_top is likely "distance dropped" (negative means gained height).
                                 # So Total Drop Height = y_initial - y_top. (e.g. 150 - (-10) = 160).
                                 # This makes sense.
    
    # 4. 计算下落时间 (Descent Time)
    # Eq (13) / (22)
    # t_drop = ... arccosh( exp( c*y_total/m + G_d_hat ) ) ...
    term_inside_exp = (c * y_total / m) + G_d_hat
    # arccosh requires input >= 1. exp(positive) >= 1. Safe.
    t_drop_hat = (1.0 / (g * gamma)) * (np.arccosh(np.exp(term_inside_exp)) - H_d_hat)
    
    # 总撞击时间
    t_im = t_top_hat + t_drop_hat
    
    # 5. 计算切换时刻 t_c (Crossover Time)
    # Eq (14) approximation
    # Note: The formula uses t_top parameter. In unified flow, we pass t_drop_hat?
    # Library Step 4 says: "t_top in Eq 14 corresponds to t_drop's start time..."
    # Actually, let's look at the structure.
    # The formula calculates the time relative to the start of the "tanh" phase (downward phase).
    # So the result is "time since top".
    # Let's call it delta_t_c.
    # delta_t_c = numerator / denominator
    # numerator = m * (g * t_top_param - Gamma * Hd + ...)
    # Here t_top_param should be t_drop_hat? No, that's the total drop time.
    # The formula (14) approximates the solution to tanh(at) = 1/t.
    # It finds the time t.
    # In the paper, t is time since start of phase.
    # So we are solving for t where v_x(t) = v_y(t).
    # The formula (14) gives t_c directly?
    # "t_c = ... (14)".
    # Let's implement Eq 14 exactly as written in Library Step 4.
    # But be careful with variables.
    # In Library Step 4: t_c_hat = t_top_hat + [ ... ]
    # The term in [ ] uses t_drop_hat?
    # "g * t_drop - Gamma * H_d ..."
    # Wait, why t_drop? t_drop is the impact time. t_c is crossover time.
    # t_c should not depend on impact time (unless impact happens before crossover).
    # The formula (14) in paper uses "t_top".
    # In paper context (Section A), t_top was "time of top".
    # But Eq 14 is derived for "crossing from vx to vy".
    # It seems Eq 14 is an approximation for the root.
    # Let's look at the paper text again.
    # "t_c = ... (14)".
    # It uses t_top.
    # In the unified section (C), it says: "t_c_hat = t_c(t_top_hat, H_d_hat)".
    # And "t_c(t_top, H_d)" refers to Eq 14.
    # So we plug in t_top_hat and H_d_hat into Eq 14.
    # Let's do that.
    
    numerator = m * (g * t_top_hat - Gamma * H_d_hat + v_x_top * (1 + (H_d_hat - g * gamma * t_top_hat)**2))
    denominator = m * g + c * v_x_top * (g * t_top_hat - Gamma * H_d_hat)
    
    # Wait, looking at Eq 14 in paper screenshot again.
    # It has t_top in it.
    # If v_yi >= 0, t_top = 0.
    # Then t_c = m(-Gamma H_d + v_xi(1+Hd^2)) / (mg + c v_xi (-Gamma Hd)).
    # This gives a time.
    # Is this time absolute or relative?
    # Since it uses t_top, it's likely absolute time from t=0.
    # So t_c_calculated is the absolute time of crossover.
    
    t_c_calculated = numerator / denominator
    
    # 6. 计算水平总距离
    # Case A: t_im <= t_c (Impact before crossover)
    # Case B: t_im > t_c (Crossover happens)
    
    # We need to handle arrays, so use np.where
    
    # Calculate x_final for Case A
    # x_final = x1 + (m/c) * ln(1 + c*v_x_top/m * (t_im - t_top_hat))
    dt_im = t_im - t_top_hat
    x_case_A = x1 + (m / c) * np.log(1 + (c * v_x_top / m) * dt_im)
    
    # Calculate x_final for Case B
    # x2 (from t_top to t_c)
    # Note: t_c_calculated is absolute time.
    dt_c = t_c_calculated - t_top_hat
    # Avoid log(negative) if t_c < t_top (should not happen if logic is right)
    dt_c = np.maximum(0, dt_c) 
    
    x2 = (m / c) * np.log(1 + (c * v_x_top / m) * dt_c)
    
    # Intermediate velocities at t_c
    # v_xc = v_x(t_c)
    # v_x(t) formula (4): v_xi / (1 + v_xi c t / m). Here t is time since v_x_top start?
    # Yes, v_x_top is the start of this phase.
    v_xc = v_x_top / (1 + (v_x_top * c / m) * dt_c)
    
    # v_yc = v_y(t_c - t_top)
    # Eq (7): Gamma * tanh(g gamma t + Hd)
    v_yc = Gamma * np.tanh(g * gamma * dt_c + H_d_hat)
    
    # Phase 3 constants
    H_c = np.arctanh(np.clip(v_yc * gamma, -0.9999, 0.9999))
    # G_c = ln(cosh(H_c)) -> cosh(arctanh(x)) = 1/sqrt(1-x^2)
    # Let's use np.log(np.cosh(H_c))
    G_c = np.log(np.cosh(H_c))
    
    # x3 (from t_c to t_im)
    dt_3 = t_im - t_c_calculated
    dt_3 = np.maximum(0, dt_3)
    
    # Eq (18) / (23)
    # term1 = arctan( sinh( g gamma dt + Hc ) )
    term1 = np.arctan(np.sinh(g * gamma * dt_3 + H_c))
    # term2 = arcsin( v_yc * gamma )
    # v_yc * gamma should be tanh( ... ). tanh is in (-1, 1). arcsin is safe.
    term2 = np.arcsin(np.clip(v_yc * gamma, -0.9999, 0.9999))
    
    pre_factor = (v_xc * np.exp(G_c)) / (g * gamma)
    x3 = pre_factor * (term1 - term2)
    
    x_case_B = x1 + x2 + x3
    
    # Final selection
    x_final = np.where(t_im <= t_c_calculated, x_case_A, x_case_B)
    
    return {
        "fall_time": t_im,
        "horizontal_distance": x_final,
        "t_c": t_c_calculated,
        "impact_before_crossover": t_im <= t_c_calculated
    }
