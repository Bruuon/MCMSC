import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
from scipy.optimize import linprog

class EquipmentOptimizationModel:
    def __init__(self):
        # 设备数据库 - 控制中心设备
        self.control_equipment = {
            'backup_drone': {
                'name': '备用无人机',
                'acquisition_cost': 30000,  # 元
                'maintenance_cost': 4500,   # 元/年
                'readiness_cost': 2000,     # 元/年
                'usage_cost_per_hour': 500,
                'efficiency_gain': 0.85,    # 搜索效率提升
                'response_time_reduction': 0.7,  # 响应时间减少比例
                'lifespan': 3,              # 年
                'quantity_min': 1,
                'quantity_max': 5
            },
            'thermal_camera': {
                'name': '热成像相机',
                'acquisition_cost': 50000,
                'maintenance_cost': 2500,
                'readiness_cost': 1000,
                'usage_cost_per_hour': 100,
                'efficiency_gain': 0.65,
                'response_time_reduction': 0.3,
                'lifespan': 5,
                'quantity_min': 0,
                'quantity_max': 3
            },
            'gps_tracker': {
                'name': 'GPS追踪器',
                'acquisition_cost': 2000,
                'maintenance_cost': 300,
                'readiness_cost': 200,
                'usage_cost_per_hour': 50,
                'efficiency_gain': 0.75,
                'response_time_reduction': 0.6,
                'lifespan': 2,
                'quantity_min': 2,
                'quantity_max': 10
            },
            'satellite_comm': {
                'name': '卫星通信设备',
                'acquisition_cost': 15000,
                'maintenance_cost': 3000,
                'readiness_cost': 1500,
                'usage_cost_per_hour': 200,
                'efficiency_gain': 0.55,
                'response_time_reduction': 0.4,
                'lifespan': 4,
                'quantity_min': 1,
                'quantity_max': 3
            }
        }
        
        # 救援队设备
        self.rescue_equipment = {
            'rf_detector': {
                'name': '射频信号探测器',
                'acquisition_cost': 20000,
                'maintenance_cost': 1000,
                'readiness_cost': 800,
                'usage_cost_per_hour': 150,
                'efficiency_gain': 0.60,
                'response_time_reduction': 0.5,
                'lifespan': 4,
                'quantity_min': 1,
                'quantity_max': 4
            },
            'portable_gps': {
                'name': '便携式GPS导航器',
                'acquisition_cost': 3000,
                'maintenance_cost': 300,
                'readiness_cost': 200,
                'usage_cost_per_hour': 50,
                'efficiency_gain': 0.45,
                'response_time_reduction': 0.35,
                'lifespan': 3,
                'quantity_min': 3,
                'quantity_max': 8
            },
            'offroad_vehicle': {
                'name': '越野车辆',
                'acquisition_cost': 300000,
                'maintenance_cost': 30000,
                'readiness_cost': 10000,
                'usage_cost_per_hour': 500,
                'efficiency_gain': 0.70,
                'response_time_reduction': 0.65,
                'lifespan': 8,
                'quantity_min': 0,
                'quantity_max': 2
            }
        }
    
    def calculate_annual_cost(self, equipment_type, quantity, usage_hours=100):
        """计算设备的年总成本"""
        if equipment_type in self.control_equipment:
            eq = self.control_equipment[equipment_type]
        else:
            eq = self.rescue_equipment[equipment_type]
        
        annual_acquisition = eq['acquisition_cost'] / eq['lifespan']
        annual_maintenance = eq['maintenance_cost']
        annual_readiness = eq['readiness_cost']
        annual_usage = eq['usage_cost_per_hour'] * usage_hours
        
        total_annual = (annual_acquisition + annual_maintenance + 
                       annual_readiness + annual_usage) * quantity
        
        return {
            'acquisition': annual_acquisition * quantity,
            'maintenance': annual_maintenance * quantity,
            'readiness': annual_readiness * quantity,
            'usage': annual_usage * quantity,
            'total': total_annual
        }
    
    def calculate_efficiency_score(self, equipment_config, usage_hours=100):
        """计算配置方案的总效率得分"""
        total_efficiency = 0
        total_response_reduction = 0
        
        for eq_type, quantity in equipment_config.items():
            if eq_type in self.control_equipment:
                eq = self.control_equipment[eq_type]
            else:
                eq = self.rescue_equipment[eq_type]
            
            total_efficiency += eq['efficiency_gain'] * quantity
            total_response_reduction += eq['response_time_reduction'] * quantity
        
        return {
            'efficiency_score': total_efficiency,
            'response_improvement': total_response_reduction,
            'overall_score': total_efficiency * 0.6 + total_response_reduction * 0.4
        }
    
    def optimize_equipment_selection(self, budget_constraint, priority='balanced'):
        """优化设备选择方案"""
        # 合并所有设备
        all_equipment = {**self.control_equipment, **self.rescue_equipment}
        equipment_list = list(all_equipment.keys())
        
        # 目标函数系数（最小化成本，最大化效率）
        if priority == 'cost':
            # 侧重成本最小化
            c = [self.calculate_annual_cost(eq, 1)['total'] for eq in equipment_list]
        elif priority == 'efficiency':
            # 侧重效率最大化（负号因为linprog是最小化）
            c = [-all_equipment[eq]['efficiency_gain'] * 10000 for eq in equipment_list]
        else:  # balanced
            # 平衡成本与效率
            c = [self.calculate_annual_cost(eq, 1)['total'] - 
                 all_equipment[eq]['efficiency_gain'] * 5000 for eq in equipment_list]
        
        # 约束条件：预算约束
        A_ub = [[self.calculate_annual_cost(eq, 1)['total'] for eq in equipment_list]]
        b_ub = [budget_constraint]
        
        # 变量边界（设备数量约束）
        bounds = []
        for eq in equipment_list:
            bounds.append((all_equipment[eq]['quantity_min'], 
                          all_equipment[eq]['quantity_max']))
        
        # 求解线性规划
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, 
                        method='highs', integrality=1)
        
        if result.success:
            quantities = {equipment_list[i]: int(round(result.x[i])) 
                         for i in range(len(equipment_list)) if result.x[i] > 0}
            return quantities
        else:
            return self.get_fallback_solution(budget_constraint)
    
    def get_fallback_solution(self, budget):
        """获取基于经验的备选方案"""
        # 根据预算水平提供不同方案
        if budget < 100000:
            return {'backup_drone': 2, 'gps_tracker': 5, 'portable_gps': 4}
        elif budget < 300000:
            return {'backup_drone': 3, 'gps_tracker': 8, 'thermal_camera': 1, 
                   'satellite_comm': 1, 'rf_detector': 2, 'portable_gps': 6}
        else:
            return {'backup_drone': 4, 'gps_tracker': 10, 'thermal_camera': 2,
                   'satellite_comm': 2, 'rf_detector': 3, 'portable_gps': 8,
                   'offroad_vehicle': 1}
    
    def generate_cost_breakdown(self, equipment_config, years=5):
        """生成成本明细分析"""
        cost_data = {}
        total_annual_cost = 0
        
        for eq_type, quantity in equipment_config.items():
            if quantity > 0:
                cost_data[eq_type] = self.calculate_annual_cost(eq_type, quantity)
                total_annual_cost += cost_data[eq_type]['total']
        
        # 计算N年总成本
        total_cost_5years = total_annual_cost * years
        
        return {
            'annual_breakdown': cost_data,
            'total_annual_cost': total_annual_cost,
            'total_5year_cost': total_cost_5years,
            'efficiency_metrics': self.calculate_efficiency_score(equipment_config)
        }
    
    def plot_cost_analysis(self, equipment_config, save_path=None):
        """绘制成本分析图表"""
        analysis = self.generate_cost_breakdown(equipment_config)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 饼图：成本构成
        cost_categories = {'采购成本': 0, '维护成本': 0, '预备成本': 0, '使用成本': 0}
        for eq_data in analysis['annual_breakdown'].values():
            cost_categories['采购成本'] += eq_data['acquisition']
            cost_categories['维护成本'] += eq_data['maintenance']
            cost_categories['预备成本'] += eq_data['readiness']
            cost_categories['使用成本'] += eq_data['usage']
        
        ax1.pie(cost_categories.values(), labels=cost_categories.keys(), 
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('年度成本构成分析')
        
        # 柱状图：设备成本分布
        equipment_names = []
        equipment_costs = []
        for eq_type, cost_data in analysis['annual_breakdown'].items():
            if eq_type in self.control_equipment:
                equipment_names.append(self.control_equipment[eq_type]['name'])
            else:
                equipment_names.append(self.rescue_equipment[eq_type]['name'])
            equipment_costs.append(cost_data['total'])
        
        ax2.barh(equipment_names, equipment_costs)
        ax2.set_xlabel('年度成本 (元)')
        ax2.set_title('各设备年度成本分布')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return analysis

# 示例使用
if __name__ == "__main__":
    model = EquipmentOptimizationModel()
    
    # 不同预算水平的优化方案
    budgets = [80000, 200000, 500000]
    
    for budget in budgets:
        print(f"\n{'='*50}")
        print(f"预算水平: {budget:,} 元")
        print(f"{'='*50}")
        
        # 获取优化方案
        optimal_config = model.optimize_equipment_selection(budget)
        
        # 生成成本分析
        analysis = model.generate_cost_breakdown(optimal_config)
        
        print("\n推荐设备配置:")
        for eq_type, quantity in optimal_config.items():
            if eq_type in model.control_equipment:
                name = model.control_equipment[eq_type]['name']
                category = "控制中心"
            else:
                name = model.rescue_equipment[eq_type]['name']
                category = "救援队"
            print(f"  {name}: {quantity}台 ({category})")
        
        print(f"\n成本分析:")
        print(f"  年度总成本: {analysis['total_annual_cost']:,.2f} 元")
        print(f"  5年总成本: {analysis['total_5year_cost']:,.2f} 元")
        
        print(f"\n效率指标:")
        eff = analysis['efficiency_metrics']
        print(f"  综合效率得分: {eff['overall_score']:.3f}")
        print(f"  搜索效率提升: {eff['efficiency_score']:.3f}")
        print(f"  响应时间改善: {eff['response_improvement']:.3f}")
    
    # 为中等预算生成详细图表
    medium_budget_config = model.optimize_equipment_selection(200000)
    detailed_analysis = model.plot_cost_analysis(medium_budget_config, 
                                               'equipment_cost_analysis.png')