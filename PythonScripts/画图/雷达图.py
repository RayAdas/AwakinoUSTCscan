import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 数据准备
categories = ['真实-MAE', '真实-RMSE', '真实-Max Abs Error', '真实-ES',
              '真实拆分-MAE', '真实拆分-RMSE', '真实拆分-Max Abs Error', '真实拆分-ES',
              '虚拟-MAE', '虚拟-RMSE', '虚拟-Max Abs Error', '虚拟-ES']

# 两种方法的数据
methods = ['微调33代', '微调100代']
data = {
    '微调33代': [0.000017, 0.000041, 0.001304, 0.000604,
               0.000018, 0.000038, 0.000640, 0.000268,
               0.000134, 0.000389, 0.004825, 0.002249],
    '微调100代': [0.000012, 0.000013, 0.000475, 0.000593,
                0.000012, 0.000014, 0.000132, 0.000256,
                0.003103, 0.003658, 0.006078, 0.000149]
}

# 对数据进行归一化处理（因为不同维度的数值范围差异很大）
# 使用最大最小值归一化到 [0, 1] 范围
all_values = data['微调33代'] + data['微调100代']
min_val = min(all_values)
max_val = max(all_values)

def normalize(values):
    return [(v - min_val) / (max_val - min_val) for v in values]

data_normalized = {
    method: normalize(values) for method, values in data.items()
}

# 设置雷达图的角度
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合图形

# 创建图形
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

# 绘制每个方法
colors = ['#FF6B6B', '#4ECDC4']
for i, (method, values) in enumerate(data_normalized.items()):
    values_normalized = values + values[:1]  # 闭合图形
    ax.plot(angles, values_normalized, 'o-', linewidth=2, 
            label=method, color=colors[i], markersize=8)
    ax.fill(angles, values_normalized, alpha=0.1, color=colors[i])

# 设置角度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9, rotation=45)

# 设置径向刻度
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax.set_ylim(0, 1)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 添加标题和图例
plt.title('不同数据集上微调方法的性能对比雷达图\n(数值越小性能越好，已归一化)', 
          fontsize=16, pad=20, fontweight='bold')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)

# 添加区域标记
ax.text(0, 0.5, '真实数据集', transform=ax.transData, fontsize=12, 
        ha='center', va='center', fontweight='bold')
ax.text(2*pi/3, 0.5, '真实拆分', transform=ax.transData, fontsize=12, 
        ha='center', va='center', fontweight='bold')
ax.text(4*pi/3, 0.5, '虚拟数据集', transform=ax.transData, fontsize=12, 
        ha='center', va='center', fontweight='bold')

# 调整布局
plt.tight_layout()
plt.show()

# 打印原始数据对比
print("原始数据对比：")
print("-" * 80)
print(f"{'数据集-指标':<20} {'微调33代':<15} {'微调100代':<15} {'改进比例':<15}")
print("-" * 80)

for i, cat in enumerate(categories):
    val33 = data['微调33代'][i]
    val100 = data['微调100代'][i]
    improvement = (val33 - val100) / val33 * 100 if val33 != 0 else 0
    print(f"{cat:<20} {val33:<15.6f} {val100:<15.6f} {improvement:>14.2f}%")