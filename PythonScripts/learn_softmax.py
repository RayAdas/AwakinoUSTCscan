import matplotlib.pyplot as plt
import torch

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题

# 创建测试数据
L = 10
x = torch.randn(L) * 2 + 1  # 均值为1，标准差为2的正态分布
p = torch.arange(1, L+1, dtype=torch.float32)

# 计算
a1 = x / x.sum()
a2 = torch.softmax(x, dim=0)
r1 = torch.sum(a1 * p)
r2 = torch.sum(a2 * p)

# 绘图
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(range(L), x.numpy())
plt.title('原始输入 x')
plt.xlabel('位置')
plt.ylabel('数值')

plt.subplot(1, 3, 2)
plt.bar(range(L), a1.numpy(), alpha=0.5, label='线性归一化')
plt.bar(range(L), a2.numpy(), alpha=0.5, label='Softmax归一化')
plt.title('两种归一化权重对比')
plt.xlabel('位置')
plt.ylabel('权重')
plt.legend()

plt.subplot(1, 3, 3)
plt.bar(['线性(r1)', 'Softmax(r2)'], [r1.item(), r2.item()])
plt.title(f'加权平均位置\nr1={r1:.3f}, r2={r2:.3f}')
plt.ylabel('位置值')

plt.tight_layout()
plt.show()