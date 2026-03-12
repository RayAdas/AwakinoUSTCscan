import matplotlib.pyplot as plt
import numpy as np

# 数据
train_loss = [
    0.015770, 0.003454, 0.001961, 0.001376, 0.001061, 0.000835, 0.000778, 0.000593, 0.000520, 0.000459,
    0.000422, 0.000338, 0.000390, 0.000301, 0.000300, 0.000262, 0.000204, 0.000229, 0.000167, 0.000145,
    0.000220, 0.000117, 0.000180, 0.000119, 0.000122, 0.000129, 0.000152, 0.000106, 0.000113, 0.000104,
    0.000083, 0.000128, 0.000180
]

val_loss = [
    0.004720, 0.002381, 0.001687, 0.000970, 0.000882, 0.000739, 0.000679, 0.000560, 0.000386, 0.000460,
    0.000392, 0.000324, 0.000254, 0.000220, 0.000286, 0.000253, 0.000158, 0.000219, 0.000258, 0.000122,
    0.000134, 0.000170, 0.000123, 0.000098, 0.000115, 0.000295, 0.000169, 0.000204, 0.000083, 0.000111,
    0.000134, 0.000118, 0.000124
]

# 创建epoch列表
epochs = list(range(1, len(train_loss) + 1))

# 创建包含两个子图的图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 子图1：线性坐标
ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
ax1.set_title('Linear Scale', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10)

# 子图2：对数坐标
ax2.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
ax2.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
ax2.set_title('Logarithmic Scale', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss (log scale)', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=10)

# 主标题
fig.suptitle('Training and Validation Loss Over Epochs', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()