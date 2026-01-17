# Transformer Strategy for Depth Reconstruction

## 概述

基于Transformer Encoder-Only架构的深度重建策略，用于从超声波形信号重建深度图。

## 架构特点

### 1. 可学习的嵌入网络
- 使用多层感知机(MLP)网络进行波形嵌入
- 结构：`Linear -> GELU -> Dropout -> Linear -> LayerNorm`
- 将输入波形 `(wave_len,)` 映射到嵌入空间 `(embed_dim,)`

### 2. 2D位置编码
- 采用可学习的行列位置编码：`PE_{i,j} = PE_i^{row} + PE_j^{col}`
- 每个位置的编码由对应行和列的位置编码相加得到
- 初始化使用正弦位置编码，训练过程中可微调

### 3. Transformer Encoder
- 仅使用Encoder结构（无需Decoder，因为不涉及序列生成）
- 使用Pre-LN（Norm First）架构提高训练稳定性
- 多头自注意力机制捕捉空间依赖关系
- 前馈网络（FFN）进行特征变换

### 4. 输出头
- 轻量级MLP网络：`Linear -> GELU -> Dropout -> Linear`
- 将嵌入空间 `(embed_dim,)` 映射回深度值 `(1,)`

## 数据流

```
输入波形: (B, H, W, wave_len)
    ↓
嵌入网络: (B, H, W, wave_len) → (B, H×W, embed_dim)
    ↓
位置编码: (B, H×W, embed_dim) + PE
    ↓
Transformer Encoder: (B, H×W, embed_dim) → (B, H×W, embed_dim)
    ↓
输出头: (B, H×W, embed_dim) → (B, H×W, 1)
    ↓
重塑: (B, H×W, 1) → (B, H, W)
```

## 使用方法

### 快速开始

```python
from rebuild.strategies import STRATEGY_REGISTRY

# 创建Transformer策略
strategy = STRATEGY_REGISTRY["transformer"](
    input_channels=128,      # 波形长度
    spatial_size=41,         # 空间维度大小
    embed_dim=256,           # 嵌入维度
    num_heads=8,             # 注意力头数
    num_layers=6,            # Transformer层数
    mlp_ratio=4,             # FFN扩展比例
    dropout=0.1              # Dropout率
)

# 训练
history = strategy.train(
    dataloader_train=train_loader,
    dataloader_val=val_loader,
    epochs=50,
    lr=1e-4,                 # 学习率（建议比UNet低）
    weight_decay=1e-5
)

# 预测
depth_map = strategy.predict(wave_data)
```

### 完整示例

参见 `PythonScripts/train_transformer.py`

## 超参数建议

### 模型结构
- `embed_dim`: 128-512（较大值提升性能但增加计算量）
- `num_heads`: 4-16（应能整除embed_dim）
- `num_layers`: 3-12（较深网络捕捉更复杂模式）
- `mlp_ratio`: 4（标准值，可调整为2-8）
- `dropout`: 0.1-0.2

### 训练参数
- `lr`: 1e-4 到 5e-4（Transformer通常需要较低学习率）
- `batch_size`: 16-64（取决于显存大小）
- `epochs`: 50-200
- `weight_decay`: 1e-5 到 1e-4

## 性能特点

### 优势
1. **全局感受野**：自注意力机制能捕捉全局空间依赖关系
2. **灵活性强**：可学习的嵌入和位置编码适应不同数据特征
3. **可解释性**：注意力权重可视化有助于理解模型决策
4. **扩展性好**：易于调整模型深度和宽度

### 考虑事项
1. **计算开销**：自注意力的复杂度为O(n²)，空间维度较大时计算量大
2. **显存需求**：相比CNN需要更多显存
3. **训练时间**：通常需要更多训练轮次才能收敛
4. **数据需求**：可能需要更多训练数据以发挥性能

## 与UNet对比

| 特性 | Transformer | UNet |
|------|-------------|------|
| 感受野 | 全局 | 局部（受卷积核限制） |
| 参数量 | 较多 | 较少 |
| 训练速度 | 较慢 | 较快 |
| 显存占用 | 较高 | 较低 |
| 归纳偏置 | 较少 | 较强（平移不变性） |
| 适用场景 | 复杂空间关系 | 局部纹理特征 |

## 测试

```bash
# 运行测试
python PythonScripts/tests/test_transformer_strategy.py

# 训练模型
python PythonScripts/train_transformer.py
```

## 参考文献

- Vaswani et al. "Attention is All You Need" (NeurIPS 2017)
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)
