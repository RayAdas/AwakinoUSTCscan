# NetTOF 模块说明

## 1. 项目概览
NetTOF 是一个基于策略模式的合成回波参数预测框架，用于从一维超声波波形估计多个回波参数 `(fc, beta, alpha, r, tau, psi, phi)`。框架位于 `PythonScripts/NetTOF` 目录下，通过自动化的数据生成、模型训练与评估流程，快速验证不同的网络策略。

## 2. 目录结构与模块职责
- `PythonScripts/NetTOF/__init__.py`
  - 聚合包内关键对象，导出数据集、模型策略注册表等，方便外部导入。
- `PythonScripts/NetTOF/echo_model.py`
  - 定义 `EchoInfo` 参数容器与 `echo_function` 波形生成函数。
  - 提供参数范围的上下界常量，供数据集模块采样使用。
- `PythonScripts/NetTOF/dataset.py`
  - `SyntheticEchoDataset`：按指定的 `n_iter_outer` 和时间轴生成合成数据。
  - 随机采样参数、调用 `echo_function` 叠加波形，并注入高斯噪声与背景偏置。
  - 对各参数执行对数或线性归一化，提供 `normalize_params` 与 `denormalize_params` 保证可逆。
  - 支持自动拆分训练、验证、测试集并返回 `DataLoader`。
- `PythonScripts/NetTOF/strategies/`
  - `__init__.py`：维护策略注册表 `STRATEGY_REGISTRY`。
  - `base_strategy.py`：定义抽象类 `BaseNetStrategy`，约束 `build_model`、`train`、`evaluate`、`predict` 接口。
  - `regression_strategy.py`：实现 `RegressionNetStrategy`，搭建 1D CNN + 全连接的回归网络，并根据参数类型施加不同激活函数。
- `PythonScripts/NetTOF/trainer.py`
  - 提供 `Trainer` 类，封装训练循环、早停逻辑、模型保存与指标计算。
  - 使用 `MSELoss`、Adam 优化器，评估阶段输出每个参数的 MAE 与 RMSE。
- `PythonScripts/NetTOF/main.py`
  - 命令行入口，解析策略选择、样本规模、训练超参等参数。
  - 负责实例化数据集与策略、执行训练、载入最佳模型并输出测试指标。

## 3. 数据与训练流程
1. 通过 `SyntheticEchoDataset` 生成指定数量的波形样本，并完成归一化标签。
2. `get_dataloaders` 划分训练、验证、测试集并返回对应迭代器。
3. 选定的策略（如 `RegressionNetStrategy`）构建模型，调用 `Trainer.fit` 进行迭代训练。
4. 训练过程中在验证集上监控损失，触发早停并保存最佳权重。
5. 训练结束后使用 `Trainer.evaluate` 在测试集上计算参数级 MAE/RMSE。
6. `Trainer.predict` 支持单条波形推理，并可通过 `denormalize_params` 还原真实物理量。

## 4. 快速使用示例
```bash
python -m PythonScripts.NetTOF.main \
  --strategy regression \
  --n_iter_outer 2 \
  --epochs 50 \
  --batch_size 16 \
  --total_samples 192
```
上述命令将在生成的合成数据集上训练回归策略模型，自动保存最佳权重并输出测试集指标。
