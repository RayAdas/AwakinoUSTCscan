"""
测试和可视化多任务神经网络的预测结果
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from 定长回归预测高斯分布 import NormalSignalDataset, MultiTaskSignalNet
import math

def visualize_predictions(model, dataset, num_samples=4, device='cpu'):
    """可视化模型预测结果"""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    with torch.no_grad():
        for i in range(num_samples):
            # 获取数据
            test_input, test_tau, test_mask = dataset[i]
            test_input_batch = test_input.unsqueeze(0).to(device)
            
            # 预测
            tau_pred, mask_pred = model(test_input_batch)
            tau_pred = tau_pred.cpu().squeeze().numpy()
            mask_pred = mask_pred.cpu().squeeze().numpy()
            
            # 绘图
            ax = axes[i]
            t_axis = np.linspace(0, 100, len(test_input))
            
            # 绘制输入信号
            ax.plot(t_axis, test_input.numpy(), 'k-', linewidth=2, label='输入信号')
            
            # 绘制真实峰位置
            test_tau_np = test_tau.numpy()
            test_mask_np = test_mask.numpy()
            for j in range(len(test_tau_np)):
                if test_mask_np[j] > 0.5:
                    ax.axvline(test_tau_np[j], color='green', linestyle='--', 
                             linewidth=2, alpha=0.7, label=f'真实峰 {j+1}: τ={test_tau_np[j]:.1f}')
            
            # 绘制预测峰位置
            for j in range(len(tau_pred)):
                if mask_pred[j] > 0.5:  # 只显示概率大于0.5的峰
                    ax.axvline(tau_pred[j], color='red', linestyle=':', 
                             linewidth=2, alpha=0.7, 
                             label=f'预测峰 {j+1}: τ={tau_pred[j]:.1f} (prob={mask_pred[j]:.2f})')
            
            ax.set_xlabel('时间 t')
            ax.set_ylabel('信号强度')
            ax.set_title(f'样本 {i+1}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multitask_predictions.png', dpi=150, bbox_inches='tight')
    print("预测结果已保存到 multitask_predictions.png")
    plt.show()


def evaluate_metrics(model, dataset, device='cpu', num_samples=100):
    """评估模型性能指标"""
    model.eval()
    
    tau_errors = []
    mask_accuracies = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            test_input, test_tau, test_mask = dataset[i]
            test_input_batch = test_input.unsqueeze(0).to(device)
            
            tau_pred, mask_pred = model(test_input_batch)
            tau_pred = tau_pred.cpu().squeeze().numpy()
            mask_pred = mask_pred.cpu().squeeze().numpy()
            
            test_tau_np = test_tau.numpy()
            test_mask_np = test_mask.numpy()
            
            # 计算mask准确率
            mask_pred_binary = (mask_pred > 0.5).astype(float)
            mask_accuracy = (mask_pred_binary == test_mask_np).mean()
            mask_accuracies.append(mask_accuracy)
            
            # 计算tau误差（只对存在的峰）
            for j in range(len(test_tau_np)):
                if test_mask_np[j] > 0.5:
                    tau_errors.append(abs(tau_pred[j] - test_tau_np[j]))
    
    print("\n=== 模型性能评估 ===")
    print(f"Mask分类准确率: {np.mean(mask_accuracies):.4f} ± {np.std(mask_accuracies):.4f}")
    print(f"Tau回归MAE: {np.mean(tau_errors):.4f} ± {np.std(tau_errors):.4f}")
    print(f"Tau回归中位数误差: {np.median(tau_errors):.4f}")
    print(f"评估样本数: {num_samples}")
    
    return mask_accuracies, tau_errors


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据集
    print("创建测试数据集...")
    test_dataset = NormalSignalDataset(num_samples=1000, seq_len=100)
    
    # 加载模型
    print("加载模型...")
    model = MultiTaskSignalNet(seq_len=100, max_peaks=3, hidden_dim=256).to(device)
    model.load_state_dict(torch.load('best_multitask_model.pt'))
    
    # 评估性能
    evaluate_metrics(model, test_dataset, device=device, num_samples=1000)
    
    # 可视化预测结果
    print("\n生成可视化结果...")
    visualize_predictions(model, test_dataset, num_samples=6, device=device)


if __name__ == '__main__':
    main()
