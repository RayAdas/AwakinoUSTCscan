import torch.nn as nn
import torch
import math
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class NormalSignalDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=100):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.inputs = []
        self.tgts_tau = []
        self.tgts_mask = []
        t_axis = torch.linspace(0, 100, seq_len)

        a =  1 / math.sqrt(2*math.pi)
        for i in range(num_samples):
            ns_num = torch.randint(0, 4, (1,)).item()
            taus = torch.rand(ns_num) * 80 + 10  # (ns_num,)
            signal_sum = torch.zeros(seq_len)
            for j in range(ns_num):
                tau = taus[j]
                signal = a * torch.exp(-0.5 * ((t_axis - tau) / 2) ** 2)
                signal_sum += signal

            # 将taus排序，然后补长到3个元素
            taus, _ = torch.sort(taus, descending=False)
            taus_padded = torch.zeros(3)
            taus_padded[:ns_num] = taus

            mask = torch.zeros(3)
            mask[:ns_num] = 1.0

            self.inputs.append(signal_sum)
            self.tgts_tau.append(taus_padded)
            self.tgts_mask.append(mask)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.tgts_tau[idx], self.tgts_mask[idx]


class MultiTaskSignalNet(nn.Module):
    """
    多任务学习网络，用于预测信号中的高斯峰位置和存在性
    - tgts_tau: 回归任务，预测峰的位置
    - tgts_mask: 分类任务，预测峰是否存在
    """
    def __init__(self, seq_len=100, max_peaks=3, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.max_peaks = max_peaks
        
        # 特征提取器 - 使用1D CNN
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # 共享层
        self.shared_fc = nn.Sequential(
            nn.Linear(256 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 回归分支 - 预测tau位置
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, max_peaks)
        )
        
        # 分类分支 - 预测mask
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, max_peaks),
            nn.Sigmoid()  # 输出概率
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        
        # 特征提取
        features = self.feature_extractor(x)  # (batch, 256, 8)
        features = features.view(features.size(0), -1)  # (batch, 256*8)
        
        # 共享层
        shared = self.shared_fc(features)  # (batch, hidden_dim)
        
        # 回归和分类输出
        tau_pred = self.regression_head(shared)  # (batch, max_peaks)
        mask_pred = self.classification_head(shared)  # (batch, max_peaks)

        tau_pred = tau_pred * mask_pred.detach()  # 仅对存在的峰进行预测
        
        return tau_pred, mask_pred


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha  # 回归损失权重
        self.beta = beta    # 分类损失权重
        
        # 回归损失 - MSE
        self.regression_loss = nn.MSELoss()
        
        # 分类损失 - BCE
        self.classification_loss = nn.BCELoss()
        
    def forward(self, tau_pred, mask_pred, tau_target, mask_target):
        # 只对存在的峰计算回归损失
        mask_target_bool = mask_target > 0.5
        
        if mask_target_bool.any():
            # 使用mask加权回归损失
            regression_loss = ((tau_pred - tau_target) ** 2 * mask_target).sum() / mask_target.sum()
        else:
            regression_loss = torch.tensor(0.0, device=tau_pred.device)
        
        # 分类损失
        classification_loss = self.classification_loss(mask_pred, mask_target)
        
        # 总损失
        total_loss = self.alpha * regression_loss + self.beta * classification_loss
        
        return total_loss, regression_loss, classification_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    
    for inputs, tgts_tau, tgts_mask in dataloader:
        inputs = inputs.to(device)
        tgts_tau = tgts_tau.to(device)
        tgts_mask = tgts_mask.to(device)
        
        # 前向传播
        tau_pred, mask_pred = model(inputs)
        
        # 计算损失
        loss, reg_loss, cls_loss = criterion(tau_pred, mask_pred, tgts_tau, tgts_mask)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_reg_loss += reg_loss.item()
        total_cls_loss += cls_loss.item()
    
    return total_loss / len(dataloader), total_reg_loss / len(dataloader), total_cls_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_reg_loss = 0
    total_cls_loss = 0
    
    with torch.no_grad():
        for inputs, tgts_tau, tgts_mask in dataloader:
            inputs = inputs.to(device)
            tgts_tau = tgts_tau.to(device)
            tgts_mask = tgts_mask.to(device)
            
            # 前向传播
            tau_pred, mask_pred = model(inputs)
            
            # 计算损失
            loss, reg_loss, cls_loss = criterion(tau_pred, mask_pred, tgts_tau, tgts_mask)
            
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
    
    return total_loss / len(dataloader), total_reg_loss / len(dataloader), total_cls_loss / len(dataloader)


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数
    num_train_samples = 8000
    num_val_samples = 2000
    seq_len = 100
    max_peaks = 3
    batch_size = 32
    num_epochs = 130
    learning_rate = 0.001
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = NormalSignalDataset(num_samples=num_train_samples, seq_len=seq_len)
    val_dataset = NormalSignalDataset(num_samples=num_val_samples, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = MultiTaskSignalNet(seq_len=seq_len, max_peaks=max_peaks, hidden_dim=256).to(device)
    
    # 创建损失函数和优化器
    criterion = MultiTaskLoss(alpha=1.0, beta=10.0)  # 分类损失权重更高
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss, train_reg, train_cls = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_reg, val_cls = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Reg: {train_reg:.4f}, Cls: {train_cls:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Reg: {val_reg:.4f}, Cls: {val_cls:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multitask_model.pt')
            print(f"  -> 保存最佳模型 (val_loss: {val_loss:.4f})")
    
    print("\n训练完成！")
    
    # 测试示例
    print("\n测试示例：")
    model.load_state_dict(torch.load('best_multitask_model.pt'))
    model.eval()
    
    with torch.no_grad():
        test_input, test_tau, test_mask = val_dataset[0]
        test_input = test_input.unsqueeze(0).to(device)
        
        tau_pred, mask_pred = model(test_input)
        tau_pred = tau_pred.cpu().squeeze().numpy()
        mask_pred = mask_pred.cpu().squeeze().numpy()
        
        print(f"真实 tau: {test_tau.numpy()}")
        print(f"预测 tau: {tau_pred}")
        print(f"真实 mask: {test_mask.numpy()}")
        print(f"预测 mask: {mask_pred}")


if __name__ == '__main__':
    main()
