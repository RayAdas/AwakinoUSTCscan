import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from utils import MyFilter, FileIO

# 定义LSTM网络结构
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        input_size = 1
        self.lstm = nn.LSTM(input_size, config['hidden_size'], config['num_layers'], batch_first=True)
        self.fc = nn.Linear(config['hidden_size'], 1)
        
    def forward(self, x):
        # 获取当前设备信息
        device = x.device
        
        # 初始化隐藏状态（自动匹配设备）
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))  # 此时所有张量都在相同设备
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# 超参数设置
config = {
    'batch_size': 32,
    'num_workers': 4,
    'hidden_size': 64,
    'num_layers': 2,
    'learning_rate': 0.001,
    'num_epochs': 5,
    'seq_length': 1000  # 根据实际波形长度调整
}

# 数据加载
def prepare_dataloader(dataset):
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )
    return dataloader

# 训练函数
def train_model(model, train_dataset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")
    model.to(torch.device(device))

    train_loader = prepare_dataloader(train_dataset)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    losses = []
    accuracies = []
    
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(-1).to(device)  # [batch, seq_len] -> [batch, seq_len, 1]
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.show()
    
    return model
def test_model(model, test_dataset):
    # 确定设备并统一管理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")
    model.to(torch.device(device))

    test_loader = prepare_dataloader(test_dataset)
    
    # 存储预测结果
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 统一设备转移
            inputs = inputs.unsqueeze(-1).to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # 收集结果时转回CPU
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())  # 确保标签也在CPU
            all_probs.extend(probs)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)
