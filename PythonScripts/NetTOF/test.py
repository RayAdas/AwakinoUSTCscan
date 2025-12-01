from copy import deepcopy
import review_visual
from sklearn.metrics import classification_report
import torch
import CNN
import os
from utils import FileIO, EchoDataset

if __name__ == "__main__":
    fio = FileIO()
    model = CNN.CNNModel()
    model_path = fio.join_datapath(fio.algorithm + '_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # 加载数据集
    echo_train_dataset = EchoDataset()
    echo_train_dataset.load_file(fio.join_datapath('echo_train_dataset.pt'))
    echo_val_dataset = EchoDataset()
    echo_val_dataset.load_file(fio.join_datapath('echo_val_dataset.pt'))
    echo_test_dataset = EchoDataset()
    echo_test_dataset.load_file(fio.join_datapath('echo_test_dataset.pt'))

    # 获取原始样本数据用于可视化
    # sample_data = echo_dataset.dataset

    # 测试集可视化
    preds, labels, probs = CNN.test_model(model, echo_test_dataset)
    review_visual.visualize_results(deepcopy(labels), deepcopy(preds), deepcopy(probs), title = "test")
    
    # 验证集可视化
    preds, labels, probs = CNN.test_model(model, echo_val_dataset)
    review_visual.visualize_results(deepcopy(labels), deepcopy(preds), deepcopy(probs), title = "val")

    # 训练集可视化
    preds, labels, probs = CNN.test_model(model, echo_train_dataset)
    review_visual.visualize_results(deepcopy(labels), deepcopy(preds), deepcopy(probs), title = "train")
    
    # 输出详细评估指标
    print("Detailed Performance Metrics:")
    print(classification_report(labels, preds, target_names=['Negative', 'Positive']))