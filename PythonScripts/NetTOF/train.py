import CNN
from utils import FileIO, EchoDataset
import torch
from torch.utils.data import random_split
    
if __name__ == "__main__":
    fio = FileIO()

    # 加载数据集
    echo_dataset = EchoDataset()
    echo_dataset.load_file(fio.join_datapath('echo_dataset.pt'))

    # 按照0.7:0.15:0.15的比例划分数据集
    total_size = len(echo_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        echo_dataset, [train_size, val_size, test_size]
    )
    
    trained_model = None
    # 开始训练
    match fio.algorithm:
        case 'CNN':
            trained_model:CNN.CNNModel = CNN.train_model(train_dataset, val_dataset)
        case 'LSTM':
            pass
        case _:
            raise ValueError("Unsupported algorithm!")

    def subset_to_dict(subset):
        data = [subset.dataset.data[i] for i in subset.indices]
        tgt = [subset.dataset.tgt[i] for i in subset.indices]
        return {'data': data, 'tgt': tgt}

    try:
        model_path = fio.join_datapath(fio.algorithm + '_model.pth')
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        train_dataset_path = fio.join_datapath('echo_train_dataset.pt')
        torch.save(subset_to_dict(train_dataset), train_dataset_path)
        print(f"Train dataset saved to {train_dataset_path}")

        val_dataset_path = fio.join_datapath('echo_val_dataset.pt')
        torch.save(subset_to_dict(val_dataset), val_dataset_path)
        print(f"Validation dataset saved to {val_dataset_path}")

        test_dataset_path = fio.join_datapath('echo_test_dataset.pt')
        torch.save(subset_to_dict(test_dataset), test_dataset_path)
        print(f"Test dataset saved to {test_dataset_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

