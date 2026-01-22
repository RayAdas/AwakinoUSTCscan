import time
import os

import torch
from rebuild.dataset import DeepImgDataset

from utils.file_io import FileIO
N_SAMPLES = 1000
ENABLE_VISUALIZATION = True

if __name__ == "__main__":
    FileIO.init()
    dataset = DeepImgDataset(n_samples=N_SAMPLES)

    # save dataset to a file using YYYYMMDDhhmmss filename
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    dataset_name = f"sim_{N_SAMPLES}_{timestamp}"

    os.makedirs(os.path.join(FileIO.rebuild_dataset_path, dataset_name), exist_ok=True)
    torch.save(dataset, os.path.join(FileIO.rebuild_dataset_path, dataset_name, "dataset.pt"))
    print(f"Dataset with {N_SAMPLES} samples saved to {dataset_name}")

    if ENABLE_VISUALIZATION:
        sample = dataset[0]
        input_wave = sample["input"]
        target_depth = sample["target"]
        
        print(f"Input wave shape: {input_wave.shape}")
        print(f"Target depth shape: {target_depth.shape}")
        
        import matplotlib.pyplot as plt
        
        # input_wave is (receptive_field_size, receptive_field_size, wave_len)
        # We'll visualize it as a 2D heatmap of the wave at each spatial position
        # Let's show the first few wavelength samples
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Show a few wavelength slices of the input wave
        for i in range(6):
            ax = axes[i // 3, i % 3]
            wave_idx = i * input_wave.shape[2] // 6
            ax.imshow(input_wave[:, :, wave_idx].cpu().numpy())
            ax.set_title(f"Wave slice {wave_idx}")
            ax.axis('off')
        
        # Show a few wave of the input wave
        plt.figure(figsize=(6, 6))
        center_wave = input_wave[input_wave.shape[0]//2, input_wave.shape[1]//2, :].cpu().numpy()
        plt.plot(center_wave)
        integration = sum(center_wave)
        plt.title(f'Input Wave at Center Position {integration:.2f}')

        # Show target depth in a separate figure
        plt.figure(figsize=(6, 6))
        plt.imshow(target_depth.cpu().numpy())
        plt.colorbar(label='Depth')
        plt.title('Target Depth')
        plt.show()