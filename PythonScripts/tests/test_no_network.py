import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from tests.train_unet import TGT_KERNEL, heatmap_loss
from utils.file_io import FileIO
from rebuild.unet import UNet3D
from rebuild.dataset import DeepImgDataset


# Match training hyperparameters for evaluation/visualization
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
	FileIO.init()
	dataset_path = os.path.join(FileIO.curr_rebuild_dataset_path, "dataset.pt")
	dataset = torch.load(dataset_path, weights_only=False)
	return dataset

def predict_depth(wave_input):
	"""Convert predicted heatmap to depth index via soft-argmax.

	wave_input: (B, H, W, T) float tensor
	return: (B, H, W) float indices
	"""
	x = wave_input  # (B,H,W,T)
	peak_idx = torch.max(x, dim=-1).indices  # (B,H,W)

	# 以tau=peak_idx, sigma=1e-4为初始值，进行参数优化，使得生成的正态分布拟合x中的波形
	B, H, W, T = x.shape
	tau = peak_idx.float().unsqueeze(-1).detach().requires_grad_(True)  # (B,H,W,1)
	sigma = torch.full_like(tau, 1e-4, requires_grad=True)  # (B,H,W,1)
	optimizer = torch.optim.Adam([tau, sigma], lr=1e-1)
	for _ in range(100):
		optimizer.zero_grad()
		t = torch.arange(T, device=x.device, dtype=x.dtype).view(1, 1, 1, T)  # (1,1,1,T)
		gauss = torch.exp(-0.5 * ((t - tau) / sigma) ** 2)  # (B,H,W,T)
		gauss = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
		loss = heatmap_loss(gauss, x)
		loss.backward()
		optimizer.step()
		# 防止sigma过小
		sigma.data.clamp_(min=1e-4)

	return DeepImgDataset.wave_pos2real_depth(tau).detach().squeeze(-1)

def visualize_samples(dataset, device):
	"""Visualize samples one-by-one: 2D/3D GT and prediction per window."""
	for i in range(len(dataset)):
		sample = dataset[i]
		wave_input = sample["input"].unsqueeze(0).to(device)  # (1,H,W,T)
		r_target = sample["target"].cpu().numpy()
		r_pred = predict_depth(wave_input).squeeze(0).cpu().numpy()

		vmin = float(min(r_target.min(), r_pred.min()))
		vmax = float(max(r_target.max(), r_pred.max()))

		h, w = r_target.shape
		x = np.arange(0, w, 1)
		y = np.arange(0, h, 1)
		X, Y = np.meshgrid(x, y)

		fig = plt.figure(figsize=(16, 12))
		fig.suptitle(f"Sample {i+1}/{len(dataset)}", fontsize=14)

		ax_2d_t = fig.add_subplot(2, 2, 1)
		im_t = ax_2d_t.imshow(r_target, cmap='viridis', vmin=vmin, vmax=vmax)
		ax_2d_t.set_title(f"2D Ground Truth (max: {r_target.max():.2f})")
		ax_2d_t.axis('off')
		fig.colorbar(im_t, ax=ax_2d_t, shrink=0.8)

		ax_2d_p = fig.add_subplot(2, 2, 2)
		im_p = ax_2d_p.imshow(r_pred, cmap='viridis', vmin=vmin, vmax=vmax)
		ax_2d_p.set_title(f"2D Prediction (max: {r_pred.max():.2f})")
		ax_2d_p.axis('off')
		fig.colorbar(im_p, ax=ax_2d_p, shrink=0.8)

		ax_3d_t = fig.add_subplot(2, 2, 3, projection='3d')
		surf_t = ax_3d_t.plot_surface(
			X, Y, r_target, cmap='viridis', edgecolor='none', alpha=0.9, rstride=1, cstride=1
		)
		ax_3d_t.set_title("3D Ground Truth")
		ax_3d_t.set_xlabel('X')
		ax_3d_t.set_ylabel('Y')
		ax_3d_t.set_zlabel('Index')
		ax_3d_t.view_init(elev=30, azim=45)
		fig.colorbar(surf_t, ax=ax_3d_t, shrink=0.6, aspect=8)

		ax_3d_p = fig.add_subplot(2, 2, 4, projection='3d')
		surf_p = ax_3d_p.plot_surface(
			X, Y, r_pred, cmap='viridis', edgecolor='none', alpha=0.9, rstride=1, cstride=1
		)
		ax_3d_p.set_title("3D Prediction")
		ax_3d_p.set_xlabel('X')
		ax_3d_p.set_ylabel('Y')
		ax_3d_p.set_zlabel('Index')
		ax_3d_p.view_init(elev=30, azim=45)
		fig.colorbar(surf_p, ax=ax_3d_p, shrink=0.6, aspect=8)

		plt.tight_layout()
		plt.show(block=True)
		plt.close(fig)


def main():
	dataset = load_dataset()

	visualize_samples(dataset, DEVICE)


if __name__ == "__main__":
	main()

