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


def load_model():
	model = UNet3D(in_ch=1).to(DEVICE)
	ckpt_path = FileIO.rebuild_model_path
	ckpt = torch.load(ckpt_path, map_location=DEVICE)
	state = ckpt.get("model_state", ckpt)
	model.load_state_dict(state)
	model.eval()
	return model

@torch.no_grad()
def predict_depth(model, wave_input):
	"""Convert predicted heatmap to depth index via soft-argmax.

	wave_input: (B, H, W, T) float tensor
	return: (B, H, W) float indices
	"""
	x = wave_input.unsqueeze(1)  # (B,1,H,W,T)
	heat = model(x)              # (B,1,H,W,T)

	# Soft-argmax over time axis T
	heat = heat.squeeze(1)       # (B,H,W,T)
	prob = torch.softmax(heat, dim=-1)
	T = prob.shape[-1]
	t = torch.arange(T, device=prob.device, dtype=prob.dtype).view(1, 1, 1, T)
	r_pred = (prob * t).sum(dim=-1)  # (B,H,W)
	r_pred = DeepImgDataset.wave_pos2real_depth(r_pred)
	return r_pred


@torch.no_grad()
def evaluate(model, loader):
	total_loss = 0.0
	n_batches = 0
	for batch in loader:
		wave = batch["input"].to(DEVICE)   # (B,H,W,T)
		r = batch["target"].to(DEVICE)     # (B,H,W)

		x = wave.unsqueeze(1)               # (B,1,H,W,T)
		s_gt = DeepImgDataset.defects_to_waves(
            r,
            TGT_KERNEL,
            receptive_field_size=r.shape[1],
        ).unsqueeze(1)                     # (B,1,H,W,T)
		pred = model(x)
		loss = heatmap_loss(pred, s_gt)
		total_loss += float(loss.item())
		n_batches += 1

	return total_loss / max(n_batches, 1)


@torch.no_grad()
def visualize_samples(model, dataset, device):
	"""Visualize sample predictions: 2D maps and 3D surfaces."""
	# Show one 2D comparison
	fig1 = plt.figure(figsize=(12, 6))
	sample = dataset[0]
	wave_input = sample["input"].unsqueeze(0).to(device)  # (1,H,W,T)
	r_target = sample["target"].cpu().numpy()
	r_pred = predict_depth(model, wave_input).squeeze(0).cpu().numpy()
	fig1.add_subplot(1, 2, 1)
	im0 = plt.imshow(r_target, cmap='viridis')
	plt.title(f"Ground Truth (max: {r_target.max():.2f})")
	plt.axis('off')
	plt.colorbar(im0, ax=plt.gca())
	fig1.add_subplot(1, 2, 2)
	im1 = plt.imshow(r_pred, cmap='viridis')
	plt.title(f"Prediction (max: {r_pred.max():.2f})")
	plt.axis('off')
	plt.colorbar(im1, ax=plt.gca())
	plt.tight_layout()

	# Show 3D surfaces for first 3 samples
	fig2 = plt.figure(figsize=(18, 12))
	for i in range(3):
		sample_i = dataset[i]
		wave_i = sample_i["input"].unsqueeze(0).to(device)
		r_t = sample_i["target"].cpu().numpy()
		r_p = predict_depth(model, wave_i).squeeze(0).cpu().numpy()

		h, w = r_t.shape
		x = np.arange(0, w, 1)
		y = np.arange(0, h, 1)
		X, Y = np.meshgrid(x, y)

		ax_t = fig2.add_subplot(2, 3, i+1, projection='3d')
		surf_t = ax_t.plot_surface(X, Y, r_t, cmap='viridis',
								   edgecolor='none', alpha=0.9, rstride=1, cstride=1)
		ax_t.set_title(f"Sample {i+1}: Ground Truth")
		ax_t.set_xlabel('X')
		ax_t.set_ylabel('Y')
		ax_t.set_zlabel('Index')
		ax_t.view_init(elev=30, azim=45)
		fig2.colorbar(surf_t, ax=ax_t, shrink=0.5, aspect=5)

		ax_p = fig2.add_subplot(2, 3, i+4, projection='3d')
		surf_p = ax_p.plot_surface(X, Y, r_p, cmap='viridis',
								   edgecolor='none', alpha=0.9, rstride=1, cstride=1)
		ax_p.set_title(f"Sample {i+1}: Prediction")
		ax_p.set_xlabel('X')
		ax_p.set_ylabel('Y')
		ax_p.set_zlabel('Index')
		ax_p.view_init(elev=30, azim=45)
		fig2.colorbar(surf_p, ax=ax_p, shrink=0.5, aspect=5)

	plt.tight_layout()
	plt.show()


def main():
	dataset = load_dataset()
	model = load_model()

	# Small loader for evaluation
	loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
	loss = evaluate(model, loader)
	print(f"Test Heatmap MSE Loss: {loss:.6f}")

	visualize_samples(model, dataset, DEVICE)


if __name__ == "__main__":
	main()

