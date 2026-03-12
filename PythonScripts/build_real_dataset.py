import os
import time

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn.functional as F

from rebuild.dataset import DeepImgDataset
from utils.file_io import FileIO


SAMPLE_RATE_HZ = 5.0e8
LOW_CUT_HZ = 1.0e6
HIGH_CUT_HZ = 50.0e6
WINDOW_SIZE = 41
TARGET_T = 128

# Index-depth anchors for the original T axis.
ANCHOR_INDEX_DEPTH_0 = 458.0
ANCHOR_DEPTH_0 = 0.0
ANCHOR_INDEX_DEPTH_1 = 431.0
ANCHOR_DEPTH_1 = 1.38e-3

# Physical depth range to keep before resampling.
DEPTH_RANGE_MIN = -2e-3
DEPTH_RANGE_MAX = 12e-3
MAX_DEPTH_FROM_MASK = 0.012


def analytic_envelope_torch(x: torch.Tensor) -> torch.Tensor:
    """Compute envelope via analytic signal (Hilbert transform) on last axis."""
    n = x.shape[-1]
    x_fft = torch.fft.fft(x, dim=-1)

    h = torch.zeros(n, dtype=x_fft.dtype, device=x.device)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    analytic = torch.fft.ifft(x_fft * h, dim=-1)
    return analytic.abs()


def bandpass_fft_torch(x: torch.Tensor, sample_rate_hz: float, low_cut_hz: float, high_cut_hz: float) -> torch.Tensor:
    """Band-pass filter using frequency-domain masking on the last axis."""
    t_len = x.shape[-1]
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(t_len, d=1.0 / sample_rate_hz).to(x.device)
    band_mask = (freqs >= low_cut_hz) & (freqs <= high_cut_hz)
    spec_filtered = spec * band_mask
    return torch.fft.irfft(spec_filtered, n=t_len, dim=-1)


def load_depth_from_mask(mask_path: str, h: int, w: int) -> np.ndarray:
    mask = mpimg.imread(mask_path)

    if mask.ndim == 3:
        rgb = mask[..., :3]
        if not (np.allclose(rgb[..., 0], rgb[..., 1], atol=1e-6) and np.allclose(rgb[..., 0], rgb[..., 2], atol=1e-6)):
            raise ValueError("mask.png RGB channels must be identical for grayscale convention")
        gray = rgb[..., 0]
    elif mask.ndim == 2:
        gray = mask
    else:
        raise ValueError(f"Unsupported mask image shape: {mask.shape}")

    if gray.shape != (h, w):
        raise ValueError(f"mask.png shape {gray.shape} does not match waveform (H, W)=({h}, {w})")

    gray = gray.astype(np.float32, copy=False)
    if gray.max() <= 1.0 + 1e-6:
        gray = gray * 255.0

    return gray / 256.0 * MAX_DEPTH_FROM_MASK


def _index_to_depth(index: torch.Tensor) -> torch.Tensor:
    slope = (ANCHOR_DEPTH_1 - ANCHOR_DEPTH_0) / (ANCHOR_INDEX_DEPTH_1 - ANCHOR_INDEX_DEPTH_0)
    intercept = ANCHOR_DEPTH_0 - slope * ANCHOR_INDEX_DEPTH_0
    return slope * index + intercept


def _depth_to_index(depth: torch.Tensor) -> torch.Tensor:
    slope = (ANCHOR_DEPTH_1 - ANCHOR_DEPTH_0) / (ANCHOR_INDEX_DEPTH_1 - ANCHOR_INDEX_DEPTH_0)
    intercept = ANCHOR_DEPTH_0 - slope * ANCHOR_INDEX_DEPTH_0
    return (depth - intercept) / slope


def remap_t_axis_by_physical_depth(envelope: torch.Tensor, target_t: int) -> torch.Tensor:
    """Map original T axis to target_t using physical depth anchors and depth-range interpolation."""
    h, w, t = envelope.shape
    device = envelope.device

    idx_min = _depth_to_index(torch.tensor(DEPTH_RANGE_MIN, device=device, dtype=envelope.dtype))
    idx_max = _depth_to_index(torch.tensor(DEPTH_RANGE_MAX, device=device, dtype=envelope.dtype))

    idx_start = torch.minimum(idx_min, idx_max)
    idx_end = torch.maximum(idx_min, idx_max)
    if idx_end < 0 or idx_start > (t - 1):
        raise ValueError(f"Depth range [{DEPTH_RANGE_MIN}, {DEPTH_RANGE_MAX}] has no overlap with source T-axis")

    src_positions = torch.linspace(
        _depth_to_index(torch.tensor(DeepImgDataset.DEPTH_MIN, device=device, dtype=envelope.dtype)),
        _depth_to_index(torch.tensor(DeepImgDataset.DEPTH_MAX, device=device, dtype=envelope.dtype)),
        target_t,
        device=device,
        dtype=envelope.dtype,
    ).clamp(0, t - 1)

    i0 = torch.floor(src_positions).long()
    i1 = torch.clamp(i0 + 1, max=t - 1)
    w1 = (src_positions - i0.to(src_positions.dtype)).unsqueeze(0)
    w0 = 1.0 - w1

    flat = envelope.reshape(-1, t)
    sampled = flat[:, i0] * w0 + flat[:, i1] * w1
    return sampled.reshape(h, w, target_t)


def to_windows_hwt(data_hwt: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert (H, W, T) to (N, window_size, window_size, T) with stride=1."""
    h, w, t = data_hwt.shape
    if h < window_size or w < window_size:
        raise ValueError(f"Input size {(h, w)} is smaller than window size {window_size}")

    x = data_hwt.permute(2, 0, 1).unsqueeze(0)  # (1,T,H,W)
    patches = F.unfold(x, kernel_size=(window_size, window_size), stride=1)
    n = patches.shape[-1]
    return patches.squeeze(0).transpose(0, 1).reshape(n, t, window_size, window_size).permute(0, 2, 3, 1).contiguous()


def to_windows_hw(depth_hw: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert (H, W) to (N, window_size, window_size) with stride=1."""
    h, w = depth_hw.shape
    if h < window_size or w < window_size:
        raise ValueError(f"Depth size {(h, w)} is smaller than window size {window_size}")

    y = depth_hw.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    patches = F.unfold(y, kernel_size=(window_size, window_size), stride=1)
    n = patches.shape[-1]
    return patches.squeeze(0).transpose(0, 1).reshape(n, window_size, window_size).contiguous()


def build_dataset_from_real_data() -> DeepImgDataset:
    FileIO.init()
    if FileIO.curr_CS_data_path is None:
        raise ValueError("No current C-scan database selected in config.ini")

    waveform_path = os.path.join(FileIO.curr_CS_data_path, "waveform_data.npy")
    mask_path = os.path.join(FileIO.curr_CS_data_path, "mask.png")

    waveform_np = np.load(waveform_path)
    if waveform_np.ndim != 3:
        raise ValueError(f"Expected waveform_data shape (H, W, T), got {waveform_np.shape}")

    h, w, t = waveform_np.shape
    if t < 2:
        raise ValueError(f"T dimension must be >= 2, got {t}")

    src_idx = torch.arange(t, dtype=torch.float32)
    src_depth = _index_to_depth(src_idx)
    d_min = float(src_depth.min().item())
    d_max = float(src_depth.max().item())
    if DEPTH_RANGE_MAX < d_min or DEPTH_RANGE_MIN > d_max:
        raise ValueError(
            f"Depth range [{DEPTH_RANGE_MIN}, {DEPTH_RANGE_MAX}] is outside source mapped depth range [{d_min}, {d_max}]"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform = torch.from_numpy(waveform_np.astype(np.float32)).to(device)

    waveform = waveform - waveform.mean(dim=-1, keepdim=True)
    waveform = bandpass_fft_torch(waveform, SAMPLE_RATE_HZ, LOW_CUT_HZ, HIGH_CUT_HZ)
    envelope = analytic_envelope_torch(waveform)

    mapped = remap_t_axis_by_physical_depth(envelope, TARGET_T)

    depth_np = load_depth_from_mask(mask_path, h, w)
    depth_map = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)

    input_windows = to_windows_hwt(mapped, WINDOW_SIZE)
    target_windows = to_windows_hw(depth_map, WINDOW_SIZE)

    dataset = DeepImgDataset.__new__(DeepImgDataset)
    dataset.input = input_windows.cpu()
    dataset.tgt = target_windows.cpu()
    dataset.n_samples = dataset.input.shape[0]
    dataset.defects_meta = [None] * dataset.n_samples
    return dataset


def save_dataset(dataset: DeepImgDataset) -> str:
    FileIO.init()
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    dataset_name = f"real_{dataset.n_samples}_{timestamp}"
    save_dir = os.path.join(FileIO.rebuild_dataset_path, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "dataset.pt")
    torch.save(dataset, save_path)
    return save_path


if __name__ == "__main__":
    dataset = build_dataset_from_real_data()
    save_path = save_dataset(dataset)
    print(f"Saved dataset to: {save_path}")
    print(f"input shape: {tuple(dataset.input.shape)}")
    print(f"target shape: {tuple(dataset.tgt.shape)}")
