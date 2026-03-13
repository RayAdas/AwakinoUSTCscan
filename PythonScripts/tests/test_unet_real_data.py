import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from build_real_dataset import (
    HIGH_CUT_HZ,
    LOW_CUT_HZ,
    SAMPLE_RATE_HZ,
    TARGET_T,
    analytic_envelope_torch,
    bandpass_fft_torch,
    load_depth_from_mask,
    remap_t_axis_by_physical_depth,
)
from rebuild.dataset import DeepImgDataset
from rebuild.unet import UNet3D
from utils.file_io import FileIO


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test UNet model on real C-scan waveform data.")
    parser.add_argument("--window", type=int, default=41, help="Sliding window size in H/W.")
    parser.add_argument("--stride", type=int, default=20, help="Sliding stride in H/W.")
    parser.add_argument("--target-t", type=int, default=TARGET_T, help="Target T length after interpolation.")
    parser.add_argument(
        "--enhance-alpha",
        type=float,
        default=40.0,
        help="Exponential enhancement strength after T-axis mapping and before scaling.",
    )
    parser.add_argument("--save-name", type=str, default="real_data_depth_prediction.png", help="Output figure name.")
    parser.add_argument(
        "--save-3d-name",
        type=str,
        default="real_data_depth_prediction_3d.png",
        help="Output 3D figure name (shown after main window is closed).",
    )
    parser.add_argument("--waveform-save-name", type=str, default="real_data_waveform_samples.png", help="Output waveform comparison figure name.")
    parser.add_argument("--n-samples", type=int, default=6, help="Number of random (H, W) points for waveform comparison.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling points.")
    parser.add_argument("--no-show", action="store_true", help="Do not display matplotlib window.")
    return parser.parse_args()


def load_ground_truth_depth(h: int, w: int) -> np.ndarray:
    """Load ground-truth depth from mask.png and return it in millimetres."""
    mask_path = os.path.join(FileIO.curr_CS_data_path, "mask.png")
    depth_m = load_depth_from_mask(mask_path, h, w)
    return depth_m.astype(np.float32)


def compute_metrics(pred_depth_mm: np.ndarray, gt_depth_mm: np.ndarray) -> dict:
    """Compute evaluation metrics between predicted and ground-truth depth maps (both in mm)."""
    diff = pred_depth_mm - gt_depth_mm
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    abs_max_err = float(np.max(np.abs(diff)))

    # Edge Sharpness: mean gradient magnitude of predicted depth
    # evaluated over pixels that lie in ground-truth edge regions
    # (gradient magnitude > mean gt gradient).
    gy_gt, gx_gt = np.gradient(gt_depth_mm)
    grad_mag_gt = np.hypot(gx_gt, gy_gt)
    edge_mask = grad_mag_gt > grad_mag_gt.mean()

    gy_pred, gx_pred = np.gradient(pred_depth_mm)
    grad_mag_pred = np.hypot(gx_pred, gy_pred)
    es = float(grad_mag_pred[edge_mask].mean()) if edge_mask.any() else float(grad_mag_pred.mean())

    return {
        "MAE (mm)": mae,
        "RMSE (mm)": rmse,
        "Max Abs Error (mm)": abs_max_err,
        "Edge Sharpness (mm/px)": es,
    }


def print_metrics(metrics: dict) -> None:
    print("\n=== Evaluation Metrics ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("==========================\n")


def load_real_data() -> np.ndarray:
    FileIO.init()  # 初始化文件路径和配置
    waveform_data = np.load(os.path.join(FileIO.curr_CS_data_path, "waveform_data.npy"))
    if waveform_data.ndim != 3:
        raise ValueError(f"Expected waveform_data with shape (H, W, T), got {waveform_data.shape}")
    return waveform_data.astype(np.float32)


def preprocess_to_envelope(
    waveform_data: np.ndarray,
) -> np.ndarray:
    data = torch.from_numpy(waveform_data.astype(np.float32, copy=False)).to(DEVICE)
    data = data - data.mean(dim=-1, keepdim=True)
    filtered = bandpass_fft_torch(data, SAMPLE_RATE_HZ, LOW_CUT_HZ, HIGH_CUT_HZ)
    envelope = analytic_envelope_torch(filtered)
    return envelope.detach().cpu().numpy().astype(np.float32, copy=False)


def remap_t_axis(data: np.ndarray, target_t: int) -> np.ndarray:
    data_tensor = torch.from_numpy(data.astype(np.float32, copy=False)).to(DEVICE)
    out = remap_t_axis_by_physical_depth(data_tensor, target_t)
    return out.detach().cpu().numpy().astype(np.float32, copy=False)

def scale_each_waveform_to_max(data_hwt: np.ndarray, target_max: float) -> np.ndarray:
    if target_max < 0:
        raise ValueError("target_max must be non-negative")

    max_per_wave = np.max(data_hwt, axis=-1, keepdims=True)
    scale = np.where(max_per_wave > 1e-12, target_max / max_per_wave, 0.0).astype(np.float32)
    out = data_hwt * scale
    return np.maximum(out, 0.0).astype(np.float32, copy=False)


def enhance_waveforms_exponential(data_hwt: np.ndarray, alpha: float) -> np.ndarray:
    # data_tensor = torch.from_numpy(data_hwt).to(DEVICE)
    # enhanced = torch.softmax(data_tensor * alpha, dim=-1) * data_tensor
    # return enhanced.cpu().numpy().astype(np.float32)
    return data_hwt


def load_model() -> UNet3D:
    model = UNet3D(in_ch=1).to(DEVICE)
    ckpt_path = FileIO.rebuild_model_path
    if ckpt_path is None:
        raise ValueError("No rebuild model selected. Check [ModelSelect] CurrentRebuildModel in config.ini.")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


@torch.no_grad()
def predict_depth(model: UNet3D, wave_input: torch.Tensor) -> torch.Tensor:
    x = wave_input.unsqueeze(1)  # (B,1,H,W,T)

    heat = model(x).squeeze(1)   # (B,H,W,T)
    heat = heat.clamp(min=0)
    # heat = x.squeeze(1)

    beta = 1e1
    prob = torch.softmax(heat * beta, dim=-1)

    t = torch.arange(prob.shape[-1], device=prob.device, dtype=prob.dtype).view(1, 1, 1, -1)
    pred_pos = (prob * t).sum(dim=-1)  # (B,H,W)
    pred_depth = DeepImgDataset.wave_pos2real_depth(pred_pos)
    # pred_depth = pred_depth * 1000
    return pred_depth

def generate_starts(size: int, window: int, stride: int) -> List[int]:
    if size <= window:
        return [0]

    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def sliding_window_predict(
    model: UNet3D,
    data_hwt: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    h, w, t = data_hwt.shape

    pad_h = max(0, window - h)
    pad_w = max(0, window - w)
    if pad_h > 0 or pad_w > 0:
        data_hwt = np.pad(data_hwt, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    hp, wp, _ = data_hwt.shape
    acc = np.zeros((hp, wp), dtype=np.float32)
    cnt = np.zeros((hp, wp), dtype=np.float32)

    h_starts = generate_starts(hp, window, stride)
    w_starts = generate_starts(wp, window, stride)

    for hs in h_starts:
        for ws in w_starts:
            patch = data_hwt[hs:hs + window, ws:ws + window, :]
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
            pred_patch = predict_depth(model, patch_tensor).squeeze(0).cpu().numpy().astype(np.float32)

            acc[hs:hs + window, ws:ws + window] += pred_patch
            cnt[hs:hs + window, ws:ws + window] += 1.0

    cnt = np.maximum(cnt, 1.0)
    pred_full = acc / cnt
    return pred_full[:h, :w]


@torch.no_grad()
def predict_patch_output_waveform(model: UNet3D, patch_hwt: np.ndarray) -> np.ndarray:
    patch_tensor = torch.from_numpy(patch_hwt).unsqueeze(0).to(DEVICE)  # (1,H,W,T)
    x = patch_tensor.unsqueeze(1)  # (1,1,H,W,T)
    heat = model(x).squeeze(0).squeeze(0)  # (H,W,T)
    heat = heat.clamp(min=0)
    prob = torch.softmax(heat * 1e1, dim=-1)
    return prob.detach().cpu().numpy().astype(np.float32)


def extract_point_output_waveform(
    model: UNet3D,
    data_hwt: np.ndarray,
    y: int,
    x: int,
    window: int,
) -> np.ndarray:
    h, w, _ = data_hwt.shape
    half = window // 2

    hs = int(np.clip(y - half, 0, max(0, h - window)))
    ws = int(np.clip(x - half, 0, max(0, w - window)))
    patch = data_hwt[hs:hs + window, ws:ws + window, :]

    ph = max(0, window - patch.shape[0])
    pw = max(0, window - patch.shape[1])
    if ph > 0 or pw > 0:
        patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)), mode="edge")

    out_prob = predict_patch_output_waveform(model, patch)
    ly = int(np.clip(y - hs, 0, out_prob.shape[0] - 1))
    lx = int(np.clip(x - ws, 0, out_prob.shape[1] - 1))
    return out_prob[ly, lx, :]


def choose_random_points(h: int, w: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = h * w
    n_pick = max(1, min(n_samples, total))
    flat_idx = rng.choice(total, size=n_pick, replace=False)
    ys = flat_idx // w
    xs = flat_idx % w
    return np.stack([ys, xs], axis=1)


def extract_input_output_waveforms(
    model: UNet3D,
    data_hwt: np.ndarray,
    points_yx: np.ndarray,
    window: int,
) -> List[dict]:
    h, w, _ = data_hwt.shape
    half = window // 2
    records: List[dict] = []

    for y, x in points_yx:
        hs = int(np.clip(y - half, 0, max(0, h - window)))
        ws = int(np.clip(x - half, 0, max(0, w - window)))
        patch = data_hwt[hs:hs + window, ws:ws + window, :]

        # Pad if H/W is smaller than window.
        ph = max(0, window - patch.shape[0])
        pw = max(0, window - patch.shape[1])
        if ph > 0 or pw > 0:
            patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)), mode="edge")

        out_prob = predict_patch_output_waveform(model, patch)
        ly = int(y - hs)
        lx = int(x - ws)
        ly = int(np.clip(ly, 0, out_prob.shape[0] - 1))
        lx = int(np.clip(lx, 0, out_prob.shape[1] - 1))

        in_wave = data_hwt[int(y), int(x), :]
        out_wave = out_prob[ly, lx, :]
        records.append({"y": int(y), "x": int(x), "input": in_wave, "output": out_wave})

    return records


def visualize_random_waveform_pairs(records: List[dict], save_path: str, show: bool) -> None:
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(12, max(3 * n, 4)), squeeze=False)

    for i, rec in enumerate(records):
        y, x = rec["y"], rec["x"]
        in_wave = rec["input"]
        out_wave = rec["output"]

        axes[i, 0].plot(in_wave, "b-", linewidth=1.6)
        axes[i, 0].set_title(f"Input Envelope @ (H={y}, W={x})")
        axes[i, 0].set_xlabel("T")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.25)

        axes[i, 1].plot(out_wave, "r-", linewidth=1.6)
        axes[i, 1].set_title(f"Model Output Waveform @ (H={y}, W={x})")
        axes[i, 1].set_xlabel("T")
        axes[i, 1].set_ylabel("Probability")
        axes[i, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved waveform comparison to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_result(pred_depth: np.ndarray, save_path: str, show: bool) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pred_depth, cmap="viridis")
    plt.colorbar(im, label="Predicted Depth (m)")
    plt.title("UNet Prediction on Real Data")
    plt.xlabel("W")
    plt.ylabel("H")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def visualize_hover_waveform_view(
    pred_depth: np.ndarray,
    raw_hwt_128: np.ndarray,
    model_input_hwt: np.ndarray,
    model: UNet3D,
    window: int,
    save_path: str,
    show: bool,
) -> None:
    h, w, _ = model_input_hwt.shape
    fig, (ax_map, ax_raw, ax_out) = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.2, 1, 1]})

    im = ax_map.imshow(pred_depth, cmap="viridis")
    fig.colorbar(im, ax=ax_map, label="Predicted Depth (m)")
    ax_map.set_title("UNet Prediction (Hover to Inspect)")
    ax_map.set_xlabel("W")
    ax_map.set_ylabel("H")

    init_y, init_x = 0, 0
    marker = ax_map.scatter([init_x], [init_y], c="r", s=18)

    raw_line, = ax_raw.plot(raw_hwt_128[init_y, init_x, :], "b-", linewidth=1.6)
    ax_raw.set_title(f"Raw Waveform (128) @ (H={init_y}, W={init_x})")
    ax_raw.set_xlabel("T")
    ax_raw.set_ylabel("Amplitude")
    ax_raw.grid(True, alpha=0.25)

    init_out = extract_point_output_waveform(model, model_input_hwt, init_y, init_x, window)
    out_line, = ax_out.plot(init_out, "r-", linewidth=1.6)
    ax_out.set_title(f"Model Output @ (H={init_y}, W={init_x})")
    ax_out.set_xlabel("T")
    ax_out.set_ylabel("Probability")
    ax_out.grid(True, alpha=0.25)

    ax_raw.relim()
    ax_raw.autoscale_view()
    ax_out.relim()
    ax_out.autoscale_view()

    point_cache = {(init_y, init_x): init_out}
    state = {"last": (init_y, init_x)}

    def on_move(event) -> None:
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return

        x = int(np.clip(round(event.xdata), 0, w - 1))
        y = int(np.clip(round(event.ydata), 0, h - 1))
        if state["last"] == (y, x):
            return
        state["last"] = (y, x)

        marker.set_offsets(np.array([[x, y]], dtype=np.float32))

        raw_wave = raw_hwt_128[y, x, :]
        raw_line.set_ydata(raw_wave)
        ax_raw.set_title(f"Raw Waveform (128) @ (H={y}, W={x})")
        ax_raw.relim()
        ax_raw.autoscale_view()

        key = (y, x)
        if key not in point_cache:
            point_cache[key] = extract_point_output_waveform(model, model_input_hwt, y, x, window)
        out_wave = point_cache[key]
        out_line.set_ydata(out_wave)
        ax_out.set_title(f"Model Output @ (H={y}, W={x})")
        ax_out.relim()
        ax_out.autoscale_view()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved interactive figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_total_depth_3d(pred_depth: np.ndarray, save_path: str, show: bool) -> None:
    h, w = pred_depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z_min = float(np.min(pred_depth))
    z_max = float(np.max(pred_depth))
    z_span = max(z_max - z_min, 1e-6)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        x,
        y,
        pred_depth,
        cmap="viridis",
        edgecolor="none",
        alpha=0.95,
        rstride=2,
        cstride=2,
    )
    ax.set_title("Predicted Depth 3D Surface")
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    ax.set_zlabel("Depth (m)")
    # Keep X/Y/Z rendered with a unified data scale.
    ax.set_box_aspect((max(w - 1, 1), max(h - 1, 1), z_span))
    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.7, aspect=12, label="Depth (m)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved 3D figure to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()

    print("Loading real waveform data...")
    waveform_data = load_real_data()
    print(f"Raw data shape: {waveform_data.shape}")

    print("Preprocessing: DC removal + low-pass + envelope...")
    envelope = preprocess_to_envelope(waveform_data)
    print(
        f"Envelope shape: {envelope.shape}, min={envelope.min():.6f}, max={envelope.max():.6f}"
    )

    print(f"Remapping T axis with dataset physical-depth mapping to {args.target_t}...")
    model_input = remap_t_axis(envelope, target_t=args.target_t)
    print(f"Enhancing waveform contrast with exponential alpha={args.enhance_alpha}...")
    model_input = enhance_waveforms_exponential(model_input, alpha=args.enhance_alpha)
    print(f"Model input shape: {model_input.shape}")

    print("Loading model...")
    model = load_model()

    print(f"Sliding-window inference (window={args.window}, stride={args.stride})...")
    pred_depth = sliding_window_predict(
        model=model,
        data_hwt=model_input,
        window=args.window,
        stride=args.stride,
    )

    print("Computing evaluation metrics...")
    h_out, w_out = pred_depth.shape
    try:
        gt_depth_mm = load_ground_truth_depth(h_out, w_out)
        metrics = compute_metrics(pred_depth, gt_depth_mm)
        print_metrics(metrics)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Warning: Could not compute evaluation metrics: {exc}")

    save_path = os.path.join(FileIO.curr_CS_data_path, args.save_name)
    visualize_hover_waveform_view(
        pred_depth=pred_depth,
        raw_hwt_128=model_input,
        model_input_hwt=model_input,
        model=model,
        window=args.window,
        save_path=save_path,
        show=not args.no_show,
    )

    # Show the 3D total depth map after the main interactive window is closed.
    save_3d_path = os.path.join(FileIO.curr_CS_data_path, args.save_3d_name)
    visualize_total_depth_3d(
        pred_depth=pred_depth,
        save_path=save_3d_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
