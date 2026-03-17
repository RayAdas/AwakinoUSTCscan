import numpy as np
import cv2
from pathlib import Path


def compute_metrics(pred_depth_mm: np.ndarray, gt_depth_mm: np.ndarray) -> dict:
    """Compute evaluation metrics between predicted and ground-truth depth maps."""
    diff = pred_depth_mm - gt_depth_mm

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    abs_max_err = float(np.max(np.abs(diff)))

    # Edge Sharpness
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


def solve_linear_fit(depth_map: np.ndarray, mask: np.ndarray):
    """
    Solve min ||a * depth_map + b - mask||^2
    """
    x = depth_map.reshape(-1)
    y = mask.reshape(-1)

    A = np.vstack([x, np.ones_like(x)]).T
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    a, b = sol
    return float(a), float(b)


def map_mask(mask: np.ndarray):
    """
    Map mask values 0~255 -> 0~0.012 linearly
    """
    return mask.astype(np.float32) / 255.0 * 0.012


def main():

    data_dir = Path("./data/NpWaveData/20250716_151236")

    depth_path = data_dir / "depth_map.jpg"
    mask_path = data_dir / "mask.png"

    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)

    if depth_map.shape != mask.shape:
        raise ValueError("Image shapes do not match")

    # -------- mask mapping --------
    mask_mapped = map_mask(mask)

    # -------- optimization --------
    a, b = solve_linear_fit(depth_map, mask_mapped)

    img_best_match = a * depth_map + b

    # -------- metrics --------
    metrics = compute_metrics(img_best_match, mask_mapped)

    print("Optimization result")
    print("-------------------")
    print(f"a = {a}")
    print(f"b = {b}")

    print("\nMetrics")
    print("-------------------")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # optional save
    np.save(data_dir / "img_best_match.npy", img_best_match)


if __name__ == "__main__":
    main()