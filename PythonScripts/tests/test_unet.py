import os
import tkinter as tk
from tkinter import ttk

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

    heat = heat.clamp(min=0)

    beta = 1e1
    prob = torch.softmax(heat * beta, dim=-1)  # (B,H,W,T)

    T = prob.shape[-1]
    t = torch.arange(T, device=prob.device, dtype=prob.dtype).view(1, 1, 1, T)
    r_pred = (prob * t).sum(dim=-1)  # (B,H,W)
    r_pred = DeepImgDataset.wave_pos2real_depth(r_pred)
    return r_pred


class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D UNet 深度预测可视化")
        self.root.geometry("1600x1000")
        
        # 加载数据
        print("加载数据集...")
        self.dataset = load_dataset()
        print("数据集加载完成")
        self.defects_meta = getattr(self.dataset, "defects_meta", None)
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=1,  # 每次一个样本
            shuffle=False,
            num_workers=0
        )
        self.data_iterator = iter(self.dataloader)
        
        # 加载模型
        print("加载模型...")
        self.model = load_model()
        print("模型加载完成")
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 初始化变量
        self.current_idx = 0
        self.H_idx = 15  # 默认H位置
        self.W_idx = 20  # 默认W位置
        
        # 初始化image/surface和colorbar成员变量
        self.im_2d_t = None
        self.cbar_2d_t = None
        self.im_2d_p = None
        self.cbar_2d_p = None
        self.surf_3d_t = None
        self.cbar_3d_t = None
        self.surf_3d_p = None
        self.cbar_3d_p = None
        
        # 存储所有预测结果
        print("开始推理...")
        self.all_predictions = []
        self.all_targets = []
        self.all_waves = []
        self.all_heatmaps = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                (waves, targets) = (batch["input"], batch["target"])
                waves = waves.to(self.device)
                predictions = predict_depth(self.model, waves)
                self.all_predictions.append(predictions.cpu())
                self.all_targets.append(targets)
                self.all_waves.append(waves.cpu())
                self.all_heatmaps.append(self.model(waves.unsqueeze(1)).squeeze(1).cpu())
                if batch_idx % 10 == 0:
                    print(f"已处理 {batch_idx+1}/{len(self.dataloader)} 个样本")
        
        print("推理完成")
        
        # 转换为numpy数组
        self.all_predictions = torch.cat(self.all_predictions, dim=0).numpy()
        self.all_targets = torch.cat(self.all_targets, dim=0).numpy()
        self.all_waves = torch.cat(self.all_waves, dim=0).numpy()
        self.all_heatmaps = torch.cat(self.all_heatmaps, dim=0).numpy()

        print("计算评价指标...")
        self.compute_and_print_metrics()

        # 创建UI
        self.create_widgets()
        
        # 显示第一帧
        self.update_display()
    
    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ==========================================================
        # 修改 1: 先创建并打包底部的控制面板 (Control Frame)
        # 这样可以保证底部按钮区优先获得空间，不会被图表挤出去
        # ==========================================================
        control_frame = ttk.Frame(main_frame)
        # 注意：这里先pack，占据底部空间
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=5, pady=(10, 0))
        
        # --- 将原有的控件代码移动到这里 ---
        
        # 创建一个子框架来更好地组织控件
        control_subframe = ttk.Frame(control_frame)
        control_subframe.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # H坐标输入
        h_frame = ttk.Frame(control_subframe)
        h_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(h_frame, text="H坐标:").pack(side=tk.LEFT, padx=(0, 5))
        self.h_entry = ttk.Entry(h_frame, width=10)
        self.h_entry.insert(0, str(self.H_idx))
        self.h_entry.pack(side=tk.LEFT)
        
        # W坐标输入
        w_frame = ttk.Frame(control_subframe)
        w_frame.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(w_frame, text="W坐标:").pack(side=tk.LEFT, padx=(0, 5))
        self.w_entry = ttk.Entry(w_frame, width=10)
        self.w_entry.insert(0, str(self.W_idx))
        self.w_entry.pack(side=tk.LEFT)
        
        # 更新按钮
        self.update_btn = ttk.Button(
            control_subframe, 
            text="更新坐标", 
            command=self.update_coordinates
        )
        self.update_btn.pack(side=tk.LEFT, padx=20, pady=5)
        
        # 上一帧按钮
        self.prev_btn = ttk.Button(
            control_subframe, 
            text="上一帧", 
            command=self.prev_frame
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 下一帧按钮
        self.next_btn = ttk.Button(
            control_subframe, 
            text="下一帧", 
            command=self.next_frame
        )
        self.next_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 帧信息标签
        self.frame_label = ttk.Label(control_frame, text="")
        self.frame_label.pack(side=tk.RIGHT, padx=20, pady=5)

        # 缺陷类型标签
        self.defect_label = ttk.Label(control_frame, text="Defect: N/A")
        self.defect_label.pack(side=tk.RIGHT, padx=20, pady=5)

        # 帧号跳转输入
        jump_frame = ttk.Frame(control_subframe)
        jump_frame.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Label(jump_frame, text="跳转帧:").pack(side=tk.LEFT, padx=(0, 5))
        self.frame_entry = ttk.Entry(jump_frame, width=10)
        self.frame_entry.insert(0, str(self.current_idx))
        self.frame_entry.pack(side=tk.LEFT)

        self.jump_btn = ttk.Button(
            jump_frame,
            text="跳转",
            command=self.jump_to_frame
        )
        self.jump_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # 绑定回车键
        self.h_entry.bind('<Return>', lambda e: self.update_coordinates())
        self.w_entry.bind('<Return>', lambda e: self.update_coordinates())
        self.frame_entry.bind('<Return>', lambda e: self.jump_to_frame())

        # ==========================================================
        # 修改 2: 后创建图表区域，并调整 Figure 尺寸
        # 它会占据主框架中“剩余”的所有空间
        # ==========================================================
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 修改 figsize：原来的 (15, 10) 太大了，改为 (12, 8) 或根据需要调整
        # dpi=100 时，(15, 10) = 1500x1000像素，这会把底部挤出去
        self.fig = Figure(figsize=(12, 8), dpi=100) 

        # ... (以下图表子图创建代码保持不变) ...
        # 上排：数据集三张图
        self.ax_2d_t = self.fig.add_subplot(2, 3, 1)
        self.ax_2d_t.set_title("Dataset 2D Ground Truth")
        self.ax_2d_t.axis('off')

        self.ax_3d_t = self.fig.add_subplot(2, 3, 2, projection='3d')
        self.ax_3d_t.set_title("Dataset 3D Ground Truth")

        self.ax_wave_in = self.fig.add_subplot(2, 3, 3)
        self.ax_wave_in.set_title(f"Dataset Input Waveform")
        self.ax_wave_in.set_xlabel("Time")
        self.ax_wave_in.set_ylabel("Amplitude")

        # 下排：预测三张图
        self.ax_2d_p = self.fig.add_subplot(2, 3, 4)
        self.ax_2d_p.set_title("Prediction 2D")
        self.ax_2d_p.axis('off')

        self.ax_3d_p = self.fig.add_subplot(2, 3, 5, projection='3d')
        self.ax_3d_p.set_title("Prediction 3D")

        self.ax_heatmap = self.fig.add_subplot(2, 3, 6)
        self.ax_heatmap.set_title("Prediction Heatmap")
        self.ax_heatmap.set_xlabel("Time")
        self.ax_heatmap.set_ylabel("Probability")
        
        self.fig.tight_layout(pad=3.0)
        
        # 将图表嵌入Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, chart_frame)
        toolbar.update()
    
    def update_coordinates(self):
        try:
            new_H = int(self.h_entry.get())
            new_W = int(self.w_entry.get())
            
            # 检查边界
            H_max = self.all_targets.shape[1] - 1
            W_max = self.all_targets.shape[2] - 1
            
            if 0 <= new_H <= H_max and 0 <= new_W <= W_max:
                self.H_idx = new_H
                self.W_idx = new_W
                self.update_display()
            else:
                print(f"坐标超出范围！H范围: 0-{H_max}, W范围: 0-{W_max}")
        except ValueError:
            print("请输入有效的整数坐标！")
    
    def prev_frame(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def next_frame(self):
        if self.current_idx < len(self.all_predictions) - 1:
            self.current_idx += 1
            self.update_display()

    def jump_to_frame(self):
        try:
            new_idx = int(self.frame_entry.get())
        except ValueError:
            print("请输入有效的整数帧号！")
            return

        max_idx = len(self.all_predictions) - 1
        if 0 <= new_idx <= max_idx:
            self.current_idx = new_idx
            self.update_display()
        else:
            print(f"帧号超出范围！有效范围: 0-{max_idx}")

    def get_current_defect_type(self):
        defects_meta = self.defects_meta
        if defects_meta is None:
            return "N/A"
        if not isinstance(defects_meta, (list, tuple)):
            return "N/A"
        if not (0 <= self.current_idx < len(defects_meta)):
            return "N/A"

        defect = defects_meta[self.current_idx]
        if defect is None:
            return "None"
        if isinstance(defect, str):
            return defect

        name = defect.__class__.__name__
        if name.endswith("DefectType"):
            name = name[:-10]
        return name
    
    def update_display(self):
        # 获取当前帧数据
        target = self.all_targets[self.current_idx]
        prediction = self.all_predictions[self.current_idx]
        wave = self.all_waves[self.current_idx]
        
        # 计算共同的颜色范围
        vmin = min(target.min(), prediction.min())
        vmax = max(target.max(), prediction.max())
        
        # 清空所有子图
        for ax in [self.ax_2d_t, self.ax_2d_p, self.ax_3d_t, 
                  self.ax_3d_p, self.ax_wave_in, self.ax_heatmap]:
            ax.clear()
        
        # ========== 1. 2D真实值 ==========
        im_t = self.ax_2d_t.imshow(target, cmap='viridis', vmin=vmin, vmax=vmax)
        self.ax_2d_t.set_title(f"Dataset 2D Ground Truth (帧: {self.current_idx}, max: {target.max():.2f})")
        self.ax_2d_t.axis('off')
        if self.im_2d_t is None:
            self.im_2d_t = im_t
            self.cbar_2d_t = self.fig.colorbar(im_t, ax=self.ax_2d_t, shrink=0.8)
        else:
            self.im_2d_t.set_data(target)
            self.im_2d_t.set_clim(vmin, vmax)
        
        # ========== 2. 2D预测值 ==========
        im_p = self.ax_2d_p.imshow(prediction, cmap='viridis', vmin=vmin, vmax=vmax)
        self.ax_2d_p.set_title(f"Prediction 2D (帧: {self.current_idx}, max: {prediction.max():.2f})")
        self.ax_2d_p.axis('off')
        if self.im_2d_p is None:
            self.im_2d_p = im_p
            self.cbar_2d_p = self.fig.colorbar(im_p, ax=self.ax_2d_p, shrink=0.8)
        else:
            self.im_2d_p.set_data(prediction)
            self.im_2d_p.set_clim(vmin, vmax)
        
        # ========== 3. 3D真实值 ==========
        H, W = target.shape
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        
        surf_t = self.ax_3d_t.plot_surface(
            X, Y, target, cmap='viridis', edgecolor='none', 
            alpha=0.9, rstride=2, cstride=2
        )
        self.ax_3d_t.set_title("Dataset 3D Ground Truth")
        self.ax_3d_t.set_xlabel('X')
        self.ax_3d_t.set_ylabel('Y')
        self.ax_3d_t.set_zlabel('Depth')
        self.ax_3d_t.view_init(elev=30, azim=45)
        if self.surf_3d_t is None:
            self.surf_3d_t = surf_t
            self.cbar_3d_t = self.fig.colorbar(surf_t, ax=self.ax_3d_t, shrink=0.6, aspect=8)
        else:
            self.surf_3d_t.set_array(target.flatten())
            self.surf_3d_t.set_clim(vmin, vmax)
        
        # ========== 4. 3D预测值 ==========
        surf_p = self.ax_3d_p.plot_surface(
            X, Y, prediction, cmap='viridis', edgecolor='none', 
            alpha=0.9, rstride=2, cstride=2
        )
        self.ax_3d_p.set_title("Prediction 3D")
        self.ax_3d_p.set_xlabel('X')
        self.ax_3d_p.set_ylabel('Y')
        self.ax_3d_p.set_zlabel('Depth')
        self.ax_3d_p.view_init(elev=30, azim=45)
        if self.surf_3d_p is None:
            self.surf_3d_p = surf_p
            self.cbar_3d_p = self.fig.colorbar(surf_p, ax=self.ax_3d_p, shrink=0.6, aspect=8)
        else:
            self.surf_3d_p.set_array(prediction.flatten())
            self.surf_3d_p.set_clim(vmin, vmax)
        
        # ========== 5. 输入波形 ==========
        if self.H_idx < H and self.W_idx < W:
            wave_at_point = wave[self.H_idx, self.W_idx, :]
            self.ax_wave_in.plot(wave_at_point, 'b-', linewidth=2)
            self.ax_wave_in.set_title(f"Dataset Input Waveform at (H={self.H_idx}, W={self.W_idx})")
            self.ax_wave_in.set_xlabel("Time")
            self.ax_wave_in.set_ylabel("Amplitude")
            self.ax_wave_in.grid(True, alpha=0.3)
        
        # ========== 6. 输出热图 ==========
        try:
            h_idx = int(self.h_entry.get())
            w_idx = int(self.w_entry.get())
            if 0 <= h_idx < self.all_heatmaps.shape[1] and 0 <= w_idx < self.all_heatmaps.shape[2]:
                heat_at_point = self.all_heatmaps[self.current_idx, h_idx, w_idx, :]
            else:
                heat_at_point = np.zeros(self.all_heatmaps.shape[3])
        except (ValueError, IndexError):
            heat_at_point = np.zeros(self.all_heatmaps.shape[3])
        
        self.ax_heatmap.plot(heat_at_point, 'r-', linewidth=2)
        self.ax_heatmap.set_title(f"Prediction Heatmap, truth:{DeepImgDataset.real_depth2wave_pos(target[self.H_idx, self.W_idx])}")
        self.ax_heatmap.set_xlabel("Time")
        self.ax_heatmap.set_ylabel("Probability")
        self.ax_heatmap.grid(True, alpha=0.3)
        
        
        # 更新帧信息
        self.frame_label.config(text=f"帧: {self.current_idx}/{len(self.all_predictions)-1}")
        self.defect_label.config(text=f"Defect: {self.get_current_defect_type()}")
        
        # 重新绘制
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
        
        # 更新输入框
        self.h_entry.delete(0, tk.END)
        self.h_entry.insert(0, str(self.H_idx))
        self.w_entry.delete(0, tk.END)
        self.w_entry.insert(0, str(self.W_idx))
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(self.current_idx))

    def compute_and_print_metrics(self) -> None:
        """Compute per-sample metrics and print the averages."""
        n = len(self.all_predictions)
        mae_sum = 0.0
        rmse_sum = 0.0
        max_err_sum = 0.0
        es_sum = 0.0

        for i in range(n):
            pred = self.all_predictions[i]   # (H, W)
            gt   = self.all_targets[i]       # (H, W)
            diff = pred - gt

            mae_sum     += float(np.mean(np.abs(diff)))
            rmse_sum    += float(np.sqrt(np.mean(diff ** 2)))
            max_err_sum += float(np.max(np.abs(diff)))

            gy_gt, gx_gt = np.gradient(gt)
            grad_mag_gt  = np.hypot(gx_gt, gy_gt)
            edge_mask    = grad_mag_gt > grad_mag_gt.mean()

            gy_pred, gx_pred = np.gradient(pred)
            grad_mag_pred    = np.hypot(gx_pred, gy_pred)
            es_sum += (
                float(grad_mag_pred[edge_mask].mean())
                if edge_mask.any()
                else float(grad_mag_pred.mean())
            )

        print(f"\n=== Evaluation Metrics (averaged over {n} samples) ===")
        print(f"  MAE:                  {mae_sum / n:.6f}")
        print(f"  RMSE:                 {rmse_sum / n:.6f}")
        print(f"  Max Abs Error (avg):  {max_err_sum / n:.6f}")
        print(f"  Edge Sharpness (ES):  {es_sum / n:.6f}")
        print("======================================================\n")


def main():
    root = tk.Tk()
    app = VisualizationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()