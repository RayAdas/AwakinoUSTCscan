from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .defect_types import TYPE_REGISTRY, BaseDefectType

class DeepImgDataset(Dataset):
    DEPTH_MIN = -0.002
    DEPTH_MAX = 0.012
    WAVE_LEN = 128
    SIGMA = 3e-4

    @classmethod
    def real_depth2wave_pos(cls, depth: torch.Tensor) -> torch.Tensor:
        """Convert real-world depth (in meters) to wave position (in time bins)."""
        # 线性插值
        pos = (depth - cls.DEPTH_MIN) / (cls.DEPTH_MAX - cls.DEPTH_MIN) * (cls.WAVE_LEN - 1)
        return pos  # (...,)
    
    @classmethod
    def wave_pos2real_depth(cls, pos: torch.Tensor) -> torch.Tensor:
        """Convert wave position (in time bins) to real-world depth (in meters)."""
        depth = pos / (cls.WAVE_LEN - 1) * (cls.DEPTH_MAX - cls.DEPTH_MIN) + cls.DEPTH_MIN
        return depth  # (...,)

    @classmethod
    def build_conv_core(cls, radius: float, interval: float, sigma: float = SIGMA) -> torch.Tensor:
        """Build a 2D Gaussian convolution core."""
        size = int(2 * radius / interval)
        if size % 2 == 0:
            size -= 1  # Ensure size is odd
        
        # 创建坐标网格
        ax = torch.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # 计算高斯函数
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # 归一化
        kernel = kernel / torch.sum(kernel)
        
        return kernel
    
    @classmethod
    def defects_to_waves(
        cls,
        depth_imgs_tensor: torch.Tensor,        # (n, c, c)
        conv_kernel: torch.Tensor,            # (2a+1, 2a+1)
        receptive_field_size: int,   # 2b+1
        sigma: float = SIGMA,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Convert depth images to wave signals with batched processing to save memory.
        
        Args:
            batch_size: Number of samples to process at once to control memory usage
        """
        device = depth_imgs_tensor.device
        n, c, _ = depth_imgs_tensor.shape

        a = conv_kernel.shape[0] // 2
        b = receptive_field_size // 2

        # 卷积核展平
        kernel_flat = conv_kernel.reshape(-1).to(device)  # (k,)

        # 构造 wave 深度轴
        wave_axis = torch.linspace(
            cls.DEPTH_MIN, cls.DEPTH_MAX, cls.WAVE_LEN, device=device
        )  # (wave_len,)

        # 分批处理以节省内存/显存
        wave_results = []
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_n = batch_end - batch_start
            batch_depth = depth_imgs_tensor[batch_start:batch_end] # (batch_n, c, c)

            # 提取每个中心点的 a 邻域 (unfold)
            patches = F.unfold(
                batch_depth.unsqueeze(1),   # (batch_n,1,c,c)
                kernel_size=2*a+1
            ) # (batch_n, k, r*r)

            patches = patches.transpose(1, 2)
            patches = patches.reshape(
                batch_n,
                receptive_field_size,
                receptive_field_size,
                (2*a+1)*(2*a+1)
            )  # (batch_n, r, r, k)

            # 生成高斯波形（核心向量化）
            # patches[..., None] -> (batch_n, r, r, k, 1)
            # wave_axis -> (1,1,1,1,wave_len)
            gaussian = torch.exp(
                -0.5 * ((wave_axis - patches[..., None]) / sigma) ** 2
            )  # (batch_n, r, r, k, wave_len)

            # 乘卷积核权重并求和
            wave_batch = torch.einsum(
                'nijkw,k->nijw',
                gaussian,
                kernel_flat
            )  # (batch_n, r, r, wave_len)
            
            wave_results.append(wave_batch)
            
            # 清理中间变量
            del patches, gaussian, wave_batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # 合并所有批次的结果
        wave = torch.cat(wave_results, dim=0)
        return wave  # (n, r, r, wave_len)

    def __init__(self, receptive_field_size=41, 
                 sampling_interval=1e-3, 
                 conv_radius=5e-3, 
                 conv_kernel=None, 
                 n_samples=1000, 
                 batch_size=32,
                 device: Optional[torch.device] = None):
        """
        Args:
            batch_size: Number of samples to process at once during wave generation (controls memory usage)
            device: Device to run data generation on, None for auto selection. Anyhow, data will be "to CPU" after generation.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampling_interval = sampling_interval
        self.conv_radius = conv_radius
        conv_kernel = conv_kernel if conv_kernel is not None else self.build_conv_core(
            radius=conv_radius,
            interval=sampling_interval,
        ).to(self.device)

        a = conv_kernel.shape[0] // 2
        b = receptive_field_size // 2
        c = (a + b) * 2 + 1
        pos_img_x = torch.linspace(- (a + b) * sampling_interval, (a + b) * sampling_interval, c)
        pos_img_x = pos_img_x.unsqueeze(0).repeat(c, 1).to(self.device)
        pos_img_y = pos_img_x.t().to(self.device)

        # 生成缺陷
        n_left_samples = n_samples
        n_left_types = len(TYPE_REGISTRY)
        depth_imgs: list[torch.Tensor] = []
        defects_meta: list[BaseDefectType] = []
        for defect_type in TYPE_REGISTRY.values():
            n_type_samples = n_left_samples // n_left_types
            for _ in range(n_type_samples):
                defect = defect_type()
                defects_meta.append(defect)
                depth_imgs.append(defect.get_depth(pos_img_x, pos_img_y))
            n_left_samples -= n_type_samples
            n_left_types -= 1

        self.n_samples = len(depth_imgs)
        depth_imgs_tensor: torch.Tensor = torch.stack(depth_imgs).to(self.device)  # (n_samples, c, c)
        self.tgt = depth_imgs_tensor[:, a:-a, a:-a]  # (n_samples, receptive_field_size, receptive_field_size)
        # 生成深度序列
        print(f"Generating wave data in batches of {batch_size}...")
        self.input = self.defects_to_waves(
            depth_imgs_tensor,
            conv_kernel,
            receptive_field_size,
            batch_size=batch_size)

        self.input = self.input.to('cpu')
        self.tgt = self.tgt.to('cpu')

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {"input": self.input[idx], "target": self.tgt[idx]}
    