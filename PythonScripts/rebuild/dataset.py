import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .defect_types import TYPE_REGISTRY, BaseDefectType

class DeepImgDataset(Dataset):
    @classmethod
    def build_conv_core(cls, radius: float, interval: float, sigma: float = 1.0) -> torch.Tensor:
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
        wave_len: int,
        depth_min: float,
        depth_max: float,
        sigma: float
    ) -> torch.Tensor:
        device = depth_imgs_tensor.device
        n, c, _ = depth_imgs_tensor.shape

        a = conv_kernel.shape[0] // 2
        b = receptive_field_size // 2

        # --------------------------------------------------
        # 1. 取 receptive_field_size x receptive_field_size 区域
        # --------------------------------------------------
        center_start = a
        center_end = a + receptive_field_size
        center_defects = depth_imgs_tensor[
            :, center_start:center_end, center_start:center_end
        ]  # (n, r, r)

        # --------------------------------------------------
        # 2. 提取每个中心点的 a 邻域 (unfold)
        # --------------------------------------------------
        patches = F.unfold(
            depth_imgs_tensor.unsqueeze(1),   # (n,1,c,c)
            kernel_size=2*a+1
        )
        # (n, (2a+1)^2, r*r)
        patches = patches.transpose(1, 2)
        patches = patches.reshape(
            n,
            receptive_field_size,
            receptive_field_size,
            (2*a+1)*(2*a+1)
        )  # (n, r, r, k)

        # --------------------------------------------------
        # 3. 卷积核展平
        # --------------------------------------------------
        kernel_flat = conv_kernel.reshape(-1).to(device)  # (k,)

        # --------------------------------------------------
        # 4. 构造 wave 深度轴
        # --------------------------------------------------
        wave_axis = torch.linspace(
            depth_min, depth_max, wave_len, device=device
        )  # (wave_len,)

        # --------------------------------------------------
        # 5. 生成所有高斯波形（核心向量化）
        # --------------------------------------------------
        # patches[..., None] -> (n, r, r, k, 1)
        # wave_axis -> (1,1,1,1,wave_len)
        gaussian = torch.exp(
            -0.5 * ((wave_axis - patches[..., None]) / sigma) ** 2
        )  # (n, r, r, k, wave_len)

        # --------------------------------------------------
        # 6. 乘卷积核权重并求和
        # --------------------------------------------------
        wave = torch.einsum(
            'nijkw,k->nijw',
            gaussian,
            kernel_flat
        )

        return wave # (n, r, r, wave_len)

    def __init__(self, receptive_field_size=41, 
                 sampling_interval=1e-3, 
                 conv_radius=5e-3, 
                 conv_kernel=None, 
                 n_samples=1000, 
                 d_input=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        deepth_imgs: list[torch.Tensor] = []
        defects_meta: list[BaseDefectType] = []
        for defect_type in TYPE_REGISTRY.values():
            n_type_samples = n_left_samples // n_left_types
            for _ in range(n_type_samples):
                defect = defect_type()
                defects_meta.append(defect)
                deepth_imgs.append(defect.get_depth(pos_img_x, pos_img_y))
            n_left_samples -= n_type_samples
            n_left_types -= 1

        self.n_samples = len(deepth_imgs)
        deepth_imgs_tensor: torch.Tensor = torch.stack(deepth_imgs).to(self.device)  # (n_samples, c, c)
        self.tgt = deepth_imgs_tensor[:, a:-a, a:-a]  # (n_samples, receptive_field_size, receptive_field_size)

        # 生成深度序列
        self.input = self.defects_to_waves(
            deepth_imgs_tensor,
            conv_kernel,
            receptive_field_size,
            wave_len=d_input,
            depth_min=0.0,
            depth_max=0.01,
            sigma=1e-4
        )

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {"input": self.input[idx], "target": self.tgt[idx]}
    