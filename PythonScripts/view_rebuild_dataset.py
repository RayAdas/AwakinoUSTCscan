import os

import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from utils.file_io import FileIO
from rebuild.dataset import DeepImgDataset

TGT_KERNEL = torch.ones((1,1))

def load_dataset():
	FileIO.init()
	dataset_path = os.path.join(FileIO.curr_rebuild_dataset_path, "dataset.pt")
	dataset = torch.load(dataset_path, weights_only=False)
	return dataset

def main():
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ground_truth:DeepImgDataset = load_dataset()
	print(f"len(ground_truth): {len(ground_truth)}")

	x = ground_truth.input.to(DEVICE)      # (N,H,W,T)
	r_gt = ground_truth.tgt.to(DEVICE)     # (N,H,W)

	s_gt = DeepImgDataset.defects_to_waves(
		r_gt,
		TGT_KERNEL,
		receptive_field_size=r_gt.shape[1],
		sigma=DeepImgDataset.SIGMA,
	)                     # (N,H,W,T)

	# 转换到CPU用于绘图
	x_cpu = x.cpu()
	s_gt_cpu = s_gt.cpu()
	
	# 获取数据维度
	N, H, W, T = x_cpu.shape
	print(f"数据维度: N={N}, H={H}, W={W}, T={T}")
	
	# 创建交互式可视化窗口
	create_interactive_plot(x_cpu, s_gt_cpu, N, H, W, T)


def create_interactive_plot(x, s_gt, N, H, W, T):
	"""创建交互式绘图窗口"""
	# 初始索引
	current_indices = {'n': 0, 'h': 0, 'w': 0}
	
	# 创建图形和子图
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
	plt.subplots_adjust(bottom=0.25)
	
	# 初始绘图
	line1, = ax1.plot(x[0, 0, 0, :].numpy(), 'b-', linewidth=1.5)
	ax1.set_title(f'x波形 (N={0}, H={0}, W={0})')
	ax1.set_xlabel('时间采样点 (T)')
	ax1.set_ylabel('幅值')
	ax1.grid(True, alpha=0.3)
	
	line2, = ax2.plot(s_gt[0, 0, 0, :].numpy(), 'r-', linewidth=1.5)
	ax2.set_title(f's_gt波形 (N={0}, H={0}, W={0})')
	ax2.set_xlabel('时间采样点 (T)')
	ax2.set_ylabel('幅值')
	ax2.grid(True, alpha=0.3)
	
	# 创建文本输入框
	ax_n = plt.axes([0.15, 0.12, 0.15, 0.04])
	ax_h = plt.axes([0.15, 0.07, 0.15, 0.04])
	ax_w = plt.axes([0.15, 0.02, 0.15, 0.04])
	
	text_n = TextBox(ax_n, 'N (0-{}):'.format(N-1), initial='0')
	text_h = TextBox(ax_h, 'H (0-{}):'.format(H-1), initial='0')
	text_w = TextBox(ax_w, 'W (0-{}):'.format(W-1), initial='0')
	
	# 添加说明文本
	fig.text(0.5, 0.14, '在输入框中输入索引值并按Enter键更新波形图', 
	         ha='center', fontsize=10, style='italic')
	
	def update_plot(val=None):
		"""更新波形图"""
		try:
			n = current_indices['n']
			h = current_indices['h']
			w = current_indices['w']
			
			# 更新x波形
			line1.set_ydata(x[n, h, w, :].numpy())
			ax1.set_title(f'x波形 (N={n}, H={h}, W={w})')
			ax1.relim()
			ax1.autoscale_view()
			
			# 更新s_gt波形
			# line2.set_ydata(s_gt[n, h, w, :].numpy())

			# 查看softmax效果
			# T = torch.arange(s_gt.shape[-1])
			# line2.set_ydata((x[n, h, w, :].exp() / x[n, h, w, :].exp().sum())*T.numpy())
			# line2.set_ydata((x[n, h, w, :] / x[n, h, w, :].sum())*T.numpy())

			ax2.set_title(f's_gt波形 (N={n}, H={h}, W={w})')
			ax2.relim()
			ax2.autoscale_view()
			
			fig.canvas.draw_idle()
		except Exception as e:
			print(f"更新失败: {e}")
	
	def submit_n(text):
		"""处理N输入"""
		try:
			val = int(text)
			if 0 <= val < N:
				current_indices['n'] = val
				update_plot()
			else:
				print(f"N必须在0到{N-1}之间")
		except ValueError:
			print("请输入有效的整数")
	
	def submit_h(text):
		"""处理H输入"""
		try:
			val = int(text)
			if 0 <= val < H:
				current_indices['h'] = val
				update_plot()
			else:
				print(f"H必须在0到{H-1}之间")
		except ValueError:
			print("请输入有效的整数")
	
	def submit_w(text):
		"""处理W输入"""
		try:
			val = int(text)
			if 0 <= val < W:
				current_indices['w'] = val
				update_plot()
			else:
				print(f"W必须在0到{W-1}之间")
		except ValueError:
			print("请输入有效的整数")
	
	# 绑定事件
	text_n.on_submit(submit_n)
	text_h.on_submit(submit_h)
	text_w.on_submit(submit_w)
	
	plt.show()


if __name__ == "__main__":
	plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei'] 微软雅黑 等
	plt.rcParams['axes.unicode_minus'] = False   # 解决负号 '-' 显示为方块的问题
	main()