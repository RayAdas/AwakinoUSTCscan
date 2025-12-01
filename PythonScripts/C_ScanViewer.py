import numpy as np
import configparser
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from utils import FileIO

fio = FileIO()
waveform_data = fio.get_waveform_data()
waveform_data = waveform_data - np.mean(waveform_data) # 去除直流分量

# 创建主窗口
class WaveformViewer:
    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.current_index = 0
        self.current_point = [0, 0]

        # 获取数据维度
        self.height, self.width, self.depth = data.shape

        # 计算全局最大最小值
        self.global_min = np.min(data)
        self.global_max = np.max(data)

        # 创建Matplotlib图像
        self.fig, (self.ax, self.wave_ax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=(10, 5))
        self.image = self.ax.imshow(self.data[:, :, self.current_index], cmap='RdBu', vmin=self.global_min, vmax=self.global_max)
        self.ax.set_title(f"Slice: {self.current_index}")
        self.fig.colorbar(self.image, ax=self.ax)

        # 创建Matplotlib波形
        self.wave_ax.set_title("Waveform")
        self.wave_ax.set_xlabel("Index")
        self.wave_ax.set_ylabel("Value")
        self.waveform_line, = self.wave_ax.plot([], [], lw=2)
        self.envelope, = self.wave_ax.plot([], [], label='Envelope', lw=2)
        self.vertical_line = self.wave_ax.axvline(x=0, color='red', linestyle='--')  # 初始化红色竖线

        # 将Matplot C扫图片嵌入到Tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        # 连接鼠标事件
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_click)

        # 创建滑动条
        self.slider = ttk.Scale(root, from_=0, to=self.depth - 1, orient=tk.HORIZONTAL, command=self.update_img)
        self.slider.pack(fill=tk.X, padx=10, pady=10)

        # 创建保存按钮
        self.save_button = ttk.Button(root, text="Save Index", command=self.save_index)
        self.save_button.pack(pady=10)

    def update_wave(self, idx):
        # 更新右侧波形图
        self.current_point = idx
        waveform = self.data[idx[1], idx[0], :]
        self.waveform_line.set_data(range(waveform.shape[0]), waveform)

        filtered_waveform = self.apply_lowpass_filter(waveform, cutoff=60)

        # 计算包络线
        envelope = np.abs(hilbert(filtered_waveform))
        self.envelope.set_data(range(waveform.shape[0]), envelope)
        self.wave_ax.relim()
        self.wave_ax.autoscale_view()
        self.canvas.draw()

    def update_img(self, value):
        self.current_index = int(float(value))# 传入的value是个字符串
        self.image.set_data(self.data[:, :, self.current_index])
        self.ax.set_title(f"Slice: {self.current_index}")
        self.vertical_line.set_xdata([self.current_index, self.current_index])  # 示例红色竖线位置
        self.canvas.draw()
    
    def apply_lowpass_filter(self, waveform, cutoff):
        nyquist = 0.5 * 1000
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, waveform)

    def on_click(self, event):
    # 确保点击发生在当前的 Axes 中
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)  # 获取点击位置的索引
            print(f"Clicked at: x={x}, y={y}")
            self.update_wave(idx=[x,y])
            # value = self.data[y, x, self.current_index]  # 获取对应数据值

    def save_index(self):
            config_save = configparser.ConfigParser()
            config_save.read(fio.join_datapath('SavedIndices.ini'))
            section = str(self.current_point[0])
            option = str(self.current_point[1])
            if not config_save.has_section(section):
                config_save.add_section(section)
            config_save.set(section, option, str(self.current_index))
            with open(fio.join_datapath('SavedIndices.ini'), 'w') as configfile:
                config_save.write(configfile)
            print(f"Saved current_index={self.current_index} at section={section}, option={option}")

# 创建Tkinter主窗口
root = tk.Tk()
root.title("Waveform Viewer")
root.geometry("800x600")

# 实例化查看器
viewer = WaveformViewer(root, waveform_data)

# 运行主循环
root.mainloop()
