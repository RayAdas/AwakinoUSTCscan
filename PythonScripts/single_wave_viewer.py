"""Interactive single-wave viewer with time and frequency plots."""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils.file_io import FileIO


class SingleWaveViewer:
    def __init__(self) -> None:
        self.fio = FileIO()
        self.waveform_data = self.fio.get_waveform_data()[:,:,400:] # 舍弃前400个采样点的噪声
        self.metadata = self.fio.get_metadata()
        self.sample_rate_hz = self.fio.get_sample_rate_hz()

        self.num_x = self.metadata["numX"]
        self.num_y = self.metadata["numY"]

        self.root = tk.Tk()
        self.root.title("Single Wave Viewer")
        self._build_ui()
        self._update_plots(0, 0)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="X 索引 (0 - {})".format(self.num_x - 1)).grid(row=0, column=0, padx=5, pady=5)
        self.x_var = tk.IntVar(value=0)
        self.spin_x = ttk.Spinbox(
            control_frame,
            from_=0,
            to=max(self.num_x - 1, 0),
            textvariable=self.x_var,
            width=8,
            wrap=False,
        )
        self.spin_x.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(control_frame, text="Y 索引 (0 - {})".format(self.num_y - 1)).grid(row=0, column=2, padx=5, pady=5)
        self.y_var = tk.IntVar(value=0)
        self.spin_y = ttk.Spinbox(
            control_frame,
            from_=0,
            to=max(self.num_y - 1, 0),
            textvariable=self.y_var,
            width=8,
            wrap=False,
        )
        self.spin_y.grid(row=0, column=3, padx=5, pady=5)

        update_button = ttk.Button(control_frame, text="显示波形", command=self._on_update_clicked)
        update_button.grid(row=0, column=4, padx=10, pady=5)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax_time = self.figure.add_subplot(211)
        self.ax_freq = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=2.5)

        canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas = canvas

    def _on_update_clicked(self) -> None:
        try:
            x_idx = int(self.x_var.get())
            y_idx = int(self.y_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("输入错误", "请输入有效的整数索引。")
            return

        if not (0 <= x_idx < self.num_x):
            messagebox.showerror("输入错误", f"X 索引必须在 0 到 {self.num_x - 1} 之间。")
            return

        if not (0 <= y_idx < self.num_y):
            messagebox.showerror("输入错误", f"Y 索引必须在 0 到 {self.num_y - 1} 之间。")
            return

        self._update_plots(x_idx, y_idx)

    def _update_plots(self, x_idx: int, y_idx: int) -> None:
        signal = self.waveform_data[x_idx, y_idx, :]
        time_axis = np.arange(signal.shape[0]) / self.sample_rate_hz

        self.ax_time.clear()
        self.ax_time.plot(time_axis * 1e6, signal)
        self.ax_time.set_xlabel("时间 (µs)")
        self.ax_time.set_ylabel("幅值")
        self.ax_time.set_title(f"原始波形 (x={x_idx}, y={y_idx})")
        self.ax_time.grid(alpha=0.3)

        fft_signal = np.fft.rfft(signal - np.mean(signal))
        freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / self.sample_rate_hz)

        self.ax_freq.clear()
        self.ax_freq.plot(freqs / 1e6, np.abs(fft_signal))
        self.ax_freq.set_xlabel("频率 (MHz)")
        self.ax_freq.set_ylabel("幅值")
        self.ax_freq.set_title("频域波形")
        self.ax_freq.grid(alpha=0.3)

        self.canvas.draw_idle()

        # 用matplotlib弹出一张单独的频率图
        import matplotlib.pyplot as plt
        freq_fig = plt.figure(figsize=(6, 4), dpi=100)
        ax_freq_popup = freq_fig.add_subplot(111)
        ax_freq_popup.plot(freqs / 1e6, np.abs(fft_signal))
        ax_freq_popup.set_xlabel("频率 (MHz)")
        ax_freq_popup.set_ylabel("幅值")
        ax_freq_popup.set_title("频域波形 (弹出窗口)")
        ax_freq_popup.grid(alpha=0.3)

        freq_fig.show()


if __name__ == "__main__":
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["SimSun"],  # 宋体
        "axes.unicode_minus": False  # 解决负号显示问题
    })
    viewer = SingleWaveViewer()
    viewer.run()

