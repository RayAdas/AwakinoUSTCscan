import tkinter as tk
from tkinter import Label
import matplotlib
import time
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import numpy as np

from utils.file_io import FileIO
from EchoModel import echo_function, echo_info_default, echo_info_min, echo_info_max

class ManualFitTool(tk.Frame):
    '''
    📁 **self** (ManualFitTool) [grid]
    ├── ⚙️ **scalers container** (Frame/Widget) [grid]
    │   ├──  scaler1 (Widget)
    │   ├──  label1 (Widget)
    │   ├──  ... (more scalers/labels)
    ├── 🖼️ **matlabplot wave viewer** (Canvas/Frame/Widget)
    '''
    def __init__(self, root:tk.Tk|tk.Frame, ref_wave_data:np.ndarray, ref_wave_samplerate:float, **kwargs):
        super().__init__(root, **kwargs)

        self.set_ref_waveform(ref_wave_data, ref_wave_samplerate)

        self.parasA = echo_info_default._asdict()
        self.parasB = echo_info_default._asdict()
        self.parasMax = echo_info_max._asdict()
        self.parasMin = echo_info_min._asdict()
        
        # 重新计算tau取值范围
        self.parasMin['tau'] = 0
        self.parasMax['tau'] = len(self.ref_wave_data) / self.ref_wave_samplerate

        self.slider_labels: dict[str, dict[str, Label]] = {'A': {}, 'B': {}}
        self.last_redraw_time = 0
        self.min_redraw_interval = 0.05  # seconds
        self.redraw_id: str | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        
        self.create_widgets()
        self.update_plot()

    def set_ref_waveform(self, ref_wave_data:np.ndarray, ref_wave_samplerate:float):
        self.ref_wave_data = ref_wave_data
        self.ref_wave_samplerate = ref_wave_samplerate

    def create_label_and_slider(self,
                                root:tk.Frame,
                                from_:float,
                                to:float,
                                resolution:float,
                                default_value:float,
                                label:str,
                                command):
        slider = tk.Scale(root, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=resolution, label=label, command=command)
        slider.set(default_value)
        label_widget = Label(root, text=f'{label} Value: {default_value}')
        return slider, label_widget
    
    def create_widgets(self):
        # 创建滑条区域
        self.scalers_frame = tk.Frame(self)
        self.scalers_frame.grid(row=0, column=0, sticky='ew')
        self.scalers_frame.columnconfigure((1,3), weight=1)

        ## 创建滑条和标签
        ### A组参数
        i = 0
        for param in self.parasA.keys():
            s, l = self.create_label_and_slider(
                root=self.scalers_frame,
                from_=self.parasMin.get(param),
                to=self.parasMax.get(param),
                resolution=(self.parasMax.get(param) - self.parasMin.get(param)) / 5000,
                default_value=getattr(echo_info_default, param),
                label=f'A_ ({param})',
                command=lambda v, p=param: self.on_param_change(p, float(v), 'A')
            )
            s.grid(row=i, column=1, sticky='ew')
            l.grid(row=i, column=0)
            self.slider_labels['A'][param] = l
            i += 1
        
        ### B组参数
        i = 0
        for param in self.parasB.keys():
            s, l = self.create_label_and_slider(
                root=self.scalers_frame,
                from_=self.parasMin.get(param),
                to=self.parasMax.get(param),
                resolution=(self.parasMax.get(param) - self.parasMin.get(param)) / 5000,
                default_value=getattr(echo_info_default, param),
                label=f'B_ ({param})',
                command=lambda v, p=param: self.on_param_change(p, float(v), 'B')
            )
            s.grid(row=i, column=3, sticky='ew')
            l.grid(row=i, column=2)
            self.slider_labels['B'][param] = l
            i += 1

        # 误差值标签
        self.label_err = Label(self, text='Err Value: 0')
        self.label_err.grid(row=1, column=0, columnspan=4)

        # 创建matplotlib plot区域
        ## Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Manual Fitting Tool')
        self.waveform_line_src, self.waveform_line_A, self.waveform_line_B = self.ax.plot([], [], [], [], [], [], lw=2)

        ## Embed the matplotlib figure inside the Tk window.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=2, column=0, columnspan=4, sticky='nsew')

        ## Init the lines
        self.waveform_line_src.set_label('Ref Waveform')
        self.waveform_line_A.set_label('Params A')
        self.waveform_line_B.set_label('Params B')
        self.ax.legend()
        self.t = np.linspace(0, len(self.ref_wave_data) / self.ref_wave_samplerate, len(self.ref_wave_data), endpoint=False)
        self.waveform_line_src.set_data(self.t, self.ref_wave_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def on_param_change(self, param_name:str, value:float, group:str):
        # 更新参数变量 
        if group == 'A':
            self.parasA[param_name] = value
        elif group == 'B':
            self.parasB[param_name] = value

        label_widget = self.slider_labels[group][param_name]
        label_widget.config(text=f'{group}_ ({param_name}) Value: {value:.4f}')

        # 实时更新判断：是否超过最小间隔？
        current_time = time.time()
        if current_time - self.last_redraw_time > self.min_redraw_interval:
            # 超过间隔：立即重绘，提供实时反馈
            # 取消任何待定的延迟任务，因为现在要立即执行
            if self.redraw_id:
                self.after_cancel(self.redraw_id)
                self.redraw_id = None
                
            self.update_plot() # 立即执行重绘
            
        else:
            # 未超过间隔：抑制实时重绘，但设置一个延迟任务以确保最终更新
            
            # 如果已有延迟任务，先取消它
            if self.redraw_id:
                self.after_cancel(self.redraw_id)
            
            # 计算需要等待的时间 (转换为毫秒)
            wait_time_ms = int(self.min_redraw_interval * 1000)
            
            self.redraw_id = self.after(
                wait_time_ms, 
                self.update_plot
            )

    def update_plot(self):
        fitted_A = echo_function(self.t, **self.parasA)
        fitted_B = echo_function(self.t, **self.parasB)
        self.waveform_line_A.set_data(self.t, fitted_A)
        self.waveform_line_B.set_data(self.t, fitted_B)
        residual = self.ref_wave_data - (fitted_A + fitted_B)
        rmse = np.sqrt(np.mean(residual ** 2))
        self.label_err.config(text=f"Err Value: {rmse:.3e}")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
        self.last_redraw_time = time.time()
        self.redraw_id = None

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Manual Fitting Tool")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    fio = FileIO()
    waveform_data = fio.get_waveform_data()[0,0,:]
    waveform_data = waveform_data - np.mean(waveform_data) # 去除直流分量
    app = ManualFitTool(root, waveform_data, 1e8)
    app.grid(sticky='nsew')
    root.mainloop()
