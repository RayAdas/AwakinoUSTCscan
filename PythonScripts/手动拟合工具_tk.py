import tkinter as tk
from tkinter import Label
import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import numpy as np

from utils.file_io import FileIO
from EchoModel import echo_function, echo_info_default, echo_info_min, echo_info_max

class ManualFitTool(tk.Tk):
    def __init__(self, waveform_data:np.ndarray):
        """Expect a 2xN array where row 0 is time and row 1 is amplitude."""
        super().__init__()

        waveform_array = np.asarray(waveform_data, dtype=float)
        if waveform_array.ndim != 2 or waveform_array.shape[0] != 2:
            raise ValueError('waveform_data must be shaped (2, N) with time and amplitude rows')

        self.waveform_data = waveform_array
        self.title('Waveform Slider (Tkinter)')
        self.geometry('900x700')
        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        # Sliders and labels
        tau_min = echo_info_min.tau
        tau_max = echo_info_max.tau
        beta_min = min(echo_info_min.beta, echo_info_default.beta)
        beta_max = max(echo_info_max.beta, echo_info_default.beta)
        fc_min = echo_info_min.fc
        fc_max = echo_info_max.fc

        self.tau_scale = 1e-8
        tau_min_units = int(round(tau_min / self.tau_scale))
        tau_max_units = int(round(tau_max / self.tau_scale))
        tau_default_units = int(round(np.clip(echo_info_default.tau, tau_min, tau_max) / self.tau_scale))

        beta_range = beta_max - beta_min if beta_max > beta_min else 1.0
        beta_resolution = beta_range / 500

        self.fc_scale = 1e3
        fc_min_units = int(round(fc_min / self.fc_scale))
        fc_max_units = int(round(fc_max / self.fc_scale))
        fc_default_units = int(round(np.clip(echo_info_default.fc, fc_min, fc_max) / self.fc_scale))

        self.slider_tau_1 = tk.Scale(self, from_=tau_min_units, to=tau_max_units, orient=tk.HORIZONTAL, resolution=1, label='tau1 (x1e-8)', command=lambda e: self.update_plot())
        self.slider_tau_1.set(tau_default_units)
        self.slider_tau_1.grid(row=0, column=0, sticky='ew')
        self.label_tau_1 = Label(self, text='tau1 Value: 0 us')
        self.label_tau_1.grid(row=0, column=1)

        self.slider_tau_2 = tk.Scale(self, from_=tau_min_units, to=tau_max_units, orient=tk.HORIZONTAL, resolution=1, label='tau2 (x1e-8)', command=lambda e: self.update_plot())
        self.slider_tau_2.set(tau_default_units)
        self.slider_tau_2.grid(row=0, column=2, sticky='ew')
        self.label_tau_2 = Label(self, text='tau2 Value: 0 us')
        self.label_tau_2.grid(row=0, column=3)

        beta_default = np.clip(echo_info_default.beta, beta_min, beta_max)
        self.slider_A_1 = tk.Scale(self, from_=beta_min, to=beta_max, resolution=max(beta_resolution, 1e-6), orient=tk.HORIZONTAL, label='A1 (beta)', command=lambda e: self.update_plot())
        self.slider_A_1.set(beta_default)
        self.slider_A_1.grid(row=1, column=0, sticky='ew')
        self.label_A_1 = Label(self, text=f'A1 Value: {beta_default:.3e}')
        self.label_A_1.grid(row=1, column=1)

        self.slider_A_2 = tk.Scale(self, from_=beta_min, to=beta_max, resolution=max(beta_resolution, 1e-6), orient=tk.HORIZONTAL, label='A2 (beta)', command=lambda e: self.update_plot())
        self.slider_A_2.set(beta_default)
        self.slider_A_2.grid(row=1, column=2, sticky='ew')
        self.label_A_2 = Label(self, text=f'A2 Value: {beta_default:.3e}')
        self.label_A_2.grid(row=1, column=3)

        self.slider_fc_1 = tk.Scale(self, from_=fc_min_units, to=fc_max_units, orient=tk.HORIZONTAL, resolution=1, label='fc1 (kHz)', command=lambda e: self.update_plot())
        self.slider_fc_1.set(fc_default_units)
        self.slider_fc_1.grid(row=2, column=0, sticky='ew')
        initial_fc_1 = self.slider_fc_1.get() * self.fc_scale / 1e6
        self.label_fc_1 = Label(self, text=f"fc1 Value: {initial_fc_1:.3f} MHz")
        self.label_fc_1.grid(row=2, column=1)

        self.slider_fc_2 = tk.Scale(self, from_=fc_min_units, to=fc_max_units, orient=tk.HORIZONTAL, resolution=1, label='fc2 (kHz)', command=lambda e: self.update_plot())
        self.slider_fc_2.set(fc_default_units)
        self.slider_fc_2.grid(row=2, column=2, sticky='ew')
        initial_fc_2 = self.slider_fc_2.get() * self.fc_scale / 1e6
        self.label_fc_2 = Label(self, text=f"fc2 Value: {initial_fc_2:.3f} MHz")
        self.label_fc_2.grid(row=2, column=3)

        self.label_err = Label(self, text='Err Value: 0')
        self.label_err.grid(row=3, column=0, columnspan=4)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Manual Fitting Tool')
        self.waveform_line_src, self.waveform_line_A, self.waveform_line_B = self.ax.plot([], [], [], [], [], [], lw=2)

        # Embed the matplotlib figure inside the Tk window.
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=4, sticky='nsew')
        for col in range(4):
            self.grid_columnconfigure(col, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Init the lines
        self.waveform_line_src.set_label('Real Waveform')
        self.waveform_line_A.set_label('Params A')
        self.waveform_line_B.set_label('Params B')
        self.ax.legend()
        self.waveform_line_src.set_data(self.waveform_data[0], self.waveform_data[1])
        self.parasA = echo_info_default._asdict()
        self.parasB = echo_info_default._asdict()

    def update_plot(self):
        slider_tau_1_units = self.slider_tau_1.get()
        slider_A_1_value = self.slider_A_1.get()
        tau_1_value = slider_tau_1_units * self.tau_scale
        self.label_tau_1.config(text=f"tau1 Value: {tau_1_value * 1e6:.3f} us")
        self.label_A_1.config(text=f"A1 Value: {slider_A_1_value:.3e}")
        self.parasA['tau'] = tau_1_value
        self.parasA['beta'] = slider_A_1_value

        slider_tau_2_units = self.slider_tau_2.get()
        slider_A_2_value = self.slider_A_2.get()
        tau_2_value = slider_tau_2_units * self.tau_scale
        self.label_tau_2.config(text=f"tau2 Value: {tau_2_value * 1e6:.3f} us")
        self.label_A_2.config(text=f"A2 Value: {slider_A_2_value:.3e}")
        self.parasB['tau'] = tau_2_value
        self.parasB['beta'] = slider_A_2_value

        slider_fc_1_units = self.slider_fc_1.get()
        slider_fc_2_units = self.slider_fc_2.get()
        fc_1_value = slider_fc_1_units * self.fc_scale
        fc_2_value = slider_fc_2_units * self.fc_scale
        self.label_fc_1.config(text=f"fc1 Value: {fc_1_value / 1e6:.3f} MHz")
        self.label_fc_2.config(text=f"fc2 Value: {fc_2_value / 1e6:.3f} MHz")
        self.parasA['fc'] = fc_1_value
        self.parasB['fc'] = fc_2_value

        fitted_A = echo_function(self.waveform_data[0], **self.parasA)
        fitted_B = echo_function(self.waveform_data[0], **self.parasB)
        self.waveform_line_A.set_data(self.waveform_data[0], fitted_A)
        self.waveform_line_B.set_data(self.waveform_data[0], fitted_B)
        residual = self.waveform_data[1] - (fitted_A + fitted_B)
        rmse = np.sqrt(np.mean(residual ** 2))
        self.label_err.config(text=f"Err Value: {rmse:.3e}")
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

if __name__ == '__main__':
    fio = FileIO()
    waveform_data = fio.get_waveform_data()[0,0,:]
    waveform_data = waveform_data - np.mean(waveform_data) # 去除直流分量
    time_axis = np.linspace(0, len(waveform_data) * 1e-8, len(waveform_data), endpoint=False)
    app = ManualFitTool(np.vstack((time_axis, waveform_data)))
    app.mainloop()
