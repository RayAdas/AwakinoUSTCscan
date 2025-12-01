import torch
from utils import FileIO
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

fio = FileIO()
waveform_data = fio.get_waveform_data()

dataset_dict = torch.load(fio.join_datapath('echo_dataset.pt'))
data = dataset_dict['data']
tgt = dataset_dict['tgt']

class DataViewer:
    def __init__(self, master, data, tgt):
        self.master = master
        self.data = data
        self.tgt = tgt
        self.index = 0

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

        self.label = tk.Label(master, text=f'Target: {self.tgt[self.index].item()}')
        self.label.pack()

        self.prev_button = tk.Button(master, text="Previous", command=self.prev_data)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(master, text="Next", command=self.next_data)
        self.next_button.pack(side=tk.RIGHT)

        self.prev_tgt_button = tk.Button(master, text="Previous tgt=1", command=self.prev_tgt_data)
        self.prev_tgt_button.pack(side=tk.LEFT)

        self.next_tgt_button = tk.Button(master, text="Next tgt=1", command=self.next_tgt_data)
        self.next_tgt_button.pack(side=tk.RIGHT)

        self.plot_data()

    def plot_data(self):
        self.ax.clear()
        self.ax.plot(self.data[self.index].numpy())
        self.ax.set_title(f'Index: {self.index}')
        self.canvas.draw()
        self.label.config(text=f'Target: {self.tgt[self.index].item()}')

    def prev_data(self):
        if self.index > 0:
            self.index -= 1
            self.plot_data()

    def next_data(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.plot_data()

    def prev_tgt_data(self):
        for i in range(self.index - 1, -1, -1):
            if self.tgt[i].item() == 1:
                self.index = i
                self.plot_data()
                break

    def next_tgt_data(self):
        for i in range(self.index + 1, len(self.data)):
            if self.tgt[i].item() == 1:
                self.index = i
                self.plot_data()
                break

root = tk.Tk()
app = DataViewer(root, data, tgt)
root.mainloop()

