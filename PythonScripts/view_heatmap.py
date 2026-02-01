import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import torch


def _index_to_bhw(index: int, h: int, w: int) -> tuple[int, int, int]:
	b = index // (h * w)
	rem = index % (h * w)
	hh = rem // w
	ww = rem % w
	return b, hh, ww


def main() -> None:
	parser = argparse.ArgumentParser(description="View (B,H,W,T) tensor waveforms")
	parser.add_argument(
		"--path",
		type=str,
		default=str(Path(__file__).resolve().parents[1] / "debug_heatmap.pt"),
		help="Path to debug_heatmap.pt",
	)
	args = parser.parse_args()

	tensor_path = Path(args.path)
	if not tensor_path.exists():
		raise FileNotFoundError(f"File not found: {tensor_path}")

	data = torch.load(tensor_path, map_location="cpu")
	if not isinstance(data, torch.Tensor):
		raise TypeError("Loaded object is not a torch.Tensor")
	if data.ndim != 4:
		raise ValueError(f"Expected (B,H,W,T) tensor, got shape {tuple(data.shape)}")

	b, h, w, t = data.shape
	total = b * h * w
	current = 0
	updating_inputs = False

	fig, ax = plt.subplots(figsize=(10, 4))
	plt.subplots_adjust(bottom=0.28)
	line, = ax.plot([], [], lw=1.5)
	ax.set_xlabel("T")
	ax.set_ylabel("Amplitude")
	ax.grid(True, alpha=0.3)

	title = ax.set_title("")

	def update_plot() -> None:
		nonlocal current, updating_inputs
		current = max(0, min(current, total - 1))
		bb, hh, ww = _index_to_bhw(current, h, w)
		waveform = data[bb, hh, ww].cpu().numpy()
		line.set_data(range(t), waveform)
		ax.set_xlim(0, t - 1)
		ax.relim()
		ax.autoscale_view(scalex=False, scaley=True)
		title.set_text(
			f"Index {current + 1}/{total}  (B,H,W)=({bb},{hh},{ww})"
			f"  |  B:[0,{b - 1}] H:[0,{h - 1}] W:[0,{w - 1}]"
		)
		updating_inputs = True
		b_box.set_val(str(bb))
		h_box.set_val(str(hh))
		w_box.set_val(str(ww))
		updating_inputs = False
		fig.canvas.draw_idle()

	axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
	axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
	bprev = Button(axprev, "Prev")
	bnext = Button(axnext, "Next")

	axb = plt.axes([0.4, 0.05, 0.08, 0.06])
	axh = plt.axes([0.52, 0.05, 0.08, 0.06])
	axw = plt.axes([0.64, 0.05, 0.08, 0.06])
	axgo = plt.axes([0.76, 0.05, 0.1, 0.075])

	b_box = TextBox(axb, "B", initial="0")
	h_box = TextBox(axh, "H", initial="0")
	w_box = TextBox(axw, "W", initial="0")
	go_btn = Button(axgo, "Go")

	def on_prev(_event) -> None:
		nonlocal current
		current -= 1
		update_plot()

	def on_next(_event) -> None:
		nonlocal current
		current += 1
		update_plot()

	def _parse_int(value: str, default: int) -> int:
		try:
			return int(value)
		except ValueError:
			return default

	def go_to_bhw() -> None:
		nonlocal current
		bb = _parse_int(b_box.text, 0)
		hh = _parse_int(h_box.text, 0)
		ww = _parse_int(w_box.text, 0)
		bb = max(0, min(bb, b - 1))
		hh = max(0, min(hh, h - 1))
		ww = max(0, min(ww, w - 1))
		current = bb * h * w + hh * w + ww
		update_plot()

	def on_submit(_text: str) -> None:
		if updating_inputs:
			return
			
		go_to_bhw()

	bprev.on_clicked(on_prev)
	bnext.on_clicked(on_next)
	go_btn.on_clicked(lambda _event: go_to_bhw())
	b_box.on_submit(on_submit)
	h_box.on_submit(on_submit)
	w_box.on_submit(on_submit)

	update_plot()
	plt.show()


if __name__ == "__main__":
	main()
