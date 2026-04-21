import os
import json
import time
import sys
from pathlib import Path
from tkinter import filedialog

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
	sys.path.insert(0, str(BASE_DIR))

py_tcl_dir = Path(os.__file__).resolve().parents[1] / "tcl"
if (py_tcl_dir / "tcl8.6").exists() and "TCL_LIBRARY" not in os.environ:
	os.environ["TCL_LIBRARY"] = str(py_tcl_dir / "tcl8.6")
if (py_tcl_dir / "tk8.6").exists() and "TK_LIBRARY" not in os.environ:
	os.environ["TK_LIBRARY"] = str(py_tcl_dir / "tk8.6")

from src.config import Config
from src.data_processing import load_frames_from_video
from src.training import initialize_training_components

import customtkinter as ctk
import cv2
import torch
import numpy as np
from PIL import Image

# fit video in frame
# use dropdown for model select

MODEL_FILETYPES = [("PyTorch Model", "*.pt")]
VIDEO_FILETYPES = [("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
CATEGORY_NAMES = ["Reduced", "Mildly Reduced", "Preserved"]


class App(ctk.CTk):
	def __init__(self) -> None:
		super().__init__()
		self.cfg = Config()
		self.title("CustomTkinter App")
		self.geometry("800x500")
		self.artifacts_dir = str(BASE_DIR / "artifacts")
		self.selected_checkpoint_path = ""
		self.selected_video_path = ""
		self.video_capture = None
		self.video_after_id = None
		self.frame_interval_s = 1.0 / 30
		self.next_frame_time = 0.0
		self._video_image = None
		self.inference_busy = False
		self.protocol("WM_DELETE_WINDOW", self.on_close)

		top_bar = ctk.CTkFrame(self, fg_color="transparent")
		top_bar.place(relx=0.5, rely=0.02, relwidth=0.96, relheight=0.07, anchor="n")

		self.path_input = ctk.CTkEntry(top_bar, placeholder_text="Select a model...")
		self.path_input.pack(side="left", fill="x", expand=True, padx=(0, 8))

		browse_btn = ctk.CTkButton(top_bar, text="Browse", width=90, command=self.pick_path)
		browse_btn.pack(side="right")

		divider = ctk.CTkFrame(self, width=2, fg_color="gray35")
		divider.place(relx=0.5, rely=0.1, relheight=0.85, anchor="n")

		left_panel = ctk.CTkFrame(self)
		left_panel.place(relx=0.02, rely=0.1, relwidth=0.46, relheight=0.85, anchor="nw")

		video_row = ctk.CTkFrame(left_panel, fg_color="transparent")
		video_row.pack(fill="x", padx=10, pady=(10, 8))

		self.video_name = ctk.CTkLabel(video_row, text="No video selected")
		self.video_name.pack(side="left")

		video_btn = ctk.CTkButton(video_row, text="Browse", width=90, command=self.pick_video)
		video_btn.pack(side="right", padx=(10, 0))

		self.video_view = ctk.CTkLabel(left_panel, text="Upload a video to preview", fg_color="gray10", corner_radius=8)
		self.video_view.pack(fill="both", expand=True, padx=10, pady=(0, 10))

		right_panel = ctk.CTkFrame(self)
		right_panel.place(relx=0.52, rely=0.1, relwidth=0.46, relheight=0.85, anchor="nw")

		self.run_btn = ctk.CTkButton(right_panel, text="Run Inference", command=self.run_inference)
		self.run_btn.pack(anchor="w", padx=10, pady=(10, 8))

		self.result_text = ctk.CTkTextbox(right_panel)
		self.result_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
		self.result_text.insert("1.0", "Select a model and video, then click Run Inference.\n")
		self.result_text.configure(state="disabled")

	def pick_path(self) -> None:
		path = filedialog.askopenfilename(
			initialdir=self.artifacts_dir,
			filetypes=MODEL_FILETYPES,
		)
		if path:
			self.selected_checkpoint_path = path
			self.path_input.delete(0, "end")
			self.path_input.insert(0, Path(path).name)

	def pick_video(self) -> None:
		path = filedialog.askopenfilename(
			filetypes=VIDEO_FILETYPES,
		)
		if path:
			self.selected_video_path = path
			self.video_name.configure(text=Path(path).name)
			self.start_video(path)

	def _set_results(self, text: str) -> None:
		self.result_text.configure(state="normal")
		self.result_text.delete("1.0", "end")
		self.result_text.insert("1.0", text)
		self.result_text.configure(state="disabled")

	def _load_stats(self) -> tuple[float, float]:
		stats_path = BASE_DIR / self.cfg.STATS_CACHE_FILE
		mean = self.cfg.DEFAULT_MEAN
		std = self.cfg.DEFAULT_STD
		if stats_path.exists():
			try:
				with stats_path.open("r", encoding="utf-8") as f:
					stats = json.load(f)
				mean = float(stats.get("mean", mean))
				std = float(stats.get("std", std))
			except Exception:
				pass
		if std <= 0:
			std = self.cfg.DEFAULT_STD
		return mean, std

	def _prepare_video_tensor(self, video_path: str) -> torch.Tensor | None:
		frames = load_frames_from_video(
			video_path=video_path,
			num_frames=self.cfg.NUM_FRAMES,
			target_size=(self.cfg.TARGET_WIDTH, self.cfg.TARGET_HEIGHT),
			period=self.cfg.FRAME_SAMPLING_PERIOD,
			random_start=False,
		)
		if frames is None:
			return None
		mean, std = self._load_stats()
		frames = frames.astype(np.float32) / 255.0
		frames = (frames - mean) / std
		return torch.from_numpy(frames).unsqueeze(0).unsqueeze(0).float()

	def run_inference(self) -> None:
		if self.inference_busy:
			return
		if not self.selected_checkpoint_path:
			self._set_results("Please select a model checkpoint (.pt) first.")
			return
		if not self.selected_video_path:
			self._set_results("Please select a video first.")
			return

		self.inference_busy = True
		self.run_btn.configure(state="disabled", text="Running...")
		self.update_idletasks()

		try:
			input_tensor = self._prepare_video_tensor(self.selected_video_path)
			if input_tensor is None:
				self._set_results("Could not read enough frames from the selected video.")
				return

			components = initialize_training_components(class_weights=None)
			checkpoint = torch.load(self.selected_checkpoint_path, map_location=components.device)
			components.model.load_state_dict(checkpoint["model_state_dict"])
			components.model.eval()

			video_device = input_tensor.to(components.device, non_blocking=True)
			start = time.perf_counter()
			with torch.no_grad():
				ef_reg, ef_logits = components.model(video_device)
				probs = torch.softmax(ef_logits, dim=1).squeeze(0).cpu().numpy()
			inference_ms = (time.perf_counter() - start) * 1000.0

			pred_ef = float(ef_reg.squeeze(0).item())
			pred_idx = int(np.argmax(probs))
			pred_class = CATEGORY_NAMES[pred_idx] if pred_idx < len(CATEGORY_NAMES) else str(pred_idx)
			pred_conf = float(probs[pred_idx])
			sorted_probs = np.sort(probs)
			margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0

			cap = cv2.VideoCapture(self.selected_video_path)
			fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
			frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
			cap.release()
			duration = (frame_count / fps) if fps > 0 else 0.0

			result_lines = [
				"Inference complete",
				"",
				f"Model: {Path(self.selected_checkpoint_path).name}",
				f"Video: {Path(self.selected_video_path).name}",
				"",
				f"Predicted EF: {pred_ef:.2f}%",
				f"Classification: {pred_class}",
				f"Confidence: {pred_conf * 100:.2f}%",
				f"Confidence margin (top1-top2): {margin * 100:.2f}%",
				"",
				f"Inference time: {inference_ms:.1f} ms",
				f"Input clip: {self.cfg.NUM_FRAMES} frames @ period {self.cfg.FRAME_SAMPLING_PERIOD}",
				f"Original video: {width}x{height}, {fps:.2f} FPS, {frame_count} frames, {duration:.2f}s",
			]
			self._set_results("\n".join(result_lines))
		except Exception as exc:
			self._set_results(f"Inference failed:\n{exc}")
		finally:
			self.inference_busy = False
			self.run_btn.configure(state="normal", text="Run Inference")

	def start_video(self, path: str) -> None:
		self.stop_video()
		self.video_capture = cv2.VideoCapture(path)
		if not self.video_capture.isOpened():
			self.video_view.configure(text="Could not open video", image=None)
			self.video_capture = None
			return
		self.video_view.update_idletasks()
		self.frame_interval_s = 1.0 / 30
		fps = self.video_capture.get(cv2.CAP_PROP_FPS)
		if fps and fps > 1:
			self.frame_interval_s = 1.0 / fps
		self.next_frame_time = time.perf_counter() + self.frame_interval_s
		self.play_next_frame()

	def _fit_frame_to_view(self, frame: np.ndarray) -> np.ndarray:
		"""Resize frame to fit inside the preview area without cropping or stretching."""
		try:
			widget_scale = float(self.video_view._get_widget_scaling())
		except Exception:
			widget_scale = 1.0

		target_w_px = self.video_view.winfo_width()
		target_h_px = self.video_view.winfo_height()
		if target_w_px <= 1 or target_h_px <= 1:
			return frame

		target_w = max(1, int(target_w_px / widget_scale))
		target_h = max(1, int(target_h_px / widget_scale))
		frame_h, frame_w = frame.shape[:2]
		scale = min(target_w / frame_w, target_h / frame_h)
		display_w = max(1, int(frame_w * scale))
		display_h = max(1, int(frame_h * scale))
		interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
		return cv2.resize(frame, (display_w, display_h), interpolation=interp)

	def _schedule_next_frame(self) -> None:
		now = time.perf_counter()
		# Reset timing if the UI falls too far behind to avoid drift buildup.
		if self.next_frame_time < now - self.frame_interval_s:
			self.next_frame_time = now
		self.next_frame_time += self.frame_interval_s
		delay_ms = max(1, int((self.next_frame_time - now) * 1000))
		self.video_after_id = self.after(delay_ms, self.play_next_frame)

	def play_next_frame(self) -> None:
		if not self.video_capture:
			return
		ok, frame = self.video_capture.read()
		if not ok:
			self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
			ok, frame = self.video_capture.read()
			if not ok:
				return
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = self._fit_frame_to_view(frame)
		image = Image.fromarray(frame)
		self._video_image = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
		self.video_view.configure(text="", image=self._video_image)
		self._schedule_next_frame()

	def stop_video(self) -> None:
		if self.video_after_id is not None:
			self.after_cancel(self.video_after_id)
			self.video_after_id = None
		if self.video_capture:
			self.video_capture.release()
			self.video_capture = None

	def on_close(self) -> None:
		self.stop_video()
		self.destroy()

if __name__ == "__main__":
	app = App()
	app.mainloop()
