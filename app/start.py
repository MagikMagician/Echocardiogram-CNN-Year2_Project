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

VIDEO_FILETYPES = [("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
CATEGORY_NAMES = ["Reduced", "Mildly Reduced", "Preserved"]


class App(ctk.CTk):
	def __init__(self) -> None:
		super().__init__()
		self.cfg = Config()
		self.title("CustomTkinter App")
		self.geometry("800x500")
		self.artifacts_dir = str(BASE_DIR / "artifacts")
		self.model_folder = Path(self.artifacts_dir)
		self.selected_checkpoint_path = ""
		self.selected_video_path = ""
		self.video_capture = None
		self.preview_frames = None
		self.preview_index = 0
		self.video_after_id = None
		self.frame_interval_s = 1.0 / 30
		self.next_frame_time = 0.0
		self._video_image = None
		self.inference_busy = False
		self.protocol("WM_DELETE_WINDOW", self.on_close)

		top_bar = ctk.CTkFrame(self, fg_color="transparent")
		top_bar.place(relx=0.5, rely=0.02, relwidth=0.96, relheight=0.07, anchor="n")

		model_label = ctk.CTkLabel(top_bar, text="Model: ")
		model_label.pack(side="left", padx=(0, 8))

		self.model_dropdown = ctk.CTkComboBox(
			top_bar,
			values=["No .pt files found"],
			command=self._on_model_selected,
		)
		self.model_dropdown.pack(side="left", fill="x", expand=True, padx=(0, 8))

		browse_btn = ctk.CTkButton(top_bar, text="Choose Folder", width=120, command=self.pick_path)
		browse_btn.pack(side="right")
		self._refresh_model_dropdown(self.model_folder)

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

		self.run_btn = ctk.CTkButton(right_panel, text="Predict EF", command=self.run_inference)
		self.run_btn.pack(fill="x", padx=10, pady=(10, 8))

		self.gradcam_btn = ctk.CTkButton(right_panel, text="Generate Grad-CAM Video", command=self.generate_gradcam_video)
		self.gradcam_btn.pack(fill="x", padx=10, pady=(0, 8))

		self.result_text = ctk.CTkTextbox(right_panel)
		self.result_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
		self.result_text.configure(state="disabled")

	def pick_path(self) -> None:
		folder = filedialog.askdirectory(initialdir=str(self.model_folder))
		if folder:
			self.model_folder = Path(folder)
			self._refresh_model_dropdown(self.model_folder)

	def _refresh_model_dropdown(self, folder: Path, selected_name: str | None = None) -> None:
		pt_files = sorted(folder.glob("*.pt"))
		if not pt_files:
			self.model_dropdown.configure(values=["No .pt files found"])
			self.model_dropdown.set("No .pt files found")
			self.selected_checkpoint_path = ""
			return

		file_names = [p.name for p in pt_files]
		self.model_dropdown.configure(values=file_names)
		choice = selected_name if selected_name in file_names else file_names[0]
		self.model_dropdown.set(choice)
		self._on_model_selected(choice)

	def _on_model_selected(self, name: str) -> None:
		if name == "No .pt files found":
			self.selected_checkpoint_path = ""
			return
		self.selected_checkpoint_path = str(self.model_folder / name)

	def pick_video(self) -> None:
		path = filedialog.askopenfilename(
			filetypes=VIDEO_FILETYPES,
		)
		if path:
			self.selected_video_path = path
			self.video_name.configure(text=Path(path).name)
			self.start_video(path)

	def _compute_gradcam(self, model: torch.nn.Module, video: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
		if hasattr(model, "backbone"):
			target_layer = model.backbone[4][-1]
		else:
			target_layer = model.feature_extractor[-1]

		activations = None
		gradients = None

		def _save_activation(_module, _input, output):
			nonlocal activations
			activations = output

		def _save_gradient(_module, _grad_input, grad_output):
			nonlocal gradients
			gradients = grad_output[0]

		fwd_hook = target_layer.register_forward_hook(_save_activation)
		bwd_hook = target_layer.register_full_backward_hook(_save_gradient)
		try:
			_, logits = model(video)
			if class_idx is None:
				class_idx = int(logits.argmax(dim=1).item())
			model.zero_grad()
			logits[0, class_idx].backward()
			weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
			cam = (weights * activations).sum(dim=1, keepdim=True)
			cam = torch.relu(cam).squeeze().detach().cpu().numpy()
		finally:
			fwd_hook.remove()
			bwd_hook.remove()

		if cam.ndim == 2:
			cam = cam[np.newaxis]
		cam_min, cam_max = float(cam.min()), float(cam.max())
		if cam_max > cam_min:
			cam = (cam - cam_min) / (cam_max - cam_min)
		return cam

	def _start_preview_frames(self, frames_rgb: list[np.ndarray], fps: float) -> None:
		self.stop_video()
		if not frames_rgb:
			return
		self.preview_frames = frames_rgb
		self.preview_index = 0
		self.frame_interval_s = 1.0 / max(fps, 1.0)
		self.next_frame_time = time.perf_counter() + self.frame_interval_s
		self.play_next_frame()

	def generate_gradcam_video(self) -> None:
		if not self.selected_checkpoint_path:
			self._set_results("Please select a model checkpoint (.pt) first.")
			return
		if not self.selected_video_path:
			self._set_results("Please select a video first.")
			return

		self.gradcam_btn.configure(state="disabled", text="Generating...")
		self.update_idletasks()
		try:
			cap = cv2.VideoCapture(self.selected_video_path)
			fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
			full_frames_bgr = []
			while True:
				ok, frame = cap.read()
				if not ok:
					break
				full_frames_bgr.append(frame)
			cap.release()

			if not full_frames_bgr:
				self._set_results("Grad-CAM failed: no frames found in video.")
				return

			small_gray_frames = [
				cv2.cvtColor(
					cv2.resize(f, (self.cfg.TARGET_WIDTH, self.cfg.TARGET_HEIGHT), interpolation=cv2.INTER_AREA),
					cv2.COLOR_BGR2GRAY,
				)
				for f in full_frames_bgr
			]
			clip = np.stack(small_gray_frames).astype(np.float32)
			mean, std = self._load_stats()
			clip = clip / 255.0
			clip = (clip - mean) / std
			input_tensor = torch.from_numpy(clip).unsqueeze(0).unsqueeze(0).float()

			components = initialize_training_components(class_weights=None)
			checkpoint = torch.load(self.selected_checkpoint_path, map_location=components.device)
			components.model.load_state_dict(checkpoint["model_state_dict"])
			components.model.eval()

			video_device = input_tensor.to(components.device, non_blocking=True)
			cam = self._compute_gradcam(components.model, video_device)

			cam_t = cam.shape[0]
			overlay_frames_rgb = []
			for i, frame_bgr in enumerate(full_frames_bgr):
				idx = min(int(i * cam_t / len(full_frames_bgr)), cam_t - 1)
				cam_frame = cv2.resize(cam[idx], (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
				cam_u8 = np.clip(cam_frame * 255.0, 0, 255).astype(np.uint8)
				heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
				overlay = cv2.addWeighted(frame_bgr, 0.60, heatmap, 0.40, 0.0)
				overlay_frames_rgb.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

			out_dir = BASE_DIR / "artifacts" / "gradcam"
			out_dir.mkdir(parents=True, exist_ok=True)
			out_path = out_dir / "last_run_gradcam.mp4"
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), (full_frames_bgr[0].shape[1], full_frames_bgr[0].shape[0]))
			for frame_rgb in overlay_frames_rgb:
				writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
			writer.release()

			self._start_preview_frames(overlay_frames_rgb, fps)
			self._set_results(
				"Grad-CAM video generated and loaded in preview.\n"
				f"Saved: {out_path}"
			)
		except Exception as exc:
			self._set_results(f"Grad-CAM failed:\n{exc}")
		finally:
			self.gradcam_btn.configure(state="normal", text="Generate Grad-CAM Video")

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
			self.run_btn.configure(state="normal", text="Predict EF")

	def start_video(self, path: str) -> None:
		self.stop_video()
		self.preview_frames = None
		self.preview_index = 0
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
		if self.preview_frames is not None:
			if len(self.preview_frames) == 0:
				return
			frame = self.preview_frames[self.preview_index]
			self.preview_index = (self.preview_index + 1) % len(self.preview_frames)
		elif self.video_capture:
			ok, frame = self.video_capture.read()
			if not ok:
				self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
				ok, frame = self.video_capture.read()
				if not ok:
					return
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		else:
			return
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
