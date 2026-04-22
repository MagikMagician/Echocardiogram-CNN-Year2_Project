import os
import json
import time
import sys
import csv
import threading
from pathlib import Path
from tkinter import filedialog
from queue import Queue, Empty

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
CATEGORY_NAMES = ["Reduced (<40%)", "Mildly Reduced (40-49%)", "Preserved (≥50%)"]
QUALITY_COLORS = {"good": "#22c55e", "okay": "#f59e0b", "poor": "#ef4444"}
ROW_HEIGHT = 20
INDICATOR_WIDTH = 24
TRUE_EF_DELTA_WIDTH = 85
GRADCAM_DEFAULT_FPS = 30.0
GRADCAM_GENERATE_TEXT = "Generate Grad-CAM"
GRADCAM_EXPORT_TEXT = "Export Grad-CAM"
GRADCAM_POLL_INTERVAL_MS = 500

class App(ctk.CTk):
	def __init__(self) -> None:
		super().__init__()
		self.cfg = Config()
		self.title("Ejection Fraction Predictor")
		self.geometry("870x500")
		self.model_folder = BASE_DIR / "artifacts"
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
		self.last_predicted_ef: float | None = None
		self.last_predicted_classification: str | None = None
		self.last_pred_confidence: float | None = None
		self.last_pred_margin: float | None = None
		self.last_true_ef: float | None = None
		self.last_true_classification: str | None = None
		self.gradcam_thread: threading.Thread | None = None
		self.gradcam_poll_after_id = None
		self.gradcam_spinner_step = 0
		self.gradcam_queue: Queue = Queue()
		self.gradcam_frames: list[np.ndarray] | None = None
		self.gradcam_fps = GRADCAM_DEFAULT_FPS
		self.gradcam_source_video_path = ""
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

		self.error_banner = ctk.CTkLabel(right_panel, text="", anchor="w", wraplength=340, text_color="red")

		self.output_panel = ctk.CTkFrame(right_panel, fg_color="transparent")
		self.output_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

		self._section_header(self.output_panel, "Results")
		self.pred_ef_value, self.pred_ef_quality = self._value_row_with_indicator(self.output_panel, "Predicted EF")
		self.classification_value, self.classification_quality = self._value_row_with_indicator(self.output_panel, "Classification")
		self.confidence_value, self.confidence_quality = self._value_row_with_indicator(self.output_panel, "Confidence")
		self.margin_value, self.margin_quality = self._value_row_with_indicator(self.output_panel, "Confidence Margin")

		self._section_header(self.output_panel, "Processing Statistics")
		self.inference_time_value = self._value_row(self.output_panel, "Prediction Time")
		self.data_quality_value = self._value_row(self.output_panel, "Data Quality")
		self.video_meta_value = self._value_row(self.output_panel, "Video")

		self._section_header(self.output_panel, "True EF")
		self.true_ef_value, self.true_ef_delta_value = self._true_ef_row(self.output_panel, "True EF")
		self.true_ef_classification_value, self.true_ef_class_match_value = self._value_row_with_indicator(self.output_panel, "Classification")

		bottom_actions = ctk.CTkFrame(right_panel, fg_color="transparent")
		bottom_actions.pack(fill="x", padx=10, pady=(0, 10))

		self.gradcam_btn = ctk.CTkButton(bottom_actions, text=GRADCAM_GENERATE_TEXT, command=self.generate_gradcam_video)
		self.gradcam_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

		self.true_ef_btn = ctk.CTkButton(bottom_actions, text="Get True EF", command=self.lookup_true_ef)
		self.true_ef_btn.pack(side="right", fill="x", expand=True)

	def _section_header(self, parent: ctk.CTkFrame, title: str) -> None:
		self._section_divider(parent, pady=(5, 0))
		label = ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(weight="bold"))
		label.pack(pady=0)
		self._section_divider(parent, pady=(0, 3))

	def _section_divider(self, parent: ctk.CTkFrame, pady = 0) -> None:
		divider = ctk.CTkFrame(parent, height=2, fg_color="gray30")
		divider.pack(fill="x", padx=10, pady=pady)

	def _value_row(self, parent: ctk.CTkFrame, key: str) -> ctk.CTkLabel:
		value_label, _ = self._build_value_row(parent, key)
		return value_label

	def _build_value_row(self, parent: ctk.CTkFrame, key: str, right_widths: tuple[int, ...] = ()) -> tuple[ctk.CTkLabel, tuple[ctk.CTkLabel, ...]]:
		row = ctk.CTkFrame(parent, fg_color="transparent")
		row.pack(fill="x", padx=10, pady=0)
		key_label = ctk.CTkLabel(row, text=f"{key}:", width=130, height=ROW_HEIGHT, anchor="w")
		key_label.pack(side="left")
		value_label = ctk.CTkLabel(row, text="-", height=ROW_HEIGHT, anchor="w", justify="left")
		value_label.pack(side="left", fill="x", expand=True)
		right_labels: list[ctk.CTkLabel] = []
		for width in right_widths:
			right_label = ctk.CTkLabel(row, text="-", width=width, height=ROW_HEIGHT, anchor="e", text_color="gray70")
			right_label.pack(side="right")
			right_labels.append(right_label)
		return value_label, tuple(right_labels)

	def _value_row_with_indicator(self, parent: ctk.CTkFrame, key: str) -> tuple[ctk.CTkLabel, ctk.CTkLabel]:
		value_label, right_labels = self._build_value_row(parent, key, right_widths=(INDICATOR_WIDTH,))
		return value_label, right_labels[0]

	def _true_ef_row(self, parent: ctk.CTkFrame, key: str) -> tuple[ctk.CTkLabel, ctk.CTkLabel]:
		value_label, right_labels = self._build_value_row(parent, key, right_widths=(TRUE_EF_DELTA_WIDTH,))
		delta_label = right_labels[0]
		delta_label.configure(justify="right")
		return value_label, delta_label

	def _update_true_ef_comparison(self) -> None:
		if self.last_true_ef is None or self.last_predicted_ef is None:
			self.true_ef_delta_value.configure(text="-", text_color="gray70")
			self.true_ef_class_match_value.configure(text="-", text_color="gray70")
			return

		diff = self.last_predicted_ef - self.last_true_ef
		if diff > 0:
			arrow = "▲"
		elif diff < 0:
			arrow = "▼"
		else:
			arrow = "="
		diff_abs = abs(diff)
		if diff_abs <= 3.0:
			delta_color = "#22c55e"
		elif diff_abs <= 10.0:
			delta_color = "#f59e0b"
		else:
			delta_color = "#ef4444"
		self.true_ef_delta_value.configure(text=f"{arrow} {diff_abs:.2f}%", text_color=delta_color)

		if self.last_predicted_classification is None or self.last_true_classification is None:
			self.true_ef_class_match_value.configure(text="-", text_color="gray70")
			return

		is_match = self.last_predicted_classification == self.last_true_classification
		self.true_ef_class_match_value.configure(text="✔" if is_match else "❌", text_color="#22c55e" if is_match else "#ef4444")

	def _set_quality_circle(self, label: ctk.CTkLabel, level: str | None) -> None:
		if level in QUALITY_COLORS:
			label.configure(text="●", text_color=QUALITY_COLORS[level])
		else:
			label.configure(text="-", text_color="gray70")

	def _score_by_threshold(self, value: float | None, good_min: float, okay_min: float) -> str | None:
		if value is None:
			return None
		if value >= good_min:
			return "good"
		if value >= okay_min:
			return "okay"
		return "poor"

	def _score_probability(self, probability: float | None) -> str | None:
		return self._score_by_threshold(probability, good_min=0.35, okay_min=0.12)

	def _score_margin(self, margin: float | None) -> str | None:
		return self._score_by_threshold(margin, good_min=0.30, okay_min=0.15)

	def _score_ef_band(self, ef_value: float | None) -> str | None:
		return self._score_by_threshold(ef_value, good_min=50.0, okay_min=40.0)

	def _score_classification_band(self, classification: str | None) -> str | None:
		if classification is None:
			return None
		if classification == CATEGORY_NAMES[2]:
			return "good"
		if classification == CATEGORY_NAMES[1]:
			return "okay"
		if classification == CATEGORY_NAMES[0]:
			return "poor"
		return None

	def _update_result_quality_indicators(self) -> None:
		self._set_quality_circle(
			self.pred_ef_quality,
			self._score_ef_band(self.last_predicted_ef),
		)
		self._set_quality_circle(
			self.classification_quality,
			self._score_classification_band(self.last_predicted_classification),
		)
		self._set_quality_circle(self.confidence_quality, self._score_probability(self.last_pred_confidence))
		self._set_quality_circle(self.margin_quality, self._score_margin(self.last_pred_margin))

	def _set_error(self, message: str) -> None:
		if message:
			self.error_banner.configure(text=message)
			if self.error_banner.winfo_manager() == "":
				self.error_banner.pack(after=self.run_btn, fill="x", padx=10, pady=0)
		else:
			if self.error_banner.winfo_manager() != "":
				self.error_banner.pack_forget()

	def pick_path(self) -> None:
		folder = filedialog.askdirectory(initialdir=str(self.model_folder))
		if folder:
			self.model_folder = Path(folder)
			self._refresh_model_dropdown(self.model_folder)

	def _refresh_model_dropdown(self, folder: Path) -> None:
		pt_files = sorted(folder.glob("*.pt"))
		if not pt_files:
			self.model_dropdown.configure(values=["No .pt files found"])
			self.model_dropdown.set("No .pt files found")
			self.selected_checkpoint_path = ""
			return

		file_names = [p.name for p in pt_files]
		self.model_dropdown.configure(values=file_names)
		choice = file_names[0]
		self.model_dropdown.set(choice)
		self._on_model_selected(choice)

	def _on_model_selected(self, name: str) -> None:
		if name == "No .pt files found":
			self.selected_checkpoint_path = ""
			return
		self.selected_checkpoint_path = str(self.model_folder / name)

	def _reset_prediction_state(self) -> None:
		self.last_predicted_ef = None
		self.last_predicted_classification = None
		self.last_pred_confidence = None
		self.last_pred_margin = None
		self.pred_ef_value.configure(text="-")
		self.classification_value.configure(text="-")
		self.confidence_value.configure(text="-")
		self.margin_value.configure(text="-")
		self.inference_time_value.configure(text="-")
		self.data_quality_value.configure(text="-")
		self.video_meta_value.configure(text="-")
		self._update_result_quality_indicators()

	def _reset_true_ef_state(self) -> None:
		self.last_true_ef = None
		self.last_true_classification = None
		self.true_ef_value.configure(text="-")
		self.true_ef_classification_value.configure(text="-")
		self._update_true_ef_comparison()

	def pick_video(self) -> None:
		old_video_path = self.selected_video_path
		path = filedialog.askopenfilename(
			filetypes=VIDEO_FILETYPES,
		)
		if path:
			self.selected_video_path = path
			if path != old_video_path:
				self._set_gradcam_generate_mode()
			self._reset_prediction_state()
			self._reset_true_ef_state()
			self._set_error("")
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

	def _set_gradcam_generate_mode(self) -> None:
		self.gradcam_frames = None
		self.gradcam_fps = GRADCAM_DEFAULT_FPS
		self.gradcam_source_video_path = ""
		self.gradcam_btn.configure(state="normal", text=GRADCAM_GENERATE_TEXT, command=self.generate_gradcam_video)

	def _set_gradcam_export_mode(self, frames_rgb: list[np.ndarray], fps: float, source_video_path: str) -> None:
		self.gradcam_frames = frames_rgb
		self.gradcam_fps = fps
		self.gradcam_source_video_path = source_video_path
		self.gradcam_btn.configure(state="normal", text=GRADCAM_EXPORT_TEXT, command=self.export_gradcam_video)

	def _load_video_frames_bgr(self, video_path: str) -> tuple[list[np.ndarray], float]:
		cap = cv2.VideoCapture(video_path)
		fps = float(cap.get(cv2.CAP_PROP_FPS) or GRADCAM_DEFAULT_FPS)
		frames_bgr: list[np.ndarray] = []
		while True:
			ok, frame = cap.read()
			if not ok:
				break
			frames_bgr.append(frame)
		cap.release()
		return frames_bgr, fps

	def _prepare_gradcam_tensor(self, full_frames_bgr: list[np.ndarray]) -> torch.Tensor:
		small_gray_frames = [
			cv2.cvtColor(
				cv2.resize(frame, (self.cfg.TARGET_WIDTH, self.cfg.TARGET_HEIGHT), interpolation=cv2.INTER_AREA),
				cv2.COLOR_BGR2GRAY,
			)
			for frame in full_frames_bgr
		]
		clip = np.stack(small_gray_frames).astype(np.float32)
		mean, std = self._load_stats()
		clip = clip / 255.0
		clip = (clip - mean) / std
		return torch.from_numpy(clip).unsqueeze(0).unsqueeze(0).float()

	def _build_gradcam_overlays(self, full_frames_bgr: list[np.ndarray], cam: np.ndarray) -> list[np.ndarray]:
		cam_t = cam.shape[0]
		overlay_frames_rgb: list[np.ndarray] = []
		for i, frame_bgr in enumerate(full_frames_bgr):
			idx = min(int(i * cam_t / len(full_frames_bgr)), cam_t - 1)
			cam_frame = cv2.resize(cam[idx], (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
			cam_u8 = np.clip(cam_frame * 255.0, 0, 255).astype(np.uint8)
			heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
			overlay = cv2.addWeighted(frame_bgr, 0.60, heatmap, 0.40, 0.0)
			overlay_frames_rgb.append(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
		return overlay_frames_rgb

	def export_gradcam_video(self) -> None:
		if not self.gradcam_frames:
			self._set_error("Please generate Grad-CAM first.")
			return

		default_name = "gradcam_output.mp4"
		if self.gradcam_source_video_path:
			default_name = f"{Path(self.gradcam_source_video_path).stem}_gradcam.mp4"
		initial_dir = str(Path(self.selected_video_path).parent) if self.selected_video_path else str(BASE_DIR)

		output_path = filedialog.asksaveasfilename(
			title="Export Grad-CAM Video",
			initialdir=initial_dir,
			initialfile=default_name,
			defaultextension=".mp4",
			filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
		)
		if not output_path:
			return

		frames_rgb = self.gradcam_frames
		height, width = frames_rgb[0].shape[:2]
		fps = max(float(self.gradcam_fps), 1.0)
		ext = Path(output_path).suffix.lower()
		fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
		writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
		if not writer.isOpened():
			self._set_error(f"Could not create output file:\n{output_path}")
			return

		try:
			for frame_rgb in frames_rgb:
				writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
		finally:
			writer.release()

		self._set_error("")

	def _generate_gradcam_worker(self, video_path: str, checkpoint_path: str) -> None:
		try:
			full_frames_bgr, fps = self._load_video_frames_bgr(video_path)

			if not full_frames_bgr:
				self.gradcam_queue.put(("error", "Grad-CAM failed: no frames found in video."))
				return

			input_tensor = self._prepare_gradcam_tensor(full_frames_bgr)

			components = self._load_selected_model_components(checkpoint_path)

			video_device = input_tensor.to(components.device, non_blocking=True)
			cam = self._compute_gradcam(components.model, video_device)
			overlay_frames_rgb = self._build_gradcam_overlays(full_frames_bgr, cam)

			self.gradcam_queue.put(("ok", (overlay_frames_rgb, fps, video_path)))
		except Exception as exc:
			self.gradcam_queue.put(("error", f"Grad-CAM failed:\n{exc}"))

	def _poll_gradcam_worker(self) -> None:
		try:
			status, payload = self.gradcam_queue.get_nowait()
		except Empty:
			if self.gradcam_thread and self.gradcam_thread.is_alive():
				dots = "." * ((self.gradcam_spinner_step % 4))
				self.gradcam_spinner_step += 1
				self.gradcam_btn.configure(text=f"Generating{dots}")
				self.gradcam_poll_after_id = self.after(GRADCAM_POLL_INTERVAL_MS, self._poll_gradcam_worker)
			else:
				self._set_gradcam_generate_mode()
				self.gradcam_thread = None
				self.gradcam_poll_after_id = None
			return

		self.gradcam_poll_after_id = None
		self.gradcam_thread = None
		if status == "ok":
			overlay_frames_rgb, fps, source_video_path = payload
			if source_video_path != self.selected_video_path:
				self._set_gradcam_generate_mode()
				return
			self._start_preview_frames(overlay_frames_rgb, fps)
			self._set_gradcam_export_mode(overlay_frames_rgb, fps, source_video_path)
			self._set_error("")
		else:
			self._set_gradcam_generate_mode()
			self._set_error(str(payload))

	def generate_gradcam_video(self) -> None:
		if not self.selected_checkpoint_path:
			self._set_error("Please select a model checkpoint (.pt) first.")
			return
		if not self.selected_video_path:
			self._set_error("Please select a video first.")
			return
		if self.gradcam_thread and self.gradcam_thread.is_alive():
			return

		self.gradcam_btn.configure(state="disabled", text="Generating...")
		self.gradcam_spinner_step = 0
		while True:
			try:
				self.gradcam_queue.get_nowait()
			except Empty:
				break
		video_path = self.selected_video_path
		checkpoint_path = self.selected_checkpoint_path
		self.gradcam_thread = threading.Thread(
			target=self._generate_gradcam_worker,
			args=(video_path, checkpoint_path),
			daemon=True,
		)
		self.gradcam_thread.start()
		self._poll_gradcam_worker()

	def _load_selected_model_components(self, checkpoint_path: str | None = None):
		if checkpoint_path is None:
			checkpoint_path = self.selected_checkpoint_path
		components = initialize_training_components(class_weights=None)
		checkpoint = torch.load(checkpoint_path, map_location=components.device)
		components.model.load_state_dict(checkpoint["model_state_dict"])
		components.model.eval()
		return components

	def _video_metadata(self, video_path: str) -> tuple[float, int, int, int, float]:
		cap = cv2.VideoCapture(video_path)
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		cap.release()
		duration = (frame_count / fps) if fps > 0 else 0.0
		return fps, frame_count, width, height, duration

	def _classify_ef_value(self, ef_value: float) -> str:
		if ef_value < 40.0:
			return CATEGORY_NAMES[0]
		if ef_value < 50.0:
			return CATEGORY_NAMES[1]
		return CATEGORY_NAMES[2]

	def lookup_true_ef(self) -> None:
		if not self.selected_video_path:
			self._set_error("Please select a video first.")
			return

		video_path = Path(self.selected_video_path)
		csv_path = video_path.parent.parent / "FileList.csv"
		if not csv_path.exists():
			csv_path = video_path.parent / "FileList.csv"
		if not csv_path.exists():
			picked = filedialog.askopenfilename(
				title="Select FileList.csv",
				filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
			)
			if not picked:
				self._set_error("FileList.csv not found automatically and no CSV was selected.")
				return
			csv_path = Path(picked)

		target_stem = video_path.stem.lower()
		target_name = video_path.name.lower()

		try:
			with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
				reader = csv.DictReader(f)
				if not reader.fieldnames or "FileName" not in reader.fieldnames or "EF" not in reader.fieldnames:
					self._set_error(f"CSV missing required columns FileName/EF:\n{csv_path}")
					return

				for row in reader:
					file_name = str(row.get("FileName", "")).strip()
					if not file_name:
						continue
					candidates = {
						file_name.lower(),
						Path(file_name).name.lower(),
						Path(file_name).stem.lower(),
					}
					if target_stem in candidates or target_name in candidates:
						ef_val = str(row.get("EF", "")).strip()
						if not ef_val:
							self._set_error(f"Found video in CSV but EF is empty.\nCSV: {csv_path}")
							return
						try:
							ef_float = float(ef_val)
						except ValueError:
							self._set_error(f"Found EF value is not numeric: '{ef_val}'.\nCSV: {csv_path}")
							return

						self.true_ef_value.configure(text=f"{ef_float:.2f}%")
						true_class = self._classify_ef_value(ef_float)
						self.true_ef_classification_value.configure(text=true_class)
						self.last_true_ef = ef_float
						self.last_true_classification = true_class
						self._update_true_ef_comparison()
						self._set_error("")
						return
		except Exception as exc:
			self._set_error(f"Failed to read CSV:\n{exc}")
			return

		self._set_error(
			f"No matching FileName found for video: {video_path.name}\nCSV: {csv_path}"
		)

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

	def _compute_data_quality(self, frames: np.ndarray) -> str:
		total = int(frames.shape[0]) if frames.ndim >= 1 else 0
		if total <= 0:
			return "Unknown"

		# Padding in load_frames_from_video is zero-filled frames.
		usable_mask = np.any(frames != 0, axis=(1, 2))
		usable = int(np.sum(usable_mask))
		padded = total - usable
		usable_pct = (usable / total) * 100.0

		if padded == 0:
			label = "Good"
		elif usable_pct >= 85.0:
			label = "Okay"
		else:
			label = "Poor"

		return f"{label} ({usable}/{total} usable, {padded} padded)"

	def _prepare_video_tensor(self, video_path: str) -> tuple[torch.Tensor, str] | None:
		frames = load_frames_from_video(
			video_path=video_path,
			num_frames=self.cfg.NUM_FRAMES,
			target_size=(self.cfg.TARGET_WIDTH, self.cfg.TARGET_HEIGHT),
			period=self.cfg.FRAME_SAMPLING_PERIOD,
			random_start=False,
		)
		if frames is None:
			return None
		data_quality = self._compute_data_quality(frames)
		mean, std = self._load_stats()
		frames = frames.astype(np.float32) / 255.0
		frames = (frames - mean) / std
		video_tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0).float()
		return video_tensor, data_quality

	def run_inference(self) -> None:
		if self.inference_busy:
			return
		if not self.selected_checkpoint_path:
			self._set_error("Please select a model checkpoint (.pt) first.")
			return
		if not self.selected_video_path:
			self._set_error("Please select a video first.")
			return

		self.inference_busy = True
		self.run_btn.configure(state="disabled", text="Running...")
		self.update_idletasks()

		try:
			prepared = self._prepare_video_tensor(self.selected_video_path)
			if prepared is None:
				self._set_error("Could not read enough frames from the selected video.")
				return
			input_tensor, data_quality_text = prepared

			components = self._load_selected_model_components()

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

			fps, frame_count, width, height, duration = self._video_metadata(self.selected_video_path)
			self.pred_ef_value.configure(text=f"{pred_ef:.2f}%")
			self.classification_value.configure(text=pred_class)
			self.confidence_value.configure(text=f"{pred_conf * 100:.2f}%")
			self.margin_value.configure(text=f"{margin * 100:.2f}%")
			self.inference_time_value.configure(text=f"{inference_ms:.1f} ms")
			self.data_quality_value.configure(text=data_quality_text)
			self.video_meta_value.configure(text=f"{width}x{height}, {fps:.2f} FPS, {frame_count} frames, {duration:.2f}s")
			self.last_predicted_ef = pred_ef
			self.last_predicted_classification = pred_class
			self.last_pred_confidence = pred_conf
			self.last_pred_margin = margin
			self._update_result_quality_indicators()
			self._update_true_ef_comparison()
			self._set_error("")
		except Exception as exc:
			self._set_error(f"Prediction failed:\n{exc}")
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
		if self.gradcam_poll_after_id is not None:
			self.after_cancel(self.gradcam_poll_after_id)
			self.gradcam_poll_after_id = None
		self.stop_video()
		self.destroy()

if __name__ == "__main__":
	app = App()
	app.mainloop()
