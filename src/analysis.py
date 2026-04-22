# Core Deep Learning
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Visualization
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Utilities
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from src.CNN_model import R2Plus1DEF
from src.config import Config
from src.data_processing import (
    create_dataloaders,
    load_video_paths_and_labels,
)
from src.training import initialize_training_components, TrainingComponents

config = Config()

CATEGORY_NAMES = ["Reduced (<40%)", "Mildly Reduced (40-49%)", "Preserved (≥50%)"]
ARTIFACTS_DIR = Path("artifacts")

# =============================================================================
# Test set evaluation — loads the best checkpoint and computes regression
# metrics (MAE, RMSE, R²) and classification metrics (accuracy, ROC-AUC)
# with 95 % bootstrap confidence intervals, plus three diagnostic plots.
# =============================================================================


def _load_best_model(
    components: TrainingComponents,
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
) -> TrainingComponents:
    """Load weights from the best checkpoint into ``components.model``."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run training first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=components.device)
    components.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} "
          f"(val_loss={checkpoint.get('val_loss', float('nan')):.4f})")
    return components


def _collect_predictions(
    loader,
    model: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on all batches in *loader* and return predictions + labels.

    Returns:
        ef_true   – ground-truth EF values  (N,)
        ef_pred   – predicted EF values     (N,)
        cat_true  – ground-truth categories (N,)
        cat_pred  – predicted categories    (N,)
    """
    model.eval()
    ef_trues, ef_preds, cat_trues, cat_preds = [], [], [], []

    with torch.no_grad():
        for videos, ef_values, ef_classes in loader:
            videos = videos.to(device, non_blocking=True)
            ef_reg, ef_logits = model(videos)

            ef_trues.append(ef_values.numpy())
            ef_preds.append(ef_reg.cpu().numpy())
            cat_trues.append(ef_classes.numpy())
            cat_preds.append(ef_logits.argmax(dim=1).cpu().numpy())

    return (
        np.concatenate(ef_trues),
        np.concatenate(ef_preds),
        np.concatenate(cat_trues),
        np.concatenate(cat_preds),
    )


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Save a labelled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CATEGORY_NAMES, yticklabels=CATEGORY_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("EF Category – Confusion Matrix")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def _plot_regression_scatter(
    ef_true: np.ndarray, ef_pred: np.ndarray, save_path: Path
) -> None:
    """Save a scatter plot of predicted vs actual EF values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ef_true, ef_pred, alpha=0.4, s=8, color="steelblue")
    # Identity line
    lo, hi = ef_true.min() - 5, ef_true.max() + 5
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual EF (%)")
    ax.set_ylabel("Predicted EF (%)")
    ax.set_title("Predicted vs Actual Ejection Fraction")
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Regression scatter saved → {save_path}")


def _plot_training_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    """Save loss and MAE training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Combined Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_mae"], label="Train")
    axes[1].plot(epochs, history["val_mae"], label="Val")
    axes[1].set_title("MAE (EF %)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["train_accuracy"], label="Train")
    axes[2].plot(epochs, history["val_accuracy"], label="Val")
    axes[2].set_title("Category Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle("Training Curves")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {save_path}")


def evaluate_on_test_set(
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
    output_dir: Path = ARTIFACTS_DIR / "evaluation",
) -> Dict[str, float]:
    """
    Load the best checkpoint and evaluate comprehensively on the held-out test set.

    Produces:
    - Regression metrics: MAE, RMSE, R²
    - Classification metrics: accuracy, per-class precision/recall/F1
    - ROC-AUC (one-vs-rest)
    - Confusion matrix plot
    - Predicted vs actual EF scatter plot

    Returns a dict of scalar metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, class_weights = create_dataloaders()
    components = initialize_training_components(class_weights=class_weights)
    components = _load_best_model(components, checkpoint_path)

    ef_true, ef_pred, cat_true, cat_pred = _collect_predictions(
        test_loader, components.model, components.device
    )

    # ── Regression metrics with 95% bootstrap CIs (paper methodology) ────────
    _mae_fn  = lambda yt, yp: mean_absolute_error(yt, yp)
    _rmse_fn = lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp)))
    _r2_fn   = lambda yt, yp: r2_score(yt, yp)

    print("Computing bootstrap confidence intervals (10,000 samples)...")
    mae,  mae_lo,  mae_hi  = _bootstrap_ci(ef_true, ef_pred, _mae_fn)
    rmse, rmse_lo, rmse_hi = _bootstrap_ci(ef_true, ef_pred, _rmse_fn)
    r2,   r2_lo,   r2_hi   = _bootstrap_ci(ef_true, ef_pred, _r2_fn)

    # ── Classification metrics ────────────────────────────────────────────────
    acc = accuracy_score(cat_true, cat_pred)

    # ROC-AUC needs class probabilities, not argmax predictions, so we re-run
    # the forward pass and collect softmax outputs separately.
    components.model.eval()
    all_probs: List[np.ndarray] = []
    with torch.no_grad():
        for videos, _, _ in test_loader:
            videos = videos.to(components.device, non_blocking=True)
            _, logits = components.model(videos)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    prob_matrix = np.concatenate(all_probs, axis=0)

    try:
        roc_auc = roc_auc_score(cat_true, prob_matrix, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"  MAE  : {mae:.2f}%  (95% CI: {mae_lo:.2f}–{mae_hi:.2f}%)")
    print(f"  RMSE : {rmse:.2f}%  (95% CI: {rmse_lo:.2f}–{rmse_hi:.2f}%)")
    print(f"  R²   : {r2:.4f}   (95% CI: {r2_lo:.4f}–{r2_hi:.4f})")
    print(f"  Accuracy (categories): {acc:.4f}")
    print(f"  ROC-AUC (OvR)        : {roc_auc:.4f}")
    print()
    print(classification_report(cat_true, cat_pred, target_names=CATEGORY_NAMES))
    print("=" * 60)

    # ── Persist per-video predictions for future post-hoc analysis ───────────
    # Saves ef_true, ef_pred, cat_true, cat_pred, and per-class softmax probs
    # so any additional metric can be computed later without re-running inference.
    import pandas as pd
    pred_df = pd.DataFrame({
        "ef_true":   ef_true,
        "ef_pred":   ef_pred,
        "cat_true":  cat_true,
        "cat_pred":  cat_pred,
        "abs_error": np.abs(ef_true - ef_pred),
        "prob_reduced":       prob_matrix[:, 0],
        "prob_mildly_reduced": prob_matrix[:, 1],
        "prob_preserved":     prob_matrix[:, 2],
    })
    pred_csv = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"Per-video predictions saved → {pred_csv}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    _plot_confusion_matrix(cat_true, cat_pred, output_dir / "confusion_matrix.png")
    _plot_regression_scatter(ef_true, ef_pred, output_dir / "ef_scatter.png")
    _plot_bland_altman(ef_true, ef_pred, output_dir / "bland_altman.png")

    return {
        "mae": mae, "mae_ci": (mae_lo, mae_hi),
        "rmse": rmse, "rmse_ci": (rmse_lo, rmse_hi),
        "r2": r2, "r2_ci": (r2_lo, r2_hi),
        "accuracy": acc, "roc_auc": roc_auc,
    }


# =============================================================================
# Grad-CAM — gradient-weighted saliency maps that highlight which regions
# of the echocardiogram clip the model attends to when predicting EF.
# Helps verify that attention focuses on the left ventricle, not background.
# =============================================================================


class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3-D CNN models.

    Registers a forward/backward hook on the target layer, runs one forward
    pass, and produces a spatiotemporal saliency map of shape (T, H, W).

    Usage::

        gcam = GradCAM3D(model, target_layer=model.backbone[4][-1])
        heatmap = gcam(video_tensor, class_idx=None)   # None → argmax
    """

    def __init__(self, model: R2Plus1DEF, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output: torch.Tensor) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output: Tuple) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __call__(
        self,
        video: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute a Grad-CAM saliency map.

        Args:
            video: (1, 1, T, H, W) tensor on the same device as the model.
            class_idx: Class to back-propagate from. If None, uses the argmax.

        Returns:
            Saliency map as a numpy array of shape (T, H, W) in [0, 1].
        """
        _, logits = self.model(video)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward from the target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM: weight each activation channel by its mean gradient, sum
        # across channels, then ReLU to keep only positively-contributing regions.
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, t, h, w)
        cam = torch.relu(cam).squeeze().cpu().numpy()  # (t, h, w) or (h, w)

        if cam.ndim == 2:
            cam = cam[np.newaxis]  # (1, h, w)

        # Min-max normalise
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam


def run_gradcam_visualization(
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
    num_samples: int = 4,
    output_dir: Path = ARTIFACTS_DIR / "gradcam",
) -> None:
    """
    Generate and save Grad-CAM overlays for ``num_samples`` test videos.

    Each output figure shows a row of frames with the saliency heatmap
    blended on top, helping verify that the model attends to the left
    ventricle rather than background artefacts.
    """
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. Skipping Grad-CAM.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, class_weights = create_dataloaders()
    components = initialize_training_components(class_weights=class_weights)
    components = _load_best_model(components, checkpoint_path)
    model = components.model
    device = components.device

    # model.backbone is nn.Sequential(stem, layer1, layer2, layer3, layer4, avgpool)
    # layer4 (index 4) is the deepest residual block — best for saliency.
    target_layer = model.backbone[4][-1]
    gcam = GradCAM3D(model, target_layer)

    samples_done = 0
    for videos, ef_values, ef_classes in test_loader:
        for i in range(videos.size(0)):
            if samples_done >= num_samples:
                break

            video_single = videos[i : i + 1].to(device)  # (1, 1, T, H, W)
            true_ef = ef_values[i].item()
            true_cat = ef_classes[i].item()

            cam = gcam(video_single)  # (t_small, h_small, w_small) after pool
            # Original frames: (T, H, W)
            frames = videos[i, 0].numpy()  # unnormalized; values may be negative

            # Select evenly spaced display frames (up to 8)
            T = frames.shape[0]
            display_idx = np.linspace(0, T - 1, min(8, T), dtype=int)

            fig, axes = plt.subplots(2, len(display_idx), figsize=(2 * len(display_idx), 5))

            for col, t in enumerate(display_idx):
                # ── top row: raw frame ────────────────────────────────────────
                frame = frames[t]
                frame_disp = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-6)
                axes[0, col].imshow(frame_disp, cmap="gray", vmin=0, vmax=1)
                axes[0, col].axis("off")
                if col == 0:
                    axes[0, col].set_ylabel("Frame", fontsize=8)

                # ── bottom row: Grad-CAM overlay ──────────────────────────────
                # Resize cam slice to match frame size using nearest-neighbour
                cam_t = int(t * cam.shape[0] / T)
                cam_t = min(cam_t, cam.shape[0] - 1)
                cam_frame = cam[cam_t]  # (h_small, w_small)
                cam_resized = cv2.resize(
                    cam_frame, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                axes[1, col].imshow(frame_disp, cmap="gray", vmin=0, vmax=1)
                axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.45, vmin=0, vmax=1)
                axes[1, col].axis("off")
                if col == 0:
                    axes[1, col].set_ylabel("Grad-CAM", fontsize=8)

            fig.suptitle(
                f"Sample {samples_done + 1} | True EF: {true_ef:.1f}% "
                f"| Category: {CATEGORY_NAMES[int(true_cat)]}",
                fontsize=9,
            )
            fig.tight_layout()
            out_file = output_dir / f"gradcam_sample_{samples_done + 1:03d}.png"
            fig.savefig(out_file, dpi=150)
            plt.close(fig)
            samples_done += 1
            print(f"Grad-CAM saved → {out_file}")

        if samples_done >= num_samples:
            break

    gcam.remove_hooks()
    print(f"Grad-CAM visualizations complete ({samples_done} samples).")


# =============================================================================
# Grad-CAM with expert LV overlay — three-row visual comparison showing
# (1) raw frame, (2) cardiologist's LV contour, and (3) the model's Grad-CAM
# attention.  Only run on videos that have VolumeTracings annotations so we
# can directly compare model attention to expert anatomy frame-by-frame.
# =============================================================================


def run_gradcam_with_lv_overlay(
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
    tracings_path: Path = Path("dataset/VolumeTracings.csv"),
    num_samples: int = 8,
    output_dir: Path = ARTIFACTS_DIR / "gradcam",
) -> None:
    """
    Produce three-row Grad-CAM figures that layer the model's attention on top
    of the cardiologist's LV contour for frames that have expert annotations.

    Row 1: raw echocardiogram frame
    Row 2: frame + cyan LV contour from VolumeTracings
    Row 3: frame + Grad-CAM heatmap (model's attention)

    This makes model-vs-expert comparison intuitive at a glance: if the
    heatmap sits inside the cyan contour, the model is "looking at the heart".
    """
    import pandas as pd

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. Skipping overlay Grad-CAM.")
        return
    if not tracings_path.exists():
        print(f"VolumeTracings not found at {tracings_path}. Skipping overlay Grad-CAM.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    tracings = pd.read_csv(tracings_path)
    traced_files = set(tracings["FileName"].unique())

    _, _, test_loader, class_weights = create_dataloaders()
    components = initialize_training_components(class_weights=class_weights)
    components = _load_best_model(components, checkpoint_path)
    model = components.model
    device = components.device

    target_layer = model.backbone[4][-1]
    gcam = GradCAM3D(model, target_layer)

    # Iterate per-video via dataset so we can filter to annotated ones.
    dataset = test_loader.dataset
    samples_done = 0

    for idx in range(len(dataset)):
        if samples_done >= num_samples:
            break

        fname = Path(dataset.video_paths[idx]).name
        if fname not in traced_files:
            continue

        video_tracings = tracings[tracings["FileName"] == fname]
        annotated_frames = sorted(int(f) for f in video_tracings["Frame"].unique())
        if not annotated_frames:
            continue

        # Determine the actual clip window the dataset used for this video.
        # Test set uses random_start=False, so start = (total - window) // 2.
        cap = cv2.VideoCapture(dataset.video_paths[idx])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        num_frames = dataset.num_frames
        period = dataset.period
        window = num_frames * period
        if total_frames <= window:
            clip_start = 0
            clip_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            clip_start = (total_frames - window) // 2
            clip_indices = np.arange(clip_start, clip_start + window, period, dtype=int)

        # Keep only annotated frames that fall inside this clip window.
        def _orig_to_sampled(orig: int) -> Optional[int]:
            diffs = np.abs(clip_indices - orig)
            nearest = int(np.argmin(diffs))
            return nearest if diffs[nearest] <= period // 2 else None

        annotated_in_clip = [
            (orig, _orig_to_sampled(orig)) for orig in annotated_frames
        ]
        annotated_in_clip = [(o, s) for o, s in annotated_in_clip if s is not None]
        if not annotated_in_clip:
            continue  # Skip — no annotated frame falls inside the sampled clip

        video_tensor, ef_true, ef_class = dataset[idx]
        video_input = video_tensor.unsqueeze(0).to(device)
        cam = gcam(video_input)  # (T_cam, h, w) normalised [0,1]

        frames = video_tensor[0].cpu().numpy()  # (T, H, W)
        T = frames.shape[0]

        # Build display: the annotated sampled indices plus 4 evenly spaced fillers.
        filler = list(np.linspace(0, T - 1, 4, dtype=int))
        display_idx = sorted(set(filler + [s for _, s in annotated_in_clip]))[:6]

        # Map each sampled index back to the annotation it corresponds to (if any)
        sampled_to_orig = {s: o for o, s in annotated_in_clip}

        fig, axes = plt.subplots(3, len(display_idx),
                                 figsize=(2.2 * len(display_idx), 7))
        if len(display_idx) == 1:
            axes = axes[:, np.newaxis]

        for col, t in enumerate(display_idx):
            frame = frames[t]
            frame_disp = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-6)

            # ── Row 1: raw frame ──────────────────────────────────────────────
            axes[0, col].imshow(frame_disp, cmap="gray", vmin=0, vmax=1)
            axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
            if col == 0:
                axes[0, col].set_ylabel("Frame", fontsize=9)

            # ── Row 2: expert LV contour overlay (only on annotated frames) ───
            axes[1, col].imshow(frame_disp, cmap="gray", vmin=0, vmax=1)
            axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
            if col == 0:
                axes[1, col].set_ylabel("Expert LV", fontsize=9)

            if t in sampled_to_orig:
                orig_frame = sampled_to_orig[t]
                lv_mask = _build_lv_mask(
                    video_tracings, orig_frame,
                    height=config.TARGET_HEIGHT, width=config.TARGET_WIDTH,
                )
                if lv_mask is not None and lv_mask.sum() > 0:
                    contours, _ = cv2.findContours(
                        lv_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    for contour in contours:
                        pts = contour.squeeze()
                        if pts.ndim == 2 and len(pts) > 2:
                            axes[1, col].plot(
                                np.append(pts[:, 0], pts[0, 0]),
                                np.append(pts[:, 1], pts[0, 1]),
                                color="cyan", linewidth=2,
                            )
                    axes[1, col].set_title(f"ED/ES (f={orig_frame})",
                                            fontsize=7, color="cyan")

            # ── Row 3: Grad-CAM overlay ───────────────────────────────────────
            cam_t = min(int(t * cam.shape[0] / T), cam.shape[0] - 1)
            cam_resized = cv2.resize(
                cam[cam_t], (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            axes[2, col].imshow(frame_disp, cmap="gray", vmin=0, vmax=1)
            axes[2, col].imshow(cam_resized, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            axes[2, col].set_xticks([]); axes[2, col].set_yticks([])
            if col == 0:
                axes[2, col].set_ylabel("Grad-CAM", fontsize=9)

        fig.suptitle(
            f"Overlay Sample {samples_done + 1} | {fname} | "
            f"True EF: {float(ef_true):.1f}% | "
            f"Category: {CATEGORY_NAMES[int(ef_class)]}",
            fontsize=9,
        )
        fig.tight_layout()
        out_file = output_dir / f"gradcam_overlay_{samples_done + 1:03d}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        samples_done += 1
        print(f"Grad-CAM overlay saved → {out_file}")

    gcam.remove_hooks()
    print(f"Grad-CAM LV-overlay visualizations complete ({samples_done} samples).")


# =============================================================================
# Ablation study — trains four R(2+1)D variants to isolate the contribution
# of frame count and each loss head.  Each variant gets a fresh model and
# optimizer so results are independent.
# =============================================================================


def _ablation_variant(
    tag: str,
    num_frames: int,
    num_epochs: int,
    regression_weight: float,
    classification_weight: float,
    pretrained: bool = True,
    use_shared_head: bool = True,
) -> Dict[str, float]:
    """
    Train a single ablation variant and return its best validation metrics.

    The variant temporarily overrides ``config.NUM_FRAMES`` so the dataloader
    samples the correct number of frames.  A fresh model and optimizer are
    created for each run to keep variants independent.  ``pretrained=False``
    disables Kinetics-400 weight initialisation; ``use_shared_head=False``
    skips the 512→256→128 MLP and predicts directly from backbone features.
    """
    from src.training import train_model

    print(f"\n{'='*60}")
    print(f"ABLATION VARIANT: {tag}")
    print(f"  frames={num_frames}, reg_w={regression_weight}, "
          f"cls_w={classification_weight}, pretrained={pretrained}, "
          f"shared_head={use_shared_head}")
    print(f"{'='*60}")

    checkpoint = ARTIFACTS_DIR / "ablation" / f"{tag}.pt"

    # Skip training if a checkpoint already exists — load stored metrics instead.
    if checkpoint.exists():
        print(f"Checkpoint found at {checkpoint} — skipping training.")
        saved = torch.load(checkpoint, map_location="cpu")
        return {
            "tag": tag,
            "best_epoch": saved.get("epoch", "?"),
            "val_loss": saved.get("val_loss", float("nan")),
            "val_mae": saved.get("val_mae", float("nan")),
            "val_accuracy": saved.get("val_accuracy", float("nan")),
        }

    # Temporarily override the global frame count so the dataloader
    # samples the right number of frames for this variant.
    original_num_frames = config.NUM_FRAMES
    config.NUM_FRAMES = num_frames

    try:
        train_loader, val_loader, _, class_weights = create_dataloaders()
        components = initialize_training_components(class_weights=class_weights)

        # Swap in a model with different architectural flags if requested.
        # Rebuilds the model and reattaches a fresh optimizer so downstream
        # training code is unchanged.
        if not pretrained or not use_shared_head:
            import torch.optim as optim
            components.model = R2Plus1DEF(
                num_classes=config.NUM_CATEGORIES,
                pretrained=pretrained,
                use_shared_head=use_shared_head,
            ).to(components.device)
            components.optimizer = optim.SGD(
                components.model.parameters(),
                lr=config.LEARNING_RATE, momentum=0.9,
            )
            components.scheduler = optim.lr_scheduler.StepLR(
                components.optimizer, step_size=15, gamma=0.1,
            )

        history = train_model(
            train_loader, val_loader, components,
            num_epochs=num_epochs,
            regression_weight=regression_weight,
            classification_weight=classification_weight,
            checkpoint_path=checkpoint,
        )
    finally:
        config.NUM_FRAMES = original_num_frames

    # Return best validation metrics
    best_idx = int(np.argmin(history["val_loss"]))
    return {
        "tag": tag,
        "best_epoch": best_idx + 1,
        "val_loss": history["val_loss"][best_idx],
        "val_mae": history["val_mae"][best_idx],
        "val_accuracy": history["val_accuracy"][best_idx],
    }


def run_ablation_study(
    num_epochs_per_variant: int = 20,
    output_dir: Path = ARTIFACTS_DIR / "ablation",
) -> None:
    """
    Run a structured ablation study comparing key design choices.

    Variants tested:
    1. Baseline           – full model: 16 frames, dual-head, shared MLP, pretrained
    2. No pretraining     – identical, but backbone initialised from scratch
    3. Regression-only    – classification loss weight = 0 (replicates Stanford's
                             original single-task design)
    4. No shared head     – skips the 512→256→128 MLP; two linear heads read
                             directly from backbone features

    This isolates three orthogonal design decisions: transfer learning,
    multi-task auxiliary classification, and the shared embedding MLP.
    Results are printed as a summary table and saved as a bar chart.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        # (tag,             frames, reg_w, cls_w, pretrained, shared_head)
        ("baseline_16f",       16,   1.0,  1.0,  True,  True),
        ("no_pretrain",        16,   1.0,  1.0,  False, True),
        ("regression_only",    16,   1.0,  0.0,  True,  True),
        ("no_shared_head",     16,   1.0,  1.0,  True,  False),
    ]

    results = []
    for tag, frames, reg_w, cls_w, pretrained, shared_head in variants:
        result = _ablation_variant(
            tag=tag,
            num_frames=frames,
            num_epochs=num_epochs_per_variant,
            regression_weight=reg_w,
            classification_weight=cls_w,
            pretrained=pretrained,
            use_shared_head=shared_head,
        )
        results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print(f"{'Variant':<25} {'Best Epoch':>10} {'Val Loss':>10} {'Val MAE':>9} {'Val Acc':>9}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['tag']:<25} {r['best_epoch']:>10} "
            f"{r['val_loss']:>10.4f} {r['val_mae']:>9.4f} {r['val_accuracy']:>9.4f}"
        )
    print("=" * 70)

    # ── Bar charts ────────────────────────────────────────────────────────────
    tags = [r["tag"] for r in results]
    maes = [r["val_mae"] for r in results]
    accs = [r["val_accuracy"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(tags, maes, color="steelblue")
    axes[0].set_title("Validation MAE by Variant")
    axes[0].set_ylabel("MAE (EF %)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(tags, accs, color="seagreen")
    axes[1].set_title("Validation Accuracy by Variant")
    axes[1].set_ylabel("Accuracy")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Ablation Study Summary")
    fig.tight_layout()
    chart_path = output_dir / "ablation_summary.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Ablation chart saved → {chart_path}")


# =============================================================================
# Convenience wrapper — called from main.py when training history is available.
# =============================================================================


def plot_training_curves(history: Dict[str, List[float]]) -> None:
    """Save training curves to artifacts/evaluation/training_curves.png."""
    _plot_training_curves(history, ARTIFACTS_DIR / "evaluation" / "training_curves.png")


# =============================================================================
# Bootstrap confidence intervals — non-parametric 95 % CIs by resampling the
# test set 10,000 times (method used in the EchoNet-Dynamic paper).
# =============================================================================


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval for a scalar metric.

    Resamples the test set with replacement 10,000 times (paper methodology)
    and returns the 2.5th / 97.5th percentile range.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(42)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    alpha = (1.0 - confidence) / 2.0
    return (
        float(metric_fn(y_true, y_pred)),
        float(np.percentile(scores, 100.0 * alpha)),
        float(np.percentile(scores, 100.0 * (1.0 - alpha))),
    )


# =============================================================================
# Bland-Altman plot — standard medical imaging agreement plot that shows
# (predicted − actual) vs the mean of both.  Exposes systematic bias across
# different EF ranges (e.g. under-prediction at high EF values).
# =============================================================================


def _plot_bland_altman(ef_true: np.ndarray, ef_pred: np.ndarray, save_path: Path) -> None:
    """
    Bland-Altman agreement plot: (predicted − actual) vs mean((pred + actual)/2).

    Standard in medical imaging to detect systematic bias at different EF ranges
    (e.g. consistent under-prediction above EF=70%).  Horizontal lines show:
      • Mean difference (bias)
      • ±1.96 SD (limits of agreement — 95% of differences expected within this band)
    """
    means = (ef_true + ef_pred) / 2.0
    diffs = ef_pred - ef_true
    mean_diff = float(np.mean(diffs))
    std_diff  = float(np.std(diffs, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, alpha=0.3, s=7, color="steelblue", rasterized=True)
    ax.axhline(mean_diff, color="red",        linewidth=1.5,
               label=f"Bias: {mean_diff:+.2f}%")
    ax.axhline(loa_upper, color="darkorange", linewidth=1.2, linestyle="--",
               label=f"+1.96 SD: {loa_upper:+.2f}%")
    ax.axhline(loa_lower, color="darkorange", linewidth=1.2, linestyle="--",
               label=f"\u22121.96 SD: {loa_lower:+.2f}%")
    ax.set_xlabel("Mean of Predicted and Actual EF (%)")
    ax.set_ylabel("Predicted \u2212 Actual EF (%)")
    ax.set_title("Bland-Altman Agreement Plot")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Bland-Altman plot saved \u2192 {save_path}")


# =============================================================================
# Grad-CAM localisation vs VolumeTracings — quantifies how much of the
# model's attention falls inside the expert-annotated left ventricle.
#
# VolumeTracings.csv contains polyline segments (X1,Y1)→(X2,Y2) per frame
# that together outline the LV boundary at end-diastole and end-systole.
# We rasterise each outline into a binary mask, threshold the Grad-CAM
# heatmap at its median value, and compute IoU for each annotated frame.
# This answers: "Is the model looking at the right anatomy?"
# =============================================================================


def _build_lv_mask(
    tracings: "object",  # pandas DataFrame
    frame_idx: int,
    height: int = 112,
    width: int = 112,
) -> Optional[np.ndarray]:
    """
    Build a binary (H, W) mask of the left ventricle from polyline segments.

    The VolumeTracings format stores the LV boundary as a sequence of short
    line segments (X1,Y1)→(X2,Y2).  We collect all endpoints, form a closed
    polygon, and fill it using cv2.fillPoly.

    Returns None when fewer than 3 distinct points are found (degenerate case).
    """
    frame_rows = tracings[tracings["Frame"] == frame_idx]
    if frame_rows.empty:
        return None

    # Collect left-border and right-border points, then close the loop.
    left_pts  = frame_rows[["X1", "Y1"]].values  # (N, 2)
    right_pts = frame_rows[["X2", "Y2"]].values  # (N, 2)

    # Traverse left border top→bottom, right border bottom→top to form a
    # closed polygon (standard for EchoNet-Dynamic tracings).
    polygon = np.concatenate([left_pts, right_pts[::-1]], axis=0)
    polygon = np.clip(np.round(polygon).astype(np.int32), 0, max(height, width) - 1)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    if len(np.unique(polygon, axis=0)) < 3:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], color=1)
    return mask.astype(bool)


def _compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-Union between two boolean masks."""
    intersection = float(np.logical_and(pred_mask, gt_mask).sum())
    union        = float(np.logical_or(pred_mask, gt_mask).sum())
    return intersection / union if union > 0 else 0.0


def run_gradcam_lv_iou(
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
    tracings_path: Path = Path("dataset/VolumeTracings.csv"),
    output_dir: Path = ARTIFACTS_DIR / "evaluation",
    max_videos: int = 200,
) -> Dict[str, float]:
    """
    Quantify Grad-CAM localisation quality against expert LV annotations.

    For each test video that has VolumeTracings annotations we:
      1. Run a forward pass to get the Grad-CAM saliency map (T, H, W).
      2. Identify the annotated frames (end-diastole / end-systole).
      3. Build a binary LV mask from the polyline segments.
      4. Threshold the saliency map at its median and compute IoU.

    Outputs:
      - ``gradcam_iou_scatter.png`` — IoU vs prediction error per video
      - ``gradcam_iou_histogram.png`` — distribution of IoU scores
      - printed summary statistics

    Returns a dict with mean/median IoU and the Pearson correlation between
    IoU and absolute EF prediction error.
    """
    import pandas as pd
    from scipy import stats as scipy_stats

    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}. Skipping IoU analysis.")
        return {}
    if not tracings_path.exists():
        print(f"VolumeTracings not found at {tracings_path}. Skipping IoU analysis.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    tracings = pd.read_csv(tracings_path)
    traced_files = set(tracings["FileName"].unique())

    _, _, test_loader, class_weights = create_dataloaders()
    components = initialize_training_components(class_weights=class_weights)
    components = _load_best_model(components, checkpoint_path)
    model = components.model
    device = components.device

    # Target layer: last residual block of the R(2+1)D backbone
    target_layer = components.model.backbone[4][-1]
    gcam = GradCAM3D(model, target_layer)

    iou_scores: List[float] = []
    abs_errors: List[float] = []
    video_names: List[str] = []
    samples_done = 0

    # We need per-video access so we iterate the underlying dataset directly.
    dataset = test_loader.dataset
    indices = list(range(len(dataset)))

    for idx in indices:
        if samples_done >= max_videos:
            break

        video_tensor, ef_true, _ = dataset[idx]

        # Resolve the video filename from the dataset's path list.
        video_path = Path(dataset.video_paths[idx])
        fname = video_path.name  # e.g. "0X1234....avi"
        if fname not in traced_files:
            continue

        video_input = video_tensor.unsqueeze(0).to(device)

        # Grad-CAM gives (T, H, W) — T corresponds to the sampled frames.
        cam = gcam(video_input, class_idx=None)  # (T, H, W) in [0,1]

        # Identify annotated frame indices within the video
        video_tracings = tracings[tracings["FileName"] == fname]
        annotated_frames = sorted(video_tracings["Frame"].unique())

        # The video was sub-sampled at FRAME_SAMPLING_PERIOD.  Map annotated
        # original frame indices to the nearest sampled time step.
        num_sampled = cam.shape[0]
        sampling_period = config.FRAME_SAMPLING_PERIOD

        frame_ious: List[float] = []
        for orig_frame in annotated_frames:
            sampled_idx = min(int(orig_frame // sampling_period), num_sampled - 1)
            cam_slice   = cam[sampled_idx]  # (H, W)

            lv_mask = _build_lv_mask(
                video_tracings, orig_frame,
                height=config.TARGET_HEIGHT, width=config.TARGET_WIDTH,
            )
            if lv_mask is None or lv_mask.sum() == 0:
                continue

            # Resize CAM to match the LV mask resolution (the backbone's last
            # conv produces a smaller spatial map, e.g. 7×7, which must be
            # upsampled to 112×112 before comparing with the annotation mask).
            cam_resized = cv2.resize(
                cam_slice.astype(np.float32),
                (config.TARGET_WIDTH, config.TARGET_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            # Threshold Grad-CAM at its median to obtain a binary attention map.
            cam_binary = cam_resized >= np.median(cam_resized)
            frame_ious.append(_compute_iou(cam_binary, lv_mask))

        if not frame_ious:
            continue

        # Get model prediction for this video
        model.eval()
        with torch.no_grad():
            ef_pred, _ = model(video_input)
        ef_pred_val = float(ef_pred.cpu().item())
        abs_err = abs(ef_pred_val - float(ef_true))

        iou_scores.append(float(np.mean(frame_ious)))
        abs_errors.append(abs_err)
        video_names.append(fname)
        samples_done += 1

    gcam.remove_hooks()

    if not iou_scores:
        print("No annotated test videos found for IoU analysis.")
        return {}

    iou_arr = np.array(iou_scores)
    err_arr = np.array(abs_errors)

    # Pearson correlation: does better localisation → lower error?
    r, p_val = scipy_stats.pearsonr(iou_arr, err_arr)

    print("\n" + "=" * 60)
    print("GRAD-CAM LOCALISATION vs LV ANNOTATIONS (IoU)")
    print("=" * 60)
    print(f"  Videos analysed : {samples_done}")
    print(f"  Mean IoU        : {iou_arr.mean():.4f}")
    print(f"  Median IoU      : {np.median(iou_arr):.4f}")
    print(f"  Std IoU         : {iou_arr.std():.4f}")
    print(f"  IoU ↔ |Error| r : {r:.4f}  (p={p_val:.4f})")
    print("=" * 60)

    # ── Scatter: IoU vs absolute prediction error ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(iou_arr, err_arr, alpha=0.5, s=15, color="steelblue")
    m, b = np.polyfit(iou_arr, err_arr, 1)
    x_line = np.linspace(iou_arr.min(), iou_arr.max(), 100)
    axes[0].plot(x_line, m * x_line + b, "r--", linewidth=1.5,
                 label=f"r={r:.3f}, p={p_val:.3f}")
    axes[0].set_xlabel("Grad-CAM IoU with LV Mask")
    axes[0].set_ylabel("Absolute EF Prediction Error (%)")
    axes[0].set_title("Localisation Quality vs Prediction Error")
    axes[0].legend()

    # ── Histogram of IoU scores ───────────────────────────────────────────────
    axes[1].hist(iou_arr, bins=25, color="seagreen", edgecolor="white", linewidth=0.5)
    axes[1].axvline(iou_arr.mean(), color="red", linestyle="--",
                    linewidth=1.5, label=f"Mean = {iou_arr.mean():.3f}")
    axes[1].set_xlabel("Grad-CAM IoU with LV Mask")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of LV Localisation IoU Scores")
    axes[1].legend()

    fig.suptitle("Grad-CAM Localisation vs Expert LV Annotations")
    fig.tight_layout()
    scatter_path = output_dir / "gradcam_iou_analysis.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"IoU analysis plot saved \u2192 {scatter_path}")

    return {
        "mean_iou": float(iou_arr.mean()),
        "median_iou": float(np.median(iou_arr)),
        "iou_error_correlation": float(r),
        "iou_error_pvalue": float(p_val),
        "n_videos": samples_done,
    }


# =============================================================================
# Subgroup analysis — breaks test performance down by EF range to surface
# whether the model is reliably accurate across the full clinical spectrum.
# Clinically, errors in the "Reduced (<40%)" group are the most dangerous.
# =============================================================================


# Clinical EF bands with labels
_EF_BANDS = [
    (0.0,  20.0, "Severely Reduced\n(<20%)"),
    (20.0, 40.0, "Reduced\n(20–40%)"),
    (40.0, 50.0, "Mildly Reduced\n(40–50%)"),
    (50.0, 70.0, "Preserved\n(50–70%)"),
    (70.0, 100.0,"Hyperdynamic\n(>70%)"),
]

# Clinical cost matrix: rows=actual, cols=predicted (0=Reduced, 1=Mildly, 2=Preserved)
# Misclassifying Reduced as Preserved is the most dangerous error (weight=3).
_CLINICAL_COST_MATRIX = np.array([
    [0, 1, 3],  # Actual Reduced
    [1, 0, 1],  # Actual Mildly Reduced
    [2, 1, 0],  # Actual Preserved
], dtype=float)


def run_subgroup_analysis(
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
    output_dir: Path = ARTIFACTS_DIR / "evaluation",
) -> Dict[str, object]:
    """
    Break down test-set performance by EF range.

    For each clinical EF band (Severely Reduced → Hyperdynamic) reports:
      - Sample count
      - Mean Absolute Error (MAE)
      - Root Mean Squared Error (RMSE)
      - Category accuracy

    Also computes a cost-weighted confusion matrix where misclassifying a
    Reduced patient as Preserved is penalised more heavily than the reverse,
    reflecting real clinical consequences.

    Outputs:
      - ``subgroup_analysis.png`` — MAE and accuracy bar charts per band
      - ``clinical_cost_confusion.png`` — cost-weighted confusion matrix heatmap
      - printed summary table
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, class_weights = create_dataloaders()
    components = initialize_training_components(class_weights=class_weights)
    components = _load_best_model(components, checkpoint_path)

    ef_true, ef_pred, cat_true, cat_pred = _collect_predictions(
        test_loader, components.model, components.device
    )

    # ── Per-band metrics ───────────────────────────────────────────────────────
    band_results = []
    for lo, hi, label in _EF_BANDS:
        mask = (ef_true >= lo) & (ef_true < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mae  = float(mean_absolute_error(ef_true[mask], ef_pred[mask]))
        rmse = float(np.sqrt(mean_squared_error(ef_true[mask], ef_pred[mask])))
        acc  = float(accuracy_score(cat_true[mask], cat_pred[mask]))
        band_results.append({
            "label": label, "n": n,
            "mae": mae, "rmse": rmse, "accuracy": acc,
        })

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUBGROUP ANALYSIS BY EF RANGE")
    print(f"{'EF Band':<25} {'N':>5} {'MAE (%)':>9} {'RMSE (%)':>10} {'Accuracy':>10}")
    print("-" * 75)
    for r in band_results:
        label_flat = r["label"].replace("\n", " ")
        print(f"{label_flat:<25} {r['n']:>5} {r['mae']:>9.2f} {r['rmse']:>10.2f} {r['accuracy']:>10.4f}")
    print("=" * 75)

    # ── Bar charts ─────────────────────────────────────────────────────────────
    labels = [r["label"] for r in band_results]
    counts = [r["n"] for r in band_results]
    maes   = [r["mae"] for r in band_results]
    accs   = [r["accuracy"] for r in band_results]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars0 = axes[0].bar(x, maes, color="steelblue", width=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("MAE (EF %)")
    axes[0].set_title("MAE by EF Subgroup")
    # Annotate with sample counts
    for bar, n in zip(bars0, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f"n={n}", ha="center", va="bottom", fontsize=7)

    bars1 = axes[1].bar(x, accs, color="seagreen", width=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Category Accuracy")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Category Accuracy by EF Subgroup")
    for bar, n in zip(bars1, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"n={n}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Subgroup Analysis by Clinical EF Range")
    fig.tight_layout()
    subgroup_path = output_dir / "subgroup_analysis.png"
    fig.savefig(subgroup_path, dpi=150)
    plt.close(fig)
    print(f"Subgroup analysis plot saved \u2192 {subgroup_path}")

    # ── Clinical cost-weighted confusion matrix ────────────────────────────────
    cm = confusion_matrix(cat_true, cat_pred, labels=[0, 1, 2])
    cost_cm = cm * _CLINICAL_COST_MATRIX

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORY_NAMES, yticklabels=CATEGORY_NAMES,
                ax=axes2[0])
    axes2[0].set_xlabel("Predicted")
    axes2[0].set_ylabel("Actual")
    axes2[0].set_title("Standard Confusion Matrix")

    sns.heatmap(cost_cm, annot=True, fmt=".0f", cmap="Reds",
                xticklabels=CATEGORY_NAMES, yticklabels=CATEGORY_NAMES,
                ax=axes2[1])
    axes2[1].set_xlabel("Predicted")
    axes2[1].set_ylabel("Actual")
    axes2[1].set_title("Cost-Weighted Confusion Matrix\n(Higher = More Dangerous Error)")

    fig2.suptitle("Clinical Cost Analysis of Misclassifications")
    fig2.tight_layout()
    cost_path = output_dir / "clinical_cost_confusion.png"
    fig2.savefig(cost_path, dpi=150)
    plt.close(fig2)
    print(f"Clinical cost confusion matrix saved \u2192 {cost_path}")

    total_cost = float(cost_cm.sum())
    worst_error = float(cost_cm[0, 2])  # Reduced predicted as Preserved
    print(f"  Total clinical cost score : {total_cost:.0f}")
    print(f"  Reduced→Preserved errors  : {int(cm[0, 2])} cases (cost {worst_error:.0f})")

    return {"band_results": band_results, "total_clinical_cost": total_cost}


# =============================================================================
# Calibration analysis — checks whether the model's stated confidence (softmax
# probability) actually matches its empirical accuracy.  A well-calibrated
# model that says "80% confident" should be correct roughly 80% of the time.
#
# Reports Expected Calibration Error (ECE) and draws a reliability diagram.
# This validates whether the classification head is trustworthy for clinical use.
# =============================================================================


def run_calibration_analysis(
    checkpoint_path: Path = ARTIFACTS_DIR / "best_r2plus1d.pt",
    output_dir: Path = ARTIFACTS_DIR / "evaluation",
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Assess calibration of the classification head via a reliability diagram.

    Predictions are bucketed into ``n_bins`` equal-width confidence bins
    (0.0–0.1, 0.1–0.2, …, 0.9–1.0).  Within each bin the fraction of
    correct predictions (empirical accuracy) is plotted against the mean
    predicted probability (confidence).  A perfectly calibrated model lies
    on the diagonal.

    Expected Calibration Error (ECE) is the bin-size-weighted mean of
    |accuracy − confidence| across all bins.  Lower is better.

    Outputs:
      - ``calibration_reliability.png`` — reliability diagram + ECE annotation
      - printed ECE value

    Returns a dict with ECE and per-bin statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader, class_weights = create_dataloaders()
    components = initialize_training_components(class_weights=class_weights)
    components = _load_best_model(components, checkpoint_path)
    model = components.model
    device = components.device

    # Collect max confidence and correctness for every test prediction.
    model.eval()
    confidences: List[float] = []
    correctness: List[int]   = []

    with torch.no_grad():
        for videos, _, ef_classes in test_loader:
            videos    = videos.to(device, non_blocking=True)
            ef_classes = ef_classes.to(device, non_blocking=True)
            _, logits = model(videos)
            probs      = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            correct    = (pred == ef_classes).long()

            confidences.extend(conf.cpu().tolist())
            correctness.extend(correct.cpu().tolist())

    conf_arr = np.array(confidences)
    corr_arr = np.array(correctness, dtype=float)
    n_total  = len(conf_arr)

    # ── ECE calculation ───────────────────────────────────────────────────────
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs   = np.zeros(n_bins)
    bin_confs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf_arr >= lo) & (conf_arr < hi)
        if i == n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr <= hi)  # include 1.0 in last bin
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        bin_counts[i] = n_bin
        bin_accs[i]   = corr_arr[mask].mean()
        bin_confs[i]  = conf_arr[mask].mean()

    ece = float(np.sum(bin_counts / n_total * np.abs(bin_accs - bin_confs)))

    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  (0 = perfect calibration, 1 = worst)")
    print()
    print(f"  {'Bin':>12}  {'N':>5}  {'Conf':>7}  {'Acc':>7}  {'Gap':>7}")
    print("  " + "-" * 50)
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        gap = bin_accs[i] - bin_confs[i]
        print(f"  [{lo:.1f}–{hi:.1f}]   {bin_counts[i]:>5}  {bin_confs[i]:>7.3f}  "
              f"{bin_accs[i]:>7.3f}  {gap:>+7.3f}")
    print("=" * 60)

    # ── Reliability diagram ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))

    # Diagonal = perfect calibration
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Gap fill: area between actual and perfect
    non_empty = bin_counts > 0
    ax.bar(bin_confs[non_empty], bin_accs[non_empty],
           width=(bin_edges[1] - bin_edges[0]) * 0.8,
           alpha=0.7, color="steelblue", label="Model accuracy per bin")
    ax.bar(bin_confs[non_empty],
           bin_confs[non_empty] - bin_accs[non_empty],
           bottom=bin_accs[non_empty],
           width=(bin_edges[1] - bin_edges[0]) * 0.8,
           alpha=0.3, color="red", label="Calibration gap")

    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    cal_path = output_dir / "calibration_reliability.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)
    print(f"Calibration reliability diagram saved \u2192 {cal_path}")

    return {
        "ece": ece,
        "bin_accuracies": bin_accs.tolist(),
        "bin_confidences": bin_confs.tolist(),
        "bin_counts": bin_counts.tolist(),
    }


