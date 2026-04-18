# Core Deep Learning
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Visualization
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

from src.CNN_model import CNN3D
from src.config import Config
from src.data_processing import (
    create_dataloaders,
    load_video_paths_and_labels,
)
from src.training import initialize_training_components, TrainingComponents

config = Config()

CATEGORY_NAMES = ["Reduced (<40%)", "Mildly Reduced (40-49%)", "Preserved (≥50%)"]
ARTIFACTS_DIR = Path("artifacts")

# ==============================================================================
# STEP 7: EVALUATION ON TEST SET
# ==============================================================================


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

    # Load data and model
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

    # ROC-AUC (requires softmax probabilities) – re-run forward pass for logits
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


# ==============================================================================
# STEP 8: GRAD-CAM VISUALIZATION
# ==============================================================================


class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3-D CNN models.

    Registers a forward/backward hook on the target layer, runs one forward
    pass, and produces a spatiotemporal saliency map of shape (T, H, W).

    Usage::

        gcam = GradCAM3D(model, target_layer=model.feature_extractor[-1])
        heatmap = gcam(video_tensor, class_idx=None)   # None → argmax
    """

    def __init__(self, model: CNN3D, target_layer: nn.Module) -> None:
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
        self.model.eval()
        # Forward
        _, logits = self.model(video)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward from the target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool the gradients over spatial+temporal dims
        # activations / gradients: (1, C, t, h, w)
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

    # Hook onto the last residual block (R2+1D) or last conv block (CNN3D).
    # model.backbone is nn.Sequential(stem, layer1, layer2, layer3, layer4, avgpool)
    # so layer4 is at index 4.
    if hasattr(model, 'backbone'):
        target_layer = model.backbone[4][-1]
    else:
        target_layer = model.feature_extractor[-1]
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
                import cv2 as _cv2
                cam_resized = _cv2.resize(
                    cam_frame, (frame.shape[1], frame.shape[0]),
                    interpolation=_cv2.INTER_LINEAR,
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


# ==============================================================================
# STEP 9: ABLATION STUDY
# ==============================================================================


def _ablation_variant(
    tag: str,
    num_frames: int,
    num_epochs: int,
    regression_weight: float,
    classification_weight: float,
) -> Dict[str, float]:
    """
    Train a single ablation variant and return its best validation metrics.

    The variant temporarily overrides ``config.NUM_FRAMES`` so the dataloader
    samples the correct number of frames.  A fresh model and optimizer are
    created for each run to keep variants independent.
    """
    from src.training import train_model

    print(f"\n{'='*60}")
    print(f"ABLATION VARIANT: {tag}")
    print(f"  frames={num_frames}, reg_w={regression_weight}, cls_w={classification_weight}")
    print(f"{'='*60}")

    # Temporary config override
    original_num_frames = config.NUM_FRAMES
    config.NUM_FRAMES = num_frames

    try:
        train_loader, val_loader, _, class_weights = create_dataloaders()
        components = initialize_training_components(class_weights=class_weights)
        checkpoint = ARTIFACTS_DIR / "ablation" / f"{tag}.pt"
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

    Variants tested (following EchoNet-Dynamic ablation conventions):
    1. Baseline  – 16 frames, equal loss weights
    2. More frames – 32 frames
    3. Regression-only – classification weight = 0
    4. Classification-only – regression weight = 0

    Results are printed as a summary table and saved as a bar chart.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        # (tag,              frames, reg_w, cls_w)
        ("baseline_16f",       16,   1.0,  1.0),
        ("more_frames_32f",    32,   1.0,  1.0),
        ("regression_only",    16,   1.0,  0.0),
        ("classification_only",16,   0.0,  1.0),
    ]

    results = []
    for tag, frames, reg_w, cls_w in variants:
        result = _ablation_variant(
            tag=tag,
            num_frames=frames,
            num_epochs=num_epochs_per_variant,
            regression_weight=reg_w,
            classification_weight=cls_w,
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


# ==============================================================================
# UTILITY: Plot training curves (called from main if history is available)
# ==============================================================================

def plot_training_curves(history: Dict[str, List[float]]) -> None:
    """Save training curves to artifacts/evaluation/training_curves.png."""
    _plot_training_curves(history, ARTIFACTS_DIR / "evaluation" / "training_curves.png")


# ==============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================


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


# ==============================================================================
# BLAND-ALTMAN AGREEMENT PLOT
# ==============================================================================


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


