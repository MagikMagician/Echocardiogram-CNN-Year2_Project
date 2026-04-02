# Core Deep Learning
from sympy import im
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Data Processing
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path

# Video/Image Processing
import cv2

# Utilities
from tqdm import tqdm
import json

# Initialize configuration
from src.config import Config
config = Config()

# ==============================================================================
# STEP 1: DATA LOADING UTILITIES
# ==============================================================================


def split_data(df: pd.DataFrame, split_type: str, video_dir: Path) -> Tuple[List[str], List[float]]:
    """
    Load videos and labels for a specific split (TRAIN/VAL/TEST).

    Args:
        df: DataFrame containing video metadata
        split_type: One of 'TRAIN', 'VAL', or 'TEST'
        video_dir: Path to directory containing videos

    Returns:
        Tuple of (video_paths, ef_labels)
    """
    videos = []
    labels = []
    missing_count = 0

    # Filter by split type and check existence
    split_df = df[df["Split"] == split_type]

    for _, row in split_df.iterrows():
        video_path = video_dir / f"{row['FileName']}{config.VIDEO_EXTENSION}"

        if video_path.exists():
            videos.append(str(video_path))
            labels.append(float(row["EF"]))
        else:
            missing_count += 1
            print(f"Video not found: {video_path}")

    print(f"{split_type}: {len(videos)} videos found, {missing_count} missing")

    return videos, labels


def load_video_paths_and_labels() -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float]]:
    """
    Load EchoNet-Dynamic dataset from FileList.csv.

    Returns:
        Tuple of (train_videos, train_labels, val_videos, val_labels, test_videos, test_labels)

    Raises:
        FileNotFoundError: If FileList.csv or video directory not found
        ValueError: If required columns are missing from CSV
    """
    # Validate paths exist
    if not config.file_list_path.exists():
        raise FileNotFoundError(
            f"FileList.csv not found at {config.file_list_path}. "
            f"Please update Config.DATASET_PATH"
        )

    if not config.video_dir_path.exists():
        raise FileNotFoundError(
            f"Videos directory not found at {config.video_dir_path}"
        )

    print(f"Loading data from: {config.file_list_path}")

    # Read CSV and validate columns
    try:
        df = pd.read_csv(config.file_list_path, usecols=["FileName", "Split", "EF"])
    except ValueError as e:
        raise ValueError(f"Required columns missing from CSV: {e}")

    # Validate splits exist
    available_splits = set(df["Split"].unique())
    required_splits = {"TRAIN", "VAL", "TEST"}
    if not required_splits.issubset(available_splits):
        missing = required_splits - available_splits
        raise ValueError(f"Missing required splits in CSV: {missing}")

    # Load data for each split
    train_videos, train_labels = split_data(df, "TRAIN", config.video_dir_path)
    val_videos, val_labels = split_data(df, "VAL", config.video_dir_path)
    test_videos, test_labels = split_data(df, "TEST", config.video_dir_path)

    # Validate we have data
    if len(train_videos) == 0:
        raise ValueError("No training videos found!")

    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels


def convert_ef_to_category(ef: float) -> int:
    """
    Convert continuous EF value to clinical category.

    Args:
        ef: Ejection Fraction percentage value

    Returns:
        Category index: 0 (Reduced), 1 (Mildly Reduced), or 2 (Preserved)
    """
    if ef < config.EF_REDUCED_THRESHOLD:
        return 0  # Reduced
    if ef < config.EF_MILDLY_REDUCED_THRESHOLD:
        return 1  # Mildly Reduced
    return 2  # Preserved


def load_frames_from_video(video_path: str, num_frames: int, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Extract frames uniformly from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        target_size: (height, width) to resize frames to

    Returns:
        Array of shape (num_frames, height, width) or None if loading fails
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"Video has 0 frames: {video_path}")
            cap.release()
            return None

        # Determine frame indices to extract
        if total_frames < num_frames:
            print(f"Video has only {total_frames} frames, need {num_frames}")
            frame_indices = np.arange(total_frames)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        frame_idx_set = set(frame_indices)
        current_idx = 0

        # Read frames sequentially (more efficient than seeking)
        while len(frames) < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in frame_idx_set:
                # Convert to grayscale and resize
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, target_size)
                frames.append(frame)

            current_idx += 1

        cap.release()

        # Pad with zeros if we didn't get enough frames
        while len(frames) < num_frames:
            frames.append(np.zeros(target_size, dtype=np.uint8))
            print(f"Padded frame for {video_path}")

        return np.array(frames[:num_frames], dtype=np.float32)

    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        return None


def compute_dataset_statistics(
    video_paths: List[str],
    num_samples: int = config.NUM_SAMPLES_FOR_STATS,
    num_frames: int = config.NUM_FRAMES,
    target_size: Tuple[int, int] = config.target_size,
    cache_file: Path = config.stats_cache_path,
) -> Tuple[float, float]:
    """
    Compute mean and std across a sample of videos for normalization.
    Results are cached to avoid recomputation on subsequent runs.

    Args:
        video_paths: List of video paths
        num_samples: Number of videos to sample (default: from config)
        num_frames: Number of frames per video (default: from config)
        target_size: Frame size (default: from config)
        cache_file: Path to cache file (default: from config)

    Returns:
        Tuple of (mean, std) for normalization
    """

    # Check if cached statistics exist
    if cache_file.exists():
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                stats = json.load(f)
            mean, std = stats["mean"], stats["std"]
            print(f"Loaded cached statistics - Mean: {mean:.4f}, Std: {std:.4f}")
            return mean, std
        except Exception as e:
            print(f"Failed to load cached statistics: {e}")

    print(f"Computing dataset statistics from {num_samples} videos...")

    # Sample random videos
    num_samples = min(num_samples, len(video_paths))
    sample_indices = np.random.choice(len(video_paths), num_samples, replace=False)

    all_pixels = []

    for idx in tqdm(sample_indices, desc="Sampling videos"):
        frames = load_frames_from_video(video_paths[idx], num_frames, target_size)

        if frames is not None:
            all_pixels.append(frames.flatten())

    if len(all_pixels) == 0:
        print("No valid frames found for statistics computation")
        return config.DEFAULT_MEAN, config.DEFAULT_STD

    # Compute statistics
    all_pixels = np.concatenate(all_pixels) / 255.0  # Scale to [0, 1]
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))

    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")

    # Cache the results
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump({"mean": mean, "std": std}, f)
        print(f"Statistics cached to {cache_file}")
    except Exception as e:
        print(f"Failed to cache statistics: {e}")

    return mean, std


# ==============================================================================
# STEP 2: DATASET CLASS - Loads echocardiogram videos as temporal sequences
# ==============================================================================


class EchoDataset(Dataset):
    """PyTorch Dataset for EchoNet-Dynamic videos."""

    def __init__(
        self,
        video_paths: List[str],
        labels: List[float],
        num_frames: int = config.NUM_FRAMES,
        transform: Optional[transforms.Compose] = None,
        mean: float = 0.0,
        std: float = 1.0,
        target_size: Tuple[int, int] = config.target_size,
    ):
        """
        Initialize EchoNet dataset.

        Args:
            video_paths: List of video file paths
            labels: List of EF values (continuous, in percentage)
            num_frames: Number of frames to extract per video (default: from config)
            transform: Optional transforms for preprocessing
            mean: Dataset mean for normalization
            std: Dataset std for normalization
            target_size: Resize dimensions as (height, width) (default: from config)

        Raises:
            ValueError: If video_paths and labels have different lengths
        """
        if len(video_paths) != len(labels):
            raise ValueError(
                f"Mismatch: {len(video_paths)} videos but {len(labels)} labels"
            )

        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.mean = mean
        self.std = std
        self.target_size = target_size

    def __len__(self) -> int:
        """Return number of videos in dataset."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a single video sample.

        Args:
            idx: Index of video to load

        Returns:
            Tuple of (video_tensor, ef_continuous, ef_category)
            - video_tensor: shape (1, num_frames, H, W) for grayscale
            - ef_continuous: continuous EF value as float32
            - ef_category: categorical EF label (0, 1, or 2) as long
        """
        video_path = self.video_paths[idx]
        ef_continuous = self.labels[idx]

        # Load video frames with error handling
        frames = load_frames_from_video(video_path, self.num_frames, self.target_size)

        # Fallback to zeros if loading failed (keeps batch sizes consistent)
        if frames is None:
            print(f"Using zero frames for failed video: {video_path}")
            frames = np.zeros((self.num_frames, *self.target_size), dtype=np.float32)

        # Normalize pixel values using dataset-wide statistics
        frames = frames / 255.0  # Scale to [0, 1]
        if self.std > 0:
            frames = (frames - self.mean) / self.std  # Standardize
        else:
            print(f"Invalid std={self.std}, skipping standardization")

        # Convert to torch tensor: (1, num_frames, H, W) for grayscale
        video_tensor = torch.from_numpy(frames).unsqueeze(0).float()

        # Convert EF to clinical category
        ef_category = convert_ef_to_category(ef_continuous)

        # Apply optional transforms
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return (
            video_tensor,
            torch.tensor(ef_continuous, dtype=torch.float32),
            torch.tensor(ef_category, dtype=torch.long),
        )


# ==============================================================================
# STEP 4: CREATE DATALOADERS
# ==============================================================================
# DONE: Load data using load_video_paths_and_labels()
# DONE: Create train_dataset, val_dataset, test_dataset using EchoDataset
# DONE: Create DataLoaders with appropriate batch_size (4-8 for memory efficiency)
# DONE: Use num_workers for parallel data loading
# TODO: Apply data augmentation transforms for training (optional)


def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If dataset is invalid
    """
    print("INITIALIZING ECHONET-DYNAMIC DATASET")

    # Load video paths and labels
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = (
        load_video_paths_and_labels()
    )

    # Compute dataset-wide normalization statistics from training set
    # This ensures consistent normalization across all videos
    # Results are cached to avoid recomputation
    dataset_mean, dataset_std = compute_dataset_statistics(train_videos)

    print("Initializing datasets...")
    # Create datasets with computed statistics
    train_dataset = EchoDataset(
        video_paths=train_videos,
        labels=train_labels,
        mean=dataset_mean,
        std=dataset_std,
    )

    val_dataset = EchoDataset(
        video_paths=val_videos,
        labels=val_labels,
        mean=dataset_mean,
        std=dataset_std,
    )

    test_dataset = EchoDataset(
        video_paths=test_videos,
        labels=test_labels,
        mean=dataset_mean,
        std=dataset_std,
    )

    def build_dataloader(dataset: EchoDataset, shuffle: bool) -> DataLoader:
        """Build a dataloader with the project's common settings."""
        return DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

    # Create DataLoaders
    train_loader = build_dataloader(train_dataset, shuffle=True)
    val_loader = build_dataloader(val_dataset, shuffle=False)
    test_loader = build_dataloader(test_dataset, shuffle=False)

    print("Dataset Summary:")
    print(f"  Train: {len(train_dataset)} videos")
    print(f"  Val:   {len(val_dataset)} videos")
    print(f"  Test:  {len(test_dataset)} videos")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Normalization: mean={dataset_mean:.4f}, std={dataset_std:.4f}")

    return train_loader, val_loader, test_loader
