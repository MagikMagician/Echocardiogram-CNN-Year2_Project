# Core Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# Data Processing
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from pathlib import Path

# Video/Image Processing
import cv2
from PIL import Image

# Utilities
import os
import glob
from tqdm import tqdm
import csv
import json
import logging

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)

# ==============================================================================
# PROJECT: Automated EF Estimation from Echocardiogram Videos
# Dataset: EchoNet-Dynamic (Stanford University)
# Goal: Predict continuous EF values (regression) and classify into clinical categories
# Categories: Reduced (<40%), Mildly Reduced (40-49%), Preserved (≥50%)
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Centralize all hyperparameters and constants
# ==============================================================================
class Config:
    """Configuration for the EchoNet-Dynamic project."""
    
    # Dataset paths
    DATASET_PATH = Path(r'dataset')  # Update this to your dataset location
    FILE_LIST_NAME = 'FileList.csv'
    VIDEO_DIR_NAME = 'Videos'
    VIDEO_EXTENSION = '.avi'
    
    # Model hyperparameters
    NUM_FRAMES = 16  # Number of frames to sample per video
    TARGET_HEIGHT = 112  # Frame height after resize
    TARGET_WIDTH = 112  # Frame width after resize
    BATCH_SIZE = 4  # Batch size for DataLoader
    NUM_WORKERS = 0  # Number of workers for DataLoader (0 for OpenCV compatibility)
    
    # EF Categories
    EF_REDUCED_THRESHOLD = 40.0  # EF < 40% = Reduced
    EF_MILDLY_REDUCED_THRESHOLD = 50.0  # 40% ≤ EF < 50% = Mildly Reduced
    NUM_CATEGORIES = 3  # Reduced, Mildly Reduced, Preserved
    
    # Normalization (will be computed from training data if cache doesn't exist)
    STATS_CACHE_FILE = 'dataset_statistics.json'
    NUM_SAMPLES_FOR_STATS = 100  # Number of videos to sample for computing statistics
    DEFAULT_MEAN = 0.5  # Fallback if statistics computation fails
    DEFAULT_STD = 0.5
    
    # Training hyperparameters
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    @property
    def target_size(self) -> Tuple[int, int]:
        """Return target frame size as (height, width) tuple."""
        return (self.TARGET_HEIGHT, self.TARGET_WIDTH)
    
    @property
    def file_list_path(self) -> Path:
        """Return full path to FileList.csv."""
        return self.DATASET_PATH / self.FILE_LIST_NAME
    
    @property
    def video_dir_path(self) -> Path:
        """Return full path to Videos directory."""
        return self.DATASET_PATH / self.VIDEO_DIR_NAME
    
    @property
    def stats_cache_path(self) -> Path:
        """Return full path to statistics cache file."""
        return Path(self.STATS_CACHE_FILE)


# Initialize configuration and logging
config = Config()
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

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
    split_df = df[df['Split'] == split_type]
    
    for _, row in split_df.iterrows():
        video_path = video_dir / f"{row['FileName']}{config.VIDEO_EXTENSION}"
        
        if video_path.exists():
            videos.append(str(video_path))
            labels.append(float(row['EF']))
        else:
            missing_count += 1
            logger.warning(f"Video not found: {video_path}")
    
    logger.info(f"{split_type}: {len(videos)} videos found, {missing_count} missing")
    
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
    
    logger.info(f"Loading data from: {config.file_list_path}")
    
    # Read CSV and validate columns
    try:
        df = pd.read_csv(config.file_list_path, usecols=['FileName', 'Split', 'EF'])
    except ValueError as e:
        raise ValueError(f"Required columns missing from CSV: {e}")
    
    # Validate splits exist
    available_splits = set(df['Split'].unique())
    required_splits = {'TRAIN', 'VAL', 'TEST'}
    if not required_splits.issubset(available_splits):
        missing = required_splits - available_splits
        raise ValueError(f"Missing required splits in CSV: {missing}")
    
    # Load data for each split
    train_videos, train_labels = split_data(df, 'TRAIN', config.video_dir_path)
    val_videos, val_labels = split_data(df, 'VAL', config.video_dir_path)
    test_videos, test_labels = split_data(df, 'TEST', config.video_dir_path)
    
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
    elif ef < config.EF_MILDLY_REDUCED_THRESHOLD:
        return 1  # Mildly Reduced
    else:
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
            logger.warning(f"Cannot open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning(f"Video has 0 frames: {video_path}")
            cap.release()
            return None
        
        # Determine frame indices to extract
        if total_frames < num_frames:
            logger.debug(f"Video has only {total_frames} frames, need {num_frames}")
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
            logger.debug(f"Padded frame for {video_path}")
        
        return np.array(frames[:num_frames], dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error loading video {video_path}: {e}")
        return None


def compute_dataset_statistics(
    video_paths: List[str], 
    num_samples: int = config.NUM_SAMPLES_FOR_STATS,
    num_frames: int = config.NUM_FRAMES,
    target_size: Tuple[int, int] = config.target_size,
    cache_file: Path = config.stats_cache_path
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
            with open(cache_file, 'r') as f:
                stats = json.load(f)
            mean, std = stats['mean'], stats['std']
            logger.info(f"Loaded cached statistics - Mean: {mean:.4f}, Std: {std:.4f}")
            return mean, std
        except Exception as e:
            logger.warning(f"Failed to load cached statistics: {e}")
    
    logger.info(f"Computing dataset statistics from {num_samples} videos...")
    
    # Sample random videos
    num_samples = min(num_samples, len(video_paths))
    sample_indices = np.random.choice(len(video_paths), num_samples, replace=False)
    
    all_pixels = []
    successful = 0
    
    for idx in tqdm(sample_indices, desc="Sampling videos"):
        frames = load_frames_from_video(video_paths[idx], num_frames, target_size)
        
        if frames is not None:
            all_pixels.append(frames.flatten())
            successful += 1
    
    if len(all_pixels) == 0:
        logger.error("No valid frames found for statistics computation")
        return config.DEFAULT_MEAN, config.DEFAULT_STD
        
    # Compute statistics
    all_pixels = np.concatenate(all_pixels) / 255.0  # Scale to [0, 1]
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))
    
    logger.info(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Cache the results
    try:
        with open(cache_file, 'w') as f:
            json.dump({'mean': mean, 'std': std}, f)
        logger.info(f"Statistics cached to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache statistics: {e}")
    
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
        target_size: Tuple[int, int] = config.target_size
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
            logger.warning(f"Using zero frames for failed video: {video_path}")
            frames = np.zeros(
                (self.num_frames, *self.target_size), 
                dtype=np.float32
            )
        
        # Normalize pixel values using dataset-wide statistics
        frames = frames / 255.0  # Scale to [0, 1]
        if self.std > 0:
            frames = (frames - self.mean) / self.std  # Standardize
        else:
            logger.warning(f"Invalid std={self.std}, skipping standardization")
        
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
            torch.tensor(ef_category, dtype=torch.long)
        )


# ==============================================================================
# STEP 3: 3D CNN MODEL - Processes video volumes for EF estimation
# ==============================================================================
class CNN3D(nn.Module):
    def __init__(self, num_classes=3):
        """
        3D CNN for EF estimation from echocardiogram videos.
        Architecture inspired by EchoNet-Dynamic paper (Ouyang et al., 2020).
        Args:
            num_classes: Number of EF categories (3: Reduced, Mildly Reduced, Preserved)
        """
        super(CNN3D, self).__init__()
        # TODO: Define Conv3d layers for spatial-temporal feature extraction
        # Example: self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,3), padding=1)
        # TODO: Define BatchNorm3d for normalization
        # TODO: Define MaxPool3d layers for downsampling
        # TODO: Define fully connected layers for regression and classification
        # TODO: Regression head: outputs continuous EF value
        # TODO: Classification head: outputs 3 class probabilities
        pass
        
    def forward(self, x):
        # Input x shape: (batch, 1, time, height, width)
        # TODO: Pass through conv3d + batchnorm + relu + maxpool layers
        # TODO: Flatten spatial-temporal features
        # TODO: Pass through fully connected layers
        # TODO: Regression output: single EF value
        # TODO: Classification output: 3-class probabilities (softmax)
        # TODO: Return (ef_regression, ef_classification)
        pass


# ==============================================================================
# STEP 4: CREATE DATALOADERS
# ==============================================================================
# TODO: Load data using load_video_paths_and_labels()
# TODO: Create train_dataset, val_dataset, test_dataset using EchoDataset
# TODO: Create DataLoaders with appropriate batch_size (4-8 for memory efficiency)
# TODO: Use num_workers for parallel data loading
# TODO: Apply data augmentation transforms for training (optional)


# ==============================================================================
# STEP 5: INITIALIZE MODEL, LOSS, OPTIMIZER
# ==============================================================================
# TODO: Set device (cuda if available, else cpu)
# TODO: Initialize CNN3D model and move to device
# TODO: Define regression loss: MSELoss for continuous EF prediction
# TODO: Define classification loss: CrossEntropyLoss for clinical categories
# TODO: Define optimizer: Adam with appropriate learning rate
# TODO: Optional: Define learning rate scheduler


# ==============================================================================
# STEP 6: TRAINING LOOP
# ==============================================================================
# TODO: Loop over epochs
#   TODO: Set model to train mode
#   TODO: Loop over batches in train_loader
#       TODO: Move data to device
#       TODO: Forward pass: get regression and classification outputs
#       TODO: Calculate regression loss (MSE)
#       TODO: Calculate classification loss (CrossEntropy)
#       TODO: Combine losses (weighted sum)
#       TODO: Backward pass and optimizer step
#       TODO: Track training metrics (loss, MAE)
#   TODO: Validation loop
#       TODO: Set model to eval mode
#       TODO: Calculate validation losses and metrics
#       TODO: Calculate MAE, MSE, R² for regression
#       TODO: Calculate accuracy, precision, recall, F1 for classification
#   TODO: Save best model based on validation performance
#   TODO: Print epoch results and plot loss curves


# ==============================================================================
# STEP 7: EVALUATION ON TEST SET
# ==============================================================================
# TODO: Load best model checkpoint
# TODO: Set model to eval mode
# TODO: Evaluate on held-out test set
# TODO: Calculate regression metrics: MSE, MAE, R²
# TODO: Calculate classification metrics: accuracy, precision, recall, F1, ROC-AUC
# TODO: Plot confusion matrix for clinical categories
# TODO: Create scatter plot: predicted vs actual EF
# TODO: Emphasize recall for Reduced EF detection (minimize false negatives)
# TODO: Compare results against EchoNet-Dynamic benchmark


# ==============================================================================
# STEP 8: GRAD-CAM VISUALIZATION (OPTIONAL)
# ==============================================================================
# TODO: Implement Grad-CAM for model interpretability
# TODO: Visualize which cardiac structures the model focuses on
# TODO: Overlay attention maps on input frames
# TODO: Verify model is attending to relevant anatomical regions


# ==============================================================================
# STEP 9: ABLATION STUDY (OPTIONAL)
# ==============================================================================
# TODO: Compare different architectural components
# TODO: Test different numbers of frames, network depths, loss weights
# TODO: Document performance changes relative to baseline


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def create_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Raises:
        FileNotFoundError: If dataset files not found
        ValueError: If dataset is invalid
    """
    logger.info("INITIALIZING ECHONET-DYNAMIC DATASET")
    
    # Load video paths and labels
    train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = \
        load_video_paths_and_labels()
    
    # Compute dataset-wide normalization statistics from training set
    # This ensures consistent normalization across all videos
    # Results are cached to avoid recomputation
    dataset_mean, dataset_std = compute_dataset_statistics(train_videos)

    logger.info(f"Initializing datasets...")
    # Create datasets with computed statistics
    train_dataset = EchoDataset(
        video_paths=train_videos, 
        labels=train_labels,
        mean=dataset_mean, 
        std=dataset_std
    )
    
    val_dataset = EchoDataset(
        video_paths=val_videos, 
        labels=val_labels,
        mean=dataset_mean, 
        std=dataset_std
    )
    
    test_dataset = EchoDataset(
        video_paths=test_videos, 
        labels=test_labels,
        mean=dataset_mean, 
        std=dataset_std
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info("Dataset Summary:")
    logger.info(f"  Train: {len(train_dataset)} videos")
    logger.info(f"  Val:   {len(val_dataset)} videos")
    logger.info(f"  Test:  {len(test_dataset)} videos")
    logger.info(f"  Batch size: {config.BATCH_SIZE}")
    logger.info(f"  Normalization: mean={dataset_mean:.4f}, std={dataset_std:.4f}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders()
        
        logger.info("Data pipeline initialized.")
        logger.info(f"  GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU device: {torch.cuda.get_device_name(0)}")
        
        # TODO: Add model initialization, training, and evaluation here
        
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please update Config.DATASET_PATH in the configuration section")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise