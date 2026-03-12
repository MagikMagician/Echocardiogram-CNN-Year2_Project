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

# Video/Image Processing
import cv2
from PIL import Image

# Utilities
import os
import glob
from tqdm import tqdm
import csv

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score, roc_auc_score

# ==============================================================================
# PROJECT: Automated EF Estimation from Echocardiogram Videos
# Dataset: EchoNet-Dynamic (Stanford University)
# Goal: Predict continuous EF values (regression) and classify into clinical categories
# Categories: Reduced (<40%), Mildly Reduced (40-49%), Preserved (≥50%)
# ==============================================================================

# ==============================================================================
# STEP 1: LOAD DATA FROM CSV
# ==============================================================================
def split_data(df, split_type, video_dir='dataset/Videos'):
    """Load videos and labels for a specific split (TRAIN/VAL/TEST)."""
    videos = []
    labels = []
    
    for index, row in df.iterrows():
        if row['Split'] == split_type:
            video_path = os.path.join(video_dir, row['FileName'] + '.avi')
            if os.path.exists(video_path):
                videos.append(video_path)
                labels.append(row['EF'])
            else:
                print(f"Video not found: {video_path}")
    
    print(f"\nTotal {split_type} videos found: {len(videos)}")
    return videos, labels


def load_video_paths_and_labels():
    """
    Load EchoNet-Dynamic dataset from FileList.csv.
    Returns train/val/test splits with video paths and EF labels.
    """
    data_path = 'dataset'  # Adjust if your dataset is in a different location
    file_list_path = os.path.join(data_path, 'FileList.csv')
    video_dir_path = os.path.join(data_path, 'Videos')

    print(f"Loading data from: {file_list_path}")

    # Read CSV with FileName, Split, and EF columns
    df = pd.read_csv(file_list_path, usecols=['FileName', 'Split', 'EF'])
    
    # Split into TRAIN, VAL, and TEST sets
    train_videos, train_labels = split_data(df, 'TRAIN', video_dir_path)
    val_videos, val_labels = split_data(df, 'VAL', video_dir_path)
    test_videos, test_labels = split_data(df, 'TEST', video_dir_path)
    
    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels


def convert_ef_to_category(ef):
    """
    Convert continuous EF value to clinical category.
    Reduced: <40%, Mildly Reduced: 40-49%, Preserved: ≥50%
    """
    if ef < 40:
        return 0  # Reduced
    elif ef < 50:
        return 1  # Mildly Reduced
    else:
        return 2  # Preserved


def compute_dataset_statistics(video_paths, num_samples=100, num_frames=16):
    """
    Compute mean and std across a sample of videos for normalization.
    Computing on all 10,000 videos would be slow, so sample a subset.
    
    Args:
        video_paths: List of video paths
        num_samples: Number of videos to sample for statistics
        num_frames: Number of frames per video
    
    Returns:
        mean, std: Normalization statistics
    """
    print(f"Computing dataset statistics from {num_samples} videos...")
    
    # Sample random videos
    sample_indices = np.random.choice(len(video_paths), 
                                     min(num_samples, len(video_paths)), 
                                     replace=False)
    
    all_pixels = []
    
    for idx in tqdm(sample_indices, desc="Sampling videos"):
        try:
            cap = cv2.VideoCapture(video_paths[idx])
            if not cap.isOpened():
                continue
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            current_idx = 0
            frame_count = 0
            
            while frame_count < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if current_idx in frame_indices:
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, (112, 112))
                    all_pixels.append(frame.flatten())
                    frame_count += 1
                
                current_idx += 1
            
            cap.release()
        except Exception as e:
            print(f"Error reading video {video_paths[idx]}: {e}")
            continue
    
    if len(all_pixels) == 0:
        print("Warning: No valid frames found, using default normalization")
        return 0.5, 0.5
    
    all_pixels = np.concatenate(all_pixels) / 255.0  # Scale to [0, 1]
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std


# ==============================================================================
# STEP 2: DATASET CLASS - Loads echocardiogram videos as temporal sequences
# ==============================================================================
class EchoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, transform=None, 
                 mean=0.0, std=1.0, target_size=(112, 112)):
        """
        Dataset for echocardiogram videos.
        Args:
            video_paths: List of video file paths
            labels: List of EF values (continuous)
            num_frames: Number of frames to extract per video
            transform: Optional transforms for preprocessing
            mean: Dataset mean for normalization (compute once across dataset)
            std: Dataset std for normalization (compute once across dataset)
            target_size: Resize dimensions (height, width)
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.mean = mean
        self.std = std
        self.target_size = target_size
    
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_frames(self, video_path):
        """
        Load frames from video with error handling.
        Returns frames array or None if loading fails.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Cannot open video {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames < self.num_frames:
                print(f"Warning: Video {video_path} has only {total_frames} frames, need {self.num_frames}")
                # Adjust to available frames
                frame_indices = np.arange(total_frames)
            else:
                # Extract num_frames frames uniformly across video
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            
            # Read all frames sequentially (faster than seeking)
            current_idx = 0
            frame_idx_set = set(frame_indices)
            
            while len(frames) < len(frame_indices):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_idx in frame_idx_set:
                    # Convert to grayscale
                    if len(frame.shape) == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Resize to target size
                    frame = cv2.resize(frame, self.target_size)
                    frames.append(frame)
                
                current_idx += 1
            
            cap.release()
            
            # Pad with zeros if we didn't get enough frames
            while len(frames) < self.num_frames:
                frames.append(np.zeros(self.target_size, dtype=np.uint8))
            
            return np.array(frames[:self.num_frames], dtype=np.float32)
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None
        
    def __getitem__(self, idx):
        # Get video path and EF label at index idx
        video_path = self.video_paths[idx]
        ef_continuous = self.labels[idx]
        
        # Load video frames with error handling
        frames = self.load_video_frames(video_path)
        
        # If video loading failed, return zeros (or skip - but this keeps batch sizes consistent)
        if frames is None:
            frames = np.zeros((self.num_frames, *self.target_size), dtype=np.float32)
        
        # Normalize pixel values using dataset-wide statistics
        frames = frames / 255.0  # Scale to [0, 1]
        if self.std > 0:
            frames = (frames - self.mean) / self.std  # Standardize
        
        # Convert to torch tensor: (1, num_frames, H, W) for grayscale
        video_tensor = torch.from_numpy(frames).unsqueeze(0)  # Add channel dimension
        
        # Convert EF to clinical category
        ef_category = convert_ef_to_category(ef_continuous)
        
        # Apply optional transforms
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, torch.tensor(ef_continuous, dtype=torch.float32), torch.tensor(ef_category, dtype=torch.long)


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
# Load video paths and labels
train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = load_video_paths_and_labels()

# Compute dataset-wide normalization statistics from training set
# This ensures consistent normalization across all videos
dataset_mean, dataset_std = compute_dataset_statistics(train_videos, num_samples=100, num_frames=16)

# Create datasets with computed statistics
train_dataset = EchoDataset(video_paths=train_videos, labels=train_labels, 
                           mean=dataset_mean, std=dataset_std)
val_dataset = EchoDataset(video_paths=val_videos, labels=val_labels,
                         mean=dataset_mean, std=dataset_std)
test_dataset = EchoDataset(video_paths=test_videos, labels=test_labels,
                          mean=dataset_mean, std=dataset_std)

# Create DataLoaders
# NOTE: Use num_workers=0 to avoid OpenCV multi-threading issues
# Increase batch_size if you have enough GPU memory (start with 4-8)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

print(f"\nDataset sizes:")
print(f"Train: {len(train_dataset)} videos")
print(f"Val: {len(val_dataset)} videos")
print(f"Test: {len(test_dataset)} videos")