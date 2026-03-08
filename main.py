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


# ==============================================================================
# STEP 2: DATASET CLASS - Loads echocardiogram videos as temporal sequences
# ==============================================================================
class EchoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        """
        Dataset for echocardiogram videos.
        Args:
            video_paths: List of video file paths
            labels: List of EF values (continuous)
            num_frames: Number of frames to extract per video
            transform: Optional transforms for preprocessing
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
    
    def __len__(self):
        # TODO: Return total number of videos
        pass
        
    def __getitem__(self, idx):
        # TODO: Get video path and EF label at index idx
        # TODO: Open video with cv2.VideoCapture()
        # TODO: Extract num_frames frames uniformly across cardiac cycle
        # TODO: Convert frames to numpy array shape (num_frames, H, W)
        # TODO: Resize frames to consistent size (e.g., 112x112 for 3D CNN)
        # TODO: Normalize pixel values (mean/std normalization)
        # TODO: Convert to torch tensor: (1, num_frames, H, W) for grayscale
        # TODO: Convert EF to clinical category using convert_ef_to_category()
        # TODO: Return (video_tensor, ef_continuous, ef_category)
        pass


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
train_videos, train_labels, val_videos, val_labels, test_videos, test_labels = load_video_paths_and_labels()
train_dataset = EchoDataset(video_paths=train_videos, labels=train_labels)