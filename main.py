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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================================================================
# STEP 1: LOAD DATA FROM CSV
# ==============================================================================
def split_data(df, split_type, video_dir='dataset/Videos'):
    """Helper function to load videos and labels for a specific split type."""
    videos = []
    labels = []
    
    for index, row in df.iterrows():
        if row['Split'] == split_type:
            video_path = os.path.join(video_dir, row['FileName'] + '.avi')
            if os.path.exists(video_path):
                videos.append(video_path)
                labels.append(row['EF'])
                # print(f"Found {split_type} video: {row['FileName']}")
            else:
                print(f"Video not found: {video_path}")
    
    print(f"\nTotal {split_type} videos found: {len(videos)}")
    return videos, labels


def load_video_paths_and_labels():
    data_path = 'C:\\Users\\aaron\\Downloads\\EchoNet-Dynamic'
    file_list_path = os.path.join(data_path, 'FileList.csv')
    video_dir_path = os.path.join(data_path, 'Videos')

    print(f"Loading data from: {file_list_path}")

    df = pd.read_csv(file_list_path, usecols=['FileName', 'Split', 'EF'])
    
    train_videos, train_labels = split_data(df, 'TRAIN', video_dir_path)
    val_videos, val_labels = split_data(df, 'VAL', video_dir_path)
    
    return train_videos, train_labels, val_videos, val_labels
# TODO: Read dataset/FileList.csv
# TODO: Extract video paths and EF labels
# TODO: Split into train/val sets using train_test_split


# ==============================================================================
# STEP 2: DATASET CLASS - Loads videos and returns tensors
# ==============================================================================
class EchoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
    
    def __len__(self):
        # TODO: Return total number of videos
        pass
        
    def __getitem__(self, idx):
        # TODO: Get video path and label at index idx
        # TODO: Open video with cv2.VideoCapture()
        # TODO: Loop and extract num_frames frames
        # TODO: Convert frames to numpy array shape (num_frames, H, W)
        # TODO: Convert to torch tensor
        # TODO: Add channel dimension: (1, num_frames, H, W) for grayscale
        # TODO: Resize frames to consistent size (e.g., 128x128 or 256x256)
        # TODO: Normalize pixel values (0-1 or mean/std normalization)
        # TODO: Return (video_tensor, label)
        pass


# ==============================================================================
# STEP 3: 3D CNN MODEL - Processes video volumes
# ==============================================================================
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        # TODO: Define Conv3d layers
        # Example: self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,3))
        # TODO: Define MaxPool3d layers
        # TODO: Define fully connected layers
        # TODO: Calculate flattened size after convolutions
        pass
        
    def forward(self, x):
        # Input x shape: (batch, 1, time, height, width)
        # TODO: Pass through conv3d + relu + maxpool layers
        # TODO: Flatten tensor
        # TODO: Pass through FC layers
        # TODO: Return final prediction (EF value)
        pass


# ==============================================================================
# STEP 4: CREATE DATALOADERS
# ==============================================================================
# TODO: Create train_dataset and val_dataset using EchoDataset
# TODO: Create train_loader and val_loader with DataLoader
# NOTE: Use small batch_size (4 or 8) because videos use lots of memory


# ==============================================================================
# STEP 5: INITIALIZE MODEL, LOSS, OPTIMIZER
# ==============================================================================
# TODO: Set device (cuda or cpu)
# TODO: Initialize CNN3D model
# TODO: Define loss function (MSELoss for regression)
# TODO: Define optimizer (Adam)


# ==============================================================================
# STEP 6: TRAINING LOOP
# ==============================================================================
# TODO: Loop over epochs
#   TODO: Set model to train mode
#   TODO: Loop over batches in train_loader
#       TODO: Move data to device
#       TODO: Forward pass
#       TODO: Calculate loss
#       TODO: Backward pass
#       TODO: Update weights
#   TODO: Validation loop
#       TODO: Set model to eval mode
#       TODO: Calculate validation loss
#   TODO: Print epoch results


# ==============================================================================
# STEP 7: EVALUATION & TESTING
# ==============================================================================
# TODO: Load best model
# TODO: Evaluate on validation set
# TODO: Calculate metrics (MAE, RMSE for EF prediction)
# TODO: Visualize predictions vs actual
train_videos, train_labels, val_videos, val_labels = load_video_paths_and_labels()
train_dataset = EchoDataset(video_paths=train_videos, labels=train_labels)