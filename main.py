# Core Deep Learning
import torch
import torch.nn as nn
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

# Make a function pass in the data and make it readable for the model


# Cycle through FileList.csv to locate all videos marked as TRAIN
df = pd.read_csv('dataset/FileList.csv', usecols=['FileName', 'Split'])
train_videos = []

for index, row in df.iterrows():
    if row['Split'] == 'TRAIN':
        video_path = os.path.join('dataset/Videos', row['FileName'] + '.avi')
        if os.path.exists(video_path):
            train_videos.append(video_path)
            print(f"Found TRAIN video: {row['FileName']}")
        else:
            print(f"Video not found: {video_path}")

print(f"\nTotal TRAIN videos found: {len(train_videos)}")

# Next steps:
# - Iterate through the array of train_videos and parse through each video to extract frames


# Make the model and fit the data

# Make a function to evaluate the model and make predictions

#Make a function to visualize the results
