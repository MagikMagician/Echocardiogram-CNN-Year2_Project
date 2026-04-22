from pathlib import Path
import json
from typing import Tuple

LOCAL_CONFIG_FILE = Path("local_config.json")


def load_local_config(config_path: Path = LOCAL_CONFIG_FILE) -> dict:
    """Load optional local overrides for person-specific settings."""
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if isinstance(config_data, dict):
            return config_data
        return {}
    except Exception:
        return {}


local_config = load_local_config()

# =============================================================================
# Project configuration — all hyperparameters, thresholds, and paths in one
# place.  Override DATASET_PATH via local_config.json to avoid editing this
# file across different machines.
# =============================================================================
class Config:
    """Configuration for the EchoNet-Dynamic project."""

    # Dataset paths
    DATASET_PATH = Path(local_config.get("DATASET_PATH", "dataset"))  # Update this to your dataset location
    FILE_LIST_NAME = "FileList.csv"
    VIDEO_DIR_NAME = "Videos"
    VIDEO_EXTENSION = ".avi"

    # Model hyperparameters
    NUM_FRAMES = 32           # Number of frames to sample per video (EchoNet paper: 32)
    TARGET_HEIGHT = 112       # Frame height after resize
    TARGET_WIDTH = 112        # Frame width after resize
    BATCH_SIZE = 16           # Batch size for DataLoader
    NUM_WORKERS = 4           # Number of workers for DataLoader
    FRAME_SAMPLING_PERIOD = 2 # Sample every Nth frame (paper: 2 → 32 frames from 64-frame window)

    # EF Categories
    EF_REDUCED_THRESHOLD = 40.0  # EF < 40% = Reduced
    EF_MILDLY_REDUCED_THRESHOLD = 50.0  # 40% ≤ EF < 50% = Mildly Reduced
    NUM_CATEGORIES = 3  # Reduced, Mildly Reduced, Preserved

    # Normalization (will be computed from training data if cache doesn't exist)
    STATS_CACHE_FILE = "artifacts/dataset_statistics.json"
    NUM_SAMPLES_FOR_STATS = 100  # Number of videos to sample for computing statistics
    DEFAULT_MEAN = 0.5  # Fallback if statistics computation fails
    DEFAULT_STD = 0.5

    # Training hyperparameters
    LEARNING_RATE = 1e-4      # Scaled with batch size (linear scaling rule: 2× batch → 2× LR)
    NUM_EPOCHS = 45  # EchoNet paper trains for 45 epochs
    PATIENCE = 10  # Early stopping patience

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
