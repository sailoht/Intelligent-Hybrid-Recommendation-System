from pathlib import Path
import os

TRAINING_FILE = "Cap_Training_Data_2025.csv"
DATASET_FOLDER = "12573233-Indian_Car_Market_Capstone_2025_Dataset"

DATASET_PATH_ENV = "CAR_DATASET_PATH"
DATASET_DIR_ENV = "CAR_DATASET_DIR"


def find_training_file():
    """
    Search for Cap_Training_Data_2025.csv automatically.
    """

    # 1. Environment variable points directly to CSV
    csv_path = os.getenv(DATASET_PATH_ENV)
    if csv_path and Path(csv_path).exists():
        return Path(csv_path).resolve()

    # 2. Environment variable points to dataset folder
    dataset_dir = os.getenv(DATASET_DIR_ENV)
    if dataset_dir:
        candidate = Path(dataset_dir) / TRAINING_FILE
        if candidate.exists():
            return candidate.resolve()

    # 3. Search common locations
    search_roots = [
        Path.home() / "Downloads",
        Path.home() / "Documents",
        Path.home() / "Desktop",
        Path.cwd(),
    ]

    for root in search_roots:
        try:
            for file in root.rglob(TRAINING_FILE):
                return file.resolve()
        except PermissionError:
            pass

    return None


TRAINING_CSV_PATH = find_training_file()

if TRAINING_CSV_PATH:
    DATASET_DIR = TRAINING_CSV_PATH.parent
else:
    DATASET_DIR = None
