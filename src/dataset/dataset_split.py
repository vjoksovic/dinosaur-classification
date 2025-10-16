from sklearn.model_selection import train_test_split
import glob
import os
from pathlib import Path
from dataset.dino_dataset import DinoDataset
from utils.transformers import train_transform, test_transform
from utils.utils import load_config


def _scan_and_split():
    # Resolve dataset directory relative to project root on each call
    root = Path(__file__).resolve().parents[2]
    dataset_dir = root / "data"

    classes = sorted([d for d in os.listdir(dataset_dir) if (dataset_dir / d).is_dir()])
    config = load_config("src/config/config.py")

    file_paths = []
    labels = []

    for idx, cls in enumerate(classes):
        for img_path in sorted(glob.glob(str(dataset_dir / cls / "*"))):
            file_paths.append(img_path)
            labels.append(idx)

    # Ratios from config
    train_ratio = config["data"]["train_split"]
    val_ratio = config["data"]["val_split"]
    test_ratio = config["data"]["test_split"]
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Stratified split: first train vs temp (val+test), then val vs test
    temp_ratio = val_ratio + test_ratio
    train_fp, temp_fp, train_lbl, temp_lbl = train_test_split(
        file_paths, labels, test_size=temp_ratio, stratify=labels, random_state=42
    )
    test_size_from_temp = test_ratio / (val_ratio + test_ratio)
    val_fp, test_fp, val_lbl, test_lbl = train_test_split(
        temp_fp, temp_lbl, test_size=test_size_from_temp, stratify=temp_lbl, random_state=42
    )

    train_ds = DinoDataset(train_fp, train_lbl, transform=train_transform)
    val_ds = DinoDataset(val_fp, val_lbl, transform=test_transform)
    test_ds = DinoDataset(test_fp, test_lbl, transform=test_transform)

    return train_ds, val_ds, test_ds


def load_dataset_split():
    train_ds, val_ds, test_ds = _scan_and_split()
    return train_ds, val_ds, test_ds
