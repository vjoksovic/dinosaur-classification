"""
Utility functions for dinosaur classification CNN project.

This module contains helper functions for training, evaluation, visualization,
and general utilities for the PyTorch-based dinosaur classification model.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import yaml
import importlib.util
import sys
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict[str, Any]:
    ext = os.path.splitext(config_path)[1].lower()
    if ext == '.py':
        module_name = Path(config_path).stem
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import config module from {config_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, 'CONFIG'):
            raise AttributeError("Python config must define a top-level CONFIG dict")
        config = getattr(module, 'CONFIG')
        if not isinstance(config, dict):
            raise TypeError("CONFIG must be a dict")
        return config
    raise ValueError(f"Unsupported config extension: {ext}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to {config_path}")


def get_class_names(data_dir: str = "data") -> List[str]:
    """
    Get class names from data directory structure.
    
    Args:
        data_dir: Path to data directory containing class folders
        
    Returns:
        Sorted list of class names (dinosaur species)
    """
    root = Path(__file__).resolve().parents[2]
    dataset_dir = root / data_dir
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {dataset_dir}")
    
    classes = sorted([d for d in os.listdir(dataset_dir) if (dataset_dir / d).is_dir()])
    return classes


def print_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (3, 224, 224)) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Input size: {input_size}")
    print("=" * 50)
    
    # Print model architecture
    print(model)
    print("=" * 50)



