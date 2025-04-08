# utils.py
"""
Utility functions for Neural Architecture Search.
"""

import json
import numpy as np
import os


def convert_to_serializable(obj):
    """
    Convert numpy/pandas/non-standard objects to Python built-in types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object converted to a serializable type
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        return convert_to_serializable(obj)


def save_json(data, filepath, indent=2):
    """
    Save data to a JSON file with proper serialization.
    
    Args:
        data: Data to save
        filepath: Path to the JSON file
        indent: Indentation level for pretty-printing
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, cls=CustomJSONEncoder)


def load_json(filepath):
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded data
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def create_directories(*dirs):
    """
    Create multiple directories if they don't exist.
    
    Args:
        *dirs: Directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def calculate_combined_score(metrics, weights):
    """
    Calculate combined score from multiple metrics.
    
    Args:
        metrics (dict): Dictionary containing metrics (map50_95, fps, model_size_mb)
        weights (dict): Dictionary containing weights for different objectives
        
    Returns:
        float: Combined score
    """
    map_value = metrics.get('map50_95', 0)
    fps = metrics.get('fps', 0)
    model_size_mb = metrics.get('model_size_mb', 0)
    
    # Normalize metrics (higher is better for all)
    norm_map = map_value  # Already between 0-1
    norm_fps = min(fps / 100, 1.0)  # Normalize FPS, cap at 100 FPS
    norm_size = 1.0 - min(model_size_mb / 100, 0.9)  # Smaller is better, cap at 100MB
    
    # Calculate combined score
    combined_score = (
        weights["map_weight"] * norm_map + 
        weights["speed_weight"] * norm_fps + 
        weights["size_weight"] * norm_size
    )
    
    return combined_score