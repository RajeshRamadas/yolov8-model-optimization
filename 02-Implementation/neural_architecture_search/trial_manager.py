# trial_manager.py
"""
Module for managing trial execution in Neural Architecture Search.
"""

import os
import time
import subprocess
from pathlib import Path
from utils import save_json, load_json


def generate_trial_script(trial_id, params, data_yaml, results_dir, epochs):
    """
    Generate a Python script for running a single trial.
    
    Args:
        trial_id: ID of the trial
        params: Parameters for the trial
        data_yaml: Path to the data.yaml file
        results_dir: Directory to save results
        epochs: Number of epochs to train
        
    Returns:
        str: Path to the generated script
    """
    # Create trial directory
    trial_dir = os.path.join(results_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Save parameters for reference
    save_json(params, os.path.join(trial_dir, "params.json"))
    
    # Determine advanced parameters
    advanced_params = {k: v for k, v in params.items() 
                     if k not in ["depth_multiple", "width_multiple", "img_size", "model_type", "kernel_size"]}
    
    model_type = params.get("model_type", "yolov8n")
    
    # Create a Python script for this trial
    trial_script = os.path.join(trial_dir, "run_trial.py")
    
    with open(trial_script, 'w') as f:
        f.write(f"""
import time
from ultralytics import YOLO
import yaml
import json
import os
from pathlib import Path

# Start timer for performance measurement
start_time = time.time()

# Load the base model
model = YOLO('{model_type}.yaml')

# Set the depth and width multipliers
model.model.yaml['depth_multiple'] = {params['depth_multiple']}
model.model.yaml['width_multiple'] = {params['width_multiple']}

# Set kernel size if specified
kernel_size = {params.get('kernel_size', 3)}

# Modify kernel size in backbone and head if kernel_size is specified
for module in model.model.yaml['backbone'] + model.model.yaml['head']:
    if 'Conv' in module[2] and module[2] != 'Focus':  # Only for Conv modules, not Focus
        if 'k' in module[3]:  # If the kernel parameter exists
            module[3]['k'] = kernel_size  # Set kernel size

# Advanced parameters dictionary
advanced_params = {advanced_params}

# Train the model
results = model.train(
    data='{data_yaml}',
    epochs={epochs},
    imgsz={params['img_size']},
    project='{results_dir}',
    name='trial_{trial_id}',
    exist_ok=True,
    **advanced_params
)

# Calculate metrics
training_time = time.time() - start_time
results_dict = results.results_dict

# Get model file size
weights_path = Path('{results_dir}') / f'trial_{trial_id}' / 'weights' / 'best.pt'
model_size_mb = weights_path.stat().st_size / (1024 * 1024) if weights_path.exists() else 0

# Measure inference speed (FPS)
if weights_path.exists():
    # Load the trained model
    trained_model = YOLO(str(weights_path))
    
    # Run inference speed test
    batch_size = 1
    img_size = {params['img_size']}
    
    # Warm up
    for _ in range(10):
        trained_model.predict('https://ultralytics.com/images/bus.jpg', imgsz=img_size)
    
    # Measure speed
    t0 = time.time()
    for _ in range(50):
        trained_model.predict('https://ultralytics.com/images/bus.jpg', imgsz=img_size)
    inference_time = (time.time() - t0) / 50
    fps = 1.0 / inference_time if inference_time > 0 else 0
else:
    fps = 0

# Save additional metrics
metrics = {{
    'map50': results_dict.get('metrics/mAP50(B)', 0),
    'map50_95': results_dict.get('metrics/mAP50-95(B)', 0),
    'precision': results_dict.get('metrics/precision(B)', 0),
    'recall': results_dict.get('metrics/recall(B)', 0),
    'training_time_hours': training_time / 3600,
    'model_size_mb': model_size_mb,
    'fps': fps,
    'parameters': model.model.yaml,
    'inference_time_ms': inference_time * 1000 if 'inference_time' in locals() else 0
}}

# Calculate combined score (adjust weights based on your priorities)
map_weight = 1.0
speed_weight = 0.3
size_weight = 0.2

# Normalize metrics (higher is better for all)
norm_map = metrics['map50_95']  # Already between 0-1
norm_fps = min(fps / 100, 1.0)  # Normalize FPS, cap at 100 FPS
norm_size = 1.0 - min(model_size_mb / 100, 0.9)  # Smaller is better, cap at 100MB

combined_score = (
    map_weight * norm_map + 
    speed_weight * norm_fps + 
    size_weight * norm_size
)

metrics['combined_score'] = combined_score

# Save metrics
with open('{trial_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Training completed. Results:")
print(f"  mAP50-95: {{metrics['map50_95']:.4f}}")
print(f"  FPS: {{fps:.2f}}")
print(f"  Model Size: {{model_size_mb:.2f}} MB")
print(f"  Combined Score: {{combined_score:.4f}}")
""")
    
    return trial_script


def run_trial(trial_id, params, data_yaml, results_dir, epochs):
    """
    Run a single trial with the given parameters.
    
    Args:
        trial_id: ID of the trial
        params: Parameters for the trial
        data_yaml: Path to the data.yaml file
        results_dir: Directory to save results
        epochs: Number of epochs to train
        
    Returns:
        dict or None: Results of the trial, or None if the trial failed
    """
    print(f"\nTrial {trial_id}:")
    for key, value in params.items():
        print(f"  {key}: {value}")
        
    # Generate the trial script
    trial_script = generate_trial_script(trial_id, params, data_yaml, results_dir, epochs)
    trial_dir = os.path.join(results_dir, f"trial_{trial_id}")
    
    # Run the trial script
    print(f"Running trial script: {trial_script}")
    try:
        result = subprocess.run(["python3", trial_script], capture_output=True, text=True)
        
        with open(os.path.join(trial_dir, "output.log"), "w") as f:
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
        
        if result.returncode != 0:
            print(f"Trial {trial_id} failed with error code {result.returncode}")
            return None
        else:
            print(f"Trial {trial_id} completed successfully")
            
            # Load metrics
            metrics_file = os.path.join(trial_dir, "metrics.json")
            if os.path.exists(metrics_file):
                metrics = load_json(metrics_file)
                print(f"  mAP50-95: {metrics.get('map50_95', 0):.4f}")
                print(f"  FPS: {metrics.get('fps', 0):.2f}")
                print(f"  Model Size: {metrics.get('model_size_mb', 0):.2f} MB")
                print(f"  Combined Score: {metrics.get('combined_score', 0):.4f}")
                
                return {
                    "trial_id": trial_id,
                    "params": params,
                    "metrics": metrics
                }
    except Exception as e:
        print(f"Error running trial {trial_id}: {e}")
        return None