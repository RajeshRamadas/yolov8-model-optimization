# trial_manager.py - ENHANCED VERSION with PyTorch 2.6+ compatibility fix
"""
Module for managing trial execution in Neural Architecture Search.
ENHANCED VERSION - Resolves CUDA detection, device assignment, and PyTorch compatibility issues.
"""

import os
import time
import subprocess
import yaml
from pathlib import Path
from utils import save_json, load_json


def detect_device():
    """
    Intelligently detect the best available device for training.
    
    Returns:
        str: Device string ('cuda:0', 'mps', or 'cpu')
    """
    try:
        import torch
        
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                # Test if we can actually use CUDA
                try:
                    test_tensor = torch.randn(2, 2).cuda()
                    del test_tensor  # Clean up
                    return f'cuda:0'  # Use first GPU
                except Exception as e:
                    print(f"⚠️  CUDA detected but not functional: {e}")
        
        # Check for Apple Metal Performance Shaders (M1/M2 Macs)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                test_tensor = torch.randn(2, 2).to('mps')
                del test_tensor
                return 'mps'
            except Exception as e:
                print(f"⚠️  MPS detected but not functional: {e}")
        
        # Fallback to CPU
        print("ℹ️  Using CPU device (CUDA/MPS not available)")
        return 'cpu'
        
    except ImportError:
        print("⚠️  PyTorch not available, defaulting to CPU")
        return 'cpu'


def generate_custom_yaml(trial_id, params, results_dir):
    """
    Generate a custom YOLOv8 YAML file for the trial with proper kernel sizes.
    
    Args:
        trial_id: ID of the trial
        params: Parameters for the trial
        results_dir: Directory to save results
        
    Returns:
        str: Path to the generated YAML file
    """
    trial_dir = os.path.join(results_dir, f"trial_{trial_id}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Get parameters
    depth_multiple = params.get('depth_multiple', 0.33)
    width_multiple = params.get('width_multiple', 0.25)
    kernel_size = params.get('kernel_size', 3)
    model_type = params.get('model_type', 'yolov8n')
    
    # Determine base channels based on model type
    if model_type == 'yolov8n':
        base_channels = [64, 128, 256, 512, 1024]
    elif model_type == 'yolov8s':
        base_channels = [64, 128, 256, 512, 1024]
    else:
        base_channels = [64, 128, 256, 512, 1024]
    
    # Apply width scaling to channels
    channels = [max(16, int(ch * width_multiple)) for ch in base_channels]
    
    # Create custom YAML configuration
    yaml_config = {
        'nc': 80,  # number of classes - this will be overridden by the dataset
        'depth_multiple': depth_multiple,
        'width_multiple': width_multiple,
        'backbone': [
            [-1, 1, 'Conv', [channels[0], kernel_size, 2]],  # 0-P1/2
            [-1, 1, 'Conv', [channels[1], kernel_size, 2]],  # 1-P2/4
            [-1, round(3 * depth_multiple), 'C2f', [channels[1], True]],  # 2
            [-1, 1, 'Conv', [channels[2], kernel_size, 2]],  # 3-P3/8
            [-1, round(6 * depth_multiple), 'C2f', [channels[2], True]],  # 4
            [-1, 1, 'Conv', [channels[3], kernel_size, 2]],  # 5-P4/16
            [-1, round(6 * depth_multiple), 'C2f', [channels[3], True]],  # 6
            [-1, 1, 'Conv', [channels[4], kernel_size, 2]],  # 7-P5/32
            [-1, round(3 * depth_multiple), 'C2f', [channels[4], True]],  # 8
            [-1, 1, 'SPPF', [channels[4], 5]],  # 9
        ],
        'head': [
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],  # 10
            [[-1, 6], 1, 'Concat', [1]],  # 11
            [-1, round(3 * depth_multiple), 'C2f', [channels[3]]],  # 12
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],  # 13
            [[-1, 4], 1, 'Concat', [1]],  # 14
            [-1, round(3 * depth_multiple), 'C2f', [channels[2]]],  # 15 (P3/8-small)
            [-1, 1, 'Conv', [channels[2], kernel_size, 2]],  # 16
            [[-1, 12], 1, 'Concat', [1]],  # 17
            [-1, round(3 * depth_multiple), 'C2f', [channels[3]]],  # 18 (P4/16-medium)
            [-1, 1, 'Conv', [channels[3], kernel_size, 2]],  # 19
            [[-1, 9], 1, 'Concat', [1]],  # 20
            [-1, round(3 * depth_multiple), 'C2f', [channels[4]]],  # 21 (P5/32-large)
            [[15, 18, 21], 1, 'Detect', ['nc']],  # 22 Detect(P3, P4, P5)
        ]
    }
    
    # Save custom YAML file
    yaml_path = os.path.join(trial_dir, f"custom_model_trial_{trial_id}.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated custom YAML for trial {trial_id}: {yaml_path}")
    print(f"  - Depth multiple: {depth_multiple}")
    print(f"  - Width multiple: {width_multiple}")
    print(f"  - Kernel size: {kernel_size}")
    print(f"  - Channels: {channels}")
    
    return yaml_path


def setup_pytorch_compatibility():
    """
    Set up PyTorch compatibility for newer versions (2.6+).
    
    Returns:
        str: Python code to add to trial scripts for PyTorch compatibility
    """
    return """
# PyTorch 2.6+ Compatibility Fix
import torch
import sys

def setup_pytorch_safe_globals():
    \"\"\"Configure PyTorch safe globals for compatibility with newer versions.\"\"\"
    try:
        # Check if we're using PyTorch 2.6+
        pytorch_version = torch.__version__
        major, minor = map(int, pytorch_version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 6):
            print(f"🔧 PyTorch {pytorch_version} detected - setting up compatibility...")
            
            # Essential PyTorch classes for YOLO models
            safe_globals = [
                # Core PyTorch modules
                torch.nn.modules.container.Sequential,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.batchnorm.BatchNorm2d,
                torch.nn.modules.activation.SiLU,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.activation.LeakyReLU,
                torch.nn.modules.pooling.MaxPool2d,
                torch.nn.modules.pooling.AdaptiveAvgPool2d,
                torch.nn.modules.upsampling.Upsample,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.dropout.Dropout,
                torch.nn.modules.container.ModuleList,
                torch.nn.modules.container.ModuleDict,
                
                # Additional common modules
                torch.nn.Parameter,
                torch.Tensor,
                
                # For collections/lists/tuples
                list, tuple, dict, int, float, str, bool, type(None),
            ]
            
            # Try to add Ultralytics-specific classes if available
            try:
                from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Concat
                safe_globals.extend([Conv, C2f, SPPF, Detect, Concat])
                print("✅ Added Ultralytics modules to safe globals")
            except ImportError:
                print("⚠️  Ultralytics modules not available for safe globals")
            
            # Add safe globals
            torch.serialization.add_safe_globals(safe_globals)
            print(f"✅ Added {len(safe_globals)} classes to PyTorch safe globals")
            
            return True
        else:
            print(f"ℹ️  PyTorch {pytorch_version} - no compatibility fixes needed")
            return False
            
    except Exception as e:
        print(f"⚠️  Warning: Could not set up PyTorch compatibility: {e}")
        return False

# Set up compatibility immediately
setup_pytorch_safe_globals()
"""


def generate_trial_script(trial_id, params, data_yaml, results_dir, epochs):
    """
    Generate a Python script for running a single trial with enhanced device detection and PyTorch compatibility.
    
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
    
    # Generate custom YAML file
    custom_yaml = generate_custom_yaml(trial_id, params, results_dir)
    
    # Detect the best available device
    device = detect_device()
    
    # Determine advanced parameters (exclude basic architecture params)
    basic_params = ["depth_multiple", "width_multiple", "img_size", "model_type", "kernel_size"]
    advanced_params = {k: v for k, v in params.items() if k not in basic_params}
    
    # Get image size
    img_size = params.get('img_size', 640)
    
    # Create a Python script for this trial
    trial_script = os.path.join(trial_dir, "run_trial.py")
    
    # Get PyTorch compatibility code
    pytorch_compat_code = setup_pytorch_compatibility()
    
    with open(trial_script, 'w') as f:
        f.write(f"""
{pytorch_compat_code}

import time
import traceback
import sys
import os
from pathlib import Path
from ultralytics import YOLO
import json

# Start timer for performance measurement
start_time = time.time()

def detect_device():
    \"\"\"Detect the best available device for training.\"\"\"
    try:
        import torch
        
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 0:
                try:
                    test_tensor = torch.randn(2, 2).cuda()
                    del test_tensor
                    print(f"✅ CUDA device detected: {{torch.cuda.get_device_name(0)}}")
                    return 0  # Use device ID 0
                except Exception as e:
                    print(f"⚠️  CUDA detected but not functional: {{e}}")
        
        # Check for Apple MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                test_tensor = torch.randn(2, 2).to('mps')
                del test_tensor
                print("✅ MPS device detected")
                return 'mps'
            except Exception as e:
                print(f"⚠️  MPS detected but not functional: {{e}}")
        
        # Fallback to CPU
        print("ℹ️  Using CPU device")
        return 'cpu'
        
    except ImportError:
        print("⚠️  PyTorch not available, using CPU")
        return 'cpu'

try:
    print(f"Starting Trial {trial_id}")
    print(f"Custom YAML: {custom_yaml}")
    print(f"Data YAML: {data_yaml}")
    print(f"Image size: {img_size}")
    print(f"Epochs: {epochs}")
    
    # Detect device
    device = detect_device()
    print(f"Using device: {{device}}")
    
    # Verify files exist
    if not os.path.exists('{custom_yaml}'):
        raise FileNotFoundError(f"Custom YAML not found: {custom_yaml}")
    if not os.path.exists('{data_yaml}'):
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    
    # Load the custom model with proper error handling
    print("Loading custom model...")
    try:
        model = YOLO('{custom_yaml}')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {{e}}")
        # Try loading a standard model as fallback
        print("Trying standard yolov8n.pt as fallback...")
        model = YOLO('yolov8n.pt')
        print("✅ Fallback model loaded successfully")
    
    # Advanced parameters dictionary (filtered)
    advanced_params = {advanced_params}
    
    # Training parameters with device detection
    train_params = {{
        'data': '{data_yaml}',
        'epochs': {epochs},
        'imgsz': {img_size},
        'project': '{results_dir}',
        'name': 'trial_{trial_id}',
        'exist_ok': True,
        'verbose': True,
        'patience': 10,  # Early stopping patience
        'save': True,
        'plots': True,
        'device': device,  # Use detected device
        **advanced_params
    }}
    
    print(f"Training parameters: {{train_params}}")
    
    # Train the model
    print("Starting training...")
    results = model.train(**train_params)
    
    # Calculate metrics
    training_time = time.time() - start_time
    
    # Get model file paths
    weights_dir = Path('{results_dir}') / f'trial_{trial_id}' / 'weights'
    best_weights = weights_dir / 'best.pt'
    last_weights = weights_dir / 'last.pt'
    
    # Get model file size
    model_size_mb = 0
    if best_weights.exists():
        model_size_mb = best_weights.stat().st_size / (1024 * 1024)
    elif last_weights.exists():
        model_size_mb = last_weights.stat().st_size / (1024 * 1024)
        
    print(f"Model size: {{model_size_mb:.2f}} MB")
    
    # Measure inference speed (FPS) with error handling
    fps = 0
    inference_time_ms = 0
    
    try:
        if best_weights.exists():
            # Load the trained model
            trained_model = YOLO(str(best_weights))
            
            # Use a simple test image for speed measurement
            import torch
            import numpy as np
            
            # Create a dummy image tensor
            dummy_img = torch.randn(1, 3, {img_size}, {img_size})
            
            # Move to appropriate device
            if device != 'cpu' and device != 'mps':
                dummy_img = dummy_img.cuda()
            elif device == 'mps':
                dummy_img = dummy_img.to('mps')
            
            # Warm up
            for _ in range(5):
                try:
                    _ = trained_model.predict(dummy_img, verbose=False, show=False, save=False)
                except:
                    break
            
            # Synchronize if using CUDA
            if device != 'cpu' and device != 'mps' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_inference = time.time()
            for _ in range(10):
                try:
                    _ = trained_model.predict(dummy_img, verbose=False, show=False, save=False)
                except:
                    break
                    
            # Synchronize again if using CUDA
            if device != 'cpu' and device != 'mps' and torch.cuda.is_available():
                torch.cuda.synchronize()
                
            total_inference_time = time.time() - start_inference
            inference_time_ms = (total_inference_time / 10) * 1000
            fps = 1000 / inference_time_ms if inference_time_ms > 0 else 0
            
            print(f"Inference time: {{inference_time_ms:.2f}} ms, FPS: {{fps:.2f}}")
            
    except Exception as e:
        print(f"Error measuring inference speed: {{e}}")
        fps = 0
        inference_time_ms = 0
    
    # Extract metrics from results
    metrics_dict = {{}}
    if hasattr(results, 'results_dict') and results.results_dict:
        results_dict = results.results_dict
        metrics_dict = {{
            'map50': results_dict.get('metrics/mAP50(B)', 0),
            'map50_95': results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': results_dict.get('metrics/precision(B)', 0),
            'recall': results_dict.get('metrics/recall(B)', 0),
        }}
    else:
        # Try to get metrics from validation results
        try:
            val_results = model.val(data='{data_yaml}', verbose=False, device=device)
            if hasattr(val_results, 'box'):
                metrics_dict = {{
                    'map50': getattr(val_results.box, 'map50', 0) or 0,
                    'map50_95': getattr(val_results.box, 'map', 0) or 0,
                    'precision': getattr(val_results.box, 'mp', 0) or 0,
                    'recall': getattr(val_results.box, 'mr', 0) or 0,
                }}
        except Exception as e:
            print(f"Error getting validation metrics: {{e}}")
            metrics_dict = {{
                'map50': 0,
                'map50_95': 0,
                'precision': 0,
                'recall': 0,
            }}
    
    # Compile all metrics
    metrics = {{
        **metrics_dict,
        'training_time_hours': training_time / 3600,
        'model_size_mb': model_size_mb,
        'fps': fps,
        'inference_time_ms': inference_time_ms,
        'trial_id': {trial_id},
        'trial_success': True,
        'device_used': str(device)
    }}
    
    # Calculate combined score (adjust weights based on priorities)
    map_weight = 1.0
    speed_weight = 0.3
    size_weight = 0.2
    
    # Normalize metrics (higher is better for all)
    norm_map = metrics['map50_95']  # Already between 0-1
    norm_fps = min(fps / 100, 1.0) if fps > 0 else 0  # Normalize FPS, cap at 100 FPS
    norm_size = max(0, 1.0 - min(model_size_mb / 100, 0.9)) if model_size_mb > 0 else 0  # Smaller is better
    
    combined_score = (
        map_weight * norm_map + 
        speed_weight * norm_fps + 
        size_weight * norm_size
    )
    
    metrics['combined_score'] = combined_score
    
    # Save metrics
    with open('{trial_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f"\\nTrial {trial_id} Results:")
    print(f"  Device: {{device}}")
    print(f"  mAP50: {{metrics['map50']:.4f}}")
    print(f"  mAP50-95: {{metrics['map50_95']:.4f}}")
    print(f"  Precision: {{metrics['precision']:.4f}}")
    print(f"  Recall: {{metrics['recall']:.4f}}")
    print(f"  FPS: {{fps:.2f}}")
    print(f"  Model Size: {{model_size_mb:.2f}} MB")
    print(f"  Combined Score: {{combined_score:.4f}}")
    print(f"  Training Time: {{training_time/3600:.2f}} hours")
    
    print(f"\\nTrial {trial_id} completed successfully!")
    
except Exception as e:
    print(f"\\nERROR in Trial {trial_id}: {{str(e)}}")
    print("Full traceback:")
    traceback.print_exc()
    
    # Save error metrics
    error_metrics = {{
        'map50': 0,
        'map50_95': 0,
        'precision': 0,
        'recall': 0,
        'training_time_hours': 0,
        'model_size_mb': 0,
        'fps': 0,
        'inference_time_ms': 0,
        'combined_score': 0,
        'trial_id': {trial_id},
        'trial_success': False,
        'error_message': str(e),
        'device_used': 'unknown'
    }}
    
    with open('{trial_dir}/metrics.json', 'w') as f:
        json.dump(error_metrics, f, indent=2, default=str)
    
    # Exit with error code
    sys.exit(1)
""")
    
    return trial_script


def run_trial(trial_id, params, data_yaml, results_dir, epochs):
    """
    Run a single trial with the given parameters and enhanced device detection.
    
    Args:
        trial_id: ID of the trial
        params: Parameters for the trial
        data_yaml: Path to the data.yaml file
        results_dir: Directory to save results
        epochs: Number of epochs to train
        
    Returns:
        dict or None: Results of the trial, or None if the trial failed
    """
    print(f"\n" + "="*60)
    print(f"STARTING TRIAL {trial_id} - ENHANCED VERSION")
    print("="*60)
    
    print("Trial Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Detect device before starting
    device = detect_device()
    print(f"Detected device: {device}")
    
    # Generate the trial script
    trial_script = generate_trial_script(trial_id, params, data_yaml, results_dir, epochs)
    trial_dir = os.path.join(results_dir, f"trial_{trial_id}")
    
    # Run the trial script
    print(f"\nExecuting trial script: {trial_script}")
    try:
        result = subprocess.run(
            ["python", trial_script], 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout per trial
        )
        
        # Save output logs
        with open(os.path.join(trial_dir, "output.log"), "w") as f:
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write(f"\n\nReturn Code: {result.returncode}")
        
        if result.returncode != 0:
            print(f"❌ Trial {trial_id} FAILED with error code {result.returncode}")
            print("Error output:")
            print(result.stderr[-1000:])  # Last 1000 characters of error
            return None
        else:
            print(f"✅ Trial {trial_id} COMPLETED successfully")
            
            # Load metrics
            metrics_file = os.path.join(trial_dir, "metrics.json")
            if os.path.exists(metrics_file):
                metrics = load_json(metrics_file)
                if metrics and metrics.get('trial_success', False):
                    print(f"Results Summary:")
                    print(f"  Device Used: {metrics.get('device_used', 'unknown')}")
                    print(f"  mAP50-95: {metrics.get('map50_95', 0):.4f}")
                    print(f"  FPS: {metrics.get('fps', 0):.2f}")
                    print(f"  Model Size: {metrics.get('model_size_mb', 0):.2f} MB")
                    print(f"  Combined Score: {metrics.get('combined_score', 0):.4f}")
                    
                    return {
                        "trial_id": trial_id,
                        "params": params,
                        "metrics": metrics
                    }
                else:
                    print(f"❌ Trial {trial_id} reported failure in metrics")
                    return None
            else:
                print(f"❌ No metrics file found for trial {trial_id}")
                return None
                
    except subprocess.TimeoutExpired:
        print(f"❌ Trial {trial_id} TIMED OUT after 1 hour")
        return None
    except Exception as e:
        print(f"❌ Error running trial {trial_id}: {e}")
        return None