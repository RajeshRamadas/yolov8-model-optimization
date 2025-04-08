import os
import json
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import tempfile
import time

def load_files(metrics_path, args_path):
    """Load metrics.json and args.yaml files from specified paths."""
    # Load metrics.json
    try:
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
        print(f"Successfully loaded metrics file from {metrics_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {metrics_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: {metrics_path} is not a valid JSON file.")
        exit(1)
    
    # Load args.yaml
    try:
        with open(args_path, 'r') as f:
            args_data = yaml.safe_load(f)
        print(f"Successfully loaded args file from {args_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {args_path}")
        exit(1)
    except yaml.YAMLError:
        print(f"Error: {args_path} is not a valid YAML file.")
        exit(1)
    
    return metrics_data, args_data

def create_custom_model_yaml(metrics_data, output_path='custom_model.yaml'):
    """Create a custom model YAML file from metrics.json configuration."""
    model_config = {}
    
    # Base parameters
    model_config['nc'] = metrics_data['parameters']['nc']
    model_config['depth_multiple'] = metrics_data['parameters']['depth_multiple']
    model_config['width_multiple'] = metrics_data['parameters']['width_multiple']
    
    # Copy backbone and head configurations
    if 'backbone' in metrics_data['parameters']:
        model_config['backbone'] = metrics_data['parameters']['backbone']
    if 'head' in metrics_data['parameters']:
        model_config['head'] = metrics_data['parameters']['head']
    
    # Optional parameters if present
    if 'ch' in metrics_data['parameters']:
        model_config['ch'] = metrics_data['parameters']['ch']
    
    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(model_config, f, sort_keys=False)
    
    print(f"Created custom model configuration at {output_path}")
    return output_path

def create_custom_args_yaml(args_data, output_path, temp_dir=None, extra_args=None):
    """Create a modified args.yaml file with updated settings."""
    # Create a copy of args_data to avoid modifying the original
    modified_args = args_data.copy()
    
    # Add any extra arguments
    if extra_args:
        for key, value in extra_args.items():
            modified_args[key] = value
    
    # If a temporary directory is provided, use it for the output
    if temp_dir:
        output_path = os.path.join(temp_dir, os.path.basename(output_path))
    
    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(modified_args, f, sort_keys=False)
    
    print(f"Created modified args.yaml at {output_path}")
    return output_path

def train_yolov8(metrics_data, args_data, use_custom_model=True, epochs=None, threshold=None, threshold_metric='map50', threshold_patience=3):
    """
    Train YOLOv8 model with specified configuration.
    
    Parameters:
    - metrics_data: Data from metrics.json
    - args_data: Data from args.yaml
    - use_custom_model: Whether to use custom model from metrics.json
    - epochs: Number of epochs (overrides args.yaml)
    - threshold: Accuracy threshold for early stopping
    - threshold_metric: Metric to use for threshold checking (default: map50)
    - threshold_patience: Number of consecutive epochs threshold must be exceeded
    """
    # Create a copy of args_data to modify
    args_data_copy = args_data.copy()
    
    # Override epochs if provided
    if epochs is not None:
        args_data_copy['epochs'] = epochs
        print(f"Using command line value for epochs: {epochs}")
    
    # Create a temporary directory for modified files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Determine if we should use a custom model or standard one
        if use_custom_model:
            model_path = create_custom_model_yaml(metrics_data, 
                                                 os.path.join(temp_dir, 'custom_model.yaml'))
        else:
            # Use the model path from args.yaml
            model_path = args_data_copy.get('model', 'yolov8s.yaml')
            
            # Check if pretrained weights should be used
            if args_data_copy.get('pretrained', False):
                # Convert .yaml to .pt for pretrained weights
                if model_path.endswith('.yaml'):
                    scale = model_path.split('yolov8')[1].split('.')[0]  # Extract scale (e.g., 's' from 'yolov8s.yaml')
                    model_path = f"yolov8{scale}.pt"
        
        print(f"Using model: {model_path}")
        
        # Create model instance
        model = YOLO(model_path)
        
        # Create a modified args.yaml file for training
        modified_args_path = create_custom_args_yaml(args_data_copy, 
                                                   os.path.join(temp_dir, 'modified_args.yaml'),
                                                   temp_dir)
        
        # Start training
        print("\nStarting YOLOv8 training...")
        
        # Initialize threshold tracking variables if threshold is provided
        if threshold is not None:
            consecutive_count = 0
            best_value = 0
            early_stopped = False
            print(f"Monitoring {threshold_metric} with threshold {threshold} for {threshold_patience} consecutive epochs")
        
        # Start training with the modified args.yaml
        results = model.train(cfg=modified_args_path)
        
        # If threshold is set, we need to manually check metrics after each epoch
        # This is a workaround since we can't directly use callbacks in the current API
        if threshold is not None and hasattr(model, 'trainer') and hasattr(model.trainer, 'metrics'):
            for epoch in range(model.trainer.epoch + 1):  # +1 because we want to include the last epoch
                # Check if the metric is tracked
                if threshold_metric in model.trainer.metrics:
                    current_value = model.trainer.metrics[threshold_metric]
                    print(f"Epoch {epoch}: {threshold_metric}={current_value:.4f}, threshold={threshold}")
                    
                    # Track best value
                    if current_value > best_value:
                        best_value = current_value
                    
                    # Check if threshold is exceeded
                    if current_value >= threshold:
                        consecutive_count += 1
                        print(f"Threshold exceeded: {consecutive_count}/{threshold_patience} consecutive epochs")
                        
                        if consecutive_count >= threshold_patience:
                            print(f"\n*** Early stopping: {threshold_metric} threshold {threshold} exceeded for {threshold_patience} consecutive epochs ***")
                            early_stopped = True
                            break
                    else:
                        # Reset counter if threshold not met
                        consecutive_count = 0
                
                # If we need to check multiple epochs, add a small delay
                if epoch < model.trainer.epoch:
                    time.sleep(0.1)
        
        # Validate model
        print("\nValidating model...")
        val_results = model.val()
        
        # Print final metrics - using correct API based on available attributes
        print("\nTraining Results Summary:")
        print(f"mAP@50: {val_results.box.map50:.4f}")
        print(f"mAP@50-95: {val_results.box.map:.4f}")
        
        # Use mean_results to get mp (mean precision) and mr (mean recall) if available
        try:
            mean_results = val_results.box.mean_results()
            mp, mr = mean_results[0], mean_results[1]
            print(f"Mean Precision: {mp:.4f}")
            print(f"Mean Recall: {mr:.4f}")
        except (AttributeError, IndexError, TypeError):
            # Fallback in case mean_results is not available or doesn't return expected format
            print("Note: Detailed precision and recall metrics unavailable in this version")
        
        # Report on early stopping if applicable
        if threshold is not None:
            if early_stopped:
                print(f"\nEarly stopping was triggered: {threshold_metric} threshold of {threshold} was exceeded for {threshold_patience} consecutive epochs")
            print(f"Best {threshold_metric} value achieved: {best_value:.4f}")
        
        # Export model if specified in args.yaml
        if args_data_copy.get('format'):
            export_format = args_data_copy.get('format')
            print(f"\nExporting model to {export_format} format...")
            export_args = {
                'format': export_format,
                'keras': args_data_copy.get('keras', False),
                'optimize': args_data_copy.get('optimize', False),
                'int8': args_data_copy.get('int8', False),
                'dynamic': args_data_copy.get('dynamic', False),
                'simplify': args_data_copy.get('simplify', True),
                'opset': args_data_copy.get('opset'),
                'workspace': args_data_copy.get('workspace'),
                'nms': args_data_copy.get('nms', False)
            }
            # Filter out None values
            export_args = {k: v for k, v in export_args.items() if v is not None}
            model.export(**export_args)
        
        return model, val_results

def compare_metrics(val_results, metrics_data):
    """Compare validation results with original metrics."""
    print("\nComparison with Original Metrics:")
    
    # Define metrics mapping with fallbacks
    metrics_pairs = [
        ('mAP@50', val_results.box.map50, metrics_data.get('map50')),
        ('mAP@50-95', val_results.box.map, metrics_data.get('map50_95'))
    ]
    
    # Try to get precision and recall from mean_results if available
    try:
        mean_results = val_results.box.mean_results()
        mp, mr = mean_results[0], mean_results[1]
        metrics_pairs.extend([
            ('Mean Precision', mp, metrics_data.get('precision')),
            ('Mean Recall', mr, metrics_data.get('recall'))
        ])
    except (AttributeError, IndexError, TypeError):
        print("Note: Detailed precision and recall metrics not available for comparison")
    
    for name, new_value, old_value in metrics_pairs:
        change = ""
        if old_value is not None and new_value is not None:
            diff = new_value - old_value
            change = f"({'+' if diff >= 0 else ''}{diff:.4f})"
        
        print(f"{name}: {new_value:.4f} (Original: {old_value if old_value is not None else 'N/A'}) {change}")

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLOv8 model with configuration from metrics.json and args.yaml')
    parser.add_argument('--metrics', type=str, default='metrics.json', help='Path to metrics.json file')
    parser.add_argument('--args', type=str, default='args.yaml', help='Path to args.yaml file')
    parser.add_argument('--custom_model', action='store_true', help='Use custom model from metrics.json')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides args.yaml)')
    parser.add_argument('--project', type=str, help='Project name (overrides args.yaml)')
    parser.add_argument('--name', type=str, help='Experiment name (overrides args.yaml)')
    parser.add_argument('--threshold', type=float, help='Accuracy threshold for early stopping')
    parser.add_argument('--threshold_metric', type=str, default='map50', help='Metric to use for threshold checking (default: map50)')
    parser.add_argument('--threshold_patience', type=int, default=3, help='Number of consecutive epochs threshold must be exceeded')
    
    cli_args = parser.parse_args()
    
    # Load configuration files
    metrics_data, args_data = load_files(cli_args.metrics, cli_args.args)
    
    # Override args.yaml parameters with command line values if provided
    if cli_args.project:
        args_data['project'] = cli_args.project
        print(f"Overriding project name with: {cli_args.project}")
    
    if cli_args.name:
        args_data['name'] = cli_args.name
        print(f"Overriding experiment name with: {cli_args.name}")
    
    # Train the model
    model, results = train_yolov8(
        metrics_data=metrics_data,
        args_data=args_data,
        use_custom_model=cli_args.custom_model,
        epochs=cli_args.epochs,
        threshold=cli_args.threshold,
        threshold_metric=cli_args.threshold_metric,
        threshold_patience=cli_args.threshold_patience
    )
    
    # Compare with original metrics
    compare_metrics(results, metrics_data)
    
    # Fixed path construction to avoid f-string syntax error
    project_default = "runs/detect/train"
    name_default = "exp"
    project = args_data.get("project", project_default)
    name = args_data.get("name", name_default)
    save_dir = args_data.get('save_dir', f'{project}/{name}')
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {save_dir}")

if __name__ == "__main__":
    main()