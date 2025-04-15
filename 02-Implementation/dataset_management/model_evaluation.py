#!/usr/bin/env python
# Enhanced script for evaluating multiple YOLOv8 models and generating HTML reports
# python3 model_evaluation.py --models-dir /home/adminuser/NNI-YOLOV8/yolo_opt_project/yolo_optimizer/ --data /home/adminuser/NNI-YOLOV8/yolo_opt_project/yolo_optimizer/vehicles.v2-release.yolov8/data.yaml --task detect --device 0


import os
import json
import csv
import yaml
import time
import glob
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import torch
from datetime import datetime

def evaluate_model(model_path, data_path, task='detect', batch_size=16, device='0'):
    """
    Evaluate a YOLOv8 model and return performance metrics
    
    Args:
        model_path: Path to model weights (.pt file)
        data_path: Path to data YAML file
        task: Model task (detect, segment, classify)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on (e.g., '0' for GPU 0, 'cpu' for CPU)
    
    Returns:
        dict: Performance metrics
    """
    print(f"Evaluating model: {model_path}")
    
    # Check if CUDA is available when GPU is requested
    if device != 'cpu' and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = 'cpu'
    
    # Load model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None
    
    # Run validation
    try:
        start_time = time.time()
        results = model.val(data=data_path, batch=batch_size, device=device)
        eval_time = time.time() - start_time
        
        # Extract metrics
        metrics = {
            "model_name": Path(model_path).stem,
            "model_path": str(model_path),  # Store full path for later reference
            "task": task,
            "dataset": Path(data_path).stem,
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "eval_time_seconds": eval_time
        }
        
        # Calculate F1 score from precision and recall if f1 is not directly available
        try:
            metrics["f1"] = float(results.box.f1)
        except (AttributeError, TypeError):
            # If f1 is not available or not a scalar, calculate it from precision and recall
            if metrics["precision"] > 0 or metrics["recall"] > 0:
                metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0
            else:
                metrics["f1"] = 0.0
        
        # Additional metrics for segmentation models
        if task == 'segment' and hasattr(results, 'seg'):
            metrics.update({
                "seg_map50": float(results.seg.map50),
                "seg_map50_95": float(results.seg.map),
            })
        
        return metrics
    except Exception as e:
        print(f"Error evaluating model {model_path}: {str(e)}")
        return None

def benchmark_model(model_path, img_size=640, batch_size=1, num_iters=100, device='0'):
    """
    Benchmark a YOLOv8 model for inference speed
    
    Args:
        model_path: Path to model weights (.pt file)
        img_size: Input image size
        batch_size: Batch size for inference
        num_iters: Number of iterations for benchmarking
        device: Device to run benchmark on
        
    Returns:
        dict: Benchmark results
    """
    print(f"Benchmarking model: {model_path}")
    
    # Fix device string for newer PyTorch versions
    # Convert '0' to 'cuda:0' or similar format that PyTorch expects
    if device != 'cpu':
        if device.isdigit():
            device = f'cuda:{device}'
    
    # Check if CUDA is available when GPU is requested
    if device != 'cpu' and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = 'cpu'
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Create a random input tensor
        dummy_input = torch.rand(batch_size, 3, img_size, img_size).to(device if device != 'cpu' else 'cpu')
        
        # Warmup
        for _ in range(10):
            _ = model.predict(dummy_input, verbose=False)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iters):
            _ = model.predict(dummy_input, verbose=False)
        total_time = time.time() - start_time
        
        # Calculate metrics
        inference_time_ms = (total_time / num_iters) * 1000
        fps = 1000 / inference_time_ms
        
        # Get model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        benchmark_results = {
            "inference_time_ms": inference_time_ms,
            "fps": fps,
            "model_size_mb": model_size_mb,
        }
        
        return benchmark_results
    except Exception as e:
        print(f"Error benchmarking model {model_path}: {str(e)}")
        # Return default benchmark results so the report can still be generated
        return {
            "inference_time_ms": 0,
            "fps": 0,
            "model_size_mb": os.path.getsize(model_path) / (1024 * 1024)
        }

def find_models(root_dir, recursive=True, best_only=True):
    """
    Find model files in the given directory, optionally filtering for best.pt only
    
    Args:
        root_dir: Root directory to search for models
        recursive: Whether to search recursively in subdirectories
        best_only: If True, only include files named best.pt and exclude last.pt
    
    Returns:
        list: List of paths to model files
    """
    pattern = os.path.join(root_dir, '**/*.pt' if recursive else '*.pt')
    all_model_paths = glob.glob(pattern, recursive=recursive)
    
    if best_only:
        # Filter to include "best.pt" files and exclude "last.pt" files
        model_paths = []
        for path in all_model_paths:
            filename = os.path.basename(path)
            if filename == "best.pt" or (filename != "last.pt" and "best" in filename):
                model_paths.append(path)
        
        # Log what was found and what was filtered
        excluded_count = len(all_model_paths) - len(model_paths)
        print(f"Found {len(all_model_paths)} total model files in {root_dir}")
        print(f"After filtering: {len(model_paths)} best models kept, {excluded_count} models excluded")
    else:
        model_paths = all_model_paths
        print(f"Found {len(model_paths)} model files in {root_dir}")
    
    return model_paths

def detect_model_variant(model_path):
    """
    Attempt to detect the model variant from the filename or path
    
    Args:
        model_path: Path to model weights (.pt file)
    
    Returns:
        str: Detected variant or 'custom'
    """
    filename = Path(model_path).stem.lower()
    
    # Common YOLOv8 variants
    variants = ['n', 's', 'm', 'l', 'x']
    
    for variant in variants:
        # Check for patterns like 'yolov8n', 'yolov8n-seg', 'yolov8s-cls'
        if f'yolov8{variant}' in filename:
            return variant
    
    # Additional check for specific patterns
    if 'nano' in filename:
        return 'n'
    elif 'small' in filename:
        return 's'
    elif 'medium' in filename:
        return 'm'
    elif 'large' in filename:
        return 'l'
    elif 'xlarge' in filename:
        return 'x'
    
    # If no variant detected, return 'custom'
    return 'custom'

def generate_html_report(all_metrics, output_path, plots_dir):
    """
    Generate an HTML report for all evaluated models
    
    Args:
        all_metrics: List of dictionaries containing metrics for each model
        output_path: Path to save the HTML report
        plots_dir: Directory containing plot images
    """
    print(f"Generating HTML report at: {output_path}")
    
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get relative path from output HTML to plots directory
    plots_rel_path = os.path.relpath(plots_dir, os.path.dirname(output_path))
    
    # Generate plots
    if all_metrics:
        # Accuracy vs Speed plot
        plt.figure(figsize=(10, 6))
        
        for metrics in all_metrics:
            if 'inference_time_ms' in metrics and 'map50' in metrics:
                # Use model size for point size if available
                size = 100
                if 'model_size_mb' in metrics:
                    size = metrics['model_size_mb'] * 5  # Scale factor for better visibility
                
                plt.scatter(metrics['inference_time_ms'], metrics['map50'], 
                           s=size, alpha=0.7,
                           label=metrics['model_name'])
        
        plt.title('YOLOv8 Models: Accuracy vs Speed')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('mAP@0.5')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend if there are labeled points
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend()
        
        # Save the plot
        accuracy_speed_plot = os.path.join(plots_dir, 'accuracy_vs_speed.png')
        plt.savefig(accuracy_speed_plot)
        
        # Size vs Accuracy plot
        plt.figure(figsize=(10, 6))
        
        for metrics in all_metrics:
            if 'model_size_mb' in metrics and 'map50' in metrics:
                plt.scatter(metrics['model_size_mb'], metrics['map50'], 
                           s=100, alpha=0.7,
                           label=metrics['model_name'])
        
        plt.title('YOLOv8 Models: Size vs Accuracy')
        plt.xlabel('Model Size (MB)')
        plt.ylabel('mAP@0.5')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend if there are labeled points
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 0:
            plt.legend()
        
        # Save the plot
        size_accuracy_plot = os.path.join(plots_dir, 'size_vs_accuracy.png')
        plt.savefig(size_accuracy_plot)
    
    # Create the HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Models Performance Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e9ecef;
        }}
        .plots {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .plot {{
            max-width: 100%;
            height: auto;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .highlight {{
            background-color: #d4edda !important;
        }}
        .path-cell {{
            font-family: monospace;
            word-break: break-all;
        }}
        .model-section {{
            margin-bottom: 40px;
        }}
        .timestamp {{
            color: #6c757d;
            font-style: italic;
            margin-bottom: 20px;
        }}
        .summary {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .copy-btn {{
            background-color: #007bff;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }}
        .copy-btn:hover {{
            background-color: #0069d9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOv8 Models Performance Report</h1>
        <div class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        
        <div class="summary">
            <h2>Performance Overview</h2>
"""
    
    # Add summary information
    if all_metrics:
        # Find the best model by mAP50
        best_map50_model = max(all_metrics, key=lambda x: x.get('map50', 0))
        
        # Find the fastest model by FPS (check if fps exists before comparing)
        best_fps_model = max(all_metrics, key=lambda x: x.get('fps', 0))
        
        # Find the smallest model by size
        smallest_model = min(all_metrics, key=lambda x: x.get('model_size_mb', float('inf')))
        
        html_content += f"""
            <p><strong>Total models evaluated:</strong> {len(all_metrics)}</p>
            <p><strong>Best accuracy model (mAP@0.5):</strong> {best_map50_model['model_name']} ({best_map50_model['map50']:.3f})</p>
            <p><strong>Path:</strong> <span class="path-cell">{best_map50_model['model_path']}</span>
            <button class="copy-btn" onclick="copyToClipboard('{best_map50_model['model_path']}')">Copy Path</button></p>
            
            <p><strong>Fastest model (FPS):</strong> {best_fps_model['model_name']} ({best_fps_model.get('fps', 0):.2f} FPS)</p>
            <p><strong>Path:</strong> <span class="path-cell">{best_fps_model['model_path']}</span>
            <button class="copy-btn" onclick="copyToClipboard('{best_fps_model['model_path']}')">Copy Path</button></p>
            
            <p><strong>Smallest model:</strong> {smallest_model['model_name']} ({smallest_model.get('model_size_mb', 0):.2f} MB)</p>
            <p><strong>Path:</strong> <span class="path-cell">{smallest_model['model_path']}</span>
            <button class="copy-btn" onclick="copyToClipboard('{smallest_model['model_path']}')">Copy Path</button></p>
"""
    else:
        html_content += """
            <p>No models were successfully evaluated.</p>
"""
    
    html_content += """
        </div>
        
        <div class="plots">
"""
    
    # Add plot images
    if all_metrics:
        html_content += f"""
            <div>
                <h3>Accuracy vs Speed</h3>
                <img class="plot" src="{plots_rel_path}/accuracy_vs_speed.png" alt="Accuracy vs Speed Plot">
            </div>
            <div>
                <h3>Size vs Accuracy</h3>
                <img class="plot" src="{plots_rel_path}/size_vs_accuracy.png" alt="Size vs Accuracy Plot">
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="model-section">
            <h2>Detailed Model Comparison</h2>
"""
    
    # Add comparison table
    if all_metrics:
        # Sort models by mAP50 (descending)
        sorted_metrics = sorted(all_metrics, key=lambda x: x.get('map50', 0), reverse=True)
        
        html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model Name</th>
                        <th>Variant</th>
                        <th>mAP@0.5</th>
                        <th>mAP@0.5:0.95</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>FPS</th>
                        <th>Size (MB)</th>
                        <th>Path</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for i, metrics in enumerate(sorted_metrics):
            # Check if this is the best model (rank 1)
            highlight_class = " class='highlight'" if i == 0 else ""
            
            # Make sure we safely access values that might not exist
            fps = metrics.get('fps', 0)
            
            html_content += f"""
                    <tr{highlight_class}>
                        <td>{i+1}</td>
                        <td>{metrics['model_name']}</td>
                        <td>{metrics.get('variant', 'N/A')}</td>
                        <td>{metrics.get('map50', 0):.3f}</td>
                        <td>{metrics.get('map50_95', 0):.3f}</td>
                        <td>{metrics.get('precision', 0):.3f}</td>
                        <td>{metrics.get('recall', 0):.3f}</td>
                        <td>{metrics.get('f1', 0):.3f}</td>
                        <td>{fps:.2f}</td>
                        <td>{metrics.get('model_size_mb', 0):.2f}</td>
                        <td class="path-cell">{metrics['model_path']}
                            <button class="copy-btn" onclick="copyToClipboard('{metrics['model_path']}')">Copy</button>
                        </td>
                    </tr>
"""
        
        html_content += """
                </tbody>
            </table>
"""
    else:
        html_content += """
            <p>No models were successfully evaluated.</p>
"""
    
    html_content += """
        </div>
    </div>
    
    <script>
    function copyToClipboard(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        alert('Path copied to clipboard');
    }
    </script>
</body>
</html>
"""
    
    # Write the HTML content to a file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated at: {output_path}")

def update_benchmark_csv(all_metrics, output_path):
    """
    Update the benchmark CSV file with metrics for all models
    
    Args:
        all_metrics: List of dictionaries containing metrics for each model
        output_path: Path to the CSV file
    """
    print(f"Updating benchmark CSV at: {output_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define CSV fields
    fields = [
        'model_name', 'variant', 'task', 'dataset', 
        'map50', 'map50_95', 'precision', 'recall', 'f1',
        'inference_time_ms', 'fps', 'model_size_mb', 'model_path'
    ]
    
    # Write to CSV
    with open(output_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for metrics in all_metrics:
            # Filter dictionary to only include fields in the CSV
            row = {k: metrics.get(k, 'N/A') for k in fields}
            writer.writerow(row)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model evaluation and documentation script')
    parser.add_argument('--models-dir', type=str, help='Directory containing model weights (.pt files)')
    parser.add_argument('--model', type=str, help='Path to a specific model weight (.pt file)')
    parser.add_argument('--data', type=str, required=True, help='Path to data YAML file')
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'segment', 'classify'], help='Model task')
    parser.add_argument('--device', type=str, default='0', help='Device to run evaluation on (e.g., "0" for GPU 0, "cpu" for CPU)')
    parser.add_argument('--output-dir', type=str, default='performance', help='Directory to save output files')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--non-recursive', action='store_true', help='Do not search subdirectories for models')
    parser.add_argument('--skip-benchmark', action='store_true', help='Skip speed benchmarking')
    parser.add_argument('--all-models', action='store_true', help='Include all models (don\'t skip last.pt)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    reports_dir = os.path.join(args.output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    plots_dir = os.path.join(reports_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    evaluations_dir = os.path.join(args.output_dir, 'evaluations', args.task)
    os.makedirs(evaluations_dir, exist_ok=True)
    
    # Get model paths - updated to respect best_only parameter
    model_paths = []
    if args.models_dir:
        # Use the new best_only parameter based on all_models flag
        model_paths = find_models(args.models_dir, 
                                  recursive=not args.non_recursive,
                                  best_only=not args.all_models)
    elif args.model:
        # If a specific model is provided, always use it regardless of filename
        model_paths = [args.model]
    else:
        print("Error: Either --models-dir or --model must be specified")
        return
    
    if not model_paths:
        print("No model files found")
        return
        
    # Print the models that will be evaluated
    print(f"Will evaluate {len(model_paths)} models:")
    for i, model_path in enumerate(model_paths):
        print(f"  {i+1}. {model_path}")
    
    # Evaluate all models
    all_metrics = []
    
    for model_path in model_paths:
        try:
            # Evaluate model
            eval_metrics = evaluate_model(
                model_path=model_path,
                data_path=args.data,
                task=args.task,
                batch_size=args.batch_size,
                device=args.device
            )
            
            if eval_metrics is None:
                continue
            
            # Detect model variant
            eval_metrics['variant'] = detect_model_variant(model_path)
            
            # Run benchmark if not skipped
            if not args.skip_benchmark:
                benchmark_results = benchmark_model(
                    model_path=model_path,
                    device=args.device
                )
                
                if benchmark_results:
                    # Merge metrics
                    eval_metrics.update(benchmark_results)
            
            # Save individual evaluation results
            model_name = Path(model_path).stem
            eval_output_path = os.path.join(evaluations_dir, f"{model_name}_eval_results.json")
            with open(eval_output_path, 'w') as f:
                json.dump(eval_metrics, f, indent=2)
            
            # Add to all metrics list
            all_metrics.append(eval_metrics)
            
        except Exception as e:
            print(f"Error processing model {model_path}: {str(e)}")
    
    if not all_metrics:
        print("No models were successfully evaluated")
        return
    
    # Sort models by mAP50 for the report
    all_metrics.sort(key=lambda x: x.get('map50', 0), reverse=True)
    
    # Generate HTML report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_report_path = os.path.join(reports_dir, f"model_performance_report_{timestamp}.html")
    generate_html_report(all_metrics, html_report_path, plots_dir)
    
    # Create a symlink to the latest report
    latest_report_path = os.path.join(reports_dir, "latest_report.html")
    try:
        if os.path.exists(latest_report_path):
            os.remove(latest_report_path)
        os.symlink(html_report_path, latest_report_path)
        print(f"Created symlink to latest report at: {latest_report_path}")
    except Exception as e:
        print(f"Warning: Could not create symlink to latest report: {str(e)}")
    
    # Save all metrics to CSV
    csv_path = os.path.join(args.output_dir, f"best_models_benchmark_{timestamp}.csv")
    update_benchmark_csv(all_metrics, csv_path)
    
    # Save best model results separately
    if len(all_metrics) > 0:
        best_model = all_metrics[0]  # Already sorted by mAP50
        best_model_path = os.path.join(evaluations_dir, "best_eval_results.json")
        with open(best_model_path, 'w') as f:
            json.dump(best_model, f, indent=2)
        print(f"Best model results saved to: {best_model_path}")
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print(f"HTML report: {html_report_path}")
    print(f"CSV data: {csv_path}")

if __name__ == "__main__":
    main()