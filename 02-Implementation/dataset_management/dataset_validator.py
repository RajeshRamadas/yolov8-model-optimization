import os
import glob
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw
import random
import yaml
from collections import Counter, defaultdict
import json
import math
import shutil
import sys
from datetime import datetime
import traceback

def validate_dataset(dataset_path, yaml_path, output_dir, ci_mode=False):
    """
    Validate a YOLOv8 dataset and generate reports.
    
    Args:
        dataset_path (str): Path to the dataset directory
        yaml_path (str): Path to the dataset YAML file
        output_dir (str): Directory to save validation results
        ci_mode (bool): Whether to run in CI mode with minimal output
    
    Returns:
        dict: Statistics about the dataset
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize statistics dictionary
    statistics = {
        "total_images": 0,
        "valid_images": 0,
        "corrupt_images": 0,
        "empty_labels": 0,
        "classes": {},
        "image_sizes": {},
        "aspect_ratios": {},
        "label_issues": []
    }
    
    # Load class names from YAML
    classes = None
    if yaml_path:
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                if 'names' in yaml_data:
                    classes = yaml_data['names']
                    if not ci_mode:
                        print(f"Loaded {len(classes)} classes from {yaml_path}")
        except Exception as e:
            if not ci_mode:
                print(f"Error loading YAML file: {e}")
    
    # Process dataset
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
        
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        # Find all image files
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            image_paths.extend(glob.glob(str(images_dir / f'*{ext}')))
        
        if not image_paths:
            if not ci_mode:
                print(f"No images found in {images_dir}")
            continue
        
        if not ci_mode:
            print(f"Processing {len(image_paths)} images in {split} split...")
        
        # Process each image
        for img_path in tqdm(image_paths, desc=f"Processing {split} images", disable=ci_mode):
            img_path = Path(img_path)
            img_name = img_path.stem
            label_path = labels_dir / f"{img_name}.txt"
            
            statistics["total_images"] += 1
            
            # Check if image is valid
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                img_size = f"{img_width}x{img_height}"
                img_aspect = round(img_width / img_height, 2)
                
                # Update image size statistics
                if img_size not in statistics["image_sizes"]:
                    statistics["image_sizes"][img_size] = 0
                statistics["image_sizes"][img_size] += 1
                
                # Update aspect ratio statistics
                aspect_key = str(img_aspect)
                if aspect_key not in statistics["aspect_ratios"]:
                    statistics["aspect_ratios"][aspect_key] = 0
                statistics["aspect_ratios"][aspect_key] += 1
                
                statistics["valid_images"] += 1
            except Exception as e:
                statistics["corrupt_images"] += 1
                statistics["label_issues"].append({
                    "image": str(img_path),
                    "issue": f"Corrupt image: {str(e)}"
                })
                continue
            
            # Check if label file exists
            if not label_path.exists():
                statistics["empty_labels"] += 1
                statistics["label_issues"].append({
                    "image": str(img_path),
                    "issue": "Missing label file"
                })
                continue
            
            # Process label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.read().strip().splitlines()
                
                if not lines:
                    statistics["empty_labels"] += 1
                    statistics["label_issues"].append({
                        "image": str(img_path),
                        "issue": "Empty label file"
                    })
                    continue
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        statistics["label_issues"].append({
                            "image": str(img_path),
                            "issue": f"Invalid label format: {line}"
                        })
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Check if class_id is valid
                    class_name = str(class_id)
                    if classes and 0 <= class_id < len(classes):
                        class_name = classes[class_id]
                    
                    # Update class statistics
                    if class_name not in statistics["classes"]:
                        statistics["classes"][class_name] = 0
                    statistics["classes"][class_name] += 1
                    
            except Exception as e:
                statistics["label_issues"].append({
                    "image": str(img_path),
                    "issue": f"Error processing label: {str(e)}"
                })
    
    # Generate summary report
    create_summary_report(statistics, output_dir, dataset_path)
    
    return statistics

def create_summary_report(statistics, output_dir, dataset_path):
    """
    Create a summary report of dataset statistics.
    
    Args:
        statistics (dict): Statistics collected during validation
        output_dir (Path): Directory to save the report
        dataset_path (Path): Path to the dataset
    """
    # Calculate total objects
    total_objects = sum(statistics["classes"].values())
    
    # Create summary text
    summary = f"YOLOv8 Dataset Validation Summary\n"
    summary += f"===============================\n\n"
    summary += f"Dataset: {dataset_path}\n"
    summary += f"Total Images: {statistics['total_images']}\n"
    summary += f"Valid Images: {statistics['valid_images']}\n"
    summary += f"Corrupt Images: {statistics['corrupt_images']}\n"
    summary += f"Empty Labels: {statistics['empty_labels']}\n"
    summary += f"Total Objects: {total_objects}\n"
    summary += f"Number of Classes: {len(statistics['classes'])}\n\n"
    
    summary += f"Class Distribution:\n"
    for class_id, count in statistics["classes"].items():
        class_name = class_id  # In case class names are not loaded from YAML
        summary += f"- Class {class_id} ({class_name}): {count} instances\n"
    
    # Write summary to file
    with open(output_dir / "summary_report.txt", 'w') as f:
        f.write(summary)
    
    # Create JSON report
    json_report = {
        "dataset_path": str(dataset_path),
        "total_images": statistics["total_images"],
        "valid_images": statistics["valid_images"],
        "corrupt_images": statistics["corrupt_images"],
        "empty_labels": statistics["empty_labels"],
        "total_objects": total_objects,
        "num_classes": len(statistics["classes"]),
        "classes": statistics["classes"],
        "issues": len(statistics["label_issues"])
    }
    
    with open(output_dir / "summary_report.json", 'w') as f:
        json.dump(json_report, f, indent=4)

def main():
    """Main function to run the dataset validator."""
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Validator")
    
    parser.add_argument("--dataset_path", "-d", type=str, required=True, 
                      help="Path to the YOLOv8 dataset (should contain train/val/test folders)")
    parser.add_argument("--yaml_path", "-y", type=str, default=None,
                      help="Path to the dataset YAML file (optional)")
    parser.add_argument("--output_dir", "-o", type=str, default="validation_results",
                      help="Directory to save validation results")
    
    # Add CI-specific arguments
    parser.add_argument("--ci_mode", action="store_true",
                      help="Run in CI mode with minimal output and proper exit codes")
    parser.add_argument("--fail_on_issues", action="store_true", 
                      help="Exit with error code if issues are found")
    parser.add_argument("--issue_threshold", type=int, default=10,
                      help="Number of issues to tolerate before failing (with --fail_on_issues)")
    parser.add_argument("--json_report", action="store_true",
                      help="Generate machine-readable JSON report")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.ci_mode:
        print(f"Starting YOLOv8 dataset validation for: {args.dataset_path}")
        print(f"Results will be saved to: {args.output_dir}")
    
    try:
        # Run validation
        statistics = validate_dataset(
            args.dataset_path, 
            args.yaml_path, 
            args.output_dir,
            args.ci_mode
        )
        
        # Generate CI report if requested
        if args.json_report:
            ci_report = {
                "timestamp": datetime.now().isoformat(),
                "dataset_path": args.dataset_path,
                "total_images": statistics["total_images"],
                "valid_images": statistics["valid_images"],
                "total_objects": sum(statistics["classes"].values()),
                "num_classes": len(statistics["classes"]),
                "issues_count": len(statistics["label_issues"]),
                "status": "success" if len(statistics["label_issues"]) <= args.issue_threshold else "warning"
            }
            
            with open(os.path.join(args.output_dir, "ci_report.json"), 'w') as f:
                json.dump(ci_report, f, indent=2)
        
        # Print summary to console
        if not args.ci_mode:
            print("\nValidation Summary:")
            print(f"- Total Images: {statistics['total_images']}")
            print(f"- Valid Images: {statistics['valid_images']}")
            print(f"- Corrupt Images: {statistics['corrupt_images']}")
            print(f"- Empty Labels: {statistics['empty_labels']}")
            print(f"- Total Objects: {sum(statistics['classes'].values())}")
            print(f"- Number of Classes: {len(statistics['classes'])}")
            print(f"- Issues Found: {len(statistics['label_issues'])}")
        
        # Exit with error code if needed
        if args.fail_on_issues and len(statistics["label_issues"]) > args.issue_threshold:
            sys.stderr.write(f"ERROR: Found {len(statistics['label_issues'])} issues, threshold is {args.issue_threshold}\n")
            sys.exit(1)
        
        sys.exit(0)
        
    except Exception as e:
        if not args.ci_mode:
            print(f"Error during validation: {e}")
            traceback.print_exc()
        else:
            sys.stderr.write(f"VALIDATION_ERROR: {str(e)}\n")
        sys.exit(2)

if __name__ == "__main__":
    main()