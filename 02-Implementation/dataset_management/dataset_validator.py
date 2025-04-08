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

class YOLOv8DatasetValidator:
    def __init__(self, dataset_path, yaml_path=None, output_dir="validation_results"):
        """
        Initialize the YOLOv8 dataset validator.
        
        Args:
            dataset_path (str): Path to the dataset directory
            yaml_path (str, optional): Path to the dataset YAML file
            output_dir (str, optional): Directory to save validation results
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.yaml_path = yaml_path
        self.classes = None
        self.statistics = {
            "total_images": 0,
            "valid_images": 0,
            "corrupt_images": 0,
            "empty_labels": 0,
            "classes": {},
            "image_sizes": {},
            "aspect_ratios": {},
            "bbox_sizes": {},
            "bbox_positions": {},
            "label_issues": [],
            "class_distribution": {},
            "images_per_class": defaultdict(set),
            "bbox_area_distribution": [],
            "bbox_aspect_ratio_distribution": []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load class names from YAML if provided
        if yaml_path:
            self._load_yaml()
    
    def _load_yaml(self):
        """Load class names from YAML file."""
        try:
            with open(self.yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    self.classes = data['names']
                    
                    # Handle different formats of the 'names' field
                    if isinstance(self.classes, dict):
                        # Convert dict format (e.g. {0: 'car', 1: 'truck'}) to list
                        max_idx = max(self.classes.keys())
                        class_list = ["" for _ in range(max_idx + 1)]
                        for idx, name in self.classes.items():
                            class_list[idx] = name
                        self.classes = class_list
                    elif isinstance(self.classes, list):
                        # List format is already what we want - ['car', 'truck', ...]
                        pass
                    else:
                        print(f"Warning: Unexpected format for class names in YAML: {type(self.classes)}")
                        self.classes = None
                        return
                    
                    # Display loaded classes
                    print(f"Loaded {len(self.classes)} classes from YAML:")
                    for i, cls in enumerate(self.classes):
                        print(f"  {i}: {cls}")
                    
                    # Verify class count matches 'nc' field if present
                    if 'nc' in data and data['nc'] != len(self.classes):
                        print(f"Warning: YAML 'nc' value ({data['nc']}) doesn't match the number of classes ({len(self.classes)})")
                else:
                    print("No class names found in YAML file")
                    self.classes = None
        except Exception as e:
            print(f"Error loading YAML file: {e}")
            self.classes = None
        
    def validate(self):
        """Validate the dataset and generate statistics."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        train_dir = self.dataset_path / 'train'
        val_dir = self.dataset_path / 'val'
        test_dir = self.dataset_path / 'test'
        
        for split_dir in [d for d in [train_dir, val_dir, test_dir] if d.exists()]:
            split_name = split_dir.name
            print(f"\nValidating {split_name} split...")
            
            # Find all image files
            all_images = []
            for ext in image_extensions:
                all_images.extend(glob.glob(str(split_dir / 'images' / f'*{ext}')))
            
            if not all_images:
                print(f"No images found in {split_dir / 'images'}")
                continue
            
            # Process each image
            for img_path in tqdm(all_images, desc=f"Processing {split_name} images"):
                img_path = Path(img_path)
                img_name = img_path.stem
                label_path = split_dir / 'labels' / f"{img_name}.txt"
                
                self.statistics["total_images"] += 1
                
                # Check if image is valid
                try:
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    img_size = f"{img_width}x{img_height}"
                    img_aspect = round(img_width / img_height, 2)
                    
                    # Update image size statistics
                    if img_size not in self.statistics["image_sizes"]:
                        self.statistics["image_sizes"][img_size] = 0
                    self.statistics["image_sizes"][img_size] += 1
                    
                    # Update aspect ratio statistics
                    aspect_key = str(img_aspect)
                    if aspect_key not in self.statistics["aspect_ratios"]:
                        self.statistics["aspect_ratios"][aspect_key] = 0
                    self.statistics["aspect_ratios"][aspect_key] += 1
                    
                    self.statistics["valid_images"] += 1
                except Exception as e:
                    self.statistics["corrupt_images"] += 1
                    self.statistics["label_issues"].append({
                        "image": str(img_path),
                        "issue": f"Corrupt image: {str(e)}"
                    })
                    continue
                
                # Check if label file exists
                if not label_path.exists():
                    self.statistics["empty_labels"] += 1
                    self.statistics["label_issues"].append({
                        "image": str(img_path),
                        "issue": "Missing label file"
                    })
                    continue
                
                # Process label file
                try:
                    with open(label_path, 'r') as f:
                        lines = f.read().strip().splitlines()
                    
                    if not lines:
                        self.statistics["empty_labels"] += 1
                        self.statistics["label_issues"].append({
                            "image": str(img_path),
                            "issue": "Empty label file"
                        })
                        continue
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            self.statistics["label_issues"].append({
                                "image": str(img_path),
                                "issue": f"Invalid label format: {line}"
                            })
                            continue
                        
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Check if class_id is valid
                        if self.classes and class_id >= len(self.classes):
                            self.statistics["label_issues"].append({
                                "image": str(img_path),
                                "issue": f"Invalid class ID: {class_id}"
                            })
                        
                        # Update class statistics
                        class_name = self.classes[class_id] if self.classes and class_id < len(self.classes) else f"class_{class_id}"
                        if class_name not in self.statistics["classes"]:
                            self.statistics["classes"][class_name] = 0
                        self.statistics["classes"][class_name] += 1
                        
                        # Track images per class
                        self.statistics["images_per_class"][class_name].add(str(img_path))
                        
                        # Check if bounding box coordinates are valid
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            self.statistics["label_issues"].append({
                                "image": str(img_path),
                                "issue": f"Invalid bounding box coordinates: {x_center}, {y_center}, {width}, {height}"
                            })
                        
                        # Calculate bounding box size in pixels
                        bbox_width_px = width * img_width
                        bbox_height_px = height * img_height
                        bbox_area = bbox_width_px * bbox_height_px
                        bbox_aspect = (bbox_width_px / bbox_height_px) if bbox_height_px > 0 else 0
                        
                        # Record bbox area and aspect ratio for distribution analysis
                        self.statistics["bbox_area_distribution"].append({
                            "class": class_name,
                            "area": bbox_area,
                            "area_percentage": (bbox_area / (img_width * img_height)) * 100
                        })
                        
                        self.statistics["bbox_aspect_ratio_distribution"].append({
                            "class": class_name,
                            "aspect_ratio": bbox_aspect
                        })
                        
                        # Update bbox size statistics
                        bbox_size_category = self._categorize_bbox_size(bbox_area, img_width * img_height)
                        if bbox_size_category not in self.statistics["bbox_sizes"]:
                            self.statistics["bbox_sizes"][bbox_size_category] = 0
                        self.statistics["bbox_sizes"][bbox_size_category] += 1
                        
                        # Update bbox position statistics
                        position_category = self._categorize_bbox_position(x_center, y_center)
                        if position_category not in self.statistics["bbox_positions"]:
                            self.statistics["bbox_positions"][position_category] = 0
                        self.statistics["bbox_positions"][position_category] += 1
                        
                except Exception as e:
                    self.statistics["label_issues"].append({
                        "image": str(img_path),
                        "issue": f"Error processing label: {str(e)}"
                    })
        
        # Calculate class distribution
        total_objects = sum(self.statistics["classes"].values())
        for class_name, count in self.statistics["classes"].items():
            self.statistics["class_distribution"][class_name] = {
                "count": count,
                "percentage": (count / total_objects) * 100 if total_objects > 0 else 0,
                "images": len(self.statistics["images_per_class"][class_name])
            }
        
        # Generate visualization and reports
        self._generate_reports()
        
        return self.statistics
    
    def _categorize_bbox_size(self, area, image_area):
        """Categorize bounding box size based on area percentage."""
        percentage = (area / image_area) * 100
        if percentage < 1:
            return "tiny (<1%)"
        elif percentage < 5:
            return "small (1-5%)"
        elif percentage < 20:
            return "medium (5-20%)"
        elif percentage < 50:
            return "large (20-50%)"
        else:
            return "huge (>50%)"
    
    def _categorize_bbox_position(self, x_center, y_center):
        """Categorize bounding box position in the image."""
        x_pos = "left" if x_center < 0.33 else "center" if x_center < 0.66 else "right"
        y_pos = "top" if y_center < 0.33 else "middle" if y_center < 0.66 else "bottom"
        return f"{y_pos}-{x_pos}"
    
    def _generate_reports(self):
        """Generate reports and visualizations from the collected statistics."""
        try:
            # Create a detailed summary report
            self._generate_summary_report()
            
            # Generate class distribution visualization
            self._generate_class_distribution_chart()
            
            # Generate bounding box size distribution
            self._generate_bbox_size_chart()
            
            # Generate bounding box position heatmap
            self._generate_bbox_position_heatmap()
            
            # Generate sample visualizations
            self._generate_sample_visualizations()
            
            # Generate bounding box area distribution
            self._generate_bbox_area_distribution()
            
            # Generate bounding box aspect ratio distribution
            self._generate_bbox_aspect_ratio_distribution()
            
            # Generate image size distribution
            self._generate_image_size_distribution()
            
            # Generate issues report
            self._generate_issues_report()
        except Exception as e:
            print(f"Error generating reports: {str(e)}")
    
    def _generate_summary_report(self):
        """Generate a summary report of the dataset statistics."""
        summary = {
            "Dataset Path": str(self.dataset_path),
            "Total Images": self.statistics["total_images"],
            "Valid Images": self.statistics["valid_images"],
            "Corrupt Images": self.statistics["corrupt_images"],
            "Empty Labels": self.statistics["empty_labels"],
            "Total Objects": sum(self.statistics["classes"].values()),
            "Number of Classes": len(self.statistics["classes"]),
            "Class Distribution": self.statistics["class_distribution"],
            "Most Common Image Size": max(self.statistics["image_sizes"].items(), key=lambda x: x[1])[0] if self.statistics["image_sizes"] else "N/A",
            "Most Common Aspect Ratio": max(self.statistics["aspect_ratios"].items(), key=lambda x: x[1])[0] if self.statistics["aspect_ratios"] else "N/A",
            "Number of Issues": len(self.statistics["label_issues"])
        }
        
        # Save summary as JSON
        with open(self.output_dir / "summary_report.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Also create a readable text report
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write("YOLOv8 Dataset Validation Summary\n")
            f.write("===============================\n\n")
            
            f.write(f"Dataset Path: {summary['Dataset Path']}\n")
            f.write(f"Total Images: {summary['Total Images']}\n")
            f.write(f"Valid Images: {summary['Valid Images']} ({(summary['Valid Images']/summary['Total Images']*100 if summary['Total Images'] > 0 else 0):.2f}%)\n")
            f.write(f"Corrupt Images: {summary['Corrupt Images']} ({(summary['Corrupt Images']/summary['Total Images']*100 if summary['Total Images'] > 0 else 0):.2f}%)\n")
            f.write(f"Empty Labels: {summary['Empty Labels']} ({(summary['Empty Labels']/summary['Total Images']*100 if summary['Total Images'] > 0 else 0):.2f}%)\n")
            f.write(f"Total Objects: {summary['Total Objects']}\n")
            f.write(f"Number of Classes: {summary['Number of Classes']}\n")
            f.write(f"Most Common Image Size: {summary['Most Common Image Size']}\n")
            f.write(f"Most Common Aspect Ratio: {summary['Most Common Aspect Ratio']}\n")
            f.write(f"Number of Issues: {summary['Number of Issues']}\n\n")
            
            f.write("Class Distribution:\n")
            for class_name, info in summary["Class Distribution"].items():
                f.write(f"  - {class_name}: {info['count']} objects ({info['percentage']:.2f}%) in {info['images']} images\n")
            
            f.write("\nBounding Box Size Distribution:\n")
            for size_category, count in self.statistics["bbox_sizes"].items():
                f.write(f"  - {size_category}: {count} boxes ({count/summary['Total Objects']*100 if summary['Total Objects'] > 0 else 0:.2f}%)\n")
    
    def _generate_class_distribution_chart(self):
        """Generate a bar chart showing class distribution."""
        if not self.statistics["classes"]:
            print("Skipping class distribution chart: No classes found")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            
            # Sort classes by count
            sorted_classes = sorted(self.statistics["class_distribution"].items(), 
                                   key=lambda x: x[1]["count"], reverse=True)
            class_names = [c[0] for c in sorted_classes]
            class_counts = [c[1]["count"] for c in sorted_classes]
            image_counts = [c[1]["images"] for c in sorted_classes]
            
            # Create a grouped bar chart
            x = np.arange(len(class_names))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.bar(x - width/2, class_counts, width, label='Object Count')
            ax.bar(x + width/2, image_counts, width, label='Image Count')
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
            ax.set_xticks(x)
            
            # Rotate labels if there are many classes
            if len(class_names) > 10:
                ax.set_xticklabels(class_names, rotation=45, ha='right')
            else:
                ax.set_xticklabels(class_names)
                
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "class_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error generating class distribution chart: {e}")
    
    def _generate_bbox_size_chart(self):
        """Generate a pie chart showing bounding box size distribution."""
        if not self.statistics["bbox_sizes"]:
            print("Skipping bounding box size chart: No data available")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Sort size categories in a logical order
            size_order = ["tiny (<1%)", "small (1-5%)", "medium (5-20%)", "large (20-50%)", "huge (>50%)"]
            sizes = []
            counts = []
            
            for size in size_order:
                if size in self.statistics["bbox_sizes"]:
                    sizes.append(size)
                    counts.append(self.statistics["bbox_sizes"][size])
            
            if not sizes or not counts:
                print("Skipping bounding box size chart: No valid data")
                return
                
            # Create a pie chart
            plt.pie(counts, labels=sizes, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Bounding Box Size Distribution')
            
            plt.savefig(self.output_dir / "bbox_size_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error generating bounding box size chart: {e}")
    
    def _generate_bbox_position_heatmap(self):
        """Generate a heatmap showing bounding box position distribution."""
        if not self.statistics["bbox_positions"]:
            print("Skipping bounding box position heatmap: No data available")
            return
        
        try:
            # Create a 3x3 grid for positions
            position_grid = np.zeros((3, 3))
            
            # Position mapping
            position_to_indices = {
                "top-left": (0, 0),
                "top-center": (0, 1),
                "top-right": (0, 2),
                "middle-left": (1, 0),
                "middle-center": (1, 1),
                "middle-right": (1, 2),
                "bottom-left": (2, 0),
                "bottom-center": (2, 1),
                "bottom-right": (2, 2)
            }
            
            # Fill the grid with counts
            for position, count in self.statistics["bbox_positions"].items():
                if position in position_to_indices:
                    i, j = position_to_indices[position]
                    position_grid[i, j] = count
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(position_grid, annot=True, fmt=".1f", cmap="YlGnBu",
                       xticklabels=["Left", "Center", "Right"],
                       yticklabels=["Top", "Middle", "Bottom"])
            plt.title('Bounding Box Position Distribution')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "bbox_position_heatmap.png")
            plt.close()
        except Exception as e:
            print(f"Error generating bounding box position heatmap: {e}")
    
    def _generate_sample_visualizations(self):
        """Generate sample visualizations of images with bounding boxes."""
        # Create a directory for sample visualizations
        samples_dir = self.output_dir / "samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        # Find all image files in the dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        all_images = []
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                continue
                
            images_dir = split_dir / 'images'
            if not images_dir.exists():
                continue
                
            for ext in image_extensions:
                all_images.extend(glob.glob(str(images_dir / f'*{ext}')))
        
        if not all_images:
            print("Skipping sample visualizations: No images found")
            return
        
        # Randomly select up to 10 images
        sample_size = min(10, len(all_images))
        sample_images = random.sample(all_images, sample_size)
        
        for idx, img_path in enumerate(sample_images):
            img_path = Path(img_path)
            img_name = img_path.stem
            
            # Find the corresponding label file
            label_path = None
            for split in ['train', 'val', 'test']:
                potential_label = self.dataset_path / split / 'labels' / f"{img_name}.txt"
                if potential_label.exists():
                    label_path = potential_label
                    break
            
            if not label_path:
                continue
                
            try:
                # Load the image
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                # Create a drawing context
                draw = ImageDraw.Draw(img)
                
                # Load the labels
                with open(label_path, 'r') as f:
                    lines = f.read().strip().splitlines()
                
                # Draw each bounding box
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert normalized coordinates to pixel values
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)
                    
                    # FIX: Ensure coordinates are valid
                    if x1 >= x2:
                        x2 = x1 + 1  # Ensure at least 1 pixel width
                    if y1 >= y2:
                        y2 = y1 + 1  # Ensure at least 1 pixel height
                    
                    # Get class name
                    class_name = self.classes[class_id] if self.classes and class_id < len(self.classes) else f"class_{class_id}"
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # Draw label
                    text_width = len(class_name) * 8
                    draw.rectangle([x1, max(0, y1 - 20), min(img_width, x1 + text_width), max(0, y1)], fill="red")
                    draw.text((x1 + 5, max(0, y1 - 15)), class_name, fill="white")
                
                # Save the visualization
                img.save(samples_dir / f"sample_{idx+1}_{img_name}.jpg")
                
            except Exception as e:
                print(f"Error generating visualization for {img_path}: {e}")
    
    def _generate_bbox_area_distribution(self):
        """Generate a histogram of bounding box area distribution."""
        if not self.statistics["bbox_area_distribution"]:
            print("Skipping bbox area distribution: No data available")
            return
        
        try:
            df = pd.DataFrame(self.statistics["bbox_area_distribution"])
            
            # Check if we have enough data for a meaningful visualization
            if len(df) < 2:
                print("Skipping bbox area distribution: Insufficient data points")
                return
            
            # Check if we have at least one non-NaN value
            if df["area_percentage"].count() < 2:
                print("Skipping bbox area distribution: Insufficient valid data points")
                return
            
            # Get unique classes for histogram
            unique_classes = df["class"].unique()
            if len(unique_classes) < 1:
                print("Skipping bbox area distribution: No class information")
                return
            
            # Set explicit bins instead of auto
            bins = min(20, max(5, int(np.sqrt(len(df)))))
            
            plt.figure(figsize=(12, 8))
            # Use explicit bins and ensure KDE only when enough data
            if len(df) >= 5:  # KDE needs more data points
                sns.histplot(data=df, x="area_percentage", hue="class", kde=True, bins=bins)
            else:
                sns.histplot(data=df, x="area_percentage", hue="class", kde=False, bins=bins)
            
            plt.title("Bounding Box Area Distribution (% of Image)")
            plt.xlabel("Bounding Box Area (% of Image)")
            plt.ylabel("Count")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "bbox_area_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error generating bbox area distribution: {e}")
    
    def _generate_bbox_aspect_ratio_distribution(self):
        """Generate a histogram of bounding box aspect ratio distribution."""
        if not self.statistics["bbox_aspect_ratio_distribution"]:
            print("Skipping bbox aspect ratio distribution: No data available")
            return
        
        try:
            df = pd.DataFrame(self.statistics["bbox_aspect_ratio_distribution"])
            
            # Check for sufficient data
            if len(df) < 2:
                print("Skipping bbox aspect ratio distribution: Insufficient data")
                return
            
            # Filter out extreme values for better visualization
            df = df[df["aspect_ratio"] < 5]
            
            # Check if we still have data after filtering
            if len(df) < 2:
                print("Skipping bbox aspect ratio distribution: No data after filtering")
                return
            
            # Set explicit bins
            bins = min(20, max(5, int(np.sqrt(len(df)))))
            
            plt.figure(figsize=(12, 8))
            
            # Use explicit bins and ensure KDE only when enough data
            if len(df) >= 5:  # KDE needs more data points
                sns.histplot(data=df, x="aspect_ratio", hue="class", kde=True, bins=bins)
            else:
                sns.histplot(data=df, x="aspect_ratio", hue="class", kde=False, bins=bins)
                
            plt.title("Bounding Box Aspect Ratio Distribution")
            plt.xlabel("Aspect Ratio (width/height)")
            plt.ylabel("Count")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "bbox_aspect_ratio_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error generating bbox aspect ratio distribution: {e}")
    
    def _generate_image_size_distribution(self):
        """Generate a visualization of image size distribution."""
        if not self.statistics["image_sizes"]:
            print("Skipping image size distribution: No data available")
            return
        
        try:
            # Parse image sizes
            sizes_data = []
            for size_str, count in self.statistics["image_sizes"].items():
                try:
                    width, height = map(int, size_str.split("x"))
                    sizes_data.append({
                        "width": width,
                        "height": height,
                        "count": count,
                        "aspect_ratio": width / height
                    })
                except (ValueError, ZeroDivisionError):
                    # Skip invalid sizes
                    continue
            
            if not sizes_data:
                print("Skipping image size distribution: No valid sizes")
                return
                
            df = pd.DataFrame(sizes_data)
            
            plt.figure(figsize=(12, 12))
            
            # Create a subplot for width distribution
            plt.subplot(2, 2, 1)
            # Set explicit bins
            bins_width = min(20, max(5, int(np.sqrt(len(df)))))
            sns.histplot(data=df, x="width", weights="count", kde=True, bins=bins_width)
            plt.title("Image Width Distribution")
            plt.xlabel("Width (pixels)")
            
            # Create a subplot for height distribution
            plt.subplot(2, 2, 2)
            # Set explicit bins
            bins_height = min(20, max(5, int(np.sqrt(len(df)))))
            sns.histplot(data=df, x="height", weights="count", kde=True, bins=bins_height)
            plt.title("Image Height Distribution")
            plt.xlabel("Height (pixels)")
            
            # Create a subplot for aspect ratio distribution
            plt.subplot(2, 2, 3)
            # Set explicit bins
            bins_aspect = min(20, max(5, int(np.sqrt(len(df)))))
            sns.histplot(data=df, x="aspect_ratio", weights="count", kde=True, bins=bins_aspect)
            plt.title("Image Aspect Ratio Distribution")
            plt.xlabel("Aspect Ratio (width/height)")
            
            # Create a scatter plot of width vs height
            plt.subplot(2, 2, 4)
            plt.scatter(df["width"], df["height"], s=df["count"]*5, alpha=0.5)
            plt.title("Image Dimensions")
            plt.xlabel("Width (pixels)")
            plt.ylabel("Height (pixels)")
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "image_size_distribution.png")
            plt.close()
        except Exception as e:
            print(f"Error generating image size distribution: {e}")
    
    def _generate_issues_report(self):
        """Generate a report of issues found in the dataset."""
        if not self.statistics["label_issues"]:
            print("Skipping issues report: No issues found")
            return
        
        try:
            # Create a DataFrame from issues
            df = pd.DataFrame(self.statistics["label_issues"])
            
            # Save issues to CSV
            df.to_csv(self.output_dir / "issues.csv", index=False)
            
            # Group issues by type and create a summary
            issue_types = df["issue"].apply(lambda x: x.split(":")[0].strip())
            issue_counts = issue_types.value_counts()
            
            if len(issue_counts) == 0:
                return
            
            plt.figure(figsize=(10, 6))
            issue_counts.plot(kind="bar")
            plt.title("Dataset Issues by Type")
            plt.xlabel("Issue Type")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(self.output_dir / "issues_summary.png")
            plt.close()
        except Exception as e:
            print(f"Error generating issues report: {e}")
    
    def generate_html_report(self):
        """Generate a comprehensive HTML report with all visualizations."""
        
        # Load summary data
        try:
            with open(self.output_dir / "summary_report.json", 'r') as f:
                summary = json.load(f)
        except Exception as e:
            print(f"Error loading summary report: {e}")
            summary = {
                "Dataset Path": str(self.dataset_path),
                "Total Images": self.statistics["total_images"],
                "Valid Images": self.statistics["valid_images"],
                "Corrupt Images": self.statistics["corrupt_images"],
                "Empty Labels": self.statistics["empty_labels"],
                "Total Objects": sum(self.statistics["classes"].values()),
                "Number of Classes": len(self.statistics["classes"]),
                "Class Distribution": self.statistics["class_distribution"],
                "Most Common Image Size": max(self.statistics["image_sizes"].items(), key=lambda x: x[1])[0] if self.statistics["image_sizes"] else "N/A",
                "Most Common Aspect Ratio": max(self.statistics["aspect_ratios"].items(), key=lambda x: x[1])[0] if self.statistics["aspect_ratios"] else "N/A",
                "Number of Issues": len(self.statistics["label_issues"])
            }
        
        # Create class distribution table HTML
        class_table_rows = ""
        for class_name, info in summary["Class Distribution"].items():
            percentage = info["percentage"]
            progress_width = min(percentage, 100)  # Cap at 100% for the progress bar
            
            class_table_rows += f"""
            <tr>
                <td>{class_name}</td>
                <td>{info['count']}</td>
                <td>{info['images']}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress" style="width: {progress_width}%;"></div>
                    </div>
                    {percentage:.2f}%
                </td>
            </tr>
            """
        
        # Generate sample images HTML
        samples_html = ""
        samples_dir = self.output_dir / "samples"
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
            
            if sample_files:
                samples_html = """
                <div class="viz-container">
                    <h2>Sample Annotations</h2>
                    <div class="samples-grid">
                """
                
                for sample_file in sample_files:
                    relative_path = sample_file.relative_to(self.output_dir)
                    samples_html += f"""
                    <div class="viz-item">
                        <h3>{sample_file.stem}</h3>
                        <img src="{relative_path}" alt="{sample_file.stem}">
                    </div>
                    """
                
                samples_html += """
                    </div>
                </div>
                """
        
        # Generate issues table if issues exist
        issues_html = ""
        if self.statistics["label_issues"]:
            issues_html = """
            <div class="viz-container">
                <h2>Dataset Issues</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Image</th>
                            <th>Issue</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Limit to top 100 issues to keep the HTML file manageable
            for issue in self.statistics["label_issues"][:100]:
                issues_html += f"""
                <tr>
                    <td>{issue['image']}</td>
                    <td>{issue['issue']}</td>
                </tr>
                """
            
            issues_html += """
                    </tbody>
                </table>
                <p><a href="issues.csv">Download full issues CSV</a></p>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLOv8 Dataset Validation Report</title>
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
                .summary-box {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 20px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .stat-item {{
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .viz-container {{
                    margin-bottom: 30px;
                }}
                .viz-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .viz-item {{
                    background-color: #fff;
                    border-radius: 5px;
                    padding: 15px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .progress-bar {{
                    height: 20px;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                .progress {{
                    height: 100%;
                    background-color: #3498db;
                    border-radius: 5px;
                }}
                .samples-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>YOLOv8 Dataset Validation Report</h1>
                
                <div class="summary-box">
                    <h2>Dataset Summary</h2>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div>Total Images</div>
                            <div class="stat-value">{summary["Total Images"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Valid Images</div>
                            <div class="stat-value">{summary["Valid Images"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Total Objects</div>
                            <div class="stat-value">{summary["Total Objects"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Number of Classes</div>
                            <div class="stat-value">{summary["Number of Classes"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Corrupt Images</div>
                            <div class="stat-value">{summary["Corrupt Images"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Empty Labels</div>
                            <div class="stat-value">{summary["Empty Labels"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Most Common Image Size</div>
                            <div class="stat-value">{summary["Most Common Image Size"]}</div>
                        </div>
                        <div class="stat-item">
                            <div>Issues Found</div>
                            <div class="stat-value">{summary["Number of Issues"]}</div>
                        </div>
                    </div>
                </div>
                
                <div class="viz-container">
                    <h2>Class Distribution</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Object Count</th>
                                <th>Image Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            {class_table_rows}
                        </tbody>
                    </table>
                    
                    <div class="viz-item">
                        <h3>Class Distribution Chart</h3>
                        <img src="class_distribution.png" alt="Class Distribution">
                    </div>
                </div>
                
                <div class="viz-container">
                    <h2>Bounding Box Analysis</h2>
                    <div class="viz-grid">
                        <div class="viz-item">
                            <h3>Bounding Box Size Distribution</h3>
                            <img src="bbox_size_distribution.png" alt="Bounding Box Size Distribution">
                        </div>
                        <div class="viz-item">
                            <h3>Bounding Box Position Heatmap</h3>
                            <img src="bbox_position_heatmap.png" alt="Bounding Box Position Heatmap">
                        </div>
                        <div class="viz-item">
                            <h3>Bounding Box Area Distribution</h3>
                            <img src="bbox_area_distribution.png" alt="Bounding Box Area Distribution">
                        </div>
                        <div class="viz-item">
                            <h3>Bounding Box Aspect Ratio Distribution</h3>
                            <img src="bbox_aspect_ratio_distribution.png" alt="Bounding Box Aspect Ratio Distribution">
                        </div>
                    </div>
                </div>
                
                <div class="viz-container">
                    <h2>Image Analysis</h2>
                    <div class="viz-item">
                        <h3>Image Size Distribution</h3>
                        <img src="image_size_distribution.png" alt="Image Size Distribution">
                    </div>
                </div>
                
                {samples_html}
                
                {issues_html}
            </div>
        </body>
        </html>
        """
        
        # Write the HTML report
        try:
            with open(self.output_dir / "report.html", 'w') as f:
                f.write(html_content)
            
            print(f"HTML report generated at {self.output_dir / 'report.html'}")
            return self.output_dir / "report.html"
        except Exception as e:
            print(f"Error writing HTML report: {e}")
            return None

def main():
    """Main function to run the YOLOv8 dataset validator."""
    parser = argparse.ArgumentParser(description="YOLOv8 Dataset Validator")
    
    # Basic arguments
    parser.add_argument("--dataset_path", "-d", type=str, required=True, 
                        help="Path to the YOLOv8 dataset (should contain train/val/test folders)")
    parser.add_argument("--yaml_path", "-y", type=str, default=None,
                        help="Path to the dataset YAML file (optional)")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="Directory to save validation results (default: validation_results_<timestamp>)")
    
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
    
    # Create output directory with timestamp if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"validation_results_{timestamp}"
    
    if not args.ci_mode:
        print(f"Starting YOLOv8 dataset validation for: {args.dataset_path}")
        print(f"Results will be saved to: {args.output_dir}")
    
    # Create validator and run validation
    validator = YOLOv8DatasetValidator(args.dataset_path, args.yaml_path, args.output_dir)
    
    try:
        if not args.ci_mode:
            print("Validating dataset...")
        validator.validate()
        
        if not args.ci_mode:
            print("Generating HTML report...")
        report_path = validator.generate_html_report()
        
        # Generate machine-readable report if requested
        if args.json_report:
            with open(validator.output_dir / "ci_report.json", 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "dataset_path": str(validator.dataset_path),
                    "total_images": validator.statistics["total_images"],
                    "valid_images": validator.statistics["valid_images"],
                    "total_objects": sum(validator.statistics["classes"].values()),
                    "classes_count": len(validator.statistics["classes"]),
                    "issues_count": len(validator.statistics["label_issues"]),
                    "status": "success" if len(validator.statistics["label_issues"]) <= args.issue_threshold else "warning"
                }, f, indent=2)
        
        if not args.ci_mode:
            print("\nValidation complete!")
            if report_path:
                print(f"HTML report available at: {report_path}")
                print(f"Open this file in a web browser to view the full validation report.")
            
            # Print summary statistics to console
            try:
                with open(validator.output_dir / "summary_report.json", 'r') as f:
                    summary = json.load(f)
                    
                print("\nDataset Summary:")
                print(f"- Total Images: {summary['Total Images']}")
                print(f"- Valid Images: {summary['Valid Images']}")
                print(f"- Total Objects: {summary['Total Objects']}")
                print(f"- Number of Classes: {summary['Number of Classes']}")
                print(f"- Issues Found: {summary['Number of Issues']}")
            except Exception as e:
                if not args.ci_mode:
                    print(f"Error displaying summary: {e}")
        
        # Exit with error code if needed
        if args.fail_on_issues and len(validator.statistics["label_issues"]) > args.issue_threshold:
            print(f"ERROR: Found {len(validator.statistics['label_issues'])} issues, threshold is {args.issue_threshold}")
            sys.exit(1)
            
        sys.exit(0)
        
    except Exception as e:
        if not args.ci_mode:
            print(f"Error during validation: {e}")
            traceback.print_exc()
        else:
            print(f"VALIDATION_ERROR: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()