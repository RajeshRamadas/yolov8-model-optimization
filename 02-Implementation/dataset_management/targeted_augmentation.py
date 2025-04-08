import os
import glob
import random
import numpy as np
import yaml
import cv2
import shutil
import json
import datetime
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import albumentations as A
from concurrent.futures import ThreadPoolExecutor

class YOLODatasetAnalyzer:
    def __init__(self, dataset_path, class_names=None):
        """
        Initialize the YOLO dataset analyzer.
        
        Args:
            dataset_path (str): Path to the root of the YOLO dataset
            class_names (list, optional): List of class names if known
        """
        self.dataset_path = Path(dataset_path)
        self.train_labels = list(self.dataset_path.glob('**/train/labels/*.txt'))
        self.val_labels = list(self.dataset_path.glob('**/val/labels/*.txt'))
        self.test_labels = list(self.dataset_path.glob('**/test/labels/*.txt'))
        self.class_names = class_names
        self.class_counts = None
        self.val_class_counts = None
        self.test_class_counts = None
        self.minority_classes = None
        self.image_paths_by_class = {}
        self.metrics = {}
    
    def analyze_dataset(self):
        """Analyze the dataset to get class distribution"""
        all_classes = []
        self.image_paths_by_class = {}
        
        # Process train set
        print(f"Analyzing {len(self.train_labels)} training labels...")
        for label_path in tqdm(self.train_labels):
            classes_in_image = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # YOLO format: class x y w h
                        class_id = int(parts[0])
                        all_classes.append(class_id)
                        classes_in_image.append(class_id)
            
            # Store image paths by class
            img_path = self._get_image_path_from_label(label_path)
            for class_id in set(classes_in_image):
                if class_id not in self.image_paths_by_class:
                    self.image_paths_by_class[class_id] = []
                self.image_paths_by_class[class_id].append((img_path, label_path))
        
        # Count classes in training set
        self.class_counts = Counter(all_classes)
        
        # Analyze validation set
        val_classes = []
        if self.val_labels:
            print(f"Analyzing {len(self.val_labels)} validation labels...")
            for label_path in tqdm(self.val_labels):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO format: class x y w h
                            class_id = int(parts[0])
                            val_classes.append(class_id)
            
            self.val_class_counts = Counter(val_classes)
        
        # Analyze test set
        test_classes = []
        if self.test_labels:
            print(f"Analyzing {len(self.test_labels)} test labels...")
            for label_path in tqdm(self.test_labels):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:  # YOLO format: class x y w h
                            class_id = int(parts[0])
                            test_classes.append(class_id)
            
            self.test_class_counts = Counter(test_classes)
        
        # Load class names from yaml if available and not provided
        if self.class_names is None:
            yaml_files = list(self.dataset_path.glob('*.yaml'))
            if yaml_files:
                with open(yaml_files[0], 'r') as f:
                    data = yaml.safe_load(f)
                    if 'names' in data:
                        self.class_names = data['names']
        
        # Collect metrics
        self._collect_metrics()
        
        # Print results
        self._print_class_distribution()
        return self.class_counts
    
    def _collect_metrics(self):
        """Collect metrics for reporting"""
        # Basic dataset metrics
        self.metrics['dataset_path'] = str(self.dataset_path)
        self.metrics['num_train_files'] = len(self.train_labels)
        self.metrics['num_val_files'] = len(self.val_labels)
        self.metrics['num_test_files'] = len(self.test_labels)
        
        # Class distribution metrics
        if self.class_counts:
            self.metrics['train'] = {
                'total_objects': sum(self.class_counts.values()),
                'class_distribution': dict(self.class_counts)
            }
        
        if self.val_class_counts:
            self.metrics['val'] = {
                'total_objects': sum(self.val_class_counts.values()),
                'class_distribution': dict(self.val_class_counts)
            }
            
        if self.test_class_counts:
            self.metrics['test'] = {
                'total_objects': sum(self.test_class_counts.values()),
                'class_distribution': dict(self.test_class_counts)
            }
        
        # Class names
        if self.class_names:
            self.metrics['class_names'] = self.class_names
    
    def _print_class_distribution(self):
        """Print the class distribution statistics"""
        print("\nTraining Set Class Distribution:")
        print("-" * 40)
        
        total_objects = sum(self.class_counts.values())
        
        if self.class_names:
            for class_id, count in sorted(self.class_counts.items()):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                percentage = (count / total_objects) * 100
                print(f"{class_name:<20}: {count:5d} instances ({percentage:.2f}%)")
        else:
            for class_id, count in sorted(self.class_counts.items()):
                percentage = (count / total_objects) * 100
                print(f"Class {class_id:<15}: {count:5d} instances ({percentage:.2f}%)")
        
        print("-" * 40)
        print(f"Total objects in training set: {total_objects}")
        
        # Print validation set stats if available
        if self.val_class_counts:
            print("\nValidation Set Class Distribution:")
            print("-" * 40)
            
            val_total = sum(self.val_class_counts.values())
            
            if self.class_names:
                for class_id, count in sorted(self.val_class_counts.items()):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                    percentage = (count / val_total) * 100
                    print(f"{class_name:<20}: {count:5d} instances ({percentage:.2f}%)")
            else:
                for class_id, count in sorted(self.val_class_counts.items()):
                    percentage = (count / val_total) * 100
                    print(f"Class {class_id:<15}: {count:5d} instances ({percentage:.2f}%)")
            
            print("-" * 40)
            print(f"Total objects in validation set: {val_total}")
        
        # Print test set stats if available
        if self.test_class_counts:
            print("\nTest Set Class Distribution:")
            print("-" * 40)
            
            test_total = sum(self.test_class_counts.values())
            
            if self.class_names:
                for class_id, count in sorted(self.test_class_counts.items()):
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                    percentage = (count / test_total) * 100
                    print(f"{class_name:<20}: {count:5d} instances ({percentage:.2f}%)")
            else:
                for class_id, count in sorted(self.test_class_counts.items()):
                    percentage = (count / test_total) * 100
                    print(f"Class {class_id:<15}: {count:5d} instances ({percentage:.2f}%)")
            
            print("-" * 40)
            print(f"Total objects in test set: {test_total}")
    
    def identify_minority_classes(self, threshold_percentage=10):
        """
        Identify minority classes based on a threshold percentage.
        
        Args:
            threshold_percentage (float): Classes with percentage below this are considered minority
            
        Returns:
            list: List of minority class IDs
        """
        if self.class_counts is None:
            self.analyze_dataset()
        
        total_objects = sum(self.class_counts.values())
        self.minority_classes = []
        
        for class_id, count in self.class_counts.items():
            percentage = (count / total_objects) * 100
            if percentage < threshold_percentage:
                self.minority_classes.append(class_id)
        
        # Print minority classes
        print(f"\nIdentified {len(self.minority_classes)} minority classes:")
        if self.class_names:
            for class_id in sorted(self.minority_classes):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                count = self.class_counts[class_id]
                percentage = (count / total_objects) * 100
                print(f"{class_name:<20}: {count:5d} instances ({percentage:.2f}%)")
        else:
            for class_id in sorted(self.minority_classes):
                count = self.class_counts[class_id]
                percentage = (count / total_objects) * 100
                print(f"Class {class_id:<15}: {count:5d} instances ({percentage:.2f}%)")
        
        # Store in metrics
        self.metrics['minority_classes'] = {
            'threshold_percentage': threshold_percentage,
            'classes': []
        }
        
        if self.class_names:
            for class_id in sorted(self.minority_classes):
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                count = self.class_counts[class_id]
                percentage = (count / total_objects) * 100
                self.metrics['minority_classes']['classes'].append({
                    'id': class_id,
                    'name': class_name,
                    'count': count,
                    'percentage': percentage
                })
        else:
            for class_id in sorted(self.minority_classes):
                count = self.class_counts[class_id]
                percentage = (count / total_objects) * 100
                self.metrics['minority_classes']['classes'].append({
                    'id': class_id,
                    'count': count,
                    'percentage': percentage
                })
        
        return self.minority_classes
    
    def _get_image_path_from_label(self, label_path):
        """Convert label path to corresponding image path"""
        # YOLO structure: {dataset}/labels/{filename}.txt -> {dataset}/images/{filename}.jpg
        img_dir = str(label_path.parent).replace('labels', 'images')
        img_file = label_path.stem + '.jpg'  # Try jpg first
        img_path = os.path.join(img_dir, img_file)
        
        # Check for other extensions if jpg doesn't exist
        if not os.path.exists(img_path):
            for ext in ['.png', '.jpeg', '.bmp']:
                test_path = os.path.join(img_dir, label_path.stem + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break
        
        return img_path
    
    def plot_class_distribution(self):
        """Plot the class distribution"""
        if self.class_counts is None:
            self.analyze_dataset()
        
        # Prepare data for plotting
        class_ids = sorted(self.class_counts.keys())
        counts = [self.class_counts[c] for c in class_ids]
        
        # Use class names if available
        if self.class_names:
            labels = [self.class_names[c] if c < len(self.class_names) else f"Class {c}" for c in class_ids]
        else:
            labels = [f"Class {c}" for c in class_ids]
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        plt.bar(labels, counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in YOLOv8 Dataset')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('class_distribution.png')
        print("Class distribution plot saved as 'class_distribution.png'")
        
        # Initialize the plots dictionary
        self.metrics['plots'] = {
            'class_distribution': 'class_distribution.png'
        }
        
        # If we have val and test data, create comparison plots
        if self.val_class_counts or self.test_class_counts:
            self._plot_set_comparison(class_ids, labels)
        
    def _plot_set_comparison(self, class_ids, labels):
        """Create a comparison plot of train/val/test distributions"""
        # Prepare data
        train_counts = [self.class_counts.get(c, 0) for c in class_ids]
        val_counts = [self.val_class_counts.get(c, 0) if self.val_class_counts else 0 for c in class_ids]
        test_counts = [self.test_class_counts.get(c, 0) if self.test_class_counts else 0 for c in class_ids]
        
        # Normalize to percentages for better comparison
        train_total = sum(train_counts)
        val_total = sum(val_counts)
        test_total = sum(test_counts)
        
        train_pct = [count/train_total*100 if train_total > 0 else 0 for count in train_counts]
        val_pct = [count/val_total*100 if val_total > 0 else 0 for count in val_counts]
        test_pct = [count/test_total*100 if test_total > 0 else 0 for count in test_counts]
        
        # Set width of bars
        barWidth = 0.25
        
        # Set position of bars on X axis
        r1 = np.arange(len(class_ids))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
        
        # Create figure
        plt.figure(figsize=(16, 10))
        
        # Create bars
        plt.bar(r1, train_pct, width=barWidth, label='Train')
        if self.val_class_counts:
            plt.bar(r2, val_pct, width=barWidth, label='Validation')
        if self.test_class_counts:
            plt.bar(r3, test_pct, width=barWidth, label='Test')
        
        # Add labels
        plt.xlabel('Class', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(class_ids))], labels, rotation=45, ha='right')
        plt.ylabel('Percentage (%)')
        plt.title('Class Distribution Comparison Across Datasets')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig('dataset_comparison.png')
        print("Dataset comparison plot saved as 'dataset_comparison.png'")
        
        # Store in metrics - Fix the KeyError by ensuring 'plots' key exists
        if 'plots' not in self.metrics:
            self.metrics['plots'] = {}
        self.metrics['plots']['dataset_comparison'] = 'dataset_comparison.png'

class YOLOAugmentor:
    def __init__(self, analyzer, output_dir='augmented_dataset', augmentation_factor=5):
        """
        Initialize the YOLO dataset augmentor.
        
        Args:
            analyzer (YOLODatasetAnalyzer): Initialized and analyzed dataset
            output_dir (str): Directory to save augmented dataset
            augmentation_factor (int): How many times to augment each minority class image
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.augmentation_factor = augmentation_factor
        self.metrics = {
            'augmentation_factor': augmentation_factor,
            'output_dir': str(self.output_dir),
            'augmentation_results': {}
        }
        
        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / 'train' / 'images', exist_ok=True)
        os.makedirs(self.output_dir / 'train' / 'labels', exist_ok=True)
        
        # Create val and test directories if needed
        if analyzer.val_labels:
            os.makedirs(self.output_dir / 'val' / 'images', exist_ok=True)
            os.makedirs(self.output_dir / 'val' / 'labels', exist_ok=True)
        
        if analyzer.test_labels:
            os.makedirs(self.output_dir / 'test' / 'images', exist_ok=True)
            os.makedirs(self.output_dir / 'test' / 'labels', exist_ok=True)
        
        # Create target yaml file
        self._create_yaml_file()
    
    def _create_yaml_file(self):
        """Create dataset YAML file for the augmented dataset with specific format"""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.analyzer.class_names) if self.analyzer.class_names else 0,
            'names': self.analyzer.class_names,
            'roboflow': {
                'workspace': 'roboflow-100',
                'project': 'vehicles-q0x2v',
                'version': 2,
                'license': 'CC BY 4.0',
                'url': 'https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2'
            }
        }
        
        # Save as data.yaml instead of dataset.yaml
        with open(self.output_dir / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"Created data.yaml file at: {self.output_dir / 'data.yaml'}")
    
    def create_strong_augmentation(self):
        """Create stronger augmentation pipeline for minority classes"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomRotate90(p=0.5),
            A.Blur(blur_limit=5, p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomFog(p=0.2),
            A.Affine(scale=(0.8, 1.2), translate_percent=(0.1, 0.1), rotate=(-30, 30), p=0.5),
            A.RandomSunFlare(p=0.2),
            A.ISONoise(p=0.3),
            A.RandomSnow(p=0.1),
            A.CLAHE(p=0.3),
            # More aggressive color transforms
            A.ChannelShuffle(p=0.2),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        ], bbox_params=A.BboxParams(
            format='yolo',
            min_visibility=0.3,
            label_fields=['class_labels']
        ))
    
    def process_single_image(self, image_data, class_id, aug_index):
        """Process and augment a single image"""
        img_path, label_path = image_data
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return
        
        # Read image and labels
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = []
            class_labels = []
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id_in_file = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        bboxes.append([x, y, w, h])
                        class_labels.append(class_id_in_file)
            
            # Apply augmentation
            transform = self.create_strong_augmentation()
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']
            
            # Create output filenames
            base_name = f"{Path(img_path).stem}_aug_{class_id}_{aug_index}"
            aug_img_path = self.output_dir / 'train' / 'images' / f"{base_name}.jpg"
            aug_label_path = self.output_dir / 'train' / 'labels' / f"{base_name}.txt"
            
            # Save augmented image
            aug_image_rgb = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), aug_image_rgb)
            
            # Save augmented labels - ensure integer class IDs
            with open(aug_label_path, 'w') as f:
                for bbox, label in zip(aug_bboxes, aug_class_labels):
                    x, y, w, h = bbox
                    # Convert class label to integer if it's a float
                    label_int = int(label) if isinstance(label, (int, float)) else label
                    f.write(f"{label_int} {x} {y} {w} {h}\n")
            
            return True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False
        
    def augment_minority_classes(self):
        """Augment images containing minority classes"""
        if not self.analyzer.minority_classes:
            print("No minority classes identified. Run analyzer.identify_minority_classes() first.")
            return
        
        print(f"\nAugmenting minority classes with factor {self.augmentation_factor}...")
        
        # Process each minority class
        total_augmented = 0
        self.metrics['augmentation_results']['by_class'] = {}
        
        for class_id in self.analyzer.minority_classes:
            if class_id not in self.analyzer.image_paths_by_class:
                continue
            
            image_paths = self.analyzer.image_paths_by_class[class_id]
            class_name = (self.analyzer.class_names[class_id] 
                          if self.analyzer.class_names and class_id < len(self.analyzer.class_names) 
                          else f"Class {class_id}")
            
            print(f"Augmenting {len(image_paths)} images for {class_name}...")
            
            # Create augmentations in parallel
            augmentation_tasks = []
            
            for img_idx, img_data in enumerate(image_paths):
                for aug_idx in range(self.augmentation_factor):
                    augmentation_tasks.append((img_data, class_id, f"{img_idx}_{aug_idx}"))
            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(
                    executor.map(lambda x: self.process_single_image(*x), augmentation_tasks),
                    total=len(augmentation_tasks)
                ))
            
            successful_augs = sum(1 for r in results if r)
            total_augmented += successful_augs
            
            print(f"Successfully created {successful_augs} augmentations for {class_name}")
            
            # Store in metrics
            self.metrics['augmentation_results']['by_class'][str(class_id)] = {
                'name': class_name if self.analyzer.class_names else None,
                'original_count': len(image_paths),
                'augmented_count': successful_augs,
                'augmentation_factor': self.augmentation_factor
            }
        
        self.metrics['augmentation_results']['total_augmented'] = total_augmented
        print(f"\nAugmentation complete. Generated {total_augmented} new images.")
    
    def copy_original_dataset(self):
        """Copy original dataset to the output directory and create validation set if needed"""
        print("\nCopying original dataset...")
        self.metrics['original_dataset'] = {}
        
        # Copy train images and labels
        train_images = []
        train_labels = []
        for label_path in tqdm(self.analyzer.train_labels, desc="Copying train set"):
            img_path = self.analyzer._get_image_path_from_label(label_path)
            
            if os.path.exists(img_path):
                # Copy image
                dest_img = self.output_dir / 'train' / 'images' / Path(img_path).name
                shutil.copy2(img_path, dest_img)
                train_images.append(dest_img)
                
                # Copy label
                dest_label = self.output_dir / 'train' / 'labels' / Path(label_path).name
                shutil.copy2(label_path, dest_label)
                train_labels.append((img_path, label_path))
        
        self.metrics['original_dataset']['train_images'] = len(train_images)
        
        # Check if we need to create a validation set
        create_validation = len(self.analyzer.val_labels) == 0 and train_labels
        
        # Copy validation set if exists or create one if it doesn't
        val_images = []
        if self.analyzer.val_labels:
            # Validation set exists, copy it
            for label_path in tqdm(self.analyzer.val_labels, desc="Copying validation set"):
                img_path = self.analyzer._get_image_path_from_label(label_path)
                
                if os.path.exists(img_path):
                    # Copy image
                    dest_img = self.output_dir / 'val' / 'images' / Path(img_path).name
                    shutil.copy2(img_path, dest_img)
                    val_images.append(dest_img)
                    
                    # Copy label
                    dest_label = self.output_dir / 'val' / 'labels' / Path(label_path).name
                    shutil.copy2(label_path, dest_label)
        elif create_validation:
            # No validation set, create one from training data
            print("\nNo validation set found. Creating validation set from training data...")
            os.makedirs(self.output_dir / 'val' / 'images', exist_ok=True)
            os.makedirs(self.output_dir / 'val' / 'labels', exist_ok=True)
            
            # Use 10% of training data for validation
            val_size = max(int(len(train_labels) * 0.1), 1)  # At least 1 image
            
            # Shuffle to ensure random selection
            random.shuffle(train_labels)
            validation_samples = train_labels[:val_size]
            
            # Copy selected items to validation folder
            for img_path, label_path in tqdm(validation_samples, desc="Creating validation set"):
                # Copy image
                val_img_path = self.output_dir / 'val' / 'images' / Path(img_path).name
                train_img_path = self.output_dir / 'train' / 'images' / Path(img_path).name
                
                # Copy label
                val_label_path = self.output_dir / 'val' / 'labels' / Path(label_path).name
                train_label_path = self.output_dir / 'train' / 'labels' / Path(label_path).name
                
                # Move from train to val
                if os.path.exists(train_img_path):
                    shutil.copy2(train_img_path, val_img_path)
                    val_images.append(val_img_path)
                
                if os.path.exists(train_label_path):
                    shutil.copy2(train_label_path, val_label_path)
            
            print(f"Created validation set with {len(val_images)} images")
        
        self.metrics['original_dataset']['val_images'] = len(val_images)
        
        # Copy test set if exists
        test_images = []
        if self.analyzer.test_labels:
            for label_path in tqdm(self.analyzer.test_labels, desc="Copying test set"):
                img_path = self.analyzer._get_image_path_from_label(label_path)
                
                if os.path.exists(img_path):
                    # Copy image
                    dest_img = self.output_dir / 'test' / 'images' / Path(img_path).name
                    shutil.copy2(img_path, dest_img)
                    test_images.append(dest_img)
                    
                    # Copy label
                    dest_label = self.output_dir / 'test' / 'labels' / Path(label_path).name
                    shutil.copy2(label_path, dest_label)
        
        self.metrics['original_dataset']['test_images'] = len(test_images)
        
        print(f"Copied {len(train_images)} original training images, {len(val_images)} validation images, and {len(test_images)} test images.")
    def create_balanced_dataset(self):
        """Create a balanced dataset with original and augmented images"""
        # First copy the original dataset
        self.copy_original_dataset()
        
        # Then augment minority classes
        self.augment_minority_classes()
        
        # Generate final dataset report
        self._generate_dataset_report()
        
        print("\nBalanced dataset created at:", self.output_dir)
        print("To use this dataset with YOLOv8, specify the YAML file:")
        print(f"    {self.output_dir}/dataset.yaml")
        print(f"Detailed dataset report saved as: {self.output_dir}/dataset_report.md")
        print(f"Dataset metrics saved as: {self.output_dir}/dataset_metrics.json")
    
    def _calculate_final_class_distribution(self):
        """Calculate the final class distribution after augmentation"""
        final_distribution = {}
        
        # Count classes in the augmented training set
        train_labels_dir = self.output_dir / 'train' / 'labels'
        class_counts = Counter()
        
        for label_file in train_labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Handle both integer and float class IDs by converting to int
                            class_id = int(float(parts[0]))
                            class_counts[class_id] += 1
                        except ValueError as e:
                            # Skip invalid lines but print a warning
                            print(f"Warning: Invalid class ID in {label_file}: {parts[0]}")
                            continue
        
        # Create distribution info
        total_objects = sum(class_counts.values())
        
        for class_id, count in sorted(class_counts.items()):
            percentage = (count / total_objects) * 100
            
            class_name = (self.analyzer.class_names[class_id] 
                         if self.analyzer.class_names and class_id < len(self.analyzer.class_names) 
                         else f"Class {class_id}")
            
            final_distribution[str(class_id)] = {
                'name': class_name,
                'count': count,
                'percentage': percentage,
                'is_minority': class_id in self.analyzer.minority_classes
            }
        
        return {
            'total_objects': total_objects,
            'class_distribution': final_distribution
        }
    
    def _generate_dataset_report(self):
        """Generate a comprehensive report of the dataset"""
        # Calculate final distribution
        final_distribution = self._calculate_final_class_distribution()
        self.metrics['final_distribution'] = final_distribution
        
        # Create markdown report
        report_path = self.output_dir / 'dataset_report.md'
        
        with open(report_path, 'w') as f:
            # Header
            f.write('# YOLOv8 Balanced Dataset Report\n\n')
            f.write(f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Dataset information
            f.write('## Dataset Information\n\n')
            f.write(f'- Original dataset: {self.analyzer.metrics["dataset_path"]}\n')
            f.write(f'- Output dataset: {self.metrics["output_dir"]}\n')
            f.write(f'- Augmentation factor: {self.augmentation_factor}\n\n')
            
            # Original dataset statistics
            f.write('## Original Dataset Statistics\n\n')
            f.write('| Set | Images | Objects |\n')
            f.write('| --- | ------ | ------- |\n')
            
            train_objects = self.analyzer.metrics.get('train', {}).get('total_objects', 0)
            val_objects = self.analyzer.metrics.get('val', {}).get('total_objects', 0)
            test_objects = self.analyzer.metrics.get('test', {}).get('total_objects', 0)
            
            f.write(f'| Training | {self.analyzer.metrics["num_train_files"]} | {train_objects} |\n')
            f.write(f'| Validation | {self.analyzer.metrics["num_val_files"]} | {val_objects} |\n')
            f.write(f'| Test | {self.analyzer.metrics["num_test_files"]} | {test_objects} |\n\n')
            
            # Class imbalance
            f.write('## Class Imbalance Analysis\n\n')
            f.write(f'Threshold for minority classes: {self.analyzer.metrics["minority_classes"]["threshold_percentage"]}%\n\n')
            
            # Create table for minority classes
            if self.analyzer.metrics["minority_classes"]["classes"]:
                f.write('### Identified Minority Classes\n\n')
                f.write('| Class ID | Class Name | Original Count | Original % | Final Count | Final % | Increase Factor |\n')
                f.write('| -------- | ---------- | -------------- | ---------- | ----------- | ------- | --------------- |\n')
                
                for minority_class in self.analyzer.metrics["minority_classes"]["classes"]:
                    class_id = minority_class['id']
                    class_name = minority_class.get('name', f"Class {class_id}")
                    orig_count = minority_class['count']
                    orig_pct = minority_class['percentage']
                    
                    # Get final stats if available
                    final_stats = final_distribution['class_distribution'].get(str(class_id), {})
                    final_count = final_stats.get('count', orig_count)
                    final_pct = final_stats.get('percentage', orig_pct)
                    
                    increase_factor = round(final_count / orig_count, 2) if orig_count > 0 else 'N/A'
                    
                    f.write(f'| {class_id} | {class_name} | {orig_count} | {orig_pct:.2f}% | {final_count} | {final_pct:.2f}% | {increase_factor}x |\n')
                
                f.write('\n')
            else:
                f.write('No minority classes identified with the given threshold.\n\n')
            
            # Augmentation summary
            f.write('## Augmentation Summary\n\n')
            f.write(f'- Total original training images: {self.metrics["original_dataset"]["train_images"]}\n')
            f.write(f'- Total augmented images generated: {self.metrics["augmentation_results"].get("total_augmented", 0)}\n')
            f.write(f'- Final training set size: {self.metrics["original_dataset"]["train_images"] + self.metrics["augmentation_results"].get("total_augmented", 0)}\n\n')
            
            # Final dataset distribution
            f.write('## Final Class Distribution\n\n')
            f.write('| Class ID | Class Name | Count | Percentage | Minority? |\n')
            f.write('| -------- | ---------- | ----- | ---------- | --------- |\n')
            
            for class_id, stats in final_distribution['class_distribution'].items():
                is_minority = "Yes" if stats.get('is_minority', False) else "No"
                f.write(f'| {class_id} | {stats["name"]} | {stats["count"]} | {stats["percentage"]:.2f}% | {is_minority} |\n')
            
            f.write('\n')
            
            # Training recommendations
            f.write('## Training Recommendations\n\n')
            f.write('To train YOLOv8 with this balanced dataset, use the following command:\n\n')
            f.write('```bash\n')
            f.write(f'yolo task=detect train data={self.output_dir}/dataset.yaml model=yolov8n.pt epochs=100\n')
            f.write('```\n\n')
            
            f.write('### Additional YOLOv8 Settings for Imbalanced Data\n\n')
            f.write('Consider these additional options when training:\n\n')
            f.write('1. **Longer Training**: For imbalanced datasets, consider increasing epochs to 150-200\n')
            f.write('2. **Higher IoU Thresholds**: Use `--iou 0.7` for stricter box predictions\n')
            f.write('3. **Learning Rate Scheduling**: Try cosine scheduler with `--cos-lr`\n')
            f.write('4. **Heavy Augmentation**: Add more augmentation during training with `--augment`\n')
            f.write('5. **Class Weights**: For persistent imbalance, add class weights in the YAML file\n\n')
            
            # Visualizations
            if 'plots' in self.analyzer.metrics:
                f.write('## Visualizations\n\n')
                f.write('### Class Distribution\n\n')
                f.write(f'![Class Distribution](../class_distribution.png)\n\n')
                
                if 'dataset_comparison' in self.analyzer.metrics['plots']:
                    f.write('### Dataset Comparison\n\n')
                    f.write(f'![Dataset Comparison](../dataset_comparison.png)\n\n')
        
        # Save all metrics as JSON
        with open(self.output_dir / 'dataset_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)


def main():
    """Main function to run the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze YOLO dataset and augment minority classes')
    parser.add_argument('--dataset', type=str, required=True, help='Path to YOLO dataset directory')
    parser.add_argument('--output', type=str, default='augmented_dataset', help='Output directory for augmented dataset')
    parser.add_argument('--threshold', type=float, default=10.0, help='Percentage threshold to identify minority classes')
    parser.add_argument('--factor', type=int, default=5, help='Augmentation factor for minority classes')
    
    args = parser.parse_args()
    
    # Analyze dataset
    analyzer = YOLODatasetAnalyzer(args.dataset)
    analyzer.analyze_dataset()
    analyzer.plot_class_distribution()
    analyzer.identify_minority_classes(args.threshold)
    
    # Create augmented dataset
    augmentor = YOLOAugmentor(analyzer, args.output, args.factor)
    augmentor.create_balanced_dataset()


if __name__ == "__main__":
    main()