# YOLOv8 Model Optimization Framework

## Overview

This repository provides a comprehensive framework for optimizing YOLOv8 object detection models through Neural Architecture Search (NAS). The framework automates the entire process from dataset preparation to model deployment, helping researchers and developers find optimal network architectures that balance accuracy, speed, and model size.

![YOLOv8 Model Optimization](https://via.placeholder.com/800x400.png?text=YOLOv8+Model+Optimization)

## Features

- **Neural Architecture Search (NAS)**: Automatically explore and evaluate different model architectures
- **Dataset Validation & Augmentation**: Ensure dataset quality and apply targeted augmentation for imbalanced classes
- **Model Evaluation**: Comprehensive benchmarking of mAP, precision, recall, speed, and model size
- **CI/CD Integration**: Complete Jenkins pipeline for automated optimization workflows
- **Visualization Tools**: Generate plots and reports to analyze model performance
- **S3 Integration**: Automatically upload optimized models to S3 storage
- **Version Management**: Track model versions and performance metrics

## Project Structure

```
.
├── analyzer.py                 # Analyzes results from NAS trials
├── config_loader.py            # Loads and manages configuration for NAS
├── dataset_validator.py        # Validates dataset structure and quality
├── Jenkinsfile                 # CI/CD pipeline definition
├── main.py                     # Main module for YOLOv8 NAS
├── model_evaluation.py         # Evaluates model performance
├── rename_models.py            # Utility for renaming model weight files
├── search_space.yaml           # Defines the parameter space for architecture search
├── targeted_augmentation.py    # Performs targeted augmentation for minority classes
├── train_yolov8_model.py       # Trains YOLOv8 models with specific configurations
├── trial_manager.py            # Manages execution of NAS trials
├── upload_s3.py                # Uploads models to AWS S3
├── utils.py                    # Utility functions for the framework
├── version_manager.py          # Manages model versioning
└── visualization.py            # Creates visualizations and reports
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- PyTorch 1.8+
- Dataset in YOLOv8 format
- Jenkins server (for CI/CD pipeline)
- AWS account (for S3 integration)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/yolov8-model-optimization.git
cd yolov8-model-optimization

# Install dependencies
pip install -r requirements.txt
```

### Basic Command-Line Usage

1. **Validate your dataset**:

```bash
python dataset_validator.py --dataset_path /path/to/dataset --output_dir validation_results
```

2. **Augment imbalanced classes**:

```bash
python targeted_augmentation.py --dataset /path/to/dataset --output augmented_dataset --threshold 10 --factor 5
```

3. **Run Neural Architecture Search**:

```bash
python main.py --config search_space.yaml --data /path/to/data.yaml --trials 10 --epochs 50 --results-dir nas_results
```

4. **Evaluate models**:

```bash
python model_evaluation.py --models-dir nas_results --data /path/to/data.yaml
```

5. **Upload models to S3**:

```bash
python upload_s3.py /path/to/model model_subfolder --project my_project --notes "Model description"
```

## Neural Architecture Search

The framework performs Neural Architecture Search to find optimal YOLOv8 model architectures by exploring various hyperparameters including:

- Depth multiplier (network depth)
- Width multiplier (channel width)
- Kernel sizes
- Input image dimensions
- Optimization parameters

The search process uses a combination of random sampling and performance-based selection to efficiently explore the architecture space.

### Search Space Configuration

The `search_space.yaml` file defines the parameter space for architecture search:

```yaml
# Basic search space parameters
basic_search_space:
  depth_multiple:
    - 0.33
    - 0.5
    - 0.67
    - 1.0
  width_multiple:
    - 0.25
    - 0.5
    - 0.75
    - 1.0
  img_size:
    - 320
    - 448
    - 640
  kernel_size:
    - 1
    - 3
    - 5
    - 7

# Advanced search parameters
advanced_search_space:
  optimizer:
    - SGD
    - Adam
    - AdamW
  lr0:
    - 0.001
    - 0.01
    - 0.02
  # Other parameters...

# Objective function weights
objective_weights:
  map_weight: 1.0
  speed_weight: 0.3
  size_weight: 0.2
```

## Dataset Augmentation

The targeted augmentation module analyzes class distributions in your dataset and applies stronger augmentation to minority classes, helping to address class imbalance issues. Key features include:

- Class distribution analysis
- Minority class identification
- Strong augmentation techniques including color transforms, geometric transforms, and noise
- Validation of augmented dataset

## Model Evaluation

The evaluation module provides comprehensive performance metrics:

- mAP@0.5 and mAP@0.5-0.95 for accuracy
- Precision and recall values
- F1 score
- Inference speed (FPS)
- Model size
- Combined score balancing accuracy, speed, and size

## Jenkins CI/CD Pipeline

The included Jenkinsfile provides a complete CI/CD pipeline that automates the entire optimization process.

### Pipeline Architecture

The Jenkins pipeline consists of the following stages:

1. **Initialize Workspace**: Creates directories for outputs and displays build information
2. **Setup Virtual Environment**: Creates and configures a Python virtual environment with required dependencies
3. **Download Dataset**: Downloads the dataset from Google Drive using the provided file ID
4. **Extract and Validate Dataset**: Extracts and validates the downloaded dataset
5. **Setup Model Optimization Repository**: Clones and configures the model optimization repository
6. **Parallel Validation and Augmentation**:
   - **Dataset Validation**: Validates the dataset structure and reports issues
   - **Targeted Augmentation**: Performs targeted augmentation on minority classes
7. **Post-Augmentation Validation**: Validates the augmented dataset
8. **Neural Architecture Search (NAS)**: Executes the NAS algorithm to find optimal architectures
9. **Model Evaluation**: Evaluates the performance of the optimized models
10. **Upload Best Model to S3**: Uploads the best model to Amazon S3
11. **Generate Reports**: Creates HTML reports with model performance metrics

### Jenkins Configuration

#### Prerequisites

- Jenkins server with the following plugins:
  - Pipeline
  - Pipeline: AWS Steps
  - Email Extension
  - Timestamper
  - Credentials Binding

#### Credentials Setup

Configure the following credentials in Jenkins:

1. **AWS Credentials**:
   - Type: Username with password
   - ID: `aws-access-key-id` and `aws-secret-access-key`

2. **Google Drive Access** (if needed for dataset download):
   - Configure as needed for gdown access

#### Pipeline Parameters

The pipeline accepts the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MODEL_NAME` | YOLOv8 optimized model name | `yolov8_custom_model` |
| `GDRIVE_FILE_ID` | Google Drive file ID for dataset | `1R44tNwMYBU3kaQLB2cgqzb8HMNisEuqA` |
| `NAS_TRIALS` | Number of trials for Neural Architecture Search | Choice: `3`, `5`, `10`, `15` |
| `NAS_EPOCHS` | Number of epochs per trial | Choice: `2`, `5`, `10`, `50` |
| `SKIP_DOWNLOAD` | Skip dataset download if already available | `false` |
| `SKIP_AUGMENTATION` | Skip dataset augmentation | `false` |
| `CLEAN_WORKSPACE` | Clean workspace after build | `true` |

#### Environment Variables

The pipeline uses the following environment variables:

```
OUTPUT_ZIP = 'dataset.zip'
EXTRACT_DIR = 'extracted_dataset'
DATA_YAML_PATH_FILE = 'data_yaml_path.txt'
VENV_NAME = 'dataset_venv'
MODEL_OPT_REPO = 'yolov8-model-optimization'
ARTIFACT_DIR = "${WORKSPACE}/artifacts"
MODEL_RESULT_DIR = "${WORKSPACE}/model_evaluation_results"
AUGMENTATION_PATH_FILE = "${WORKSPACE}/augmentation_dir_path.txt"
S3_BUCKET = 'yolov8-model-repository'
S3_PREFIX = 'yolov8_model_custom'
```

### Setting Up the Jenkins Pipeline

1. **Create a new Pipeline job**:
   - From Jenkins dashboard, select "New Item"
   - Enter a name and select "Pipeline"
   - Click "OK"

2. **Configure the pipeline**:
   - Under "Pipeline", select "Pipeline script from SCM"
   - Select "Git" as the SCM
   - Enter the repository URL and credentials
   - Specify the branch to build
   - Set the script path to "Jenkinsfile"

3. **Save the configuration** and run the pipeline

## Artifacts and Reports

The pipeline and framework generate several artifacts:

1. **Validation reports**: Reports on dataset quality before and after augmentation
2. **NAS results**: CSV files, JSON configurations, and visualizations of the architecture search
3. **Model evaluation reports**: HTML reports with performance metrics and comparisons
4. **Best model**: The optimized YOLOv8 model with the best performance
5. **Build report**: A comprehensive HTML report summarizing the entire process

### Visualizations

The framework generates various visualizations to help analyze model performance:

- Class distribution plots
- Accuracy vs. speed plots
- Size vs. accuracy plots
- Parameter importance analysis
- Depth vs. width scatter plots
- Kernel size impact analysis

## Best Practices

1. **Resource Management**: 
   - Use appropriate values for `NAS_TRIALS` and `NAS_EPOCHS` based on available resources
   - Start with a smaller number of trials for initial experiments

2. **Dataset Preparation**:
   - Ensure your dataset is well-organized in YOLOv8 format
   - Use a validation set that represents the target use case
   - Check for class imbalance before starting optimization

3. **Pipeline Configuration**:
   - Keep `CLEAN_WORKSPACE` enabled to prevent disk space issues
   - Use parallel processing when available
   - Configure email notifications for long-running jobs

4. **Model Selection**:
   - Balance the objective weights based on your application needs (accuracy vs. speed vs. size)
   - Consider deployment constraints when setting optimization objectives
   - Evaluate models on representative test data

5. **AWS & Security**:
   - Ensure AWS credentials have minimal required permissions for S3 operations
   - Use versioned S3 buckets for model storage
   - Implement proper access control for model artifacts

## Troubleshooting

Common issues and solutions:

1. **Dataset Issues**:
   - **Dataset Download Failures**: Verify the Google Drive file ID and network connectivity
   - **Invalid Dataset Format**: Ensure dataset follows YOLOv8 structure with proper labels
   - **Empty Labels**: Check annotation files and dataset validation reports

2. **GPU and Training Issues**:
   - **CUDA/GPU Errors**: Ensure CUDA is properly installed and compatible with PyTorch
   - **Out of Memory Errors**: Reduce batch size, model size, or image dimensions
   - **Slow Search Process**: Decrease trials or use parallel processing

3. **Pipeline Issues**:
   - **S3 Upload Failures**: Verify AWS credentials and permissions
   - **Jenkins Workspace Issues**: Check disk space and cleanup old workspaces
   - **Timeout Errors**: Adjust the pipeline timeout (currently 8 hours)

4. **Model Performance Issues**:
   - **Low Accuracy Results**: Check dataset quality, increase epochs, or expand search space
   - **Model Conversion Errors**: Check model compatibility and export options
   - **Slow Inference**: Reduce model size or input dimensions

## Extending the Framework

To extend the framework for your needs:

1. **Custom Search Space**: Modify `search_space.yaml` to include additional parameters or constraints
2. **Additional Metrics**: Update the evaluation process in `model_evaluation.py`
3. **Custom Augmentations**: Modify `targeted_augmentation.py` to include custom augmentation techniques
4. **Integration with Other Tools**: Add stages to the Jenkinsfile for additional processing or deployment
5. **Custom Export Formats**: Add support for additional model export formats (TensorRT, ONNX, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Ultralytics for the YOLOv8 implementation
- The computer vision community for continued advancements in object detection
- Jenkins and AWS for CI/CD and cloud storage capabilities
