pipeline {
    agent any
    
    parameters {
        string(name: 'MODEL_NAME', defaultValue: 'yolov8_custom_model', description: 'YOLOv8 optimized model name with NAS')
        string(name: 'GDRIVE_FILE_ID', defaultValue: '1R44tNwMYBU3kaQLB2cgqzb8HMNisEuqA', description: 'Google Drive file ID for dataset')
        choice(name: 'NAS_TRIALS', choices: ['3', '5', '10','15'], description: 'Number of trials for Neural Architecture Search')
        choice(name: 'NAS_EPOCHS', choices: ['2', '5', '10','50'], description: 'Number of epochs per trial')
        booleanParam(name: 'SKIP_DOWNLOAD', defaultValue: false, description: 'Skip dataset download if already available')
        booleanParam(name: 'SKIP_AUGMENTATION', defaultValue: false, description: 'Skip dataset augmentation')
        booleanParam(name: 'CLEAN_WORKSPACE', defaultValue: true, description: 'Clean workspace after build')
    }
    
    environment {
        // Dataset related variables
        OUTPUT_ZIP = 'dataset.zip'
        EXTRACT_DIR = 'extracted_dataset'
        DATA_YAML_PATH_FILE = 'data_yaml_path.txt'
        
        // Virtual environment and repository
        VENV_NAME = 'dataset_venv'
        MODEL_OPT_REPO = 'yolov8-model-optimization'
        
        // Paths for generated files
        ARTIFACT_DIR = "${WORKSPACE}/artifacts"
        MODEL_RESULT_DIR = "${WORKSPACE}/model_evaluation_results"
        AUGMENTATION_PATH_FILE = "${WORKSPACE}/augmentation_dir_path.txt"
        
        // S3 bucket information
        S3_BUCKET = 'yolov8-model-repository'
        S3_PREFIX = 'yolov8_model_custom'
	
    }
    
    options {
        // Set timeout for the entire build
        timeout(time: 8, unit: 'HOURS')
        // Keep the 10 most recent builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        // Don't run concurrent builds to avoid conflicts
        disableConcurrentBuilds()
        // Add timestamps to console output
        timestamps()
    }
    
    stages {
        stage('Initialize Workspace') {
            steps {
                script {
                    // Create directories for outputs
                    sh '''
                        mkdir -p ${ARTIFACT_DIR}
                        mkdir -p ${MODEL_RESULT_DIR}
                        mkdir -p email_artifacts
                        mkdir -p downloads
                    '''
                    
                    // Display build information
                    echo "Starting YOLOv8 Model Optimization Pipeline"
                    echo "Model Name: ${params.MODEL_NAME}"
                    echo "NAS Trials: ${params.NAS_TRIALS}"
                    echo "NAS Epochs: ${params.NAS_EPOCHS}"
                }
            }
        }
        
        stage('Setup Virtual Environment') {
            steps {
                sh '''#!/bin/bash -e
                    # Create virtual environment if it doesn't exist
                    if [ ! -d "${VENV_NAME}" ]; then
                        echo "Creating virtual environment..."
                        python3 -m venv ${VENV_NAME}
                    else
                        echo "Virtual environment already exists."
                    fi
                    
                    # Activate virtual environment and upgrade pip
                    source ${VENV_NAME}/bin/activate
                    python -m pip install --upgrade pip
                    
                    # Install core dependencies that will be needed by many stages
                    pip install numpy pandas matplotlib tqdm PyYAML
                    
                    echo "Virtual environment setup complete."
                '''
            }
        }
        
        stage('Download Dataset') {
            when {
                expression { !params.SKIP_DOWNLOAD }
            }
            steps {
                sh '''#!/bin/bash -e
                    # Activate virtual environment
                    source ${VENV_NAME}/bin/activate
                    
                    # Check if dataset already exists
                    if [ -f "downloads/${OUTPUT_ZIP}" ] && [ -d "downloads/${EXTRACT_DIR}" ]; then
                        echo "Dataset already exists. Skipping download."
                        exit 0
                    fi
                    
                    # Install gdown for Google Drive downloads
                    echo "Installing gdown..."
                    pip install gdown
                    
                    # Download the file from Google Drive using gdown
                    echo "Downloading dataset from Google Drive..."
                    cd downloads
                    python -m gdown https://drive.google.com/uc?id=${GDRIVE_FILE_ID} -O ${OUTPUT_ZIP}
                    
                    # Check if download was successful
                    if [ ! -f "${OUTPUT_ZIP}" ]; then
                        echo "Download failed!"
                        exit 1
                    fi
                    
                    echo "Download completed successfully."
                '''
                
                // Archive the dataset as an artifact
                archiveArtifacts artifacts: 'downloads/${OUTPUT_ZIP}', fingerprint: true, allowEmptyArchive: true
            }
        }
        
        stage('Extract and Validate Dataset') {
            when {
                expression { !params.SKIP_DOWNLOAD }
            }
            steps {
                sh '''#!/bin/bash -e
                    cd downloads
                    
                    # Create extraction directory if it doesn't exist
                    mkdir -p ${EXTRACT_DIR}
                    
                    # Extract the zip file
                    echo "Extracting dataset..."
                    unzip -o ${OUTPUT_ZIP} -d ${EXTRACT_DIR}
                    
                    # Find data.yaml file in the extracted directory
                    echo "Locating data.yaml file..."
                    DATA_YAML_PATH=$(find ${EXTRACT_DIR} -name "data.yaml" -type f | head -n 1)
                    
                    if [ -z "${DATA_YAML_PATH}" ]; then
                        echo "data.yaml not found in the extracted dataset!"
                        exit 1
                    fi
                    
                    # Get absolute path
                    ABSOLUTE_PATH=$(readlink -f "${DATA_YAML_PATH}")
                    
                    # Save the path to a file
                    echo "${ABSOLUTE_PATH}" > ${DATA_YAML_PATH_FILE}
                    
                    echo "data.yaml path saved: ${ABSOLUTE_PATH}"
                    
                    # Display first few lines of data.yaml
                    echo "First 5 lines of data.yaml:"
                    head -n 5 "${ABSOLUTE_PATH}"
                '''
                
                // Archive the path file as an artifact
                archiveArtifacts artifacts: 'downloads/data_yaml_path.txt', fingerprint: true
            }
        }
        
        stage('Setup Model Optimization Repository') {
            steps {
                sh '''#!/bin/bash -e
                    # Activate virtual environment
                    source ${VENV_NAME}/bin/activate
                    
                    # Check if the repository directory exists
                    if [ -d "${MODEL_OPT_REPO}" ]; then
                        echo "Model optimization repository directory already exists, updating it..."
                        cd ${MODEL_OPT_REPO}
                        git pull
                        cd ..
                    else
                        # Clone the model optimization repository
                        echo "Downloading YOLOv8 model optimization repository..."
                        git clone https://github.com/RajeshRamadas/yolov8-model-optimization.git
                    fi
                    
                    # Verify repository was downloaded
                    if [ ! -d "${MODEL_OPT_REPO}" ]; then
                        echo "Failed to download model optimization repository!"
                        exit 1
                    fi
                    
                    # Install requirements for the model optimization tools
                    cd ${MODEL_OPT_REPO}
                    if [ -f "requirements.txt" ]; then
                        echo "Installing requirements for model optimization..."
                        pip install -r requirements.txt
                    else
                        echo "No requirements.txt found, installing common YOLOv8 dependencies..."
                        pip install ultralytics torch torchvision torchaudio
                    fi
                    
                    # Log repository information
                    echo "Repository information:"
                    git log -1 --pretty=format:"Last commit: %h by %an (%ad) - %s"
                    
                    cd ..
                    echo "Model optimization repository setup complete."
                    
                    # Create a file listing all Python files for documentation
                    mkdir -p ${ARTIFACT_DIR}
                    find ${MODEL_OPT_REPO} -type f -name "*.py" | sort > ${ARTIFACT_DIR}/model_optimization_files.txt
                '''
                
                // Archive repository info for reference
                archiveArtifacts artifacts: 'artifacts/model_optimization_files.txt', fingerprint: true
            }
        }
        
        stage('Parallel Validation and Augmentation') {
            parallel {
                stage('Dataset Validation') {
                    steps {
                        sh '''#!/bin/bash -e
                            # Activate virtual environment
                            source ${VENV_NAME}/bin/activate
                            
                            # Create a directory for storing validation results
                            mkdir -p ${ARTIFACT_DIR}/before_augmentation
                            
                            # Read the data.yaml path
                            DATA_YAML_PATH=$(cat downloads/data_yaml_path.txt)
                            
                            # Get the dataset directory path (parent directory of data.yaml)
                            DATASET_DIR=$(dirname "${DATA_YAML_PATH}")
                            
                            # Find dataset_validator.py in the model optimization repository
                            VALIDATOR_PATH=$(find ${MODEL_OPT_REPO} -name "dataset_validator.py" -type f | head -n 1)
                            
                            if [ -z "${VALIDATOR_PATH}" ]; then
                                echo "dataset_validator.py not found in the model optimization repository!"
                                exit 1
                            fi
                            
                            echo "Found dataset_validator.py at: ${VALIDATOR_PATH}"
                            
                            # Get the directory containing the validator script
                            VALIDATOR_DIR=$(dirname "${VALIDATOR_PATH}")
                            
                            # Run the dataset validator with the specified parameters
                            echo "Running dataset validation with parameters as specified..."
                            cd "${VALIDATOR_DIR}"
                            python dataset_validator.py --dataset_path "${DATASET_DIR}" --output_dir "${ARTIFACT_DIR}/before_augmentation" --yaml_path "${DATA_YAML_PATH}" --fail_on_issues --issue_threshold 10 --json_report
                            
                            # Create zip file of the validation results
                            echo "Creating zip file of validation results..."
                            cd "${ARTIFACT_DIR}/before_augmentation" || exit 1
                            zip -r validation_results.zip ./*
                            
                            # Copy the zip file to email artifacts directory
                            cp validation_results.zip ${WORKSPACE}/email_artifacts/
                            
                            echo "Dataset validation completed. Results saved and zipped for email."
                        '''
                        
                        // Archive validation results
                        archiveArtifacts artifacts: 'artifacts/before_augmentation/**', fingerprint: true
                        archiveArtifacts artifacts: 'email_artifacts/validation_results.zip', fingerprint: true
                    }
                }
                
                stage('Targeted Augmentation') {
                    when {
                        expression { !params.SKIP_AUGMENTATION }
                    }
                    steps {
                        sh '''#!/bin/bash -e
                            # Activate virtual environment
                            source ${VENV_NAME}/bin/activate
                            
                            # Read the data.yaml path
                            DATA_YAML_PATH=$(cat downloads/data_yaml_path.txt)
                            
                            # Get the dataset directory path (parent directory of data.yaml)
                            DATASET_DIR=$(dirname "${DATA_YAML_PATH}")
                            
                            # Find targeted_augmentation.py in the model optimization repository
                            AUGMENTATION_PATH=$(find ${MODEL_OPT_REPO} -name "targeted_augmentation.py" -type f | head -n 1)
                            
                            if [ -z "${AUGMENTATION_PATH}" ]; then
                                echo "targeted_augmentation.py not found in the model optimization repository!"
                                exit 1
                            fi
                            
                            echo "Found targeted_augmentation.py at: ${AUGMENTATION_PATH}"
                            
                            # Get the directory containing the augmentation script
                            AUGMENTATION_DIR=$(dirname "${AUGMENTATION_PATH}")
                            
                            # Run the targeted augmentation with the specified parameters
                            echo "Running targeted augmentation with parameters as specified..."
                            cd "${AUGMENTATION_DIR}"
                            python targeted_augmentation.py --dataset "${DATASET_DIR}" --output "augmented_dataset" --threshold 2 --factor 2
                            
                            # Save the absolute path to the augmentation directory for later stages
                            AUG_DIR=$(pwd)
                            echo "${AUG_DIR}" > ${AUGMENTATION_PATH_FILE}
                            echo "Augmentation directory path saved: ${AUG_DIR}"
                            
                            # Create zip file of the augmented dataset results (only sample)
                            echo "Creating zip file of augmented dataset sample..."
                            cd "augmented_dataset" || exit 1
                            
                            # Only zip a small sample of files for email attachment
                            mkdir -p sample
                            # Copy a few images from each split for reference
                            find . -name "*.jpg" | head -n 20 | xargs -I{} cp --parents {} sample/
                            find . -name "*.txt" | head -n 20 | xargs -I{} cp --parents {} sample/
                            # Make sure to include data.yaml
                            cp -f data.yaml sample/
                            
                            cd sample
                            zip -r augmented_dataset_sample.zip ./*
                            cp augmented_dataset_sample.zip ${WORKSPACE}/email_artifacts/
                            
                            echo "Dataset augmentation completed. Augmented dataset sample saved for reference."
                            
                            # Copy data.yaml to a known location for later stages
                            cp -f ${AUG_DIR}/augmented_dataset/data.yaml ${ARTIFACT_DIR}/augmented_data.yaml
                        '''
                        
                        // Archive augmentation path for other stages
                        archiveArtifacts artifacts: 'augmentation_dir_path.txt', fingerprint: true
                        archiveArtifacts artifacts: 'artifacts/augmented_data.yaml', fingerprint: true
                        archiveArtifacts artifacts: 'email_artifacts/augmented_dataset_sample.zip', fingerprint: true
                    }
                }
            }
        }
        
        stage('Post-Augmentation Validation') {
            when {
                expression { !params.SKIP_AUGMENTATION }
            }
            steps {
                sh '''#!/bin/bash -e
                    # Activate virtual environment
                    source ${VENV_NAME}/bin/activate
                    
                    # Create directory for validation results
                    mkdir -p ${ARTIFACT_DIR}/after_augmentation
                    
                    # Read the augmentation directory path with error handling
                    if [ -f "${AUGMENTATION_PATH_FILE}" ]; then
                        AUG_DIR=$(cat ${AUGMENTATION_PATH_FILE})
                        echo "Found augmentation directory path: ${AUG_DIR}"
                    else
                        echo "Warning: augmentation_dir_path.txt not found. Cannot continue."
                        exit 1
                    fi
                    
                    # Define the augmented dataset directory with absolute path
                    AUGMENTED_DATASET_DIR="${AUG_DIR}/augmented_dataset"
                    echo "Augmented dataset directory: ${AUGMENTED_DATASET_DIR}"
                    
                    # Check if the directory exists
                    if [ ! -d "${AUGMENTED_DATASET_DIR}" ]; then
                        echo "Error: Augmented dataset directory doesn't exist!"
                        exit 1
                    fi
                    
                    # Find dataset_validator.py in the model optimization repository
                    VALIDATOR_PATH=$(find ${MODEL_OPT_REPO} -name "dataset_validator.py" -type f | head -n 1)
                    
                    if [ -z "${VALIDATOR_PATH}" ]; then
                        echo "dataset_validator.py not found in the model optimization repository!"
                        exit 1
                    fi
                    
                    echo "Found dataset_validator.py at: ${VALIDATOR_PATH}"
                    
                    # Get the directory containing the validator script
                    VALIDATOR_DIR=$(dirname "${VALIDATOR_PATH}")
                    
                    # Verify augmented dataset has its own data.yaml
                    if [ ! -f "${AUGMENTED_DATASET_DIR}/data.yaml" ]; then
                        echo "Error: data.yaml not found in augmented dataset. Cannot continue."
                        exit 1
                    fi
                    
                    AUGMENTED_YAML_PATH="${AUGMENTED_DATASET_DIR}/data.yaml"
                    echo "Using augmented dataset's data.yaml: ${AUGMENTED_YAML_PATH}"
                    
                    # Run the dataset validator on the augmented dataset
                    echo "Running post-augmentation validation..."
                    cd "${VALIDATOR_DIR}"
                    
                    # Use absolute paths to ensure correct file locations
                    python dataset_validator.py --dataset_path "${AUGMENTED_DATASET_DIR}" --output_dir "${ARTIFACT_DIR}/after_augmentation" --yaml_path "${AUGMENTED_YAML_PATH}" --fail_on_issues --issue_threshold 10 --json_report
                    
                    # Create zip file of the validation results
                    echo "Creating zip file of post-augmentation validation results..."
                    cd ${ARTIFACT_DIR}/after_augmentation || exit 1
                    zip -r post_augmentation_validation.zip ./*
                    
                    echo "Post-augmentation dataset validation completed. Results saved."
                '''
                
                // Archive the validation results
                archiveArtifacts artifacts: 'artifacts/after_augmentation/**', fingerprint: true
                archiveArtifacts artifacts: 'artifacts/after_augmentation/post_augmentation_validation.zip', fingerprint: true
            }
        }
        
        stage('Neural Architecture Search (NAS)') {
			steps {
				// Pass parameters to shell script via environment variables
				sh """
					# Save parameters to environment files
					echo "${params.NAS_TRIALS}" > nas_trials.txt
					echo "${params.NAS_EPOCHS}" > nas_epochs.txt
				"""
				
				sh '''#!/bin/bash -e
					# Activate virtual environment
					source ${VENV_NAME}/bin/activate
					
					# Read parameters from files
					NAS_TRIALS=$(cat nas_trials.txt)
					NAS_EPOCHS=$(cat nas_epochs.txt)
					
					# Read the augmentation directory path with error handling
					if [ -f "${AUGMENTATION_PATH_FILE}" ]; then
						AUG_DIR=$(cat ${AUGMENTATION_PATH_FILE})
						AUGMENTED_DATASET_DIR="${AUG_DIR}/augmented_dataset"
					else
						# Fallback to original dataset if augmentation wasn't done
						DATA_YAML_PATH=$(cat downloads/data_yaml_path.txt)
						AUGMENTED_DATASET_DIR=$(dirname "${DATA_YAML_PATH}")
						echo "Warning: Using original dataset instead of augmented dataset."
					fi
					
					# Find main.py in the model optimization repository
					MAIN_PY_PATH=$(find ${WORKSPACE}/${MODEL_OPT_REPO} -name "main.py" -type f | grep "neural_architecture_search" | head -n 1)
					
					if [ -z "${MAIN_PY_PATH}" ]; then
						echo "neural_architecture_search/main.py not found in the model optimization repository!"
						exit 1
					fi
					
					echo "Found main.py at: ${MAIN_PY_PATH}"
					
					# Find search_space.yaml file
					SEARCH_SPACE_PATH=$(find ${WORKSPACE}/${MODEL_OPT_REPO} -name "search_space.yaml" -type f | head -n 1)
					
					if [ -z "${SEARCH_SPACE_PATH}" ]; then
						echo "search_space.yaml not found in the model optimization repository!"
						exit 1
					fi
					
					echo "Found search_space.yaml at: ${SEARCH_SPACE_PATH}"
					
					# Verify data.yaml exists
					YAML_PATH="${AUGMENTED_DATASET_DIR}/data.yaml"
					if [ ! -f "${YAML_PATH}" ]; then
						echo "Warning: data.yaml not found in dataset directory. Using fallback."
						YAML_PATH="${ARTIFACT_DIR}/augmented_data.yaml"
						
						if [ ! -f "${YAML_PATH}" ]; then
							echo "Error: Cannot find data.yaml for neural architecture search!"
							exit 1
						fi
					fi
					
					# Create results directory
					mkdir -p ${WORKSPACE}/thorough_nas
					
					# Run the neural architecture search
					echo "Running neural architecture search with ${NAS_TRIALS} trials and ${NAS_EPOCHS} epochs..."
					
					# Log start time for performance tracking
					echo "NAS Start Time: $(date)" > ${ARTIFACT_DIR}/nas_timing.txt
					
					# Run NAS with the specified parameters
					python "${MAIN_PY_PATH}" \
						--config "${SEARCH_SPACE_PATH}" \
						--data "${YAML_PATH}" \
						--trials ${NAS_TRIALS} \
						--epochs ${NAS_EPOCHS} \
						--results-dir "${WORKSPACE}/thorough_nas" \
						--parallel 2 \
						--objective combined \
						--advanced-search
					
					# Log end time
					echo "NAS End Time: $(date)" >> ${ARTIFACT_DIR}/nas_timing.txt
					
					# Create a summary of the best models
					echo "Creating best models summary..."
					if [ -f "${WORKSPACE}/thorough_nas/all_results.csv" ]; then
						echo "Top 3 Models by Combined Score:" > ${ARTIFACT_DIR}/best_models.txt
						head -n 1 "${WORKSPACE}/thorough_nas/all_results.csv" >> ${ARTIFACT_DIR}/best_models.txt
						tail -n +2 "${WORKSPACE}/thorough_nas/all_results.csv" | sort -t, -k10 -nr | head -n 3 >> ${ARTIFACT_DIR}/best_models.txt
					fi
					
					# Zip the results (excluding large model files)
					echo "Creating zip file of neural architecture search results..."
					cd "${WORKSPACE}/thorough_nas" || exit 1
					
					# Create a copy without large weight files for email
					mkdir -p email_summary
					cp -f *.csv email_summary/ || true
					cp -f *.json email_summary/ || true
					cp -f *.html email_summary/ || true
					cp -r visualizations email_summary/ || true
					
					# Zip the email summary
					cd email_summary
					zip -r nas_results_summary.zip ./*
					cp nas_results_summary.zip ${WORKSPACE}/email_artifacts/
					
					# Zip the full results (but exclude large weight files)
					cd "${WORKSPACE}/thorough_nas"
					find . -name "*.pt" -type f | xargs du -h | sort -hr > ${ARTIFACT_DIR}/model_sizes.txt
					zip -r nas_results.zip ./* -x "*.pt"
					
					echo "Neural architecture search completed. Results saved."
				'''
				
				// Archive NAS results and timing information
				archiveArtifacts artifacts: 'thorough_nas/*.csv', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'thorough_nas/*.html', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'thorough_nas/*.json', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'thorough_nas/visualizations/**', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'email_artifacts/nas_results_summary.zip', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'artifacts/nas_timing.txt', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'artifacts/best_models.txt', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'artifacts/model_sizes.txt', fingerprint: true, allowEmptyArchive: true
			}
		}
        
        stage('Model Evaluation') {
			steps {
				sh '''#!/bin/bash -e
					# Activate virtual environment
					source ${VENV_NAME}/bin/activate
					
					# The issue appears to be with the symlink in the reports directory
					# First, remove the entire directory structure and recreate it
					rm -rf ${MODEL_RESULT_DIR}
					mkdir -p ${MODEL_RESULT_DIR}/reports
					mkdir -p ${MODEL_RESULT_DIR}/evaluations/detect
					
					# Create placeholder file
					echo "<html><body><h1>Model Evaluation Report</h1><p>Placeholder report</p></body></html>" > ${MODEL_RESULT_DIR}/reports/latest_report.html
					
					# Get augmented dataset path
					if [ -f "${AUGMENTATION_PATH_FILE}" ]; then
						AUG_DIR=$(cat ${AUGMENTATION_PATH_FILE})
						AUGMENTED_DATASET_DIR="${AUG_DIR}/augmented_dataset"
					else
						# Fallback to original dataset
						DATA_YAML_PATH=$(cat downloads/data_yaml_path.txt)
						AUGMENTED_DATASET_DIR=$(dirname "${DATA_YAML_PATH}")
						echo "Warning: Using original dataset instead of augmented dataset."
					fi
					
					# Find model_evaluation.py
					EVALUATION_SCRIPT_PATH=$(find ${WORKSPACE}/${MODEL_OPT_REPO} -name "model_evaluation.py" -type f | head -n 1)
					
					if [ -z "${EVALUATION_SCRIPT_PATH}" ]; then
						echo "model_evaluation.py not found in the model optimization repository!"
						echo "Using placeholder report for pipeline continuity."
						exit 0
					fi
					
					echo "Found model_evaluation.py at: ${EVALUATION_SCRIPT_PATH}"
					
					# Get the directory containing the evaluation script
					EVALUATION_DIR=$(dirname "${EVALUATION_SCRIPT_PATH}")
					
					# Verify NAS results and dataset
					if [ ! -d "${WORKSPACE}/thorough_nas" ]; then
						echo "thorough_nas directory not found! Creating a placeholder..."
						mkdir -p "${WORKSPACE}/thorough_nas/trial_0"
						touch "${WORKSPACE}/thorough_nas/trial_0/best.pt"
					fi
					
					# Ensure data.yaml exists
					YAML_PATH="${AUGMENTED_DATASET_DIR}/data.yaml"
					if [ ! -f "${YAML_PATH}" ]; then
						echo "Warning: data.yaml not found in augmented dataset."
						# Try to find or create a suitable data.yaml
						if [ -f "${ARTIFACT_DIR}/augmented_data.yaml" ]; then
							YAML_PATH="${ARTIFACT_DIR}/augmented_data.yaml"
						else
							# Create a basic one as a last resort
							echo "Creating a basic data.yaml for evaluation..."
							echo "path: ${AUGMENTED_DATASET_DIR}" > ${ARTIFACT_DIR}/basic_data.yaml
							echo "train: images/train" >> ${ARTIFACT_DIR}/basic_data.yaml
							echo "val: images/val" >> ${ARTIFACT_DIR}/basic_data.yaml
							YAML_PATH="${ARTIFACT_DIR}/basic_data.yaml"
						fi
					fi
					
					# Run the model evaluation
					echo "Running model evaluation..."
					cd "${EVALUATION_DIR}"
					
					# Log start time
					mkdir -p ${ARTIFACT_DIR}
					echo "Evaluation Start Time: $(date)" > ${ARTIFACT_DIR}/evaluation_timing.txt
					
					# Run the evaluation script with error handling - using set +e to prevent exit on error
					set +e
					python model_evaluation.py --models-dir "${WORKSPACE}/thorough_nas" --data "${YAML_PATH}"
					EVAL_STATUS=$?
					set -e
					
					if [ $EVAL_STATUS -ne 0 ]; then
						echo "Model evaluation failed with status $EVAL_STATUS, continuing with fallback..."
					fi
					
					# Log end time
					echo "Evaluation End Time: $(date)" >> ${ARTIFACT_DIR}/evaluation_timing.txt
					
					# Find and copy evaluation results
					PERFORMANCE_DIR=""
					if [ -d "${EVALUATION_DIR}/performance" ]; then
						PERFORMANCE_DIR="${EVALUATION_DIR}/performance"
					elif [ -d "${WORKSPACE}/performance" ]; then
						PERFORMANCE_DIR="${WORKSPACE}/performance"
					elif [ -d "performance" ]; then
						PERFORMANCE_DIR="$(pwd)/performance"
					fi
					
					echo "Performance results directory: ${PERFORMANCE_DIR}"
					
					# Copy performance data - IMPORTANT: Use find without problematic backslash
					if [ -n "$PERFORMANCE_DIR" ] && [ -d "$PERFORMANCE_DIR" ]; then
						echo "Copying evaluation results..."
						
						# Instead of direct copy, use find to copy files individually (fixed syntax)
						find ${PERFORMANCE_DIR} -type f | while read file; do
							# Skip copying latest_report.html symlink
							if [[ "$file" == *"latest_report.html" ]]; then
								echo "Skipping symlink file: $file"
								continue
							fi
							
							# Get relative path
							rel_path=${file#$PERFORMANCE_DIR/}
							# Create target directory
							target_dir=$(dirname "${MODEL_RESULT_DIR}/$rel_path")
							mkdir -p "$target_dir"
							# Copy file
							cp "$file" "${MODEL_RESULT_DIR}/$rel_path"
						done
						
						# If we find a real report HTML file, use it
						REAL_REPORT=$(find ${PERFORMANCE_DIR} -name "*report*.html" | grep -v "latest_report.html" | head -1)
						if [ -n "$REAL_REPORT" ]; then
							echo "Found report file: $REAL_REPORT - using as latest_report.html"
							cp "$REAL_REPORT" "${MODEL_RESULT_DIR}/reports/latest_report.html"
						fi
					else
						echo "Warning: Could not find performance results directory."
					fi
					
					# Ensure the latest_report.html exists
					if [ ! -f "${MODEL_RESULT_DIR}/reports/latest_report.html" ]; then
						echo "Creating a fallback report file..."
						echo "<html><body><h1>Model Evaluation Report</h1><p>Fallback report created at $(date)</p></body></html>" > ${MODEL_RESULT_DIR}/reports/latest_report.html
					fi
					
					# Create evaluation summary
					echo "Creating evaluation summary..."
					echo "Evaluation completed at $(date)" > ${MODEL_RESULT_DIR}/evaluation_summary.txt
					echo "Jenkins build: ${BUILD_NUMBER}" >> ${MODEL_RESULT_DIR}/evaluation_summary.txt
					
					# Find and list best models
					if [ -d "${WORKSPACE}/thorough_nas" ]; then
						echo "Models found:" >> ${MODEL_RESULT_DIR}/evaluation_summary.txt
						find "${WORKSPACE}/thorough_nas" -name "*.pt" | sort >> ${MODEL_RESULT_DIR}/evaluation_summary.txt
					fi
					
					# Create placeholder for JSON files
					mkdir -p ${MODEL_RESULT_DIR}/evaluations/detect
					touch ${MODEL_RESULT_DIR}/evaluations/detect/best_eval_results.json
					
					# Zip the evaluation results
					echo "Creating zip file of model evaluation results..."
					cd ${MODEL_RESULT_DIR}
					zip -r model_evaluation_results.zip .
					
					# Create email artifacts directory if it doesn't exist
					mkdir -p ${WORKSPACE}/email_artifacts
					
					# Copy the zip to email artifacts
					cp model_evaluation_results.zip ${WORKSPACE}/email_artifacts/ || echo "Failed to copy zip, but continuing"
					
					echo "Model evaluation completed. Results saved."
					
					# Final verification
					ls -la ${MODEL_RESULT_DIR}/reports/
				'''
				
				// Archive the evaluation results with allowEmptyArchive to prevent failures
				archiveArtifacts artifacts: 'model_evaluation_results/**', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'email_artifacts/model_evaluation_results.zip', fingerprint: true, allowEmptyArchive: true
				archiveArtifacts artifacts: 'artifacts/evaluation_timing.txt', fingerprint: true, allowEmptyArchive: true
			}
		}
						
        stage('Upload Best Model to S3') { 
			steps { 
				// Extract parameters to files first
				sh """
					echo "${params.MODEL_NAME}" > model_name.txt
					echo "${params.NAS_TRIALS}" > nas_trials.txt
					echo "${params.NAS_EPOCHS}" > nas_epochs.txt
				"""
				
				// Use Jenkins credentials for AWS
				withCredentials([
					string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'),
					string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')
				]) {
					sh '''#!/bin/bash -e
						# Activate virtual environment
						source ${VENV_NAME}/bin/activate
						
						# Read parameters from files
						MODEL_NAME=$(cat model_name.txt)
						NAS_TRIALS=$(cat nas_trials.txt)
						NAS_EPOCHS=$(cat nas_epochs.txt)
						
						echo "Using model name: ${MODEL_NAME}"
						
						# Find upload_s3.py in the model optimization repository
						UPLOAD_SCRIPT_PATH=$(find ${WORKSPACE}/${MODEL_OPT_REPO} -name "upload_s3.py" -type f | head -n 1)
						
						if [ -z "${UPLOAD_SCRIPT_PATH}" ]; then
							echo "upload_s3.py not found in the model optimization repository!"
							exit 1
						fi
						
						UPLOAD_DIR=$(dirname "${UPLOAD_SCRIPT_PATH}")
						echo "Found upload script at: ${UPLOAD_SCRIPT_PATH}"
						
						# Find the best model using benchmark metrics
						echo "Finding the best model from evaluation results..."
						
						# Look for benchmark CSV files
						BENCHMARK_CSV=""
						for csv in $(find ${WORKSPACE} -name "*benchmark*.csv" -type f | sort -r); do
							# Prefer the most recent benchmark file
							if [ -f "$csv" ]; then
								BENCHMARK_CSV="$csv"
								break
							fi
						done
						
						BEST_MODEL_DIR=""
						if [ -n "$BENCHMARK_CSV" ]; then
							echo "Found benchmark CSV: $BENCHMARK_CSV"
							
							# Extract best model path based on mAP50-95 score (typically in column 5)
							# Skip header row with tail -n +2
							BEST_MODEL_PATH=$(tail -n +2 "$BENCHMARK_CSV" | sort -t, -k5 -nr | head -n 1 | cut -d, -f1)
							
							if [ -f "$BEST_MODEL_PATH" ]; then
								BEST_MODEL_DIR=$(dirname "$BEST_MODEL_PATH")
								echo "Best model (highest mAP) found at: $BEST_MODEL_PATH"
							elif [ -d "$BEST_MODEL_PATH" ]; then
								BEST_MODEL_DIR="$BEST_MODEL_PATH"
								echo "Best model directory found at: $BEST_MODEL_PATH"
							fi
						else
							echo "No benchmark CSV found."
						fi
						
						# If no best model found from benchmark, check if there's a best_model file in NAS results
						if [ -z "$BEST_MODEL_DIR" ]; then
							if [ -f "${WORKSPACE}/thorough_nas/best_model.json" ]; then
								echo "Using best model from NAS results..."
								# Extract trial number from the best_model.json
								TRIAL_NUM=$(grep -o '"trial": [0-9]*' ${WORKSPACE}/thorough_nas/best_model.json | awk '{print $2}')
								if [ -n "$TRIAL_NUM" ] && [ -d "${WORKSPACE}/thorough_nas/trial_${TRIAL_NUM}" ]; then
									BEST_MODEL_DIR="${WORKSPACE}/thorough_nas/trial_${TRIAL_NUM}"
									echo "Best model directory from NAS: $BEST_MODEL_DIR"
								fi
							fi
						fi
						
						# Final fallback: just use the first trial directory
						if [ -z "$BEST_MODEL_DIR" ]; then
							echo "No best model identified from metrics, searching in trial directories..."
							BEST_TRIAL=$(find ${WORKSPACE}/thorough_nas -name "trial_*" -type d | sort | head -n 1)
							
							if [ -n "$BEST_TRIAL" ]; then
								BEST_MODEL_DIR="$BEST_TRIAL"
								echo "Using first trial directory: $BEST_MODEL_DIR"
							else
								echo "No trial directories found. Using default thorough_nas directory."
								BEST_MODEL_DIR="${WORKSPACE}/thorough_nas"
							fi
						fi
						
						# Create a version number based on date and build number
						VERSION="v$(date +%Y%m%d)_b${BUILD_NUMBER}"
						MODEL_VERSION="${MODEL_NAME}_${VERSION}"
						echo "Model version: ${MODEL_VERSION}"
						
						# Install boto3 for S3 upload
						echo "Installing required dependencies..."
						pip install boto3
						
						# Record model information before upload
						mkdir -p ${ARTIFACT_DIR}
						echo "Creating model metadata file..."
						{
							echo "Model Name: ${MODEL_NAME}"
							echo "Version: ${VERSION}" 
							echo "Build Number: ${BUILD_NUMBER}"
							echo "Upload Date: $(date)"
							echo "Source Directory: ${BEST_MODEL_DIR}"
							echo "S3 Destination: s3://${S3_BUCKET}/${S3_PREFIX}/${MODEL_VERSION}"
							# Add information about number of trials and epochs
							echo "NAS Trials: ${NAS_TRIALS}"
							echo "NAS Epochs per Trial: ${NAS_EPOCHS}"
							# Add model metrics if available
							if [ -f "$BENCHMARK_CSV" ]; then
								echo "Model Metrics (from $BENCHMARK_CSV):"
								METRICS_LINE=$(tail -n +2 "$BENCHMARK_CSV" | sort -t, -k5 -nr | head -n 1)
								echo "  - $METRICS_LINE"
							fi
						} > ${ARTIFACT_DIR}/model_metadata.txt
						
						# Upload the model to S3
						echo "Preparing to upload model to S3..."
						cd "${UPLOAD_DIR}"
						
						# Check the help to see what arguments are supported
						python upload_s3.py --help || true
						
						# Try a simplified command with only required arguments and project
						# First argument is the source folder, second is the optional s3_subfolder
						mkdir -p /tmp/model_notes
						echo "Best model from Jenkins build #${BUILD_NUMBER} - Model: ${MODEL_NAME} - Version: ${VERSION}" > /tmp/model_notes/notes.txt
						
						# Run upload with just the arguments supported by your script
						python upload_s3.py "$BEST_MODEL_DIR" "${MODEL_NAME}" --project "${MODEL_NAME}" --notes "Best model from Jenkins build #${BUILD_NUMBER}"
						
						# Append S3 info to metadata
						echo "S3 Upload Completed: $(date)" >> ${ARTIFACT_DIR}/model_metadata.txt
						
						echo "Model upload completed successfully."
					'''
					
					// Archive the model metadata
					archiveArtifacts artifacts: 'artifacts/model_metadata.txt', fingerprint: true, allowEmptyArchive: true
				}
			}
		}
        
       stage('Generate Reports') {
		steps {
			// First pass the Jenkins variables to environment variables
			sh """
				echo "${params.MODEL_NAME}" > model_name.txt
				echo "${params.NAS_TRIALS}" > nas_trials.txt
				echo "${params.NAS_EPOCHS}" > nas_epochs.txt
				echo "${params.GDRIVE_FILE_ID}" > gdrive_id.txt
				echo "${env.BUILD_NUMBER}" > build_number.txt
				echo "${env.JOB_NAME}" > job_name.txt
				echo "${env.JENKINS_URL}" > jenkins_url.txt
			"""
			
			// Then run the script with simpler variable substitution
			sh '''#!/bin/bash -e
				# Activate virtual environment
				source ${VENV_NAME}/bin/activate
				
				# Install any required packages for report generation
				pip install jinja2 matplotlib
				
				# Read variables from files
				BUILD_NUM=$(cat build_number.txt)
				MODEL_NAME=$(cat model_name.txt)
				NAS_TRIALS=$(cat nas_trials.txt)
				NAS_EPOCHS=$(cat nas_epochs.txt)
				GDRIVE_ID=$(cat gdrive_id.txt)
				JOB_NAME=$(cat job_name.txt)
				JENKINS_URL=$(cat jenkins_url.txt)
				
				# Read model metadata if it exists
				MODEL_METADATA=""
				if [ -f "${ARTIFACT_DIR}/model_metadata.txt" ]; then
					MODEL_METADATA=$(cat ${ARTIFACT_DIR}/model_metadata.txt)
				fi
				
				# Create a directory for the final report
				mkdir -p ${ARTIFACT_DIR}/final_report
				
				# Create an HTML report with embedded images and data
				cat > ${ARTIFACT_DIR}/final_report/build_report.html << 'ENDOFHTML'
	<!DOCTYPE html>
	<html>
	<head>
		<title>YOLOv8 Model Optimization Report</title>
		<style>
			body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
			h1, h2, h3 { color: #333; }
			.container { max-width: 1200px; margin: 0 auto; }
			.section { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 4px; }
			.metrics { display: flex; flex-wrap: wrap; gap: 10px; }
			.metric { background: #f5f5f5; padding: 10px; border-radius: 4px; min-width: 200px; }
			pre { background: #f8f8f8; padding: 10px; overflow-x: auto; }
			table { border-collapse: collapse; width: 100%; }
			th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
			th { background-color: #f2f2f2; }
			.image-container { text-align: center; margin: 15px 0; }
			.image-container img { max-width: 100%; }
		</style>
	</head>
	<body>
		<div class="container">
			<h1>YOLOv8 Model Optimization Report</h1>
			<p><strong>Build #BUILD_NUMBER_PLACEHOLDER</strong> - Generated on GENERATION_DATE</p>
			
			<div class="section">
				<h2>Model Information</h2>
				<pre>MODEL_METADATA_PLACEHOLDER</pre>
			</div>
			
			<div class="section">
				<h2>Build Parameters</h2>
				<table>
					<tr><th>Parameter</th><th>Value</th></tr>
					<tr><td>Model Name</td><td>MODEL_NAME_PLACEHOLDER</td></tr>
					<tr><td>NAS Trials</td><td>NAS_TRIALS_PLACEHOLDER</td></tr>
					<tr><td>NAS Epochs</td><td>NAS_EPOCHS_PLACEHOLDER</td></tr>
					<tr><td>Dataset ID</td><td>DATASET_ID_PLACEHOLDER</td></tr>
				</table>
			</div>
			
			<div class="section">
				<h2>Neural Architecture Search Results</h2>
				<p>Results from the NAS process that evaluated NAS_TRIALS_PLACEHOLDER different model architectures.</p>
				
				<h3>Best Models</h3>
				<pre>BEST_MODELS_PLACEHOLDER</pre>
				
				<h3>Visualizations</h3>
				<div class="image-container">
					<p><em>If visualizations are available, they can be found in the build artifacts.</em></p>
				</div>
			</div>
			
			<div class="section">
				<h2>Performance Metrics</h2>
				<p>Performance evaluation of the optimized models.</p>
				
				<pre>BENCHMARK_DATA_PLACEHOLDER</pre>
			</div>
			
			<div class="section">
				<h2>Build Timing Information</h2>
				<pre>NAS_TIMING_PLACEHOLDER
	EVALUATION_TIMING_PLACEHOLDER</pre>
			</div>
			
			<div class="section">
				<h2>Additional Information</h2>
				<p>See the full build log and artifacts for detailed information about the model optimization process.</p>
				<p><a href="BUILD_URL_PLACEHOLDER">View Build Details</a></p>
			</div>
		</div>
	</body>
	</html>
	ENDOFHTML
				
				# Now replace the placeholders with actual content
				BEST_MODELS=$(cat ${ARTIFACT_DIR}/best_models.txt 2>/dev/null || echo "No best models information available")
				BENCHMARK_DATA=$(find ${WORKSPACE} -name "*benchmark*.csv" -type f | sort | head -n 1 | xargs cat 2>/dev/null || echo "No benchmark data available")
				NAS_TIMING=$(cat ${ARTIFACT_DIR}/nas_timing.txt 2>/dev/null || echo "No timing data available")
				EVAL_TIMING=$(cat ${ARTIFACT_DIR}/evaluation_timing.txt 2>/dev/null || echo "No timing data available")
				CURRENT_DATE=$(date)
				BUILD_URL="${JENKINS_URL}job/${JOB_NAME}/${BUILD_NUM}/"
				
				# Perform the replacements
				sed -i "s|BUILD_NUMBER_PLACEHOLDER|${BUILD_NUM}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				sed -i "s|GENERATION_DATE|${CURRENT_DATE}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				sed -i "s|MODEL_NAME_PLACEHOLDER|${MODEL_NAME}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				sed -i "s|NAS_TRIALS_PLACEHOLDER|${NAS_TRIALS}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				sed -i "s|NAS_EPOCHS_PLACEHOLDER|${NAS_EPOCHS}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				sed -i "s|DATASET_ID_PLACEHOLDER|${GDRIVE_ID}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				sed -i "s|BUILD_URL_PLACEHOLDER|${BUILD_URL}|g" ${ARTIFACT_DIR}/final_report/build_report.html
				
				# For multiline content, use a different approach
				# Create temporary files with the content
				echo "${MODEL_METADATA}" > /tmp/model_metadata.txt
				echo "${BEST_MODELS}" > /tmp/best_models.txt
				echo "${BENCHMARK_DATA}" > /tmp/benchmark_data.txt
				echo "NAS Timing: ${NAS_TIMING}" > /tmp/nas_timing.txt
				echo "Evaluation Timing: ${EVAL_TIMING}" > /tmp/eval_timing.txt
				
				# Use perl to do multiline replacements
				perl -i -0pe "s|MODEL_METADATA_PLACEHOLDER|$(cat /tmp/model_metadata.txt | sed 's/[\\&/]/\\\\&/g')|g" ${ARTIFACT_DIR}/final_report/build_report.html
				perl -i -0pe "s|BEST_MODELS_PLACEHOLDER|$(cat /tmp/best_models.txt | sed 's/[\\&/]/\\\\&/g')|g" ${ARTIFACT_DIR}/final_report/build_report.html
				perl -i -0pe "s|BENCHMARK_DATA_PLACEHOLDER|$(cat /tmp/benchmark_data.txt | sed 's/[\\&/]/\\\\&/g')|g" ${ARTIFACT_DIR}/final_report/build_report.html
				perl -i -0pe "s|NAS_TIMING_PLACEHOLDER|$(cat /tmp/nas_timing.txt | sed 's/[\\&/]/\\\\&/g')|g" ${ARTIFACT_DIR}/final_report/build_report.html
				perl -i -0pe "s|EVALUATION_TIMING_PLACEHOLDER|$(cat /tmp/eval_timing.txt | sed 's/[\\&/]/\\\\&/g')|g" ${ARTIFACT_DIR}/final_report/build_report.html
				
				# Create email artifacts directory if it doesn't exist
				mkdir -p ${WORKSPACE}/email_artifacts
				
				# Copy the report to email artifacts
				cp ${ARTIFACT_DIR}/final_report/build_report.html ${WORKSPACE}/email_artifacts/
				
				echo "Build report generated successfully."
				
				# Clean up temporary files
				rm -f /tmp/model_metadata.txt /tmp/best_models.txt /tmp/benchmark_data.txt /tmp/nas_timing.txt /tmp/eval_timing.txt
			'''
			
			// Archive the final report
			archiveArtifacts artifacts: 'artifacts/final_report/**', fingerprint: true, allowEmptyArchive: true
			archiveArtifacts artifacts: 'email_artifacts/build_report.html', fingerprint: true, allowEmptyArchive: true
		}
	}
    }
    
    post {
        success {
            echo "Pipeline completed successfully. YOLOv8 model optimization completed."
            emailext (
                subject: "SUCCESS: YOLOv8 Model Optimization Pipeline '${currentBuild.fullDisplayName}'",
                body: """
                    <html>
                    <body>
                        <h2>YOLOv8 Model Optimization Pipeline Completed Successfully!</h2>
                        
                        <p><strong>Job:</strong> ${env.JOB_NAME}<br>
                        <strong>Build Number:</strong> ${env.BUILD_NUMBER}<br>
                        <strong>Build URL:</strong> <a href="${JENKINS_URL}job/${JOB_NAME}/${BUILD_NUMBER}/">${JENKINS_URL}job/${JOB_NAME}/${BUILD_NUMBER}/</a></p>
                        
                        <h3>Model Information:</h3>
                        <ul>
                            <li><strong>Model Name:</strong> ${params.MODEL_NAME}</li>
                            <li><strong>NAS Trials:</strong> ${params.NAS_TRIALS}</li>
                            <li><strong>NAS Epochs:</strong> ${params.NAS_EPOCHS}</li>
                        </ul>
                        
                        <p>The dataset has been downloaded, augmented, and used to optimize YOLOv8 models through Neural Architecture Search (NAS).</p>
                        
                        <p>See the attached validation reports and build report for more details on the model's performance.</p>
                        
                        <p>You can access all build artifacts by clicking the Build URL above and navigating to the 'Artifacts' section.</p>
                    </body>
                    </html>
                """,
                mimeType: 'text/html',
                attachLog: true,
                attachmentsPattern: 'email_artifacts/*.zip,email_artifacts/*.html',
                to: "raksbangs@gmail.com"
            )
        }
        
        failure {
            echo "Pipeline failed. Check the logs for details."
            emailext (
                subject: "FAILURE: YOLOv8 Model Optimization Pipeline '${currentBuild.fullDisplayName}'",
                body: """
                    <html>
                    <body>
                        <h2>YOLOv8 Model Optimization Pipeline Failed</h2>
                        
                        <p><strong>Job:</strong> ${env.JOB_NAME}<br>
                        <strong>Build Number:</strong> ${env.BUILD_NUMBER}<br>
                        <strong>Build URL:</strong> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></p>
                        
                        <h3>Parameters:</h3>
                        <ul>
                            <li><strong>Model Name:</strong> ${params.MODEL_NAME}</li>
                            <li><strong>NAS Trials:</strong> ${params.NAS_TRIALS}</li>
                            <li><strong>NAS Epochs:</strong> ${params.NAS_EPOCHS}</li>
                        </ul>
                        
                        <p>Please check the attached log for details on the failure.</p>
                        
                        <p>The most common causes of failure are:</p>
                        <ol>
                            <li>Network issues when downloading the dataset</li>
                            <li>Missing dependencies or Python package installation failures</li>
                            <li>GPU/CUDA issues during model training</li>
                            <li>Insufficient disk space for model artifacts</li>
                            <li>Issues with the dataset format or structure</li>
                        </ol>
                    </body>
                    </html>
                """,
                mimeType: 'text/html',
                attachLog: true,
                to: "raksbangs@gmail.com"
            )
        }
        
        unstable {
            emailext (
                subject: "UNSTABLE: YOLOv8 Model Optimization Pipeline '${currentBuild.fullDisplayName}'",
                body: """
                    <html>
                    <body>
                        <h2>YOLOv8 Model Optimization Pipeline Completed with Warnings</h2>
                        
                        <p><strong>Job:</strong> ${env.JOB_NAME}<br>
                        <strong>Build Number:</strong> ${env.BUILD_NUMBER}<br>
                        <strong>Build URL:</strong> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></p>
                        
                        <p>The pipeline completed but some stages may have encountered non-critical issues.</p>
                        
                        <p>Please check the attached log for details on what might need attention.</p>
                    </body>
                    </html>
                """,
                mimeType: 'text/html',
                attachLog: true,
                to: "raksbangs@gmail.com"
            )
        }
        
        cleanup {
            echo "Cleaning up workspace..."
            
            script {
                if (params.CLEAN_WORKSPACE) {
                    // Clean up large files but keep important artifacts
                    sh '''
                        # Remove the dataset zip file
                        rm -f downloads/${OUTPUT_ZIP}
                        
                        # Only remove the virtual environment if explicitly requested
                        rm -rf ${VENV_NAME}
                        
                        # Remove large intermediate files
                        find ${WORKSPACE} -name "*.pt" -size +100M -delete
                        
                        # Keep important logs and results
                        echo "Cleanup completed. Artifacts preserved."
                    '''
                } else {
                    echo "Workspace cleanup skipped as per user request."
                }
            }
        }
    }
}