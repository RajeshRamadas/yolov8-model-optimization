import argparse
import itertools
import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('kd_combinations')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run multiple Knowledge Distillation training experiments')
    
    # Basic configuration
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--student_model', type=str, default='yolov8n.yaml', help='Student model configuration')
    parser.add_argument('--teacher_model', type=str, default='yolov8x.yaml', help='Teacher model configuration')
    parser.add_argument('--teacher_weights', type=str, help='Path to pre-trained teacher weights')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Training device (GPU ID or cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    
    # Run mode
    parser.add_argument('--skip_teacher', action='store_true', help='Skip teacher training (requires teacher_weights)')
    
    # Knowledge distillation parameters for grid search
    parser.add_argument('--alpha_values', type=float, nargs='+', default=[0.3, 0.5, 0.7], 
                        help='Alpha values to test (weight between hard and soft loss)')
    parser.add_argument('--temperature_values', type=float, nargs='+', default=[1.0, 2.0, 4.0],
                        help='Temperature values to test')
    
    # Early stopping parameter
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 to disable)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='kd_experiments', help='Directory to store experiment results')
    parser.add_argument('--log_file', type=str, default='experiments.log', help='Log file for all experiments')
    
    return parser.parse_args()

def run_experiment(args, alpha, temperature):
    """Run a single experiment with specific alpha and temperature values"""
    experiment_name = f"kd_a{alpha}_t{temperature}"
    
    # Build command
    cmd = [
        "python3", "knowledge_distillation.py",
        "--data", args.data,
        "--student_model", args.student_model,
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--device", args.device,
        "--workers", str(args.workers),
        "--alpha", str(alpha),
        "--temperature", str(temperature),
        "--student_name", experiment_name,
        "--patience", str(args.patience),  # Add patience parameter
    ]
    
    # Add optional arguments
    if args.teacher_model:
        cmd.extend(["--teacher_model", args.teacher_model])
    
    if args.teacher_weights:
        cmd.extend(["--teacher_weights", args.teacher_weights])
    
    if args.skip_teacher:
        cmd.append("--skip_teacher")
    
    # Log command
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Execute
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Stream output
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            logger.info(f"Experiment {experiment_name} completed successfully")
        else:
            logger.error(f"Experiment {experiment_name} failed with code {return_code}")
        
        return return_code
        
    except Exception as e:
        logger.exception(f"Error running experiment {experiment_name}")
        return 1

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(os.path.join(args.output_dir, args.log_file))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log configuration
    logger.info("Starting Knowledge Distillation grid search")
    logger.info(f"Alpha values: {args.alpha_values}")
    logger.info(f"Temperature values: {args.temperature_values}")
    logger.info(f"Patience: {args.patience}")
    
    # Generate parameter combinations
    combinations = list(itertools.product(args.alpha_values, args.temperature_values))
    logger.info(f"Will run {len(combinations)} experiments")
    
    # Run experiments
    for i, (alpha, temperature) in enumerate(combinations):
        logger.info(f"Experiment {i+1}/{len(combinations)}")
        run_experiment(args, alpha, temperature)
    
    logger.info("All experiments completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
