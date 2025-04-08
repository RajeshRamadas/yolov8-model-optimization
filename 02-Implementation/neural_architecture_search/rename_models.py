# rename_models.py
"""
Script to rename model weights files by adding trial numbers to the filenames.
This can be run after the Neural Architecture Search completes.
"""

import os
import shutil
import argparse
import glob
from pathlib import Path


def rename_models(results_dir):
    """
    Rename model weight files in trial directories by adding trial numbers to filenames.
    
    Args:
        results_dir (str): Path to the results directory containing trial folders
    """
    # Get all trial directories
    trial_dirs = glob.glob(os.path.join(results_dir, "trial_*"))
    
    print(f"Found {len(trial_dirs)} trial directories in {results_dir}")
    
    renamed_count = 0
    for trial_dir in trial_dirs:
        # Extract trial number from directory name
        trial_id = os.path.basename(trial_dir).split('_')[1]
        
        # Check for weights directory
        weights_dir = os.path.join(trial_dir, "weights")
        if not os.path.exists(weights_dir):
            print(f"No weights directory found in {trial_dir}")
            continue
        
        # Rename best.pt to best_trial_{trial_id}.pt if it exists
        best_model_path = os.path.join(weights_dir, "best.pt")
        if os.path.exists(best_model_path):
            new_best_path = os.path.join(weights_dir, f"best_trial_{trial_id}.pt")
            if not os.path.exists(new_best_path):  # Only copy if target doesn't exist
                shutil.copy(best_model_path, new_best_path)
                print(f"Copied: {best_model_path} → {new_best_path}")
                renamed_count += 1
        
        # Rename last.pt to last_trial_{trial_id}.pt if it exists
        last_model_path = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_model_path):
            new_last_path = os.path.join(weights_dir, f"last_trial_{trial_id}.pt")
            if not os.path.exists(new_last_path):  # Only copy if target doesn't exist
                shutil.copy(last_model_path, new_last_path)
                print(f"Copied: {last_model_path} → {new_last_path}")
                renamed_count += 1
    
    print(f"Renamed {renamed_count} model files")
    
    # Find best model directory
    best_model_file = os.path.join(results_dir, "best_model.json")
    if os.path.exists(best_model_file):
        import json
        with open(best_model_file, 'r') as f:
            best_model_data = json.load(f)
        
        if 'trial_id' in best_model_data:
            best_trial_id = best_model_data['trial_id']
            best_trial_dir = os.path.join(results_dir, f"trial_{best_trial_id}")
            best_model_path = os.path.join(best_trial_dir, "weights", f"best_trial_{best_trial_id}.pt")
            
            # Create a copy in the main results directory
            if os.path.exists(os.path.join(best_trial_dir, "weights", "best.pt")):
                final_best_path = os.path.join(results_dir, "best_model.pt")
                shutil.copy(os.path.join(best_trial_dir, "weights", "best.pt"), final_best_path)
                print(f"Copied best model to: {final_best_path}")
                
                final_best_trial_path = os.path.join(results_dir, f"best_model_trial_{best_trial_id}.pt")
                shutil.copy(os.path.join(best_trial_dir, "weights", "best.pt"), final_best_trial_path)
                print(f"Copied best model with trial ID to: {final_best_trial_path}")


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Rename model weights files by adding trial numbers')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing NAS results')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rename_models(args.results_dir)