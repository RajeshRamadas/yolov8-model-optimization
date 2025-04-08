# main.py
"""
Main module for YOLOv8 Neural Architecture Search.
"""

import os
import random
import argparse
import pandas as pd
import concurrent.futures
import shutil
from pathlib import Path

from config_loader import load_search_config, get_search_space, get_default_args
from utils import save_json, create_directories
from trial_manager import run_trial
from analyzer import analyze_results


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Advanced YOLOv8 Neural Architecture Search')
    
    parser.add_argument('--config', type=str, default='search_space.yaml',
                        help='Path to search space configuration YAML file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--trials', type=int,
                        help='Number of trials to run')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs per trial')
    parser.add_argument('--results-dir', type=str,
                        help='Directory to save results')
    parser.add_argument('--parallel', type=int,
                        help='Number of trials to run in parallel (use with caution)')
    parser.add_argument('--objective', type=str,
                        choices=['map', 'latency', 'size', 'combined'],
                        help='Optimization objective')
    parser.add_argument('--advanced-search', action='store_true',
                        help='Use advanced search space with more parameters')
    parser.add_argument('--no-rename', action='store_true',
                        help='Skip renaming model files with trial numbers')
    
    return parser.parse_args()


def rename_model_files(results_dir):
    """
    Rename model weight files by adding trial numbers to filenames.
    
    Args:
        results_dir (str): Path to the results directory
    """
    print("\nRenaming model files to include trial numbers...")
    
    # Process each trial directory
    trial_dirs = [d for d in os.listdir(results_dir) if d.startswith("trial_")]
    renamed_count = 0
    
    for trial_dir in trial_dirs:
        # Extract trial number
        trial_id = trial_dir.split("_")[1]
        weights_dir = os.path.join(results_dir, trial_dir, "weights")
        
        if not os.path.exists(weights_dir):
            continue
            
        # Rename best.pt to best_trial_{trial_id}.pt if it exists
        best_model = os.path.join(weights_dir, "best.pt")
        if os.path.exists(best_model):
            new_best_name = os.path.join(weights_dir, f"best_trial_{trial_id}.pt")
            shutil.copy(best_model, new_best_name)
            renamed_count += 1
            
        # Rename last.pt to last_trial_{trial_id}.pt if it exists
        last_model = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_model):
            new_last_name = os.path.join(weights_dir, f"last_trial_{trial_id}.pt")
            shutil.copy(last_model, new_last_name)
            renamed_count += 1
    
    print(f"Renamed {renamed_count} model files")


def main():
    """Main function to run the Neural Architecture Search."""
    
    # Parse command-line arguments
    args = parse_args()
    
    # Load search configuration
    config = load_search_config(args.config)
    
    # Load default arguments from config
    default_args = get_default_args(config)
    
    # Override defaults with command-line arguments if provided
    trials = args.trials if args.trials is not None else default_args['trials']
    epochs = args.epochs if args.epochs is not None else default_args['epochs']
    results_dir = args.results_dir if args.results_dir is not None else default_args['results_dir']
    parallel = args.parallel if args.parallel is not None else default_args['parallel']
    objective = args.objective if args.objective is not None else default_args['objective']
    advanced_search = args.advanced_search or default_args.get('advanced_search', False)
    
    # Get search space
    search_space = get_search_space(config, advanced_search)
    
    # Create results directory
    create_directories(results_dir)
    
    # Save search space configuration
    save_json(search_space, os.path.join(results_dir, "search_space.json"))
    
    # Generate trial configurations
    trial_configs = []
    for i in range(trials):
        # Sample parameters from search space
        params = {key: random.choice(value) for key, value in search_space.items()}
        trial_configs.append((i, params))
    
    # Save all trial configurations
    save_json([{"trial_id": i, "params": params} for i, params in trial_configs], 
              os.path.join(results_dir, "trial_configs.json"))
    
    print(f"Starting architecture search with {trials} trials")
    print(f"Search space: {len(search_space)} parameters")
    print(f"Results directory: {results_dir}")
    
    # Run trials
    trials_results = []
    
    if parallel > 1:
        # Run trials in parallel
        print(f"Running {parallel} trials in parallel")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for trial_id, params in trial_configs:
                future = executor.submit(
                    run_trial, trial_id, params, args.data, results_dir, epochs
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    trials_results.append(result)
    else:
        # Run trials sequentially
        for trial_id, params in trial_configs:
            result = run_trial(trial_id, params, args.data, results_dir, epochs)
            if result is not None:
                trials_results.append(result)
    
    # Analyze results
    best_model = analyze_results(trials_results, results_dir, objective)
    
    # Rename model files to include trial numbers
    if not args.no_rename:
        rename_model_files(results_dir)
    
    if best_model:
        trial_id = best_model['trial_id']
        print("\n===== Architecture Search Results =====")
        print(f"Best model found (trial {trial_id}):")
        print("Parameters:")
        for key, value in best_model['params'].items():
            print(f"  {key}: {value}")
        print("Metrics:")
        for key, value in best_model['metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        # Save best model information
        save_json(best_model, os.path.join(results_dir, "best_model.json"))
        
        # Create a copy of the best model in the root results directory
        if not args.no_rename:
            best_weights_dir = os.path.join(results_dir, f"trial_{trial_id}", "weights")
            if os.path.exists(os.path.join(best_weights_dir, "best.pt")):
                # Copy with descriptive name
                best_model_name = os.path.join(results_dir, f"best_model_trial_{trial_id}.pt")
                shutil.copy(os.path.join(best_weights_dir, "best.pt"), best_model_name)
                print(f"Copied best model to: {best_model_name}")
        
        print(f"\nBest model saved at:")
        print(f"  Standard path: {results_dir}/trial_{trial_id}/weights/best.pt")
        if not args.no_rename:
            print(f"  With trial ID: {results_dir}/trial_{trial_id}/weights/best_trial_{trial_id}.pt")
            print(f"  Root copy:    {results_dir}/best_model_trial_{trial_id}.pt")
        print(f"Results report: {results_dir}/nas_report.html")
    else:
        print("No successful trials found")


if __name__ == "__main__":
    main()