# main.py
"""
Main module for YOLOv8 Neural Architecture Search - FIXED VERSION.
"""

import os
import random
import argparse
import pandas as pd
import concurrent.futures
import shutil
import traceback
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
    parser = argparse.ArgumentParser(description='Advanced YOLOv8 Neural Architecture Search - FIXED')
    
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
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per trial in seconds (default: 3600)')
    
    return parser.parse_args()


def validate_data_yaml(data_yaml_path):
    """
    Validate that the data.yaml file exists and is accessible.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(data_yaml_path):
        print(f"❌ ERROR: Data YAML file not found: {data_yaml_path}")
        return False
    
    try:
        import yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['train', 'val', 'nc']
        missing_keys = [key for key in required_keys if key not in data_config]
        
        if missing_keys:
            print(f"❌ ERROR: Missing required keys in data.yaml: {missing_keys}")
            return False
        
        print(f"✅ Data YAML validated: {data_yaml_path}")
        print(f"   Classes: {data_config.get('nc', 'unknown')}")
        print(f"   Train: {data_config.get('train', 'unknown')}")
        print(f"   Val: {data_config.get('val', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Could not read data YAML: {e}")
        return False


def rename_model_files(results_dir):
    """
    Rename original model weight files by adding trial numbers to filenames.
    
    Args:
        results_dir (str): Path to the results directory
    """
    print("\n" + "="*60)
    print("RENAMING MODEL FILES")
    print("="*60)
    
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
            try:
                shutil.copy2(best_model, new_best_name)
                print(f"✅ Renamed: best.pt -> best_trial_{trial_id}.pt")
                renamed_count += 1
            except Exception as e:
                print(f"❌ Error renaming {best_model}: {e}")
            
        # Rename last.pt to last_trial_{trial_id}.pt if it exists
        last_model = os.path.join(weights_dir, "last.pt")
        if os.path.exists(last_model):
            new_last_name = os.path.join(weights_dir, f"last_trial_{trial_id}.pt")
            try:
                shutil.copy2(last_model, new_last_name)
                print(f"✅ Renamed: last.pt -> last_trial_{trial_id}.pt")
                renamed_count += 1
            except Exception as e:
                print(f"❌ Error renaming {last_model}: {e}")
    
    print(f"\n📊 Renamed {renamed_count} model files total")


def main():
    """Main function to run the Neural Architecture Search."""
    
    print("=" * 80)
    print("🚀 YOLOv8 NEURAL ARCHITECTURE SEARCH - FIXED VERSION")
    print("=" * 80)
    
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Validate data.yaml first
        if not validate_data_yaml(args.data):
            print("❌ Data validation failed. Exiting.")
            return 1
        
        # Load search configuration
        try:
            config = load_search_config(args.config)
            print(f"✅ Loaded search configuration: {args.config}")
        except Exception as e:
            print(f"❌ Error loading config {args.config}: {e}")
            print("Creating default configuration...")
            
            # Create default config if missing
            default_config = {
                'basic_search_space': {
                    'depth_multiple': [0.33, 0.5, 0.67],
                    'width_multiple': [0.25, 0.5, 0.75],
                    'img_size': [320, 640],
                    'kernel_size': [3, 5],
                    'model_type': ['yolov8n']
                },
                'defaults': {
                    'trials': 3,
                    'epochs': 5,
                    'results_dir': 'nas_results',
                    'parallel': 1,
                    'objective': 'combined'
                }
            }
            config = default_config
        
        # Load default arguments from config
        default_args = get_default_args(config)
        
        # Override defaults with command-line arguments if provided
        trials = args.trials if args.trials is not None else default_args['trials']
        epochs = args.epochs if args.epochs is not None else default_args['epochs']
        results_dir = args.results_dir if args.results_dir is not None else default_args['results_dir']
        parallel = args.parallel if args.parallel is not None else default_args['parallel']
        objective = args.objective if args.objective is not None else default_args['objective']
        advanced_search = args.advanced_search or default_args.get('advanced_search', False)
        
        # Validate parameters
        if trials <= 0:
            print(f"❌ Invalid number of trials: {trials}")
            return 1
        if epochs <= 0:
            print(f"❌ Invalid number of epochs: {epochs}")
            return 1
        
        # Get search space
        search_space = get_search_space(config, advanced_search)
        
        # Print configuration summary
        print(f"\n📋 SEARCH CONFIGURATION:")
        print(f"   Trials: {trials}")
        print(f"   Epochs per trial: {epochs}")
        print(f"   Results directory: {results_dir}")
        print(f"   Parallel workers: {parallel}")
        print(f"   Objective: {objective}")
        print(f"   Advanced search: {advanced_search}")
        print(f"   Search space parameters: {len(search_space)}")
        
        # Create results directory
        create_directories(results_dir)
        
        # Save search space configuration
        save_json(search_space, os.path.join(results_dir, "search_space.json"))
        
        # Generate trial configurations
        print(f"\n🎲 GENERATING {trials} TRIAL CONFIGURATIONS...")
        trial_configs = []
        for i in range(trials):
            # Sample parameters from search space
            params = {}
            for key, value in search_space.items():
                if isinstance(value, list):
                    params[key] = random.choice(value)
                else:
                    params[key] = value
            trial_configs.append((i, params))
            
            print(f"Trial {i}: {params}")
        
        # Save all trial configurations
        save_json([{"trial_id": i, "params": params} for i, params in trial_configs], 
                  os.path.join(results_dir, "trial_configs.json"))
        
        print(f"\n🚀 STARTING ARCHITECTURE SEARCH")
        print(f"Total trials: {trials}")
        print(f"Search space: {len(search_space)} parameters")
        print(f"Results directory: {results_dir}")
        
        # Run trials
        trials_results = []
        
        if parallel > 1:
            # Run trials in parallel
            print(f"\n⚡ Running {parallel} trials in parallel...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = []
                for trial_id, params in trial_configs:
                    future = executor.submit(
                        run_trial, trial_id, params, args.data, results_dir, epochs
                    )
                    futures.append(future)
                
                # Collect results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        result = future.result(timeout=args.timeout)
                        if result is not None:
                            trials_results.append(result)
                            print(f"✅ Completed {len(trials_results)}/{trials} trials")
                        else:
                            print(f"❌ Trial failed - {len(trials_results)}/{trials} completed")
                    except concurrent.futures.TimeoutError:
                        print(f"❌ Trial timed out after {args.timeout} seconds")
                    except Exception as e:
                        print(f"❌ Trial error: {e}")
        else:
            # Run trials sequentially
            print(f"\n🔄 Running trials sequentially...")
            for trial_id, params in trial_configs:
                try:
                    result = run_trial(trial_id, params, args.data, results_dir, epochs)
                    if result is not None:
                        trials_results.append(result)
                        print(f"✅ Completed {len(trials_results)}/{trials} trials")
                    else:
                        print(f"❌ Trial {trial_id} failed - {len(trials_results)}/{trials} completed")
                except Exception as e:
                    print(f"❌ Error in trial {trial_id}: {e}")
        
        print(f"\n📊 TRIAL EXECUTION SUMMARY:")
        print(f"   Total trials attempted: {trials}")
        print(f"   Successful trials: {len(trials_results)}")
        print(f"   Failed trials: {trials - len(trials_results)}")
        print(f"   Success rate: {len(trials_results)/trials*100:.1f}%")
        
        # Analyze results
        if trials_results:
            print(f"\n🔍 ANALYZING RESULTS...")
            best_model = analyze_results(trials_results, results_dir, objective)
            
            # Rename model files to include trial numbers
            if not args.no_rename:
                rename_model_files(results_dir)
            
            if best_model:
                trial_id = best_model['trial_id']
                print(f"\n🏆 BEST MODEL FOUND (Trial {trial_id}):")
                print("=" * 60)
                
                print("📋 Parameters:")
                for key, value in best_model['params'].items():
                    print(f"   {key}: {value}")
                
                print("\n📊 Metrics:")
                for key, value in best_model['metrics'].items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
                
                # Save best model information
                save_json(best_model, os.path.join(results_dir, "best_model.json"))
                
                # Copy the renamed best model to the root results directory
                if not args.no_rename:
                    best_weights_dir = os.path.join(results_dir, f"trial_{trial_id}", "weights")
                    renamed_best_model = os.path.join(best_weights_dir, f"best_trial_{trial_id}.pt")
                    if os.path.exists(renamed_best_model):
                        # Copy with descriptive name to root directory
                        best_model_name = os.path.join(results_dir, f"best_model_trial_{trial_id}.pt")
                        shutil.copy(renamed_best_model, best_model_name)
                        print(f"\n📁 Model files:")
                        print(f"   Best model copied to: {best_model_name}")
                
                print(f"\n📍 Model Locations:")
                if not args.no_rename:
                    print(f"   Renamed path: {results_dir}/trial_{trial_id}/weights/best_trial_{trial_id}.pt")
                    print(f"   Root copy: {results_dir}/best_model_trial_{trial_id}.pt")
                else:
                    print(f"   Standard path: {results_dir}/trial_{trial_id}/weights/best.pt")
                
                print(f"   Results report: {results_dir}/nas_report.html")
                
                # Final success message
                print(f"\n🎉 NEURAL ARCHITECTURE SEARCH COMPLETED SUCCESSFULLY!")
                print(f"   Best trial: {trial_id}")
                print(f"   Best mAP50-95: {best_model['metrics'].get('map50_95', 0):.4f}")
                print(f"   Best combined score: {best_model['metrics'].get('combined_score', 0):.4f}")
                
            else:
                print(f"\n❌ No best model could be determined from results")
                return 1
        else:
            print(f"\n❌ NO SUCCESSFUL TRIALS FOUND")
            print("Possible issues:")
            print("   - Dataset path incorrect")
            print("   - Insufficient GPU memory")
            print("   - Dependencies missing")
            print("   - Training timeout too short")
            print("\nCheck individual trial logs in the results directory for details.")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Search interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)