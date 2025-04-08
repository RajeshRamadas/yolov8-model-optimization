# analyzer.py
"""
Module for analyzing results from Neural Architecture Search.
"""

import os
import pandas as pd
from utils import save_json, convert_to_serializable
from visualization import create_visualizations, create_html_report


def analyze_results(trials_results, results_dir, objective='map'):
    """
    Analyze results from the NAS trials and find the best model.
    
    Args:
        trials_results: List of trial results
        results_dir: Directory containing results
        objective: Optimization objective ('map', 'latency', 'size', or 'combined')
        
    Returns:
        dict or None: Best model data, or None if no successful trials
    """
    if not trials_results:
        print("No successful trials to analyze")
        return None
    
    # Create DataFrame for analysis
    results_data = []
    for result in trials_results:
        if result is not None:
            data = {
                "trial_id": result["trial_id"],
                **{f"param_{k}": v for k, v in result["params"].items()},
                **{f"metric_{k}": v for k, v in result["metrics"].items() 
                  if isinstance(v, (int, float, bool, str))}
            }
            results_data.append(data)
    
    if not results_data:
        print("No valid results data")
        return None
    
    df = pd.DataFrame(results_data)
    
    # Save the full results table
    results_csv = os.path.join(results_dir, "all_results.csv")
    df.to_csv(results_csv, index=False)
    print(f"All results saved to {results_csv}")
    
    # Determine which metric to optimize
    if objective == 'map':
        metric_col = 'metric_map50_95'
        higher_is_better = True
    elif objective == 'latency':
        metric_col = 'metric_fps'
        higher_is_better = True
    elif objective == 'size':
        metric_col = 'metric_model_size_mb'
        higher_is_better = False
    else:  # combined
        metric_col = 'metric_combined_score'
        higher_is_better = True
    
    # Sort by the selected metric
    if higher_is_better:
        df_sorted = df.sort_values(by=metric_col, ascending=False)
    else:
        df_sorted = df.sort_values(by=metric_col, ascending=True)
    
    # Get the best model
    best_model = df_sorted.iloc[0].to_dict()
    
    # Extract parameters and metrics for the best model
    best_params = {k.replace('param_', ''): convert_to_serializable(v) for k, v in best_model.items() if k.startswith('param_')}
    best_metrics = {k.replace('metric_', ''): convert_to_serializable(v) for k, v in best_model.items() if k.startswith('metric_')}
    
    # Create visualizations
    viz_dir = create_visualizations(df, results_dir, metric_col)
    
    # Create HTML report
    create_html_report(df, results_dir, viz_dir, metric_col)
    
    return {
        "trial_id": convert_to_serializable(best_model["trial_id"]),
        "params": best_params,
        "metrics": best_metrics
    }