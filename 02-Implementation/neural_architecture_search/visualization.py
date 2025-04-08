# visualization.py
"""
Module for creating visualizations and reports from Neural Architecture Search results.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import save_json, convert_to_serializable, CustomJSONEncoder


def create_visualizations(df, results_dir, metric_col):
    """
    Create visualizations of the NAS results.
    
    Args:
        df: DataFrame containing results
        results_dir: Directory to save visualizations
        metric_col: Column name of the metric to optimize
        
    Returns:
        str: Path to the visualizations directory
    """
    viz_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Depth vs Width scatter plot colored by the objective metric
    if 'param_depth_multiple' in df.columns and 'param_width_multiple' in df.columns:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['param_depth_multiple'], 
            df['param_width_multiple'],
            c=df[metric_col], 
            cmap='viridis', 
            s=100,
            alpha=0.7
        )
        plt.colorbar(scatter, label=metric_col.replace('metric_', ''))
        plt.xlabel('Depth Multiple')
        plt.ylabel('Width Multiple')
        plt.title('Impact of Depth and Width on Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'depth_width_scatter.png'), dpi=300)
        plt.close()
    
    # 2. Performance vs Model Size
    if 'metric_model_size_mb' in df.columns and metric_col != 'metric_model_size_mb':
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['metric_model_size_mb'], 
            df[metric_col],
            c=df['param_img_size'] if 'param_img_size' in df.columns else 'blue', 
            cmap='plasma', 
            s=80,
            alpha=0.8
        )
        if 'param_img_size' in df.columns:
            plt.colorbar(scatter, label='Image Size')
        plt.xlabel('Model Size (MB)')
        plt.ylabel(metric_col.replace('metric_', ''))
        plt.title('Performance vs Model Size')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'performance_vs_size.png'), dpi=300)
        plt.close()
    
    # 3. Performance vs Inference Speed
    if 'metric_fps' in df.columns and metric_col != 'metric_fps':
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df['metric_fps'], 
            df[metric_col],
            c=df['param_img_size'] if 'param_img_size' in df.columns else 'blue', 
            cmap='plasma', 
            s=80,
            alpha=0.8
        )
        if 'param_img_size' in df.columns:
            plt.colorbar(scatter, label='Image Size')
        plt.xlabel('Inference Speed (FPS)')
        plt.ylabel(metric_col.replace('metric_', ''))
        plt.title('Performance vs Inference Speed')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(viz_dir, 'performance_vs_speed.png'), dpi=300)
        plt.close()
        
    # 4. Kernel Size Impact (New Visualization)
    if 'param_kernel_size' in df.columns:
        try:
            # Group by kernel size and calculate mean of objective
            grouped = df.groupby('param_kernel_size')[metric_col].mean().reset_index()
            
            plt.figure(figsize=(10, 6))
            plt.bar(grouped['param_kernel_size'].astype(str), grouped[metric_col], color='lightblue')
            plt.xlabel('Kernel Size')
            plt.ylabel(metric_col.replace('metric_', ''))
            plt.title('Impact of Kernel Size on Performance')
            plt.grid(True, linestyle='--', alpha=0.5, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'kernel_size_impact.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not create kernel size impact plot: {e}")
    
    # 5. Parameter importance (correlation with objective)
    # Filter numeric parameters only for correlation calculation
    param_cols = [col for col in df.columns if col.startswith('param_')]
    numeric_param_cols = []
    correlations = []
    
    for col in param_cols:
        # Check if the parameter is numeric and has more than one unique value
        try:
            values = df[col].astype(float)
            # Skip if all values are the same (zero variance)
            if values.nunique() <= 1:
                continue
                
            # Skip if there are NaN values
            if values.isnull().any():
                continue
                
            numeric_param_cols.append(col)
            
            # Safely calculate correlation
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings
                corr = df[col].corr(df[metric_col])
                
            # Only include correlation if it's a valid number
            if np.isfinite(corr):
                correlations.append(corr)
            else:
                # If correlation is nan or inf, don't include this parameter
                numeric_param_cols.pop()
        except (ValueError, TypeError):
            # Skip non-numeric parameters
            continue
    
    # Create parameter importance plot for numeric parameters
    if numeric_param_cols and correlations:
        plt.figure(figsize=(12, 6))
        bars = plt.barh(
            [col.replace('param_', '') for col in numeric_param_cols], 
            correlations,
            color='skyblue'
        )
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.7)
        plt.xlabel('Correlation with Objective')
        plt.title('Parameter Importance (Numeric Parameters)')
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'parameter_importance.png'), dpi=300)
        plt.close()
    
    # 6. Categorical parameters analysis (if any)
    categorical_params = [col.replace('param_', '') for col in param_cols if col not in numeric_param_cols]
    
    for param in categorical_params:
        param_col = f'param_{param}'
        if param_col in df.columns:
            try:
                # Check if there are multiple unique values
                if df[param_col].nunique() <= 1:
                    continue
                    
                # Group by categorical parameter and calculate mean of objective
                grouped = df.groupby(param_col)[metric_col].mean().reset_index()
                
                if len(grouped) > 1:  # Only create plot if there are multiple categories
                    plt.figure(figsize=(10, 6))
                    plt.bar(grouped[param_col].astype(str), grouped[metric_col], color='lightgreen')
                    plt.xlabel(param)
                    plt.ylabel(metric_col.replace('metric_', ''))
                    plt.title(f'Impact of {param} on Performance')
                    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f'{param}_impact.png'), dpi=300)
                    plt.close()
            except Exception as e:
                print(f"Could not create plot for parameter {param}: {e}")
    
    return viz_dir

# This is a partial update focusing on the create_html_report functiondef create_html_report(df, results_dir, viz_dir, metric_col):    """    Create an HTML report summarizing the NAS results.        Args:        df: DataFrame containing results        results_dir: Directory to save the report        viz_dir: Directory containing visualizations        metric_col: Column name of the metric to optimize            Returns:        str: Path to the HTML report    """    html_file = os.path.join(results_dir, "nas_report.html")        # Get top 5 models    if metric_col == 'metric_model_size_mb':        # For size, smaller is better        top_models = df.sort_values(by=metric_col, ascending=True).head(5)    else:        # For other metrics, larger is better        top_models = df.sort_values(by=metric_col, ascending=False).head(5)        # Create HTML content    html_content = f"""    <!DOCTYPE html>    <html>    <head>        <title>YOLOv8 Neural Architecture Search Results</title>        <style>            body {{ font-family: Arial, sans-serif; margin: 20px; }}            h1, h2, h3 {{ color: #333; }}            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}            th {{ background-color: #f2f2f2; }}            tr:hover {{ background-color: #f5f5f5; }}            .visualization {{ margin: 20px 0; max-width: 100%; }}            .viz-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}            .viz-item {{ width: 48%; margin-bottom: 20px; }}            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}            .model-paths {{ background-color: #f8f8f8; padding: 12px; border-radius: 5px; margin-top: 10px; }}            .model-path {{ font-family: monospace; margin: 5px 0; }}            .top-models-table {{ margin-top: 20px; }}            .highlight {{ background-color: #fff8e1; }}        </style>    </head>    <body>        <h1>YOLOv8 Neural Architecture Search Results</h1>                <h2>Search Summary</h2>        <ul>            <li><strong>Total Trials:</strong> {len(df)}</li>            <li><strong>Optimization Objective:</strong> {metric_col.replace('metric_', '')}</li>            <li><strong>Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</li>        </ul>                <h2>Top Models</h2>        <table class="top-models-table">            <tr>                <th>Rank</th>                <th>Trial ID</th>                <th>mAP50-95</th>                <th>FPS</th>                <th>Size (MB)</th>    """        # Add optional columns for depth, width, img_size, and kernel_size if they exist    optional_columns = ['depth_multiple', 'width_multiple', 'img_size', 'kernel_size']    for column in optional_columns:        if f'param_{column}' in df.columns:            html_content += f"<th>{column.replace('_', ' ').title()}</th>\n"        html_content += """                <th>Model Paths</th>            </tr>    """        # Add rows for top models    for i, (idx, row) in enumerate(top_models.iterrows()):        trial_id = convert_to_serializable(row['trial_id'])                # Determine if this is the top model (for highlighting)        is_top_model = i == 0        row_class = 'highlight' if is_top_model else ''                html_content += f"""            <tr class="{row_class}">                <td>{i+1}</td>                <td>{trial_id}</td>                <td>{float(row.get('metric_map50_95', 0)):.4f}</td>                <td>{float(row.get('metric_fps', 0)):.2f}</td>                <td>{float(row.get('metric_model_size_mb', 0)):.2f}</td>        """                # Add values for optional columns if they exist        for column in optional_columns:            param_col = f'param_{column}'            if param_col in row:                html_content += f"<td>{convert_to_serializable(row.get(param_col, 'N/A'))}</td>\n"                # Add model paths        trial_path = f"trial_{trial_id}"        html_content += f"""                <td>                    <details>                        <summary>Show paths</summary>                        <div class="model-paths">                            <div class="model-path">Original: {trial_path}/weights/best.pt</div>                            <div class="model-path">With trial: {trial_path}/weights/best_trial_{trial_id}.pt</div>                        </div>                    </details>                </td>            </tr>        """        html_content += """        </table>                <h2>Visualizations</h2>        <div class="viz-container">    """        # Add visualizations    viz_files = os.listdir(viz_dir)    for viz_file in viz_files:        if viz_file.endswith('.png'):            html_content += f"""            <div class="viz-item">                <h3>{viz_file.replace('.png', '').replace('_', ' ').title()}</h3>                <img src="visualizations/{viz_file}" class="visualization" />            </div>            """        html_content += """        </div>                <h2>Best Model Details</h2>    """        # Get best model    if metric_col == 'metric_model_size_mb':        best_model_row = df.sort_values(by=metric_col, ascending=True).iloc[0]    else:        best_model_row = df.sort_values(by=metric_col, ascending=False).iloc[0]        # Add best model parameters - Convert all values to serializable types    best_params = {}    for k, v in best_model_row.items():        if k.startswith('param_'):            best_params[k.replace('param_', '')] = convert_to_serializable(v)        # Same for metrics    best_metrics = {}    for k, v in best_model_row.items():        if k.startswith('metric_'):            best_metrics[k.replace('metric_', '')] = convert_to_serializable(v)        # Best model paths    trial_id = best_model_row['trial_id']        html_content += f"""        <h3>Parameters</h3>        <pre>{CustomJSONEncoder().encode(best_params)}</pre>                <h3>Metrics</h3>        <pre>{CustomJSONEncoder().encode(best_metrics)}</pre>                <h3>Model Paths</h3>        <div class="model-paths">            <div class="model-path"><strong>Standard path:</strong> {results_dir}/trial_{trial_id}/weights/best.pt</div>            <div class="model-path"><strong>With trial ID:</strong> {results_dir}/trial_{trial_id}/weights/best_trial_{trial_id}.pt</div>            <div class="model-path"><strong>Root copy:</strong> {results_dir}/best_model_trial_{trial_id}.pt</div>        </div>    """        html_content += """    </body>    </html>    """        # Write HTML file    with open(html_file, 'w') as f:        f.write(html_content)        print(f"HTML report generated: {html_file}")    return html_file
    """
    Create an HTML report summarizing the NAS results.
    
    Args:
        df: DataFrame containing results
        results_dir: Directory to save the report
        viz_dir: Directory containing visualizations
        metric_col: Column name of the metric to optimize
        
    Returns:
        str: Path to the HTML report
    """
    html_file = os.path.join(results_dir, "nas_report.html")
    
    # Get top 5 models
    if metric_col == 'metric_model_size_mb':
        # For size, smaller is better
        top_models = df.sort_values(by=metric_col, ascending=True).head(5)
    else:
        # For other metrics, larger is better
        top_models = df.sort_values(by=metric_col, ascending=False).head(5)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Neural Architecture Search Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .visualization {{ margin: 20px 0; max-width: 100%; }}
            .viz-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .viz-item {{ width: 48%; margin-bottom: 20px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>YOLOv8 Neural Architecture Search Results</h1>
        
        <h2>Search Summary</h2>
        <ul>
            <li><strong>Total Trials:</strong> {len(df)}</li>
            <li><strong>Optimization Objective:</strong> {metric_col.replace('metric_', '')}</li>
            <li><strong>Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>
        
        <h2>Top Models</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Trial ID</th>
                <th>mAP50-95</th>
                <th>FPS</th>
                <th>Size (MB)</th>
    """
    
    # Add optional columns for depth, width, img_size, and kernel_size if they exist
    optional_columns = ['depth_multiple', 'width_multiple', 'img_size', 'kernel_size']
    for column in optional_columns:
        if f'param_{column}' in df.columns:
            html_content += f"<th>{column.replace('_', ' ').title()}</th>\n"
    
    html_content += """
            </tr>
    """
    
    # Add rows for top models
    for i, (idx, row) in enumerate(top_models.iterrows()):
        html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{convert_to_serializable(row['trial_id'])}</td>
                <td>{float(row.get('metric_map50_95', 0)):.4f}</td>
                <td>{float(row.get('metric_fps', 0)):.2f}</td>
                <td>{float(row.get('metric_model_size_mb', 0)):.2f}</td>
        """
        
        # Add values for optional columns if they exist
        for column in optional_columns:
            param_col = f'param_{column}'
            if param_col in row:
                html_content += f"<td>{convert_to_serializable(row.get(param_col, 'N/A'))}</td>\n"
        
        html_content += """
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visualizations</h2>
        <div class="viz-container">
    """
    
    # Add visualizations
    viz_files = os.listdir(viz_dir)
    for viz_file in viz_files:
        if viz_file.endswith('.png'):
            html_content += f"""
            <div class="viz-item">
                <h3>{viz_file.replace('.png', '').replace('_', ' ').title()}</h3>
                <img src="visualizations/{viz_file}" class="visualization" />
            </div>
            """
    
    html_content += """
        </div>
        
        <h2>Best Model Details</h2>
    """
    
    # Get best model
    if metric_col == 'metric_model_size_mb':
        best_model_row = df.sort_values(by=metric_col, ascending=True).iloc[0]
    else:
        best_model_row = df.sort_values(by=metric_col, ascending=False).iloc[0]
    
    # Add best model parameters - Convert all values to serializable types
    best_params = {}
    for k, v in best_model_row.items():
        if k.startswith('param_'):
            best_params[k.replace('param_', '')] = convert_to_serializable(v)
    
    # Same for metrics
    best_metrics = {}
    for k, v in best_model_row.items():
        if k.startswith('metric_'):
            best_metrics[k.replace('metric_', '')] = convert_to_serializable(v)
    
    html_content += f"""
        <h3>Parameters</h3>
        <pre>{CustomJSONEncoder().encode(best_params)}</pre>
        
        <h3>Metrics</h3>
        <pre>{CustomJSONEncoder().encode(best_metrics)}</pre>
        
        <h3>Model Path</h3>
        <p>{results_dir}/trial_{best_model_row['trial_id']}/weights/best.pt</p>
    """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {html_file}")
    return html_file