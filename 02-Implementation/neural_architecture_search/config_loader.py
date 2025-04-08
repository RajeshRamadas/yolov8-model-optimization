# config_loader.py
"""
Module for loading and managing configuration for Neural Architecture Search.
"""

import os
import yaml


def load_search_config(config_path):
    """
    Load search configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config


def get_search_space(config, advanced=False):
    """
    Get the search space from the configuration.
    
    Args:
        config (dict): Configuration dictionary
        advanced (bool): Whether to include advanced search parameters
        
    Returns:
        dict: Search space parameters
    """
    # Start with basic search space
    if 'basic_search_space' not in config:
        raise ValueError("Configuration must contain 'basic_search_space'")
        
    search_space = config['basic_search_space'].copy()
    
    # Add advanced parameters if requested
    if advanced and 'advanced_search_space' in config:
        for key, value in config['advanced_search_space'].items():
            search_space[key] = value
            
    return search_space


def get_objective_weights(config):
    """
    Get objective weights from the configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Objective weights
    """
    if 'objective_weights' not in config:
        # Return default weights if not specified
        return {
            "map_weight": 1.0,
            "speed_weight": 0.3,
            "size_weight": 0.2
        }
        
    return config['objective_weights']


def get_default_args(config):
    """
    Get default arguments from the configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Default arguments
    """
    if 'defaults' not in config:
        # Return hardcoded defaults if not specified
        return {
            "trials": 10,
            "epochs": 10,
            "results_dir": "nas_results",
            "parallel": 1,
            "objective": "map",
            "advanced_search": False
        }
        
    defaults = config['defaults'].copy()
    # Add advanced_search if not present
    if "advanced_search" not in defaults:
        defaults["advanced_search"] = False
        
    return defaults


def save_search_config(config, config_path):
    """
    Save search configuration to a YAML file.
    
    Args:
        config (dict): Configuration dictionary
        config_path (str): Path to save the YAML configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)