# AutoYOLO Implementation Protocols & Code Snippets
# Ready-to-use code for paper experiments

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from ultralytics import YOLO
from scipy import stats
import mlflow
import time
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1. EXPERIMENTAL CONFIGURATION
# =============================================================================

class ExperimentConfig:
    """Centralized configuration for all experiments"""
    
    def __init__(self):
        self.datasets = {
            'coco': {
                'data_yaml': 'data/coco.yaml',
                'train_images': 118287,
                'val_images': 5000,
                'classes': 80
            },
            'openimages': {
                'data_yaml': 'data/openimages_subset.yaml', 
                'train_images': 50000,
                'val_images': 5000,
                'classes': 100
            },
            'industrial': {
                'data_yaml': 'data/industrial.yaml',
                'train_images': 10000, 
                'val_images': 2000,
                'classes': 15
            }
        }
        
        self.models = {
            'yolov8n': {'params': '3.2M', 'size': '6.2MB'},
            'yolov8s': {'params': '11.2M', 'size': '21.5MB'},
            'yolov8m': {'params': '25.9M', 'size': '49.7MB'},
            'yolov8l': {'params': '43.7M', 'size': '83.7MB'}
        }
        
        self.nas_search_space = {
            'depth_multiple': [0.33, 0.5, 0.67, 1.0, 1.33],
            'width_multiple': [0.25, 0.5, 0.75, 1.0, 1.25],
            'kernel_sizes': [3, 5, 7],
            'activation': ['SiLU', 'ReLU', 'Hardswish'],
            'head_channels': [64, 128, 256, 512]
        }
        
        self.kd_grid = {
            'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
            'temperature': [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        }

# =============================================================================
# 2. BASELINE ESTABLISHMENT PROTOCOL
# =============================================================================

def establish_baselines(config: ExperimentConfig, num_seeds: int = 3) -> pd.DataFrame:
    """
    Establish baseline performance for all model-dataset combinations
    
    Returns:
        DataFrame with baseline results for statistical comparison
    """
    results = []
    
    for dataset_name, dataset_info in config.datasets.items():
        for model_name in config.models.keys():
            for seed in range(num_seeds):
                # Set random seeds for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                print(f"Training {model_name} on {dataset_name} (seed {seed})")
                
                # Initialize model
                model = YOLO(f'{model_name}.pt')
                
                # Training configuration
                train_args = {
                    'data': dataset_info['data_yaml'],
                    'epochs': 100,
                    'imgsz': 640,
                    'batch': 16,
                    'device': 0,
                    'project': f'baselines/{dataset_name}',
                    'name': f'{model_name}_seed{seed}',
                    'exist_ok': True,
                    'save': True,
                    'patience': 20,
                    'seed': seed
                }
                
                # Train model
                start_time = time.time()
                training_results = model.train(**train_args)
                training_time = time.time() - start_time
                
                # Evaluate model
                val_results = model.val()
                
                # Collect metrics
                result = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'seed': seed,
                    'map50': val_results.box.map50,
                    'map50_95': val_results.box.map,
                    'precision': val_results.box.mp,
                    'recall': val_results.box.mr,
                    'training_time_hours': training_time / 3600,
                    'model_size_mb': Path(f'{train_args["project"]}/{train_args["name"]}/weights/best.pt').stat().st_size / (1024*1024)
                }
                
                # Measure inference speed
                result['fps'] = measure_inference_speed(model, dataset_info['data_yaml'])
                
                results.append(result)
                
                # Log to MLflow
                with mlflow.start_run(run_name=f"baseline_{model_name}_{dataset_name}_seed{seed}"):
                    mlflow.log_params({
                        'model': model_name,
                        'dataset': dataset_name, 
                        'seed': seed,
                        'experiment_type': 'baseline'
                    })
                    mlflow.log_metrics(result)
    
    baseline_df = pd.DataFrame(results)
    baseline_df.to_csv('results/baseline_results.csv', index=False)
    return baseline_df

def measure_inference_speed(model, data_yaml: str, num_iterations: int = 100) -> float:
    """Measure inference speed in FPS"""
    
    # Load a sample image from validation set
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    val_path = Path(data_config['path']) / data_config['val']
    sample_images = list(val_path.glob('*.jpg'))[:10]
    
    # Warm up
    for _ in range(10):
        _ = model(sample_images[0], verbose=False)
    
    # Measure speed
    start_time = time.time()
    for i in range(num_iterations):
        img_path = sample_images[i % len(sample_images)]
        _ = model(img_path, verbose=False)
    
    total_time = time.time() - start_time
    fps = num_iterations / total_time
    
    return fps

# =============================================================================
# 3. NEURAL ARCHITECTURE SEARCH IMPLEMENTATION
# =============================================================================

class AutoYOLONAS:
    """Neural Architecture Search for YOLO models"""
    
    def __init__(self, search_space: Dict, population_size: int = 20):
        self.search_space = search_space
        self.population_size = population_size
        self.generation = 0
        
    def sample_architecture(self) -> Dict:
        """Sample a random architecture from search space"""
        arch = {}
        for param, options in self.search_space.items():
            if isinstance(options, list):
                arch[param] = np.random.choice(options)
            else:
                arch[param] = options
        return arch
    
    def generate_yolo_config(self, arch_params: Dict, num_classes: int) -> Dict:
        """Generate YOLOv8 configuration from architecture parameters"""
        
        # Calculate scaled channels
        base_channels = [64, 128, 256, 512, 1024]
        width_mult = arch_params['width_multiple']
        channels = [max(16, int(ch * width_mult)) for ch in base_channels]
        
        # Calculate scaled depths
        depth_mult = arch_params['depth_multiple']
        base_depths = [3, 6, 6, 3]  # C2f depths for each stage
        depths = [max(1, round(d * depth_mult)) for d in base_depths]
        
        kernel_size = arch_params['kernel_sizes']
        
        config = {
            'nc': num_classes,
            'depth_multiple': depth_mult,
            'width_multiple': width_mult,
            'backbone': [
                [-1, 1, 'Conv', [channels[0], kernel_size, 2]],  # 0-P1/2
                [-1, 1, 'Conv', [channels[1], kernel_size, 2]],  # 1-P2/4
                [-1, depths[0], 'C2f', [channels[1], True]],      # 2
                [-1, 1, 'Conv', [channels[2], kernel_size, 2]],  # 3-P3/8
                [-1, depths[1], 'C2f', [channels[2], True]],      # 4
                [-1, 1, 'Conv', [channels[3], kernel_size, 2]],  # 5-P4/16
                [-1, depths[2], 'C2f', [channels[3], True]],      # 6
                [-1, 1, 'Conv', [channels[4], kernel_size, 2]],  # 7-P5/32
                [-1, depths[3], 'C2f', [channels[4], True]],      # 8
                [-1, 1, 'SPPF', [channels[4], 5]],               # 9
            ],
            'head': [
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [channels[3]]],
                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                [[-1, 4], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [channels[2]]],
                [-1, 1, 'Conv', [channels[2], kernel_size, 2]],
                [[-1, 12], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [channels[3]]],
                [-1, 1, 'Conv', [channels[3], kernel_size, 2]],
                [[-1, 9], 1, 'Concat', [1]],
                [-1, 3, 'C2f', [channels[4]]],
                [[15, 18, 21], 1, 'Detect', [num_classes]]
            ]
        }
        
        return config
    
    def evaluate_architecture(self, arch_params: Dict, dataset_config: Dict, 
                            fast_eval: bool = True) -> Dict:
        """Evaluate architecture performance"""
        
        # Generate YAML config
        yaml_config = self.generate_yolo_config(arch_params, dataset_config['classes'])
        
        # Save temporary config file
        config_path = f'temp_configs/arch_{hash(str(arch_params))}.yaml'
        Path('temp_configs').mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(yaml_config, f)
        
        # Train model with reduced epochs for fast evaluation
        epochs = 25 if fast_eval else 100
        
        model = YOLO(config_path)
        
        train_args = {
            'data': dataset_config['data_yaml'],
            'epochs': epochs,
            'imgsz': 640,
            'batch': 16,
            'device': 0,
            'project': 'nas_trials',
            'name': f'trial_{hash(str(arch_params))}',
            'exist_ok': True,
            'patience': 10,
            'verbose': False
        }
        
        try:
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            # Evaluate
            val_results = model.val()
            
            # Calculate parameters and FLOPs
            total_params = sum(p.numel() for p in model.model.parameters())
            
            metrics = {
                'map50_95': val_results.box.map,
                'map50': val_results.box.map50,
                'precision': val_results.box.mp,
                'recall': val_results.box.mr,
                'parameters': total_params,
                'training_time': training_time,
                'converged': True
            }
            
            # Combined efficiency score
            param_efficiency = metrics['map50_95'] / (total_params / 1e6)  # mAP per million params
            time_efficiency = metrics['map50_95'] / (training_time / 3600)  # mAP per hour
            
            metrics['efficiency_score'] = 0.7 * param_efficiency + 0.3 * time_efficiency
            
        except Exception as e:
            print(f"Architecture evaluation failed: {e}")
            metrics = {
                'map50_95': 0.0,
                'map50': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'parameters': 0,
                'training_time': 0,
                'efficiency_score': 0.0,
                'converged': False
            }
        
        # Clean up temp files
        if Path(config_path).exists():
            Path(config_path).unlink()
        
        return metrics
    
    def evolutionary_search(self, dataset_config: Dict, num_generations: int = 10) -> List[Dict]:
        """Run evolutionary architecture search"""
        
        # Initialize population
        population = [self.sample_architecture() for _ in range(self.population_size)]
        best_architectures = []
        
        for generation in range(num_generations):
            print(f"Generation {generation + 1}/{num_generations}")
            
            # Evaluate population
            fitness_scores = []
            for i, arch in enumerate(population):
                print(f"  Evaluating architecture {i + 1}/{len(population)}")
                metrics = self.evaluate_architecture(arch, dataset_config, fast_eval=True)
                fitness_scores.append(metrics['efficiency_score'])
                
                # Track best architectures
                if metrics['converged']:
                    best_architectures.append({
                        'generation': generation,
                        'architecture': arch,
                        'metrics': metrics
                    })
            
            # Select parents (tournament selection)
            parents = self._tournament_selection(population, fitness_scores, k=3)
            
            # Generate offspring
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = np.random.choice(parents, 2, replace=False)
                child = self._crossover(population[parent1], population[parent2])
                child = self._mutate(child)
                offspring.append(child)
            
            population = offspring
            self.generation += 1
        
        return best_architectures
    
    def _tournament_selection(self, population: List[Dict], fitness: List[float], k: int = 3) -> List[int]:
        """Tournament selection for parent selection"""
        selected = []
        for _ in range(len(population) // 2):
            tournament_indices = np.random.choice(len(population), k, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(winner_idx)
        return selected
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Single-point crossover between two architectures"""
        child = {}
        for param in self.search_space.keys():
            child[param] = parent1[param] if np.random.random() < 0.5 else parent2[param]
        return child
    
    def _mutate(self, architecture: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate architecture with given probability"""
        mutated = architecture.copy()
        for param, options in self.search_space.items():
            if np.random.random() < mutation_rate and isinstance(options, list):
                mutated[param] = np.random.choice(options)
        return mutated

# =============================================================================
# 4. KNOWLEDGE DISTILLATION IMPLEMENTATION
# =============================================================================

class KnowledgeDistillationTrainer:
    """Knowledge Distillation trainer for YOLO models"""
    
    def __init__(self, teacher_model_path: str, alpha: float = 0.5, temperature: float = 2.0):
        self.teacher = YOLO(teacher_model_path)
        self.teacher.model.eval()
        
        # Freeze teacher parameters
        for param in self.teacher.model.parameters():
            param.requires_grad = False
            
        self.alpha = alpha
        self.temperature = temperature
    
    def compute_distillation_loss(self, student_outputs, teacher_outputs):
        """Compute knowledge distillation loss for multi-scale detection"""
        
        # Ensure outputs are lists (for multi-scale)
        if not isinstance(student_outputs, list):
            student_outputs = [student_outputs]
        if not isinstance(teacher_outputs, list):
            teacher_outputs = [teacher_outputs]
        
        total_kd_loss = 0
        valid_scales = 0
        
        for s_out, t_out in zip(student_outputs, teacher_outputs):
            if s_out.shape != t_out.shape:
                continue
                
            # Extract classification predictions (skip bbox coordinates)
            s_cls = s_out[..., 4:]  # [batch, anchors, num_classes]
            t_cls = t_out[..., 4:]
            
            # Apply temperature scaling
            s_log_softmax = F.log_softmax(s_cls / self.temperature, dim=-1)
            t_softmax = F.softmax(t_cls / self.temperature, dim=-1).detach()
            
            # Compute KL divergence
            kd_loss = F.kl_div(s_log_softmax, t_softmax, reduction='batchmean')
            
            # Temperature compensation
            total_kd_loss += kd_loss * (self.temperature ** 2)
            valid_scales += 1
        
        if valid_scales > 0:
            return total_kd_loss / valid_scales
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def train_student(self, student_config_path: str, dataset_config: Dict, 
                     epochs: int = 100) -> Dict:
        """Train student model with knowledge distillation"""
        
        # Custom training loop with KD
        student = YOLO(student_config_path)
        
        # Training configuration
        train_args = {
            'data': dataset_config['data_yaml'],
            'epochs': epochs,
            'imgsz': 640,
            'batch': 16,
            'device': 0,
            'project': 'kd_experiments',
            'name': f'student_alpha{self.alpha}_temp{self.temperature}',
            'exist_ok': True
        }
        
        # Override loss computation for KD
        original_criterion = student.model.criterion
        
        def kd_criterion(predictions, targets):
            # Standard detection loss (hard targets)
            hard_loss = original_criterion(predictions, targets)
            
            # Get teacher predictions (soft targets)
            with torch.no_grad():
                # Assuming targets contain image tensors
                teacher_predictions = self.teacher.model(targets['img'])
            
            # Knowledge distillation loss
            soft_loss = self.compute_distillation_loss(predictions, teacher_predictions)
            
            # Combined loss
            total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            
            return total_loss
        
        # Replace criterion
        student.model.criterion = kd_criterion
        
        # Train with modified loss
        start_time = time.time()
        results = student.train(**train_args)
        training_time = time.time() - start_time
        
        # Evaluate student
        val_results = student.val()
        
        metrics = {
            'alpha': self.alpha,
            'temperature': self.temperature,
            'map50_95': val_results.box.map,
            'map50': val_results.box.map50,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
            'training_time_hours': training_time / 3600
        }
        
        return metrics

# =============================================================================
# 5. COMPREHENSIVE EVALUATION PROTOCOL
# =============================================================================

def run_full_evaluation_protocol(config: ExperimentConfig, 
                                nas_trials: int = 50,
                                kd_grid_samples: int = 25) -> pd.DataFrame:
    """Run complete AutoYOLO evaluation protocol"""
    
    all_results = []
    
    for dataset_name, dataset_info in config.datasets.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # 1. Neural Architecture Search
        print("Running Neural Architecture Search...")
        nas = AutoYOLONAS(config.nas_search_space)
        best_architectures = nas.evolutionary_search(dataset_info, num_generations=10)
        
        # Select top 5 architectures for detailed evaluation
        best_architectures.sort(key=lambda x: x['metrics']['efficiency_score'], reverse=True)
        top_architectures = best_architectures[:5]
        
        # 2. Knowledge Distillation Grid Search
        print("Running Knowledge Distillation experiments...")
        
        # Sample from KD grid
        alpha_samples = np.random.choice(config.kd_grid['alpha'], kd_grid_samples//5, replace=True)
        temp_samples = np.random.choice(config.kd_grid['temperature'], kd_grid_samples//5, replace=True)
        
        for arch_info in top_architectures:
            arch = arch_info['architecture']
            
            # Generate student config
            yaml_config = nas.generate_yolo_config(arch, dataset_info['classes'])
            config_path = f'best_configs/{dataset_name}_arch_{hash(str(arch))}.yaml'
            Path('best_configs').mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(yaml_config, f)
            
            # Test KD combinations
            for alpha, temperature in zip(alpha_samples, temp_samples):
                print(f"  Testing α={alpha}, T={temperature}")
                
                # Setup KD trainer
                teacher_path = 'yolov8l.pt'  # Use large model as teacher
                kd_trainer = KnowledgeDistillationTrainer(teacher_path, alpha, temperature)
                
                # Train with KD
                kd_results = kd_trainer.train_student(config_path, dataset_info, epochs=50)
                
                # Combine architecture and KD results
                result = {
                    'dataset': dataset_name,
                    'experiment_type': 'AutoYOLO',
                    'architecture': arch,
                    'nas_efficiency_score': arch_info['metrics']['efficiency_score'],
                    **kd_results
                }
                
                all_results.append(result)
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/autoyolo_comprehensive_results.csv', index=False)
    
    return results_df

# =============================================================================
# 6. STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def perform_statistical_analysis(baseline_df: pd.DataFrame, 
                                autoyolo_df: pd.DataFrame) -> Dict:
    """Perform comprehensive statistical analysis"""
    
    results = {}
    
    for dataset in baseline_df['dataset'].unique():
        print(f"\nStatistical Analysis for {dataset}:")
        
        # Filter data
        baseline_data = baseline_df[baseline_df['dataset'] == dataset]
        autoyolo_data = autoyolo_df[autoyolo_df['dataset'] == dataset]
        
        # Compare best AutoYOLO vs best baseline
        best_baseline = baseline_data.loc[baseline_data['map50_95'].idxmax()]
        best_autoyolo = autoyolo_data.loc[autoyolo_data['map50_95'].idxmax()]
        
        # Paired t-test (if multiple seeds available)
        if len(baseline_data) > 1 and len(autoyolo_data) > 1:
            t_stat, p_value = stats.ttest_ind(
                baseline_data['map50_95'], 
                autoyolo_data['map50_95']
            )
            
            # Cohen's d for effect size
            pooled_std = np.sqrt(((len(baseline_data) - 1) * baseline_data['map50_95'].std()**2 + 
                                 (len(autoyolo_data) - 1) * autoyolo_data['map50_95'].std()**2) / 
                                (len(baseline_data) + len(autoyolo_data) - 2))
            
            cohens_d = (autoyolo_data['map50_95'].mean() - baseline_data['map50_95'].mean()) / pooled_std
            
            results[dataset] = {
                'baseline_mean_map': baseline_data['map50_95'].mean(),
                'autoyolo_mean_map': autoyolo_data['map50_95'].mean(),
                'improvement_pct': ((autoyolo_data['map50_95'].mean() - baseline_data['map50_95'].mean()) / 
                                   baseline_data['map50_95'].mean()) * 100,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            }
            
            print(f"  Mean mAP improvement: {results[dataset]['improvement_pct']:.2f}%")
            print(f"  Statistical significance: p = {p_value:.4f}")
            print(f"  Effect size (Cohen's d): {cohens_d:.3f} ({results[dataset]['effect_size']})")
    
    return results

# =============================================================================
# 7. VISUALIZATION FUNCTIONS
# =============================================================================

def create_paper_figures(baseline_df: pd.DataFrame, autoyolo_df: pd.DataFrame, 
                        nas_results: List[Dict]) -> None:
    """Generate all figures for the paper"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Figure 1: Performance comparison across datasets
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    datasets = baseline_df['dataset'].unique()
    metrics = ['map50_95', 'parameters']
    
    for i, dataset in enumerate(datasets[:4]):  # Max 4 datasets
        ax = axes[i//2, i%2]
        
        baseline_data = baseline_df[baseline_df['dataset'] == dataset]
        autoyolo_data = autoyolo_df[autoyolo_df['dataset'] == dataset]
        
        # Scatter plot: accuracy vs parameters
        ax.scatter(baseline_data['parameters']/1e6, baseline_data['map50_95'], 
                  alpha=0.7, label='Baseline', s=60)
        ax.scatter(autoyolo_data['parameters']/1e6, autoyolo_data['map50_95'], 
                  alpha=0.7, label='AutoYOLO', s=60)
        
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('mAP@0.5:0.95')
        ax.set_title(f'{dataset.title()} Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: NAS convergence analysis
    if nas_results:
        plt.figure(figsize=(10, 6))
        
        generations = [r['generation'] for r in nas_results]
        efficiency_scores = [r['metrics']['efficiency_score'] for r in nas_results]
        
        # Plot convergence
        for gen in range(max(generations) + 1):
            gen_scores = [score for g, score in zip(generations, efficiency_scores) if g == gen]
            if gen_scores:
                plt.scatter([gen] * len(gen_scores), gen_scores, alpha=0.6, s=30)
        
        # Best scores per generation
        best_scores = []
        for gen in range(max(generations) + 1):
            gen_scores = [score for g, score in zip(generations, efficiency_scores) if g == gen]
            if gen_scores:
                best_scores.append(max(gen_scores))
            else:
                best_scores.append(0)
        
        plt.plot(range(len(best_scores)), best_scores, 'r-', linewidth=2, label='Best Score')
        
        plt.xlabel('Generation')
        plt.ylabel('Efficiency Score')
        plt.title('Neural Architecture Search Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('figures/nas_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Figures saved to figures/ directory")

# =============================================================================
# 8. MAIN EXECUTION PROTOCOL
# =============================================================================

def main():
    """Main execution function for paper experiments"""
    
    # Initialize configuration
    config = ExperimentConfig()
    
    # Initialize MLflow
    mlflow.set_experiment("AutoYOLO_Paper_Experiments")
    
    # Create directories
    Path('results').mkdir(exist_ok=True)
    Path('figures').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    print("AutoYOLO Paper Experiments")
    print("=" * 50)
    
    # Step 1: Establish baselines
    print("Step 1: Establishing baselines...")
    baseline_df = establish_baselines(config, num_seeds=3)
    
    # Step 2: Run full AutoYOLO evaluation
    print("Step 2: Running AutoYOLO experiments...")
    autoyolo_df = run_full_evaluation_protocol(config, nas_trials=50, kd_grid_samples=25)
    
    # Step 3: Statistical analysis
    print("Step 3: Performing statistical analysis...")
    stats_results = perform_statistical_analysis(baseline_df, autoyolo_df)
    
    # Save statistical results
    with open('results/statistical_analysis.json', 'w') as f:
        json.dump(stats_results, f, indent=2)
    
    # Step 4: Generate figures
    print("Step 4: Creating figures...")
    # Note: nas_results would come from the actual NAS runs
    create_paper_figures(baseline_df, autoyolo_df, [])
    
    print("\nExperiment protocol completed!")
    print("Results saved in results/ directory")
    print("Figures saved in figures/ directory")

if __name__ == "__main__":
    main()