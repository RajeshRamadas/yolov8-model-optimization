import torch
import torch.nn.functional as F
import os
import argparse
import logging
import sys
import copy
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('kd_training.log')
    ])
logger = logging.getLogger('kd_trainer')

def train_teacher_model(cfg):
    """Train the teacher model first"""
    logger.info("="*50)
    logger.info("TRAINING TEACHER MODEL")
    logger.info("="*50)
    
    # Create a copy of the config for the teacher
    teacher_cfg = cfg.copy()
    teacher_model = teacher_cfg.pop('teacher_model', 'yolov8x.yaml')  # Use larger model for teacher
    teacher_name = teacher_cfg.pop('teacher_name', 'yolov8x_custom')  # Unique experiment name
    
    # Remove KD-specific parameters that YOLO doesn't understand
    for param in ['alpha', 'temperature', 'student_model', 'patience']:
        if param in teacher_cfg:
            teacher_cfg.pop(param)
    
    # Train the teacher model using default YOLO trainer
    teacher = YOLO(teacher_model)
    teacher.train(
        data=teacher_cfg['data'],
        epochs=teacher_cfg['epochs'],
        imgsz=teacher_cfg['imgsz'],
        batch=teacher_cfg['batch'],
        device=teacher_cfg['device'],
        workers=teacher_cfg['workers'],
        name=teacher_name
    )
    
    # Return the path to the best weights
    best_weights_path = os.path.join('runs/detect', teacher_name, 'weights/best.pt')
    logger.info(f"Teacher model trained. Best weights saved to: {best_weights_path}")
    return best_weights_path

class KDTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, teacher_weights=None, kd_params=None):
        # Store KD parameters separately
        self.kd_params = kd_params or {}
        
        # Create a clean copy of overrides without KD-specific parameters
        if overrides is None:
            overrides = {}
        clean_overrides = copy.deepcopy(overrides)
        
        # Remove KD-specific and custom parameters from overrides
        custom_params = ['alpha', 'temperature', 'teacher_model', 'teacher_name', 'student_model', 'patience']
        for param in custom_params:
            if param in clean_overrides:
                clean_overrides.pop(param)
        
        # Initialize the parent with clean overrides
        super().__init__(cfg=cfg, overrides=clean_overrides, _callbacks=_callbacks)
        
        # Device handling
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load teacher model
        self.teacher = YOLO(teacher_weights).model
        self.teacher.eval().to(self.device)
        for param in self.teacher.parameters():
            param.requires_grad = False  # Freeze teacher
        
        logger.info(f"Loaded teacher model from: {teacher_weights}")
        
        # KD hyperparameters
        self.alpha = float(self.kd_params.get('alpha', 0.5))  # Weight for hard loss
        self.temperature = float(self.kd_params.get('temperature', 2.0))  # Temperature for softening
        
        # Early stopping parameters
        self.patience = int(self.kd_params.get('patience', 0))  # Default: no early stopping
        self.best_fitness = 0
        self.no_improvement_count = 0
        
        logger.info(f"KD parameters - Alpha: {self.alpha}, Temperature: {self.temperature}")
        if self.patience > 0:
            logger.info(f"Early stopping enabled with patience: {self.patience}")
    
    def get_distillation_loss(self, student_outputs, teacher_outputs, batch):
        # Standard YOLO detection loss
        hard_loss = self.criterion(student_outputs, batch)
        
        # Initialize soft loss
        soft_loss = 0
        valid_outputs = 0
        
        # Handle single output case
        student_outputs = [student_outputs] if not isinstance(student_outputs, list) else student_outputs
        teacher_outputs = [teacher_outputs] if not isinstance(teacher_outputs, list) else teacher_outputs
        
        # Process each detection head
        for s_out, t_out in zip(student_outputs, teacher_outputs):
            if s_out.shape != t_out.shape:
                continue
            
            # Extract class predictions (YOLOv8 uses index 4 for class start)
            s_cls = s_out[..., 4:]  # Shape: [batch, anchors, classes]
            t_cls = t_out[..., 4:]
            
            # Compute KL divergence with numerical stability
            s_log_softmax = F.log_softmax(s_cls / self.temperature, dim=-1)
            t_softmax = F.softmax(t_cls / self.temperature, dim=-1).detach()
            
            # Clamp to avoid NaN
            t_softmax = torch.clamp(t_softmax, min=1e-7, max=1.0)
            
            soft_loss += F.kl_div(
                s_log_softmax,
                t_softmax,
                reduction='batchmean'
            ) * (self.temperature ** 2)
            valid_outputs += 1
        
        # Combine losses
        if valid_outputs == 0:
            return hard_loss  # Fallback if no matching outputs
        
        return self.alpha * hard_loss + (1 - self.alpha) * (soft_loss / valid_outputs)
    
    def training_step(self, batch):
        # Move batch to device if needed
        if batch['img'].device != self.device:
            batch['img'] = batch['img'].to(self.device)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(batch['img'])
        
        # Student forward pass
        student_outputs = self.model(batch['img'])
        
        # Compute combined loss
        loss = self.get_distillation_loss(student_outputs, teacher_outputs, batch)
        return loss
    
    def _do_eval(self, *args, **kwargs):
        """Override validation handling to add early stopping"""
        # Call parent method to perform validation
        results = super()._do_eval(*args, **kwargs)
        
        # Check for early stopping if enabled
        if self.patience > 0:
            # Get the current fitness value
            current_fitness = results[0].fitness
            
            if current_fitness > self.best_fitness:
                # Found a better model
                self.best_fitness = current_fitness
                self.no_improvement_count = 0
                logger.info(f"New best fitness: {self.best_fitness:.4f}")
            else:
                # No improvement
                self.no_improvement_count += 1
                logger.info(f"No improvement in fitness: {self.no_improvement_count}/{self.patience}")
                
                if self.no_improvement_count >= self.patience:
                    logger.info(f"Early stopping triggered after {self.epoch} epochs")
                    self.epoch = self.epochs  # Set current epoch to max to stop training
        
        return results

def train_student_model(cfg, teacher_weights):
    """Train the student model with knowledge distillation"""
    logger.info("="*50)
    logger.info("TRAINING STUDENT MODEL WITH KNOWLEDGE DISTILLATION")
    logger.info("="*50)
    
    # Separate KD parameters from YOLO parameters
    kd_params = {}
    clean_cfg = copy.deepcopy(cfg)
    
    # Extract KD-specific parameters
    for param in ['alpha', 'temperature', 'teacher_model', 'teacher_name', 'patience']:
        if param in clean_cfg:
            kd_params[param] = clean_cfg.pop(param)
    
    # Initialize and train
    trainer = KDTrainer(
        overrides=clean_cfg, 
        teacher_weights=teacher_weights, 
        kd_params=kd_params
    )
    results = trainer.train()
    
    return results

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLOv8 Knowledge Distillation Training')
    
    # Basic configuration
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--student_model', type=str, default='yolov8n.yaml', help='Student model configuration')
    parser.add_argument('--teacher_model', type=str, default='yolov8x.yaml', help='Teacher model configuration')
    parser.add_argument('--teacher_weights', type=str, help='Path to pre-trained teacher weights, if empty teacher will be trained')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Training device (GPU ID or cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    
    # Knowledge distillation parameters
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight balance between hard and soft loss')
    parser.add_argument('--temperature', type=float, default=2.0, help='Softening temperature for KD')
    
    # Early stopping parameter
    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 to disable)')
    
    # Output naming
    parser.add_argument('--teacher_name', type=str, default='yolov8x_custom', help='Teacher experiment name')
    parser.add_argument('--student_name', type=str, default='yolov8n_kd', help='Student experiment name')
    
    # Run mode
    parser.add_argument('--skip_teacher', action='store_true', help='Skip teacher training (requires teacher_weights)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Create configuration dictionary
    cfg = {
        'model': args.student_model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'name': args.student_name,
        'teacher_model': args.teacher_model,
        'teacher_name': args.teacher_name,
        'alpha': args.alpha,
        'temperature': args.temperature,
        'patience': args.patience,  # Added patience parameter
    }
    
    # Log configuration
    logger.info("Configuration:")
    for key, value in cfg.items():
        logger.info(f"  {key}: {value}")
    
    # Handle teacher model
    if args.skip_teacher:
        if not args.teacher_weights:
            logger.error("Teacher weights must be provided when skipping teacher training")
            sys.exit(1)
        teacher_weights = args.teacher_weights
        logger.info(f"Using pre-trained teacher weights: {teacher_weights}")
    else:
        teacher_weights = args.teacher_weights if args.teacher_weights else train_teacher_model(cfg)
    
    # Train student model
    results = train_student_model(cfg, teacher_weights)
    
    # Save results and artifacts
    results_path = os.path.join('runs/detect', args.student_name, 'results.csv')
    logger.info(f"Training completed. Results saved to: {results_path}")
    
    # Return success code
    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.exception("An error occurred during training")
        sys.exit(1)
