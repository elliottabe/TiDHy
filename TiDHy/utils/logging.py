import logging
import wandb
import jax.numpy as jnp
from flax import nnx
from typing import Dict, Optional, Any
import numpy as np

def init_wandb(config: Any, project_name: str = "TiDHy", run_name: Optional[str] = None, **kwargs):
    """
    Initialize wandb run with configuration.
    
    Note: This function is primarily for standalone use.
    When using with Run_TiDHy_NNX_vmap.py, wandb is initialized there.
    """
    # Convert config to dict if it's not already
    config_dict = config if isinstance(config, dict) else vars(config)
    
    # Initialize wandb
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        **kwargs
    )
    return run

def is_wandb_available() -> bool:
    """Check if wandb is available and has an active run."""
    return wandb.run is not None

def log_training_step(
    metrics: Dict[str, float], 
    step: int, 
    prefix: str = "train",
    log_sparsity: bool = True,
    log_gradnorm: bool = False
):
    """
    Log training step metrics to wandb.
    
    Args:
        metrics: Dictionary of metrics from training step
        step: Current step/epoch number
        prefix: Prefix for metric names (e.g., 'train', 'val')
        log_sparsity: Whether to log sparsity metrics
        log_gradnorm: Whether to log gradnorm metrics
    """
    log_dict = {}
    
    # Core loss components
    core_metrics = ['loss', 'spatial_loss_rhat', 'spatial_loss_rbar', 'temp_loss', 'cos_reg']
    for key in core_metrics:
        if key in metrics:
            log_dict[f'{prefix}/{key}'] = float(metrics[key])
    
    # Sparsity metrics
    if log_sparsity:
        sparsity_metrics = ['sparsity_reg', 'r_l0_norm', 'r2_l0_norm', 'w_l0_norm', 
                           'r_l1_norm', 'r2_l1_norm', 'w_l1_norm', 'r_l2_norm', 'r2_l2_norm', 'w_l2_norm']
        for key in sparsity_metrics:
            if key in metrics:
                log_dict[f'{prefix}/sparsity/{key}'] = float(metrics[key])
    
    # GradNorm metrics
    if log_gradnorm:
        gradnorm_metrics = ['grad_norm_spatial_rhat', 'grad_norm_spatial_rbar', 'grad_norm_temp',
                           'weight_spatial_rhat', 'weight_spatial_rbar', 'weight_temp']
        for key in gradnorm_metrics:
            if key in metrics:
                log_dict[f'{prefix}/gradnorm/{key}'] = float(metrics[key])
    
    # Learning rates if available
    lr_keys = [k for k in metrics.keys() if 'learning_rate' in k]
    for key in lr_keys:
        log_dict[f'{prefix}/{key}'] = float(metrics[key])
    
    wandb.log(log_dict, step=step)

def log_training_history_to_wandb(cfg, history, trained_model):
    # Log training history to wandb
    for epoch, loss in enumerate(history['loss']):
        log_dict = {
            'epoch': epoch,
            'train/total_loss': float(loss),
            'train/spatial_loss_rhat': float(history['spatial_loss_rhat'][epoch]),
            'train/spatial_loss_rbar': float(history['spatial_loss_rbar'][epoch]),
            'train/temp_loss': float(history['temp_loss'][epoch]),
            'train/cos_reg': float(history['cos_reg'][epoch])
        }
        
        # Add validation loss if available
        if 'val_loss' in history and epoch < len(history['val_loss']):
            log_dict['val/total_loss'] = float(history['val_loss'][epoch])
        
        wandb.log(log_dict, step=epoch)

    # Log final model parameters
    log_model_parameters(trained_model, step=len(history['loss']))

def log_sparsity_analysis(
    model: nnx.Module,
    r_values: jnp.array, 
    r2_values: jnp.array, 
    w_values: jnp.array, 
    step: int,
    threshold: float = 1e-3
):
    """
    Log detailed sparsity analysis.
    
    Args:
        model: The model (for accessing sparsity parameters)
        r_values: r values from forward pass
        r2_values: r2 values from forward pass  
        w_values: w values from forward pass
        step: Current step
        threshold: Threshold for considering values as "active"
    """
    log_dict = {}
    
    # Compute sparsity statistics for r
    r_flat = r_values.flatten()
    r_active_ratio = float(jnp.mean(jnp.abs(r_flat) > threshold))
    r_mean_magnitude = float(jnp.mean(jnp.abs(r_flat)))
    r_max_magnitude = float(jnp.max(jnp.abs(r_flat)))
    
    log_dict.update({
        'sparsity_analysis/r_active_ratio': r_active_ratio,
        'sparsity_analysis/r_mean_magnitude': r_mean_magnitude,
        'sparsity_analysis/r_max_magnitude': r_max_magnitude,
    })
    
    # Compute sparsity statistics for r2
    r2_flat = r2_values.flatten()
    r2_active_ratio = float(jnp.mean(jnp.abs(r2_flat) > threshold))
    r2_mean_magnitude = float(jnp.mean(jnp.abs(r2_flat)))
    r2_max_magnitude = float(jnp.max(jnp.abs(r2_flat)))
    
    log_dict.update({
        'sparsity_analysis/r2_active_ratio': r2_active_ratio,
        'sparsity_analysis/r2_mean_magnitude': r2_mean_magnitude,
        'sparsity_analysis/r2_max_magnitude': r2_max_magnitude,
    })
    
    # Compute sparsity statistics for w (hypernetwork output)
    w_flat = w_values.flatten()
    w_active_ratio = float(jnp.mean(jnp.abs(w_flat) > threshold))
    w_mean_magnitude = float(jnp.mean(jnp.abs(w_flat)))
    w_max_magnitude = float(jnp.max(jnp.abs(w_flat)))
    
    # Critical: Check if hypernetwork is collapsing
    w_collapse_risk = w_active_ratio < 0.1  # Less than 10% active is risky
    
    log_dict.update({
        'sparsity_analysis/w_active_ratio': w_active_ratio,
        'sparsity_analysis/w_mean_magnitude': w_mean_magnitude,
        'sparsity_analysis/w_max_magnitude': w_max_magnitude,
        'sparsity_analysis/w_collapse_risk': float(w_collapse_risk),
    })
    
    # Log target sparsity levels from model
    if hasattr(model, 'target_sparsity_r'):
        log_dict['sparsity_targets/r_target'] = model.target_sparsity_r
        log_dict['sparsity_targets/r_achieved'] = 1.0 - r_active_ratio
        
    if hasattr(model, 'target_sparsity_r2'):  
        log_dict['sparsity_targets/r2_target'] = model.target_sparsity_r2
        log_dict['sparsity_targets/r2_achieved'] = 1.0 - r2_active_ratio
        
    if hasattr(model, 'target_sparsity_w'):
        log_dict['sparsity_targets/w_target'] = model.target_sparsity_w  
        log_dict['sparsity_targets/w_achieved'] = 1.0 - w_active_ratio
    
    wandb.log(log_dict, step=step)

def log_optimization_metrics(
    optimizer: nnx.Optimizer,
    step: int,
    grad_norms: Optional[Dict[str, float]] = None
):
    """Log optimization-related metrics."""
    log_dict = {}
    
    # Log current learning rates
    if hasattr(optimizer.tx, 'learning_rate'):
        log_dict['optimization/learning_rate'] = float(optimizer.tx.learning_rate)
    elif hasattr(optimizer, 'tx') and hasattr(optimizer.tx, '_learning_rate'):
        log_dict['optimization/learning_rate'] = float(optimizer.tx._learning_rate)
    
    # Log gradient norms if provided
    if grad_norms:
        for component, norm in grad_norms.items():
            log_dict[f'optimization/grad_norm_{component}'] = float(norm)
    
    # Log optimizer state statistics if available
    if hasattr(optimizer, 'opt_state'):
        # This would depend on the specific optimizer being used
        pass
    
    if log_dict:
        wandb.log(log_dict, step=step)

