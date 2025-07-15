import asyncio
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import concurrent.futures
from contextlib import asynccontextmanager

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from bubbles_core import SystemContext, Actions
from DreamerV3Bubble import DreamerV3Bubble, DreamerV3Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'tuning_log_{int(datetime.now().timestamp())}.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    num_iterations: int = 10
    num_transitions: int = 1000
    num_trials: int = 30
    parallel_workers: int = 4
    warmup_iterations: int = 5
    eval_iterations: int = 20
    checkpoint_interval: int = 5
    timeout_seconds: int = 600
    retry_attempts: int = 3
    resource_limit_gb: float = 8.0
    
    # Dynamic thresholds
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    
    # Multi-objective weights
    return_weight: float = 1.0
    validation_weight: float = 0.3
    stability_weight: float = 0.1
    efficiency_weight: float = 0.05


class ConfigValidator:
    """Validates hyperparameter configurations."""
    
    @staticmethod
    def validate(config: Dict[str, Any], state_dim: int, action_dim: int) -> Tuple[bool, Optional[str]]:
        """
        Validate a hyperparameter configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check divisibility constraints
        if config['hidden_dim'] % config['num_heads'] != 0:
            return False, f"hidden_dim ({config['hidden_dim']}) must be divisible by num_heads ({config['num_heads']})"
        
        # Check memory constraints
        estimated_memory = (
            config['batch_size'] * config['sequence_length'] * 
            config['hidden_dim'] * config['num_transformer_layers'] * 4 / 1e9
        )
        if estimated_memory > 16.0:  # 16GB limit
            return False, f"Estimated memory usage ({estimated_memory:.1f}GB) exceeds limit"
        
        # Check reasonable ranges
        if config['learning_rate'] > 0.1 or config['learning_rate'] < 1e-7:
            return False, f"Learning rate {config['learning_rate']} outside reasonable range"
        
        if config['batch_size'] < 1 or config['batch_size'] > 512:
            return False, f"Batch size {config['batch_size']} outside reasonable range"
        
        if config['sequence_length'] < 1 or config['sequence_length'] > 100:
            return False, f"Sequence length {config['sequence_length']} outside reasonable range"
        
        return True, None


class CheckpointManager:
    """Manages checkpointing and recovery for tuning runs."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint = self.checkpoint_dir / "latest.pkl"
        
    async def save_state(self, state: Dict[str, Any], iteration: int):
        """Save tuning state to checkpoint."""
        checkpoint = {
            'state': state,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save with iteration number
        iter_path = self.checkpoint_dir / f"checkpoint_{iteration:04d}.pkl"
        await asyncio.to_thread(self._save_pickle, checkpoint, iter_path)
        
        # Update latest symlink
        await asyncio.to_thread(self._save_pickle, checkpoint, self.latest_checkpoint)
        
        logger.info(f"Saved checkpoint at iteration {iteration}")
    
    async def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint if available."""
        if self.latest_checkpoint.exists():
            try:
                checkpoint = await asyncio.to_thread(self._load_pickle, self.latest_checkpoint)
                logger.info(f"Loaded checkpoint from {checkpoint['timestamp']}")
                return checkpoint
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None
        return None
    
    def _save_pickle(self, obj: Any, path: Path):
        """Thread-safe pickle save."""
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    
    def _load_pickle(self, path: Path) -> Any:
        """Thread-safe pickle load."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TuningVisualizer:
    """Modular visualization components for tuning results."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: Path):
        self.df = results_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set(style="whitegrid", palette="muted")
    
    def plot_parameter_impact(self, param: str, metric: str = 'avg_return') -> plt.Figure:
        """Create a single parameter impact plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle log scale for certain parameters
        log_params = ['learning_rate', 'entropy_coeff', 'weight_decay']
        
        if param in log_params:
            ax.set_xscale('log')
        
        # Create scatter plot with color gradient for performance
        scatter = ax.scatter(
            self.df[param], 
            self.df[metric],
            c=self.df[metric],
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(
            np.log10(self.df[param]) if param in log_params else self.df[param],
            self.df[metric]
        )
        
        ax.set_xlabel(param.replace('_', ' ').title())
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{param} vs {metric} (R² = {r_value**2:.3f})')
        
        plt.colorbar(scatter, ax=ax, label=metric)
        plt.tight_layout()
        
        return fig
    
    def plot_optimization_trajectory(self) -> plt.Figure:
        """Plot the optimization trajectory over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Add trial numbers
        self.df['trial'] = range(len(self.df))
        
        # Plot objective value
        ax1.plot(self.df['trial'], self.df['objective_value'], 'b-', label='Objective')
        ax1.scatter(self.df['trial'], self.df['objective_value'], c='blue', alpha=0.6)
        ax1.set_ylabel('Objective Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot individual metrics
        ax2.plot(self.df['trial'], self.df['avg_return'], 'g-', label='Return')
        ax2.plot(self.df['trial'], -self.df['avg_validation_loss'], 'r-', label='-Val Loss')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Metric Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Optimization Trajectory')
        plt.tight_layout()
        
        return fig
    
    def plot_parallel_coordinates(self, top_k: int = 10) -> plt.Figure:
        """Create parallel coordinates plot for top configurations."""
        from pandas.plotting import parallel_coordinates
        
        # Get top k configurations
        top_configs = self.df.nlargest(top_k, 'objective_value')
        
        # Select key parameters for visualization
        param_cols = ['learning_rate', 'batch_size', 'horizon', 'entropy_coeff', 
                      'hidden_dim', 'num_transformer_layers']
        
        # Normalize parameters for visualization
        normalized_df = top_configs.copy()
        for col in param_cols:
            if col in normalized_df.columns:
                normalized_df[col] = (
                    (normalized_df[col] - self.df[col].min()) / 
                    (self.df[col].max() - self.df[col].min())
                )
        
        # Add performance class
        normalized_df['performance'] = pd.qcut(
            normalized_df['objective_value'], 
            q=[0, 0.5, 1.0], 
            labels=['Good', 'Best']
        )
        
        fig, ax = plt.subplots(figsize=(14, 8))
        parallel_coordinates(
            normalized_df[param_cols + ['performance']], 
            'performance',
            alpha=0.4,
            ax=ax
        )
        
        plt.xticks(rotation=45)
        plt.title(f'Parallel Coordinates - Top {top_k} Configurations')
        plt.tight_layout()
        
        return fig
    
    def generate_report(self) -> Path:
        """Generate a comprehensive HTML report."""
        from matplotlib.backends.backend_pdf import PdfPages
        
        report_path = self.output_dir / f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        with PdfPages(report_path) as pdf:
            # Overview page
            fig = self._create_overview_page()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Parameter impacts
            for param in ['learning_rate', 'batch_size', 'entropy_coeff']:
                if param in self.df.columns:
                    fig = self.plot_parameter_impact(param)
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Trajectory
            fig = self.plot_optimization_trajectory()
            pdf.savefig(fig)
            plt.close(fig)
            
            # Parallel coordinates
            fig = self.plot_parallel_coordinates()
            pdf.savefig(fig)
            plt.close(fig)
        
        logger.info(f"Generated report: {report_path}")
        return report_path
    
    def _create_overview_page(self) -> plt.Figure:
        """Create an overview page with key statistics."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'Hyperparameter Tuning Report', 
                ha='center', va='top', fontsize=20, weight='bold')
        
        # Best configuration
        best_idx = self.df['objective_value'].idxmax()
        best_config = self.df.iloc[best_idx]
        
        # Summary statistics
        summary_text = f"""
        Total Trials: {len(self.df)}
        Best Objective Value: {best_config['objective_value']:.4f}
        Best Average Return: {best_config['avg_return']:.4f}
        Best Validation Loss: {best_config['avg_validation_loss']:.4f}
        
        Best Configuration:
        - Learning Rate: {best_config['learning_rate']:.2e}
        - Batch Size: {int(best_config['batch_size'])}
        - Hidden Dim: {int(best_config['hidden_dim'])}
        - Entropy Coeff: {best_config['entropy_coeff']:.2e}
        """
        
        fig.text(0.1, 0.7, summary_text, ha='left', va='top', fontsize=12, 
                family='monospace')
        
        return fig


class HyperparameterTuner:
    """
    Production-ready hyperparameter tuner for DreamerV3Bubble.
    
    Features:
    - Parallel evaluation with resource management
    - Checkpointing and recovery
    - Dynamic configuration validation
    - Comprehensive visualization
    - Robust error handling
    """
    
    def __init__(self, context: SystemContext, config: Optional[TuningConfig] = None,
                 checkpoint_dir: Optional[Path] = None):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            context: SystemContext for creating DreamerV3Bubble instances
            config: TuningConfig object with tuning parameters
            checkpoint_dir: Directory for checkpointing (auto-created if None)
        """
        if not isinstance(context, SystemContext):
            raise TypeError(f"HyperparameterTuner requires a SystemContext instance, got {type(context)}")
        
        self.context = context
        self.config = config or TuningConfig()
        self.results = []
        self.failed_configs = []
        
        # Setup directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(f'tuning_run_{timestamp}')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir or self.run_dir / 'checkpoints'
        )
        
        self.plot_dir = self.run_dir / 'plots'
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Resource management
        self.resource_semaphore = asyncio.Semaphore(self.config.parallel_workers)
        self.active_evaluations = set()
        
        # Best configuration tracking
        self.best_config = None
        self.best_score = float('-inf')
        
        # State dimensions (will be set dynamically)
        self.state_dim = None
        self.action_dim = None
        
        logger.info(f"Initialized HyperparameterTuner with {self.config.parallel_workers} workers")
        logger.info(f"Run directory: {self.run_dir}")

    async def _get_bubble_dimensions(self, bubble_id: str) -> Tuple[int, int]:
        """Get state and action dimensions from a bubble."""
        bubble = self.context.get_bubble(bubble_id)
        if isinstance(bubble, DreamerV3Bubble):
            return bubble.config.state_dim, bubble.config.action_dim
        else:
            # Try to create a temporary bubble to get dimensions
            temp_bubble = DreamerV3Bubble(
                object_id=f"temp_{bubble_id}",
                context=self.context
            )
            await temp_bubble.ensure_initialized()
            dims = temp_bubble.config.state_dim, temp_bubble.config.action_dim
            del temp_bubble
            return dims

    @asynccontextmanager
    async def _resource_managed_evaluation(self, config_id: str):
        """Context manager for resource-managed evaluation."""
        await self.resource_semaphore.acquire()
        self.active_evaluations.add(config_id)
        
        try:
            yield
        finally:
            self.active_evaluations.discard(config_id)
            self.resource_semaphore.release()
            
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def evaluate_config(self, config: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate a hyperparameter configuration with retries and resource management.
        """
        config_id = self._generate_config_id(config)
        
        # Validate configuration
        is_valid, error_msg = ConfigValidator.validate(config, self.state_dim, self.action_dim)
        if not is_valid:
            logger.warning(f"Invalid config {config_id}: {error_msg}")
            return self._create_failed_result(config, error_msg)
        
        async with self._resource_managed_evaluation(config_id):
            for attempt in range(self.config.retry_attempts):
                try:
                    result = await self._evaluate_config_impl(config, config_id)
                    
                    # Validate result
                    if self._is_valid_result(result):
                        return result
                    else:
                        logger.warning(f"Invalid result for {config_id}, attempt {attempt + 1}")
                        
                except asyncio.TimeoutError:
                    logger.error(f"Timeout evaluating {config_id}, attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"Error evaluating {config_id}, attempt {attempt + 1}: {e}")
                
                # Exponential backoff
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
            
            # All attempts failed
            self.failed_configs.append(config)
            return self._create_failed_result(config, "Max retries exceeded")

    async def _evaluate_config_impl(self, config: Dict[str, float], config_id: str) -> Dict[str, float]:
        """Implementation of config evaluation."""
        # Create configuration object
        config_obj = DreamerV3Config(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=int(config['hidden_dim']),
            num_categories=32,
            num_classes=32,
            horizon=int(config['horizon']),
            num_transformer_layers=int(config['num_transformer_layers']),
            num_heads=int(config['num_heads']),
            batch_size=int(config['batch_size']),
            sequence_length=int(config['sequence_length']),
            learning_rate=config['learning_rate'],
            gradient_clip_norm=config['gradient_clip_norm'],
            entropy_coeff=config['entropy_coeff'],
            dropout_rate=config['dropout_rate'],
            weight_decay=config['weight_decay'],
            debug_mode=False
        )
        
        # Create bubble with timeout
        bubble = await asyncio.wait_for(
            self._create_bubble(config_id, config_obj),
            timeout=30.0
        )
        
        try:
            # Warmup phase
            logger.info(f"Starting warmup for {config_id}")
            for _ in range(self.config.warmup_iterations):
                bubble.collect_transitions(100)
                await bubble.train_world_model()
                await bubble.train_actor_critic()
            
            # Clear metrics after warmup
            bubble.training_metrics.clear()
            
            # Evaluation phase
            metrics_history = {
                'actor_loss': [], 'critic_loss': [], 'avg_return': [], 
                'validation_loss': [], 'entropy_variance': [], 'memory_usage': [],
                'training_time': [], 'throughput': []
            }
            
            start_time = time.time()
            transitions_processed = 0
            
            for i in range(self.config.eval_iterations):
                iter_start = time.time()
                
                # Collect transitions
                bubble.collect_transitions(self.config.num_transitions)
                transitions_processed += self.config.num_transitions
                
                # Train with timeout
                await asyncio.wait_for(
                    asyncio.gather(
                        bubble.train_world_model(),
                        bubble.train_actor_critic()
                    ),
                    timeout=60.0
                )
                
                # Collect metrics
                metrics = bubble.training_metrics
                for key in ['actor_loss', 'critic_loss', 'avg_return', 'validation_loss']:
                    if key in metrics and metrics[key] is not None:
                        metrics_history[key].append(float(metrics[key]))
                
                # Performance metrics
                iter_time = time.time() - iter_start
                metrics_history['training_time'].append(iter_time)
                metrics_history['throughput'].append(self.config.num_transitions / iter_time)
                
                # Memory usage
                if hasattr(bubble, '_get_memory_usage'):
                    memory = bubble._get_memory_usage()
                    if memory is not None:
                        metrics_history['memory_usage'].append(memory)
            
            # Calculate final metrics
            total_time = time.time() - start_time
            
            result = {
                'config': config,
                'avg_actor_loss': np.mean(metrics_history['actor_loss']) if metrics_history['actor_loss'] else float('inf'),
                'std_actor_loss': np.std(metrics_history['actor_loss']) if metrics_history['actor_loss'] else 0.0,
                'avg_critic_loss': np.mean(metrics_history['critic_loss']) if metrics_history['critic_loss'] else float('inf'),
                'avg_return': np.mean(metrics_history['avg_return']) if metrics_history['avg_return'] else -1000.0,
                'std_return': np.std(metrics_history['avg_return']) if metrics_history['avg_return'] else 0.0,
                'avg_validation_loss': np.mean(metrics_history['validation_loss']) if metrics_history['validation_loss'] else float('inf'),
                'avg_entropy_variance': np.mean(metrics_history['entropy_variance']) if metrics_history['entropy_variance'] else 0.0,
                'avg_memory_usage': np.mean(metrics_history['memory_usage']) if metrics_history['memory_usage'] else 0.0,
                'avg_throughput': transitions_processed / total_time,
                'total_time': total_time,
                'stability_ratio': self._calculate_stability(metrics_history)
            }
            
            # Calculate objective value
            result['objective_value'] = self._calculate_objective(result)
            
            logger.info(f"Completed {config_id}: return={result['avg_return']:.3f}, "
                       f"obj={result['objective_value']:.3f}, time={total_time:.1f}s")
            
            return result
            
        finally:
            # Ensure cleanup
            del bubble
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def _create_bubble(self, config_id: str, config: DreamerV3Config) -> DreamerV3Bubble:
        """Create and initialize a bubble instance."""
        bubble = DreamerV3Bubble(
            object_id=f"tune_{config_id}",
            context=self.context,
            config=config
        )
        await bubble.ensure_initialized()
        return bubble

    def _generate_config_id(self, config: Dict[str, float]) -> str:
        """Generate a unique ID for a configuration."""
        return (f"lr{config['learning_rate']:.1e}_"
                f"bs{int(config['batch_size'])}_"
                f"h{int(config['horizon'])}_"
                f"{int(time.time() % 10000)}")

    def _calculate_stability(self, metrics_history: Dict[str, List[float]]) -> float:
        """Calculate training stability from metrics history."""
        if not metrics_history['actor_loss'] or len(metrics_history['actor_loss']) < 2:
            return 0.0
        
        # Calculate coefficient of variation for losses
        actor_cv = np.std(metrics_history['actor_loss']) / (np.mean(metrics_history['actor_loss']) + 1e-8)
        critic_cv = np.std(metrics_history['critic_loss']) / (np.mean(metrics_history['critic_loss']) + 1e-8) if metrics_history['critic_loss'] else 0
        
        # Lower CV means more stable
        stability = 1.0 / (1.0 + actor_cv + critic_cv)
        return float(np.clip(stability, 0, 1))

    def _calculate_objective(self, result: Dict[str, float]) -> float:
        """Calculate multi-objective optimization value."""
        obj = (
            self.config.return_weight * result['avg_return']
            - self.config.validation_weight * result['avg_validation_loss']
            - self.config.stability_weight * (1 - result['stability_ratio'])
            + self.config.efficiency_weight * result['avg_throughput'] / 1000
        )
        
        # Penalize high memory usage
        if result['avg_memory_usage'] > self.config.resource_limit_gb:
            obj -= 10.0
        
        return float(obj)

    def _is_valid_result(self, result: Dict[str, float]) -> bool:
        """Check if evaluation result is valid."""
        required_keys = ['avg_return', 'avg_validation_loss', 'objective_value']
        
        for key in required_keys:
            if key not in result or not np.isfinite(result[key]):
                return False
        
        # Sanity checks
        if result['avg_return'] < -10000 or result['avg_return'] > 10000:
            return False
        
        return True

    def _create_failed_result(self, config: Dict[str, float], error_msg: str) -> Dict[str, float]:
        """Create a result dict for failed evaluations."""
        return {
            'config': config,
            'avg_actor_loss': float('inf'),
            'std_actor_loss': 0.0,
            'avg_critic_loss': float('inf'),
            'avg_return': -1000.0,
            'std_return': 0.0,
            'avg_validation_loss': float('inf'),
            'avg_entropy_variance': 0.0,
            'avg_memory_usage': 0.0,
            'avg_throughput': 0.0,
            'total_time': 0.0,
            'stability_ratio': 0.0,
            'objective_value': -1000.0,
            'error': error_msg
        }

    async def run_bayesian_search(self, target_bubble_id: Optional[str] = None) -> Optional[Dict]:
        """
        Run Bayesian optimization with parallel evaluation and checkpointing.
        """
        try:
            # Get state/action dimensions if not set
            if self.state_dim is None or self.action_dim is None:
                if target_bubble_id:
                    self.state_dim, self.action_dim = await self._get_bubble_dimensions(target_bubble_id)
                else:
                    # Default dimensions
                    self.state_dim, self.action_dim = 24, 5
                    
            logger.info(f"Using dimensions: state_dim={self.state_dim}, action_dim={self.action_dim}")
            
            # Try to load checkpoint
            checkpoint = await self.checkpoint_manager.load_latest()
            start_iteration = 0
            
            if checkpoint:
                self.results = checkpoint['state'].get('results', [])
                self.best_config = checkpoint['state'].get('best_config')
                self.best_score = checkpoint['state'].get('best_score', float('-inf'))
                start_iteration = checkpoint['iteration']
                logger.info(f"Resuming from iteration {start_iteration}")
            
            # Parameter bounds
            pbounds = {
                'learning_rate': (1e-5, 1e-3),
                'batch_size': (16, 128),
                'horizon': (10, 20),
                'entropy_coeff': (1e-4, 1e-2),
                'hidden_dim': (256, 1024),
                'num_transformer_layers': (1, 4),
                'num_heads': (4, 16),
                'sequence_length': (5, 20),
                'gradient_clip_norm': (1.0, 10.0),
                'dropout_rate': (0.05, 0.3),
                'weight_decay': (1e-5, 1e-3)
            }
            
            # Setup Bayesian optimization
            optimizer = BayesianOptimization(
                f=None,  # We'll use suggest/register pattern
                pbounds=pbounds,
                random_state=42,
                verbose=2
            )
            
            # Setup logging
            json_logger = JSONLogger(path=str(self.run_dir / "optimization_log.json"))
            optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)
            
            # Load previous logs if resuming
            if checkpoint and (self.run_dir / "optimization_log.json").exists():
                load_logs(optimizer, logs=[str(self.run_dir / "optimization_log.json")])
            
            # Run optimization loop
            for iteration in range(start_iteration, self.config.num_trials):
                # Get next suggestions (up to parallel_workers at once)
                suggestions = []
                for _ in range(min(self.config.parallel_workers, 
                                 self.config.num_trials - iteration)):
                    if iteration + len(suggestions) < self.config.num_trials:
                        suggestion = optimizer.suggest()
                        suggestions.append(suggestion)
                
                # Evaluate in parallel
                tasks = [self.evaluate_config(config) for config in suggestions]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for config, result in zip(suggestions, results):
                    if isinstance(result, Exception):
                        logger.error(f"Evaluation failed: {result}")
                        result = self._create_failed_result(config, str(result))
                    
                    # Register with optimizer
                    optimizer.register(params=config, target=result['objective_value'])
                    
                    # Store result
                    self.results.append(result)
                    
                    # Update best
                    if result['objective_value'] > self.best_score:
                        self.best_score = result['objective_value']
                        self.best_config = result['config']
                        logger.info(f"New best score: {self.best_score:.4f}")
                
                iteration += len(suggestions)
                
                # Checkpoint
                if iteration % self.config.checkpoint_interval == 0:
                    await self.checkpoint_manager.save_state({
                        'results': self.results,
                        'best_config': self.best_config,
                        'best_score': self.best_score,
                        'optimizer_state': optimizer.res
                    }, iteration)
                
                # Generate intermediate visualizations
                if len(self.results) >= 10 and iteration % 10 == 0:
                    await self._generate_intermediate_report()
            
            # Final analysis
            return await self._finalize_results()
            
        except Exception as e:
            logger.error(f"Bayesian search failed: {e}", exc_info=True)
            raise

    async def _generate_intermediate_report(self):
        """Generate intermediate visualization report."""
        try:
            valid_results = [r for r in self.results if 'error' not in r]
            if len(valid_results) < 5:
                return
                
            df = pd.DataFrame([{
                **r['config'],
                'avg_return': r['avg_return'],
                'avg_validation_loss': r['avg_validation_loss'],
                'objective_value': r['objective_value'],
                'stability_ratio': r['stability_ratio'],
                'avg_throughput': r['avg_throughput']
            } for r in valid_results])
            
            visualizer = TuningVisualizer(df, self.plot_dir / 'intermediate')
            await asyncio.to_thread(visualizer.generate_report)
            
        except Exception as e:
            logger.error(f"Failed to generate intermediate report: {e}")

    async def _finalize_results(self) -> Optional[Dict]:
        """Finalize results and generate final report."""
        valid_results = [r for r in self.results if 'error' not in r]
        
        if not valid_results:
            logger.error("No valid results to finalize")
            return None
        
        # Find best result
        best_result = max(valid_results, key=lambda r: r['objective_value'])
        
        # Generate final visualizations
        df = pd.DataFrame([{
            **r['config'],
            'avg_return': r['avg_return'],
            'avg_validation_loss': r['avg_validation_loss'],
            'objective_value': r['objective_value'],
            'stability_ratio': r['stability_ratio'],
            'avg_throughput': r['avg_throughput'],
            'avg_memory_usage': r['avg_memory_usage']
        } for r in valid_results])
        
        visualizer = TuningVisualizer(df, self.plot_dir)
        report_path = await asyncio.to_thread(visualizer.generate_report)
        
        # Save detailed results
        results_path = self.run_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'best_config': best_result['config'],
                'best_metrics': {k: v for k, v in best_result.items() if k != 'config'},
                'all_results': valid_results,
                'failed_configs': self.failed_configs,
                'report_path': str(report_path)
            }, f, indent=2)
        
        logger.info(f"Tuning complete. Best objective: {best_result['objective_value']:.4f}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Report saved to: {report_path}")
        
        return best_result
