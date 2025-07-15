import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import psutil

import torch
import numpy as np
from bubbles_core import (
    UniversalBubble, Actions, Event, UniversalCode, Tags, 
    SystemContext, logger, EventService
)
from tuning_utils import HyperparameterTuner, TuningConfig, ConfigValidator
from DreamerV3Bubble import DreamerV3Bubble, DreamerV3Config


@dataclass
class AdaptationConfig:
    """Configuration for dynamic adaptation based on system conditions."""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    gpu_threshold: float = 90.0
    
    # Adaptation multipliers
    high_cpu_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'learning_rate_multiplier': 0.9,
        'batch_size_divisor': 2,
        'entropy_coeff_multiplier': 1.5,
        'gradient_clip_multiplier': 0.8
    })
    
    high_memory_adjustments: Dict[str, float] = field(default_factory=lambda: {
        'batch_size_divisor': 2,
        'sequence_length_divisor': 2,
        'hidden_dim_divisor': 1.5
    })
    
    # Adaptation limits
    min_batch_size: int = 8
    min_sequence_length: int = 3
    max_entropy_coeff: float = 0.01
    min_learning_rate: float = 1e-6


class TuningScheduler:
    """Manages tuning schedule and decides when to run tuning."""
    
    def __init__(self, initial_interval: timedelta = timedelta(hours=1),
                 max_interval: timedelta = timedelta(days=1),
                 performance_threshold: float = 0.1):
        self.initial_interval = initial_interval
        self.max_interval = max_interval
        self.performance_threshold = performance_threshold
        
        self.last_tuning_time = None
        self.last_performance = None
        self.current_interval = initial_interval
        self.tuning_history = []
    
    def should_tune(self, current_performance: Optional[float] = None) -> bool:
        """Determine if tuning should run based on schedule and performance."""
        now = datetime.now()
        
        # First run
        if self.last_tuning_time is None:
            return True
        
        # Check time interval
        if now - self.last_tuning_time >= self.current_interval:
            return True
        
        # Check performance degradation
        if (current_performance is not None and 
            self.last_performance is not None and
            self.last_performance - current_performance > self.performance_threshold):
            logger.info(f"Performance degradation detected: {self.last_performance:.3f} -> {current_performance:.3f}")
            return True
        
        return False
    
    def update(self, tuning_result: Dict[str, Any]):
        """Update scheduler after tuning completes."""
        self.last_tuning_time = datetime.now()
        self.last_performance = tuning_result.get('avg_return', 0)
        self.tuning_history.append({
            'timestamp': self.last_tuning_time,
            'performance': self.last_performance,
            'config': tuning_result.get('config')
        })
        
        # Adaptive interval adjustment
        if len(self.tuning_history) >= 2:
            recent_improvements = [
                self.tuning_history[i]['performance'] - self.tuning_history[i-1]['performance']
                for i in range(-min(3, len(self.tuning_history)-1), 0)
            ]
            avg_improvement = np.mean(recent_improvements)
            
            if avg_improvement < 0.01:  # Minimal improvement
                # Increase interval up to max
                self.current_interval = min(
                    self.current_interval * 1.5,
                    self.max_interval
                )
                logger.info(f"Adjusted tuning interval to {self.current_interval}")
            elif avg_improvement > 0.1:  # Significant improvement
                # Decrease interval to tune more frequently
                self.current_interval = max(
                    self.current_interval * 0.75,
                    self.initial_interval
                )
                logger.info(f"Adjusted tuning interval to {self.current_interval}")


class ConfigCache:
    """Manages caching and retrieval of tuning configurations."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "config_cache.json"
        self.performance_history = self.cache_dir / "performance_history.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk."""
        self.cache = {}
        self.history = []
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config cache: {e}")
        
        if self.performance_history.exists():
            try:
                with open(self.performance_history, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load performance history: {e}")
    
    def save_config(self, bubble_id: str, config: Dict[str, Any], metrics: Dict[str, float]):
        """Save a configuration and its performance metrics."""
        self.cache[bubble_id] = {
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'version': 1
        }
        
        self.history.append({
            'bubble_id': bubble_id,
            'timestamp': datetime.now().isoformat(),
            'performance': metrics.get('avg_return', 0),
            'objective': metrics.get('objective_value', 0)
        })
        
        self._persist()
    
    def get_config(self, bubble_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached configuration for a bubble."""
        return self.cache.get(bubble_id)
    
    def get_best_config(self) -> Optional[Dict[str, Any]]:
        """Get the best performing configuration across all bubbles."""
        if not self.cache:
            return None
        
        best_bubble = max(
            self.cache.items(),
            key=lambda x: x[1]['metrics'].get('objective_value', float('-inf'))
        )
        
        return best_bubble[1]
    
    def _persist(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            
            with open(self.performance_history, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist cache: {e}")


class TuningBubble(UniversalBubble):
    """
    Production-ready hyperparameter tuning bubble for DreamerV3Bubble.
    
    Features:
    - Automatic scheduling of tuning runs
    - Dynamic adaptation based on system conditions
    - Configuration caching and reuse
    - Multi-bubble tuning support
    - Comprehensive monitoring and reporting
    """
    
    def __init__(self, object_id: str, context: SystemContext, 
                 target_bubble_ids: Optional[List[str]] = None,
                 dreamer_bubble_id: Optional[str] = None,  # For backward compatibility
                 tuning_config: Optional[TuningConfig] = None,
                 adaptation_config: Optional[AdaptationConfig] = None,
                 cache_dir: Optional[Path] = None,
                 auto_tune: bool = True):
        """
        Initialize the TuningBubble.
        
        Args:
            object_id: Unique identifier for this bubble
            context: System context for bubble operations
            target_bubble_ids: List of DreamerV3Bubble IDs to tune (None = tune all)
            dreamer_bubble_id: Single DreamerV3Bubble ID to tune (for backward compatibility)
            tuning_config: Configuration for hyperparameter tuning
            adaptation_config: Configuration for dynamic adaptation
            cache_dir: Directory for caching configurations
            auto_tune: Whether to automatically schedule tuning runs
        """
        super().__init__(object_id=object_id, context=context)
        
        # Configuration - handle both target_bubble_ids and dreamer_bubble_id
        if dreamer_bubble_id:
            # If single bubble ID provided, convert to list
            self.target_bubble_ids = [dreamer_bubble_id]
        else:
            self.target_bubble_ids = target_bubble_ids or []
        self.tuning_config = tuning_config or TuningConfig()
        self.adaptation_config = adaptation_config or AdaptationConfig()
        self.auto_tune = auto_tune
        
        # Setup directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = Path(cache_dir or f'tuning_{timestamp}')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.tuner = HyperparameterTuner(
            context=self.context,
            config=self.tuning_config,
            checkpoint_dir=self.base_dir / 'checkpoints'
        )
        
        self.scheduler = TuningScheduler()
        self.config_cache = ConfigCache(self.base_dir / 'cache')
        
        # State tracking
        self.active_tuning_tasks = {}
        self.adaptation_history = []
        self.system_metrics_buffer = []
        
        # Async initialization
        self._initialized = False
        self._init_task = asyncio.create_task(self._async_init())
        
        logger.info(f"{self.object_id}: Initialized TuningBubble")
        logger.info(f"Target bubbles: {self.target_bubble_ids or 'All DreamerV3Bubbles'}")
        logger.info(f"Auto-tune: {self.auto_tune}, Base directory: {self.base_dir}")

    async def _async_init(self):
        """Perform async initialization."""
        try:
            await self._subscribe_to_events()
            
            # Discover target bubbles if not specified
            if not self.target_bubble_ids:
                self.target_bubble_ids = await self._discover_dreamer_bubbles()
            
            # Load cached configurations
            for bubble_id in self.target_bubble_ids:
                cached = self.config_cache.get_config(bubble_id)
                if cached:
                    await self._apply_cached_config(bubble_id, cached)
            
            self._initialized = True
            logger.info(f"{self.object_id}: Async initialization complete")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed async initialization: {e}", exc_info=True)
            raise

    async def ensure_initialized(self):
        """Ensure async initialization is complete."""
        if not self._initialized:
            await self._init_task

    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        events_to_subscribe = [
            Actions.SYSTEM_STATE_UPDATE,
            Actions.DREAMER_PERFORMANCE_UPDATE,
            Actions.TUNING_REQUEST
        ]
        
        for event_type in events_to_subscribe:
            try:
                await EventService.subscribe(event_type, self.handle_event)
                logger.debug(f"{self.object_id}: Subscribed to {event_type}")
            except Exception as e:
                logger.error(f"{self.object_id}: Failed to subscribe to {event_type}: {e}")

    async def _discover_dreamer_bubbles(self) -> List[str]:
        """Discover all DreamerV3Bubble instances in the system."""
        dreamer_bubbles = []
        
        for bubble_id, bubble in self.context.bubbles.items():
            if isinstance(bubble, DreamerV3Bubble):
                dreamer_bubbles.append(bubble_id)
        
        logger.info(f"{self.object_id}: Discovered {len(dreamer_bubbles)} DreamerV3Bubbles")
        return dreamer_bubbles

    async def handle_event(self, event: Event):
        """Handle events for system monitoring and tuning requests."""
        try:
            if event.type == Actions.SYSTEM_STATE_UPDATE:
                await self._handle_system_state(event)
            
            elif event.type == Actions.DREAMER_PERFORMANCE_UPDATE:
                await self._handle_performance_update(event)
            
            elif event.type == Actions.TUNING_REQUEST:
                await self._handle_tuning_request(event)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event {event.type}: {e}")
        
        await super().handle_event(event)

    async def _handle_system_state(self, event: Event):
        """Handle system state updates for dynamic adaptation."""
        if isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
            state = event.data.value
            
            # Buffer system metrics
            self.system_metrics_buffer.append({
                'timestamp': datetime.now(),
                'cpu_percent': state.get('cpu_percent', 0),
                'memory_percent': state.get('memory_percent', 0),
                'gpu_percent': self._get_gpu_usage()
            })
            
            # Keep buffer size reasonable
            if len(self.system_metrics_buffer) > 100:
                self.system_metrics_buffer.pop(0)
            
            # Check for adaptation needs
            await self._check_adaptation_needed(state)

    async def _handle_performance_update(self, event: Event):
        """Handle performance updates from DreamerV3Bubbles."""
        if isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
            data = event.data.value
            bubble_id = data.get('bubble_id', event.origin)
            performance = data.get('avg_return', 0)
            
            # Check if tuning is needed based on performance
            if self.auto_tune and bubble_id in self.target_bubble_ids:
                if self.scheduler.should_tune(performance):
                    asyncio.create_task(self.run_tuning_for_bubble(bubble_id))

    async def _handle_tuning_request(self, event: Event):
        """Handle explicit tuning requests."""
        if isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
            data = event.data.value
            bubble_id = data.get('bubble_id')
            
            if bubble_id:
                asyncio.create_task(self.run_tuning_for_bubble(bubble_id))
            else:
                asyncio.create_task(self.run_tuning_all())

    async def _check_adaptation_needed(self, state: Dict[str, Any]):
        """Check if dynamic adaptation is needed based on system state."""
        cpu_usage = state.get('cpu_percent', 0)
        memory_usage = state.get('memory_percent', 0)
        gpu_usage = self._get_gpu_usage()
        
        # Determine if adaptation is needed
        adapt_needed = (
            cpu_usage > self.adaptation_config.cpu_threshold or
            memory_usage > self.adaptation_config.memory_threshold or
            gpu_usage > self.adaptation_config.gpu_threshold
        )
        
        if adapt_needed:
            # Apply adaptations to all target bubbles
            for bubble_id in self.target_bubble_ids:
                await self._apply_dynamic_adaptation(
                    bubble_id, cpu_usage, memory_usage, gpu_usage
                )

    async def _apply_dynamic_adaptation(self, bubble_id: str, 
                                      cpu_usage: float, memory_usage: float, 
                                      gpu_usage: float):
        """Apply dynamic parameter adjustments to a bubble."""
        bubble = self.context.get_bubble(bubble_id)
        if not isinstance(bubble, DreamerV3Bubble):
            return
        
        adaptations = []
        original_config = {
            'learning_rate': bubble.config.learning_rate,
            'batch_size': bubble.config.batch_size,
            'sequence_length': bubble.config.sequence_length,
            'entropy_coeff': bubble.config.entropy_coeff,
            'gradient_clip_norm': bubble.config.gradient_clip_norm
        }
        
        # High CPU adaptations
        if cpu_usage > self.adaptation_config.cpu_threshold:
            adj = self.adaptation_config.high_cpu_adjustments
            
            # Reduce learning rate
            new_lr = max(
                bubble.config.learning_rate * adj['learning_rate_multiplier'],
                self.adaptation_config.min_learning_rate
            )
            if new_lr != bubble.config.learning_rate:
                bubble.config.learning_rate = new_lr
                adaptations.append(f"lr: {original_config['learning_rate']:.2e} -> {new_lr:.2e}")
            
            # Increase entropy for exploration
            new_entropy = min(
                bubble.config.entropy_coeff * adj['entropy_coeff_multiplier'],
                self.adaptation_config.max_entropy_coeff
            )
            if new_entropy != bubble.config.entropy_coeff:
                bubble.config.entropy_coeff = new_entropy
                adaptations.append(f"entropy: {original_config['entropy_coeff']:.2e} -> {new_entropy:.2e}")
        
        # High memory adaptations
        if memory_usage > self.adaptation_config.memory_threshold or gpu_usage > self.adaptation_config.gpu_threshold:
            adj = self.adaptation_config.high_memory_adjustments
            
            # Reduce batch size
            new_batch_size = max(
                int(bubble.config.batch_size / adj['batch_size_divisor']),
                self.adaptation_config.min_batch_size
            )
            if new_batch_size != bubble.config.batch_size:
                bubble.config.batch_size = new_batch_size
                adaptations.append(f"batch_size: {original_config['batch_size']} -> {new_batch_size}")
            
            # Reduce sequence length
            new_seq_len = max(
                int(bubble.config.sequence_length / adj['sequence_length_divisor']),
                self.adaptation_config.min_sequence_length
            )
            if new_seq_len != bubble.config.sequence_length:
                bubble.config.sequence_length = new_seq_len
                adaptations.append(f"seq_length: {original_config['sequence_length']} -> {new_seq_len}")
        
        if adaptations:
            # Update optimizers with new parameters
            bubble._update_optimizer_params()
            
            # Log adaptation
            adaptation_record = {
                'timestamp': datetime.now().isoformat(),
                'bubble_id': bubble_id,
                'system_state': {
                    'cpu': cpu_usage,
                    'memory': memory_usage,
                    'gpu': gpu_usage
                },
                'adaptations': adaptations,
                'original_config': original_config
            }
            self.adaptation_history.append(adaptation_record)
            
            logger.info(f"{self.object_id}: Applied adaptations to {bubble_id}: {', '.join(adaptations)}")
            
            # Emit adaptation event
            await self._emit_adaptation_event(bubble_id, adaptation_record)

    async def run_tuning_for_bubble(self, bubble_id: str):
        """Run hyperparameter tuning for a specific bubble."""
        if bubble_id in self.active_tuning_tasks:
            logger.warning(f"{self.object_id}: Tuning already active for {bubble_id}")
            return
        
        await self.ensure_initialized()
        
        try:
            logger.info(f"{self.object_id}: Starting tuning for {bubble_id}")
            self.active_tuning_tasks[bubble_id] = True
            
            # Check cache first
            cached_config = self.config_cache.get_config(bubble_id)
            if cached_config and self._is_recent_config(cached_config):
                logger.info(f"{self.object_id}: Using recent cached config for {bubble_id}")
                await self._apply_cached_config(bubble_id, cached_config)
                return
            
            # Run tuning
            result = await self.tuner.run_bayesian_search(target_bubble_id=bubble_id)
            
            if result:
                # Apply best configuration
                await self._apply_tuning_result(bubble_id, result)
                
                # Update scheduler
                self.scheduler.update(result)
                
                # Cache configuration
                self.config_cache.save_config(
                    bubble_id,
                    result['config'],
                    {k: v for k, v in result.items() if k != 'config'}
                )
                
                # Emit tuning complete event
                await self._emit_tuning_complete_event(bubble_id, result)
                
                # Add chat message
                await self.add_chat_message(
                    f"Tuning completed for {bubble_id}. "
                    f"Best return: {result['avg_return']:.2f}, "
                    f"Objective: {result['objective_value']:.2f}"
                )
            else:
                logger.error(f"{self.object_id}: Tuning failed for {bubble_id}")
                await self.add_chat_message(f"Tuning failed for {bubble_id}")
        
        except Exception as e:
            logger.error(f"{self.object_id}: Error tuning {bubble_id}: {e}", exc_info=True)
            await self.add_chat_message(f"Tuning error for {bubble_id}: {str(e)}")
        
        finally:
            self.active_tuning_tasks.pop(bubble_id, None)
            
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def run_tuning_all(self):
        """Run tuning for all target bubbles."""
        await self.ensure_initialized()
        
        logger.info(f"{self.object_id}: Starting tuning for all bubbles")
        
        # Run tuning for each bubble sequentially to avoid resource conflicts
        for bubble_id in self.target_bubble_ids:
            if bubble_id not in self.active_tuning_tasks:
                await self.run_tuning_for_bubble(bubble_id)

    async def _apply_tuning_result(self, bubble_id: str, result: Dict[str, Any]):
        """Apply tuning result to a bubble."""
        bubble = self.context.get_bubble(bubble_id)
        if not isinstance(bubble, DreamerV3Bubble):
            logger.error(f"{self.object_id}: Bubble {bubble_id} not found or wrong type")
            return
        
        # Create new config from result
        new_config = self._create_config_from_dict(
            result['config'],
            bubble.config.state_dim,
            bubble.config.action_dim
        )
        
        # Apply configuration
        bubble.config = new_config
        bubble._rebuild_models()
        
        logger.info(f"{self.object_id}: Applied tuning result to {bubble_id}")

    async def _apply_cached_config(self, bubble_id: str, cached: Dict[str, Any]):
        """Apply a cached configuration to a bubble."""
        bubble = self.context.get_bubble(bubble_id)
        if not isinstance(bubble, DreamerV3Bubble):
            return
        
        config_dict = cached['config']
        new_config = self._create_config_from_dict(
            config_dict,
            bubble.config.state_dim,
            bubble.config.action_dim
        )
        
        bubble.config = new_config
        bubble._rebuild_models()
        
        logger.info(f"{self.object_id}: Applied cached config to {bubble_id} "
                   f"(from {cached['timestamp']})")

    def _create_config_from_dict(self, config_dict: Dict[str, Any], 
                               state_dim: int, action_dim: int) -> DreamerV3Config:
        """Create a DreamerV3Config object from a dictionary."""
        return DreamerV3Config(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(config_dict['hidden_dim']),
            num_categories=32,
            num_classes=32,
            horizon=int(config_dict['horizon']),
            num_transformer_layers=int(config_dict['num_transformer_layers']),
            num_heads=int(config_dict['num_heads']),
            batch_size=int(config_dict['batch_size']),
            sequence_length=int(config_dict['sequence_length']),
            learning_rate=config_dict['learning_rate'],
            gradient_clip_norm=config_dict['gradient_clip_norm'],
            entropy_coeff=config_dict['entropy_coeff'],
            dropout_rate=config_dict['dropout_rate'],
            weight_decay=config_dict['weight_decay'],
            debug_mode=False
        )

    def _is_recent_config(self, cached: Dict[str, Any], max_age_hours: int = 24) -> bool:
        """Check if a cached configuration is recent enough to reuse."""
        timestamp = datetime.fromisoformat(cached['timestamp'])
        age = datetime.now() - timestamp
        return age < timedelta(hours=max_age_hours)

    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        if torch.cuda.is_available():
            try:
                # This is a simplified version - you might want to use nvidia-ml-py
                return torch.cuda.utilization()
            except:
                return 0.0
        return 0.0

    async def _emit_adaptation_event(self, bubble_id: str, adaptation_record: Dict[str, Any]):
        """Emit an event about dynamic adaptation."""
        event_data = UniversalCode(
            Tags.DICT,
            {
                'bubble_id': bubble_id,
                'adaptations': adaptation_record['adaptations'],
                'system_state': adaptation_record['system_state']
            },
            description="Dynamic adaptation applied"
        )
        
        event = Event(
            type=Actions.ADAPTATION_UPDATE,
            data=event_data,
            origin=self.object_id,
            priority=3
        )
        
        await self.context.dispatch_event(event)

    async def _emit_tuning_complete_event(self, bubble_id: str, result: Dict[str, Any]):
        """Emit an event when tuning completes."""
        event_data = UniversalCode(
            Tags.DICT,
            {
                'bubble_id': bubble_id,
                'best_config': result['config'],
                'metrics': {k: v for k, v in result.items() if k != 'config'}
            },
            description="Hyperparameter tuning completed"
        )
        
        event = Event(
            type=Actions.TUNING_COMPLETE,
            data=event_data,
            origin=self.object_id,
            priority=2
        )
        
        await self.context.dispatch_event(event)

    async def autonomous_step(self):
        """Periodic tasks including scheduled tuning and monitoring."""
        await super().autonomous_step()
        await self.ensure_initialized()
        
        # Check for scheduled tuning every 300 steps (2.5 minutes at 0.5s/step)
        if self.auto_tune and self.execution_count % 300 == 0:
            current_performance = self._get_average_performance()
            
            if self.scheduler.should_tune(current_performance):
                logger.info(f"{self.object_id}: Scheduled tuning triggered")
                asyncio.create_task(self.run_tuning_all())
        
        # Generate status report every 1200 steps (10 minutes)
        if self.execution_count % 1200 == 0:
            await self._generate_status_report()
        
        # Cleanup old adaptation history every 600 steps (5 minutes)
        if self.execution_count % 600 == 0:
            self._cleanup_old_data()
        
        # Memory cleanup
        if self.execution_count % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        await asyncio.sleep(0.5)

    def _get_average_performance(self) -> Optional[float]:
        """Get average performance across all target bubbles."""
        performances = []
        
        for bubble_id in self.target_bubble_ids:
            bubble = self.context.get_bubble(bubble_id)
            if isinstance(bubble, DreamerV3Bubble):
                metrics = bubble.training_metrics
                if 'avg_return' in metrics:
                    performances.append(metrics['avg_return'])
        
        return np.mean(performances) if performances else None

    async def _generate_status_report(self):
        """Generate a status report of tuning activities."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'target_bubbles': len(self.target_bubble_ids),
            'active_tuning': len(self.active_tuning_tasks),
            'cached_configs': len(self.config_cache.cache),
            'adaptations_last_hour': len([
                a for a in self.adaptation_history
                if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)
            ]),
            'average_performance': self._get_average_performance(),
            'next_scheduled_tuning': (
                self.scheduler.last_tuning_time + self.scheduler.current_interval
                if self.scheduler.last_tuning_time else 'Now'
            )
        }
        
        logger.info(f"{self.object_id}: Status Report - {json.dumps(report, indent=2)}")
        
        # Save to file
        report_file = self.base_dir / 'status_reports' / f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

    def _cleanup_old_data(self):
        """Cleanup old adaptation history to prevent memory growth."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.adaptation_history = [
            a for a in self.adaptation_history
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]
        
        # Also cleanup system metrics buffer if too large
        if len(self.system_metrics_buffer) > 1000:
            self.system_metrics_buffer = self.system_metrics_buffer[-500:]
