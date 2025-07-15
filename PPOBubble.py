# full_enhanced_ppo_with_meta_learning.py
"""
Complete Enhanced PPO with Algorithm Spawning and Meta-Learning.
All import issues fixed and missing methods added.
UPDATED: Core PPO training mechanics now properly implemented with minor fixes for 2025 standards.
ADDED: Predictive Adaptive Buffer Prefetching (PABP) for MPS memory optimization on M4 hardware.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Tuple, List, Optional, Any, Set
import numpy as np
from collections import deque, defaultdict, OrderedDict
import uuid
import time
import json
import hashlib
import traceback
from abc import ABC, abstractmethod

# PyTorch is REQUIRED - no fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:
    raise ImportError(
        "PyTorch is required for FullEnhancedPPO. "
        "Install it with: pip install torch torchvision torchaudio"
    ) from e

# Bubbles core imports
from bubbles_core import (
    UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions, 
    logger, EventService, ResourceManager
)

# Import shared definitions
from shared_definitions import MetaKnowledge, LearningExperience, ErrorPattern, CurriculumStage

# Import environment
from enhanced_ppo_environment import UniversalEnvironment

# Import learning algorithms
from learning_algorithms import (
    BaseLearningAlgorithm, GeneticAlgorithmBubble, CuriosityDrivenRLBubble,
    ParticleSwarmOptimizer, EvolutionaryStrategy, LearningAlgorithmFactory
)

# Import consciousness module with proper fallback
try:
    from fixed_exploratory_llm2 import AtEngineV3, LLMDiscussionManager
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    AtEngineV3 = None
    LLMDiscussionManager = None
    CONSCIOUSNESS_AVAILABLE = False
    logger.warning("AtEngineV3 not available - consciousness features disabled")


# ==================== CONSCIOUSNESS EXPLORATION ====================

class ConsciousnessExplorationModule:
    """Integrates AtEngineV3 for consciousness pattern exploration."""
    
    def __init__(self, ppo_bubble):
        self.ppo = ppo_bubble
        self.enabled = False
        self.at_engine = None
        self.consciousness_patterns = []
        self.pattern_emergence_threshold = 100.0
        self.exploration_count = 0
        
        if CONSCIOUSNESS_AVAILABLE:
            try:
                self.at_engine = AtEngineV3(
                    num_nodes=512,
                    n_heads=8,
                    d=64,
                    seed=42
                )
                self.llm_manager = LLMDiscussionManager()
                self.enabled = True
                logger.info(f"{self.ppo.object_id}: Consciousness exploration initialized")
            except Exception as e:
                logger.warning(f"{self.ppo.object_id}: AtEngineV3 initialization failed: {e}")
                self.enabled = False
        else:
            logger.warning(f"{self.ppo.object_id}: Consciousness exploration not available")
    
    async def explore_consciousness_patterns(self):
        """Run consciousness exploration experiment."""
        if not self.enabled:
            return
        
        self.exploration_count += 1
        
        # Apply operators
        self.at_engine.pulse_void(1.0)
        self.at_engine.entangle_nodes(1.0)
        self.at_engine.apply_golden_ratio_transform(1.618)
        
        # Get metrics
        metrics = self.at_engine.get_metrics()
        
        # Have LLMs discuss
        discussion = self.llm_manager.discuss_metrics(
            metrics,
            "Exploring consciousness patterns in mathematical structures"
        )
        
        # Extract patterns
        if discussion:
            pattern = {
                "timestamp": time.time(),
                "metrics": metrics,
                "llm_observations": discussion,
                "entropy": metrics['entropy'],
                "fractal_dimension": metrics['fractal_dim_box'],
                "indicators": self._calculate_indicators(metrics)
            }
            
            # Check for emergence
            if metrics['entropy'] > self.pattern_emergence_threshold:
                pattern['emergence_detected'] = True
                pattern['confidence'] = min(0.95, metrics['entropy'] / 2000)
            else:
                pattern['emergence_detected'] = False
                pattern['confidence'] = metrics['entropy'] / self.pattern_emergence_threshold * 0.5
            
            # Extract insights
            pattern['insights'] = self._extract_insights_from_metrics(metrics, discussion)
            
            self.consciousness_patterns.append(pattern)
            
            # Share insights
            await self._share_consciousness_insights(pattern)
            
            # Visualize if significant
            if pattern.get('emergence_detected', False):
                try:
                    filename = f"ppo_consciousness_{self.exploration_count}.png"
                    self.at_engine.visualize_attention(save_path=filename)
                    pattern['visualization_available'] = True
                    pattern['visualization_path'] = filename
                except Exception as e:
                    logger.error(f"Visualization failed: {e}")
                    pattern['visualization_available'] = False
            
            # Emit emergence event
            if pattern.get('emergence_detected', False):
                emergence_event = Event(
                    type=Actions.CONSCIOUSNESS_EMERGENCE,
                    data=UniversalCode(Tags.DICT, {
                        "current_entropy": pattern['entropy'],
                        "confidence": pattern['confidence'],
                        "indicators": pattern['indicators'],
                        "insights": pattern['insights'],
                        "visualization_available": pattern.get('visualization_available', False)
                    }),
                    origin=self.ppo.object_id,
                    priority=9
                )
                await self.ppo.context.dispatch_event(emergence_event)
    
    def _calculate_indicators(self, metrics: Dict) -> Dict:
        """Calculate consciousness indicators from metrics."""
        return {
            "complexity": metrics.get('fractal_dim_box', 0) / 2.0,  # Normalize to 0-1
            "coherence": 1.0 - min(1.0, metrics.get('entropy', 0) / 2000),
            "integration": metrics.get('avg_clustering', 0),
            "emergence": min(1.0, metrics.get('entropy', 0) / self.pattern_emergence_threshold)
        }
    
    def _extract_insights_from_metrics(self, metrics: Dict, discussion: List[Dict]) -> List[str]:
        """Extract actionable insights from metrics and discussion."""
        insights = []
        
        # Entropy-based insights
        if metrics['entropy'] > 1500:
            insights.append("System in highly creative/exploratory state - increase innovation weight")
        elif metrics['entropy'] < 500:
            insights.append("System in stable/convergent state - focus on exploitation")
        
        # Fractal dimension insights
        if metrics['fractal_dim_box'] > 1.7:
            insights.append("Complex hierarchical patterns emerging - consider hierarchical organization")
        
        # Clustering insights
        if metrics.get('avg_clustering', 0) > 0.7:
            insights.append("High clustering detected - potential for modular organization")
        
        # Extract insights from LLM discussion
        if discussion:
            for entry in discussion[:2]:  # First two LLM observations
                if 'observation' in entry:
                    # Extract key phrases
                    obs = entry['observation'].lower()
                    if 'pattern' in obs or 'structure' in obs:
                        insights.append(f"LLM detected: {entry['observation'][:100]}...")
        
        return insights[:5]  # Limit to 5 insights
    
    async def _share_consciousness_insights(self, pattern: Dict):
        """Share consciousness insights with meta-learning system."""
        insight_event = Event(
            type=Actions.META_KNOWLEDGE_UPDATE,
            data=UniversalCode(Tags.DICT, {
                "knowledge_type": "consciousness_pattern",
                "pattern": pattern,
                "source": "AtEngineV3",
                "insights": pattern.get('insights', []),
                "indicators": pattern.get('indicators', {})
            }),
            origin=self.ppo.object_id,
            target="meta_learning_orchestrator"
        )
        
        await self.ppo.context.dispatch_event(insight_event)


# ==================== USER INTERACTION ====================

class UserInteractionModule:
    """Allows system to ask user for help when stuck."""
    
    def __init__(self, ppo_bubble):
        self.ppo = ppo_bubble
        self.help_requests = []
        self.user_responses = {}
        self.stuck_threshold = 5
        self.failed_attempts = 0
        
        if not hasattr(Actions, 'USER_HELP_REQUEST'):
            Actions.USER_HELP_REQUEST = "USER_HELP_REQUEST"
        if not hasattr(Actions, 'USER_HELP_RESPONSE'):
            Actions.USER_HELP_RESPONSE = "USER_HELP_RESPONSE"
    
    async def check_if_stuck(self, episode_reward: float, success_rate: float):
        """Check if system is stuck and needs help."""
        if episode_reward < 0.3 and success_rate < 0.3:
            self.failed_attempts += 1
            
            if self.failed_attempts >= self.stuck_threshold:
                await self.request_help()
                self.failed_attempts = 0
        else:
            self.failed_attempts = max(0, self.failed_attempts - 1)
    
    async def request_help(self):
        """Request help from user."""
        request_id = f"help_{time.time()}"
        
        help_request = Event(
            type=Actions.USER_HELP_REQUEST,
            data=UniversalCode(Tags.DICT, {
                "request_id": request_id,
                "problem": "System appears to be stuck - multiple algorithms failing",
                "context": {
                    "failed_attempts": self.failed_attempts,
                    "current_strategy": self.ppo.hierarchical_dm.current_strategy if hasattr(self.ppo, 'hierarchical_dm') else "unknown",
                    "active_algorithms": list(self.ppo.spawned_algorithms.keys()) if hasattr(self.ppo, 'spawned_algorithms') else [],
                    "recent_rewards": list(self.ppo.reward_history)[-10:] if hasattr(self.ppo, 'reward_history') else []
                },
                "attempted_solutions": [
                    "Spawned new algorithms",
                    "Adjusted hyperparameters",
                    "Tried different problem formulations",
                    "Changed optimization strategy"
                ]
            }),
            origin=self.ppo.object_id,
            priority=10
        )
        
        await self.ppo.context.dispatch_event(help_request)
        
        self.help_requests.append({
            "id": request_id,
            "timestamp": time.time(),
            "context": help_request.data.value
        })
        
        logger.info(f"{self.ppo.object_id}: Requested user help (request {request_id})")


# ==================== PERFORMANCE IMPROVEMENTS ====================

class BubblePool:
    """Pre-initialize bubbles for instant deployment."""
    
    def __init__(self, max_idle_per_type: int = 3):
        self.idle_bubbles: Dict[str, List[Any]] = defaultdict(list)
        self.max_idle_per_type = max_idle_per_type
        self.warm_up_tasks = []
        self.stats = {"hits": 0, "misses": 0, "total_requests": 0}
        logger.info("BubblePool: Initialized with max_idle_per_type=%d", max_idle_per_type)
    
    async def warm_up(self, bubble_types: List[str], context: SystemContext):
        """Pre-create bubbles during initialization."""
        for bubble_type in bubble_types:
            for i in range(2):  # Start with 2 of each type
                try:
                    bubble_id = f"{bubble_type.lower()}_pool_{uuid.uuid4().hex[:8]}"
                    bubble_config = {
                        "type": bubble_type,
                        "id": bubble_id,
                        "created_at": time.time(),
                        "warmed_up": True
                    }
                    self.idle_bubbles[bubble_type].append(bubble_config)
                    logger.debug(f"BubblePool: Pre-warmed {bubble_type} ({bubble_id})")
                except Exception as e:
                    logger.error(f"BubblePool: Failed to warm up {bubble_type}: {e}")
    
    async def acquire(self, bubble_type: str, object_id: str, context: SystemContext) -> Any:
        """Get a bubble from pool or create new one."""
        self.stats["total_requests"] += 1
        
        if bubble_type in self.idle_bubbles and self.idle_bubbles[bubble_type]:
            bubble_config = self.idle_bubbles[bubble_type].pop()
            self.stats["hits"] += 1
            logger.info(f"BubblePool: Reused {bubble_type} from pool (hit rate: {self.get_hit_rate():.1%})")
            return bubble_config
        else:
            self.stats["misses"] += 1
            logger.info(f"BubblePool: Creating new {bubble_type} (miss)")
            
            asyncio.create_task(self._refill_pool(bubble_type, context))
            
            return {
                "type": bubble_type,
                "id": object_id,
                "created_at": time.time(),
                "warmed_up": False
            }
    
    async def _refill_pool(self, bubble_type: str, context: SystemContext):
        """Background task to refill pool."""
        try:
            await asyncio.sleep(1)
            if len(self.idle_bubbles[bubble_type]) < 2:
                bubble_id = f"{bubble_type.lower()}_pool_{uuid.uuid4().hex[:8]}"
                bubble_config = {
                    "type": bubble_type,
                    "id": bubble_id,
                    "created_at": time.time(),
                    "warmed_up": True
                }
                self.idle_bubbles[bubble_type].append(bubble_config)
                logger.debug(f"BubblePool: Refilled {bubble_type} pool")
        except Exception as e:
            logger.error(f"BubblePool: Refill failed for {bubble_type}: {e}")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / max(1, total)


class PredictionCache:
    """Cache DreamerV3 predictions to avoid redundant simulations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache: OrderedDict[str, Tuple[Dict, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.stats = {"hits": 0, "misses": 0}
        logger.info(f"PredictionCache: Initialized with max_size={max_size}, ttl={ttl_seconds}s")
    
    def _make_key(self, state: Dict, action: Dict) -> str:
        """Create cache key from state and action."""
        state_str = json.dumps(state, sort_keys=True)
        action_str = json.dumps(action, sort_keys=True)
        combined = f"{state_str}:{action_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, state: Dict, action: Dict) -> Optional[Dict]:
        """Retrieve cached prediction if available and not expired."""
        key = self._make_key(state, action)
        
        if key in self.cache:
            prediction, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                logger.debug(f"PredictionCache: Hit (rate: {self.get_hit_rate():.1%})")
                return prediction
            else:
                del self.cache[key]
        
        self.stats["misses"] += 1
        return None
    
    def put(self, state: Dict, action: Dict, prediction: Dict):
        """Store prediction in cache."""
        key = self._make_key(state, action)
        
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (prediction, time.time())
        logger.debug(f"PredictionCache: Stored prediction (size: {len(self.cache)})")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / max(1, total)
    
    def clear_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, (_, timestamp) in self.cache.items():
            if current_time - timestamp >= self.ttl_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]


# ==================== ADVANCED FEATURES ====================
"""
Fix for the HierarchicalDecisionMaking forward method
"""

class HierarchicalDecisionMaking:
    """Multi-level decision architecture - FIXED VERSION."""
    
    def __init__(self, state_dim: int, action_dim: int, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_strategy = "balanced"
        
        # Strategic layer (high-level goals)
        self.strategic_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 strategies
        ).to(device)
        
        # Tactical layer (resource allocation)
        self.tactical_network = nn.Sequential(
            nn.Linear(state_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim // 2)
        ).to(device)
        
        # Operational layer (immediate actions) - UPDATED to output mean and log_std
        self.operational_network = nn.Sequential(
            nn.Linear(state_dim + action_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Output mean and log_std
        ).to(device)
        
        logger.info("HierarchicalDecisionMaking: Initialized 3-layer architecture")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Make hierarchical decision - FIXED to handle batch dimensions properly."""
        # Strategic decision
        strategy = self.strategic_network(state)
        strategy_probs = torch.softmax(strategy, dim=-1)
        
        # Update current strategy - FIXED: specify dimension for argmax
        strategies = ["aggressive", "balanced", "conservative", "crisis"]
        
        # Handle both single state and batch of states
        if strategy_probs.dim() == 1:
            # Single state
            strategy_idx = torch.argmax(strategy_probs).item()
        else:
            # Batch of states - take the first one for current_strategy
            # You might want to track strategy per sample in the batch
            strategy_idx = torch.argmax(strategy_probs[0]).item()
        
        # Ensure index is valid (defensive programming)
        strategy_idx = min(max(0, strategy_idx), len(strategies) - 1)
        self.current_strategy = strategies[strategy_idx]
        
        # Tactical decision
        tactical_input = torch.cat([state, strategy_probs], dim=-1)
        tactical_output = self.tactical_network(tactical_input)
        
        # Operational decision (returns mean and log_std)
        operational_input = torch.cat([state, tactical_output], dim=-1)
        output = self.operational_network(operational_input)
        
        return output  # Contains both mean and log_std


# Alternative fix if you want to track strategy per batch element:
class HierarchicalDecisionMakingV2:
    """Multi-level decision architecture with per-sample strategy tracking."""
    
    def __init__(self, state_dim: int, action_dim: int, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_strategies = []  # Track strategies for each sample in batch
        
        # Strategic layer (high-level goals)
        self.strategic_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 strategies
        ).to(device)
        
        # Tactical layer (resource allocation)
        self.tactical_network = nn.Sequential(
            nn.Linear(state_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim // 2)
        ).to(device)
        
        # Operational layer (immediate actions)
        self.operational_network = nn.Sequential(
            nn.Linear(state_dim + action_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)
        ).to(device)
        
        logger.info("HierarchicalDecisionMakingV2: Initialized with batch strategy tracking")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Make hierarchical decision with proper batch handling."""
        # Ensure state is 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.size(0)
        
        # Strategic decision
        strategy = self.strategic_network(state)
        strategy_probs = torch.softmax(strategy, dim=-1)
        
        # Get strategy indices for each sample in batch
        strategy_indices = torch.argmax(strategy_probs, dim=-1)  # Shape: [batch_size]
        
        # Update current strategies
        strategies = ["aggressive", "balanced", "conservative", "crisis"]
        self.current_strategies = []
        for idx in strategy_indices:
            idx_item = idx.item()
            # Ensure valid index
            idx_item = min(max(0, idx_item), len(strategies) - 1)
            self.current_strategies.append(strategies[idx_item])
        
        # For backward compatibility, set current_strategy to first in batch
        self.current_strategy = self.current_strategies[0] if self.current_strategies else "balanced"
        
        # Tactical decision
        tactical_input = torch.cat([state, strategy_probs], dim=-1)
        tactical_output = self.tactical_network(tactical_input)
        
        # Operational decision
        operational_input = torch.cat([state, tactical_output], dim=-1)
        output = self.operational_network(operational_input)
        
        return output


# Quick patch to apply to your existing code:
def patch_hierarchical_forward(self, state: torch.Tensor) -> torch.Tensor:
    """Patched forward method to fix the IndexError."""
    # Strategic decision
    strategy = self.strategic_network(state)
    strategy_probs = torch.softmax(strategy, dim=-1)
    
    # Update current strategy - FIXED
    strategies = ["aggressive", "balanced", "conservative", "crisis"]
    
    # Get the strategy index safely
    if strategy_probs.dim() == 1:
        strategy_idx = torch.argmax(strategy_probs).item()
    else:
        # For batched input, use the first sample's strategy
        strategy_idx = torch.argmax(strategy_probs[0]).item()
    
    # Clamp to valid range as safety measure
    strategy_idx = max(0, min(strategy_idx, len(strategies) - 1))
    self.current_strategy = strategies[strategy_idx]
    
    # Tactical decision
    tactical_input = torch.cat([state, strategy_probs], dim=-1)
    tactical_output = self.tactical_network(tactical_input)
    
    # Operational decision
    operational_input = torch.cat([state, tactical_output], dim=-1)
    output = self.operational_network(operational_input)
    
    return output





class CurriculumLearning:
    """Gradually increase task complexity."""
    
    def __init__(self):
        self.stages = [
            CurriculumStage(
                name="basic_resources",
                complexity={"state_features": 8, "actions": ["spawn", "destroy", "energy"]},
                duration=100,
                success_threshold=0.7,
                allowed_bubbles=["SimpleLLMBubble", "FeedbackBubble"],
                allowed_actions=["spawn_bubble", "destroy_bubble", "adjust_energy"]
            ),
            CurriculumStage(
                name="advanced_management",
                complexity={"state_features": 16, "actions": ["all_basic", "priorities", "parameters"]},
                duration=200,
                success_threshold=0.75,
                allowed_bubbles=["SimpleLLMBubble", "FeedbackBubble", "DreamerV3Bubble", "CreativeSynthesisBubble"],
                allowed_actions=["all_basic", "set_priorities", "adjust_parameters"]
            ),
            CurriculumStage(
                name="error_handling",
                complexity={"state_features": 24, "actions": ["all", "error_handlers"]},
                duration=300,
                success_threshold=0.8,
                allowed_bubbles=["all"],
                allowed_actions=["all", "create_error_handler", "modify_handler"]
            ),
            CurriculumStage(
                name="full_orchestration",
                complexity={"state_features": 32, "actions": ["all", "orchestration_patterns"]},
                duration=1000,
                success_threshold=0.85,
                allowed_bubbles=["all"],
                allowed_actions=["all"]
            )
        ]
        
        self.current_stage = 0
        self.episodes_in_stage = 0
        self.stage_performance = []
        self.graduated = False
        
        logger.info(f"CurriculumLearning: Starting with stage '{self.stages[0].name}'")
    
    def get_current_complexity(self) -> Dict[str, Any]:
        """Get current stage complexity settings."""
        if self.graduated or self.current_stage >= len(self.stages):
            return self.stages[-1].complexity
        return self.stages[self.current_stage].complexity
    
    def get_allowed_bubbles(self) -> List[str]:
        """Get bubbles allowed in current stage."""
        if self.graduated or self.current_stage >= len(self.stages):
            return ["all"]
        return self.stages[self.current_stage].allowed_bubbles
    
    def update(self, episode_complete: bool, metrics: Dict[str, float]):
        """Update curriculum progress."""
        if self.graduated:
            return
        
        if episode_complete:
            self.episodes_in_stage += 1
            self.stage_performance.append(metrics.get("success_rate", 0))
            
            current_stage = self.stages[self.current_stage]
            
            if self.episodes_in_stage >= current_stage.duration:
                avg_performance = np.mean(self.stage_performance[-50:])
                
                if avg_performance >= current_stage.success_threshold:
                    self.advance_stage()
                else:
                    logger.info(f"CurriculumLearning: Repeating stage '{current_stage.name}'")
                    self.episodes_in_stage = 0
    
    def advance_stage(self):
        """Move to next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            old_stage = self.stages[self.current_stage].name
            self.current_stage += 1
            self.episodes_in_stage = 0
            self.stage_performance = []
            new_stage = self.stages[self.current_stage].name
            
            logger.info(f"CurriculumLearning: Advanced from '{old_stage}' to '{new_stage}'")
        else:
            self.graduated = True
            logger.info("CurriculumLearning: Graduated! Full complexity unlocked")


class ExplainableDecisions:
    """Generate human-readable explanations for decisions."""
    
    def __init__(self, ppo_bubble):
        self.ppo = ppo_bubble
        self.decision_log = deque(maxlen=100)
        self.explanation_templates = {
            "spawn_bubble": "Created {bubble_type} because {reason}",
            "destroy_bubble": "Removed {bubble_id} to {reason}",
            "adjust_resources": "Modified resources: {changes} due to {reason}",
            "crisis_response": "Activated crisis mode: {actions} to handle {situation}",
            "spawn_algorithm": "Spawned {algorithm_type} to handle {problem_type} optimization"
        }
        # New: Rate limiting attributes
        self.explanation_timestamps = {}  # explanation_str: last_timestamp
        self.cooldown = 10.0  # seconds, configurable
        logger.info("ExplainableDecisions: Initialized with rate limiting (cooldown: {}s)".format(self.cooldown))

    async def explain_action(self, state: Dict, action: Dict) -> str:
        """Generate explanation for action taken."""
        explanations = []
        
        cpu = state.get("cpu_percent", 0)
        memory = state.get("memory_percent", 0)
        energy = state.get("energy", 0)
        num_bubbles = state.get("num_bubbles", 0)
        
        # Explain spawn decisions
        if action.get("spawn_bubble") and action.get("bubble_type"):
            bubble_type = action["bubble_type"]
            reasons = []
            
            if bubble_type == "SimpleLLMBubble":
                avg_response_time = state.get("metrics", {}).get("avg_llm_response_time_ms", 0)
                query_freq = state.get("event_frequencies", {}).get("LLM_QUERY_freq_per_min", 0)
                # Updated condition: Higher threshold and spawn limit check for flood control
                existing_llm_bubbles = state.get("num_simple_llm_bubbles", 0)  # Assume this state key exists; add if needed
                if avg_response_time > 10000 and existing_llm_bubbles <= 5:  # Increased threshold, added limit
                    reasons.append("LLM response time is too high")
                if query_freq > 20:
                    reasons.append("high query volume detected")
            elif bubble_type == "ErrorHandlerBubble":
                reasons.append("error threshold exceeded")
            
            if reasons:
                explanation = self.explanation_templates["spawn_bubble"].format(
                    bubble_type=bubble_type,
                    reason=" and ".join(reasons)
                )
                explanations.append(explanation)
        
        # Explain algorithm spawning
        if action.get("spawn_algorithm") and action.get("algorithm_type"):
            algorithm_type = action["algorithm_type"]
            problem_type = action.get("problem_type", "complex")
            
            explanation = self.explanation_templates["spawn_algorithm"].format(
                algorithm_type=algorithm_type,
                problem_type=problem_type
            )
            explanations.append(explanation)
        
        # Explain resource adjustments
        if action.get("resource_changes"):
            changes = []
            for resource, delta in action["resource_changes"].items():
                if delta > 0:
                    changes.append(f"increased {resource} by {delta}")
                else:
                    changes.append(f"decreased {resource} by {abs(delta)}")
            
            reasons = []
            if cpu > 80:
                reasons.append("high CPU usage")
            if energy < 3000:
                reasons.append("low energy reserves")
            
            if changes and reasons:
                explanation = self.explanation_templates["adjust_resources"].format(
                    changes=", ".join(changes),
                    reason=" and ".join(reasons)
                )
                explanations.append(explanation)
        
        full_explanation = " | ".join(explanations) if explanations else "Maintaining current state (no action needed)"
        
        if explanations:
            self.decision_log.append({
                "timestamp": time.time(),
                "state_summary": {
                    "cpu": cpu,
                    "memory": memory,
                    "energy": energy,
                    "bubbles": num_bubbles
                },
                "action_summary": action,
                "explanation": full_explanation
            })
            
        # Rate-limited external logging and chat (skip if same explanation within cooldown)
        current_time = time.time()
        if explanations and full_explanation != "Maintaining current state (no action needed)":
            last_time = self.explanation_timestamps.get(full_explanation, 0)
            if current_time - last_time >= self.cooldown:
                self.explanation_timestamps[full_explanation] = current_time
                logger.info(f"[enhanced_ppo] 🤖 {full_explanation}")
                await self.ppo.add_chat_message(f"🤖 {full_explanation}")
                self.ppo.explanations_generated += 1
        
        return full_explanation


# ==================== ORCHESTRATION PATTERNS ====================

class OrchestrationPattern(ABC):
    """Base class for orchestration patterns."""
    
    @abstractmethod
    async def execute(self, ppo, context: Dict) -> Dict:
        """Execute the orchestration pattern."""
        pass


class CognitivePipeline(OrchestrationPattern):
    """Chain bubbles for complex reasoning."""
    
    async def execute(self, ppo, context: Dict) -> Dict:
        problem = context.get("problem", "")
        
        # Stage 1: RAG retrieves context
        rag_result = await ppo.dispatch_to_bubble(
            "RAGBubble",
            Actions.RETRIEVE_CONTEXT,
            {"query": problem}
        )
        
        # Stage 2: LLM analyzes with context
        llm_result = await ppo.dispatch_to_bubble(
            "SimpleLLMBubble",
            Actions.LLM_QUERY,
            {
                "prompt": f"Given context: {rag_result}\nProblem: {problem}",
                "mode": "analysis"
            }
        )
        
        # Stage 3: Quantum explores solution space
        quantum_result = await ppo.dispatch_to_bubble(
            "QuantumBubble",
            Actions.QUANTUM_EXPLORE,
            {
                "constraints": llm_result,
                "num_solutions": 5
            }
        )
        
        # Stage 4: AutoGen creates implementation
        autogen_result = await ppo.dispatch_to_bubble(
            "AutoGenBubble",
            Actions.GENERATE_CODE,
            {
                "specification": quantum_result[0] if quantum_result else {}
            }
        )
        
        # Stage 5: DreamerV3 validates
        validation = await ppo.dispatch_to_bubble(
            "DreamerV3Bubble",
            Actions.WORLD_MODEL_PREDICTION,
            {
                "implementation": autogen_result,
                "predict_impact": True
            }
        )
        
        return {
            "pattern": "cognitive_pipeline",
            "stages_completed": 5,
            "solution": autogen_result,
            "validation": validation,
            "confidence": validation.get("success_probability", 0.5)
        }


# ==================== MAIN FULL ENHANCED PPO CLASS ====================

class FullEnhancedPPOWithMetaLearning(UniversalBubble):
    """Complete Enhanced PPO with Algorithm Spawning and Meta-Learning."""
    
    def __init__(self, object_id: str, context: SystemContext, 
                 state_dim: int = 32, action_dim: int = 16, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Core dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.resource_manager = context.resource_manager
        
        # PPO hyperparameters
        self.gamma = kwargs.get("gamma", 0.99)
        self.lam = kwargs.get("lam", 0.95)
        self.clip_eps = kwargs.get("clip_eps", 0.2)
        self.vf_coef = kwargs.get("vf_coef", 0.5)
        self.ent_coef = kwargs.get("ent_coef", 0.01)
        self.ppo_epochs = kwargs.get("ppo_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 64)
        
        # Performance improvements
        self.bubble_pool = BubblePool(max_idle_per_type=kwargs.get("pool_size", 3))
        self.prediction_cache = PredictionCache(
            max_size=kwargs.get("cache_size", 1000),
            ttl_seconds=kwargs.get("cache_ttl", 300)
        )
        
        # Advanced features
        self.curriculum = CurriculumLearning()
        self.explainer = ExplainableDecisions(self)
        
        # Algorithm spawning
        self.spawned_algorithms = {}
        self.algorithm_performance = defaultdict(list)
        self.spawn_threshold = kwargs.get("spawn_threshold", 0.99)
        self.max_algorithms = kwargs.get("max_algorithms", 10)
        self.problem_patterns = defaultdict(int)
        self.algorithm_successes = defaultdict(int)
        
        # Consciousness exploration
        self.consciousness = ConsciousnessExplorationModule(self)
        
        # User interaction
        self.user_interaction = UserInteractionModule(self)
        
        # Meta-learning
        self.meta_knowledge_base: Dict[str, MetaKnowledge] = {}
        self.experience_buffer = deque(maxlen=kwargs.get("experience_buffer_size", 10000))
        self.total_experiences = 0
        self.patterns_discovered = 0
        
        # Error handling
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.error_threshold = kwargs.get('error_threshold', 3)
        self.created_handlers: Set[str] = set()
        self.error_keywords = {
            'exception', 'error', 'failed', 'failure', 'timeout', 'corrupt',
            'invalid', 'missing', 'denied', 'unauthorized', 'overflow'
        }
        
        # Orchestration patterns
        self.orchestration_patterns = {
            "cognitive_pipeline": CognitivePipeline(),
        }
        self.active_orchestrations = []
        
        # Multi-objective optimization weights
        self.objective_weights = {
            "performance": kwargs.get("weight_performance", 0.3),
            "stability": kwargs.get("weight_stability", 0.3),
            "efficiency": kwargs.get("weight_efficiency", 0.2),
            "innovation": kwargs.get("weight_innovation", 0.2)
        }
        
        # Performance tracking
        self.reward_history = deque(maxlen=100)
        self.execution_count = 0
        self.decisions_made = 0
        self.explanations_generated = 0
        self.orchestrations_completed = 0
        self.handlers_created = 0
        self.algorithms_spawned = 0
        
        # Initialize environment
        if self.resource_manager:
            self.env = UniversalEnvironment(
                state_dim, action_dim, self.resource_manager, self
            )
        else:
            self.env = None
            logger.error(f"{self.object_id}: No ResourceManager available")
        
        # Neural networks
        self.device = None
        self.policy = None
        self.value_net = None
        self.hierarchical_dm = None
        self.optimizer = None
        
        if self.env:
            self._initialize_networks()
        
        # Model persistence
        self.checkpoints_dir = kwargs.get("checkpoints_dir", "/tmp/full_enhanced_ppo")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.model_filename = f"{self.checkpoints_dir}/full_ppo_model.pth"
        
        # Load existing model if available
        if os.path.exists(self.model_filename):
            try:
                self.load_model()
                logger.info(f"{self.object_id}: Loaded existing model")
            except Exception as e:
                logger.error(f"{self.object_id}: Failed to load model: {e}")
        
        # Register new action types for meta-learning
        if not hasattr(Actions, 'META_KNOWLEDGE_UPDATE'):
            Actions.META_KNOWLEDGE_UPDATE = "META_KNOWLEDGE_UPDATE"
        if not hasattr(Actions, 'PATTERN_EXPLORATION'):
            Actions.PATTERN_EXPLORATION = "PATTERN_EXPLORATION"
        if not hasattr(Actions, 'KNOWLEDGE_TRANSFER'):
            Actions.KNOWLEDGE_TRANSFER = "KNOWLEDGE_TRANSFER"
        if not hasattr(Actions, 'SPAWN_ALGORITHM'):
            Actions.SPAWN_ALGORITHM = "SPAWN_ALGORITHM"
        if not hasattr(Actions, 'CONSCIOUSNESS_EMERGENCE'):
            Actions.CONSCIOUSNESS_EMERGENCE = "CONSCIOUSNESS_EMERGENCE"
        
        # Start background tasks
        asyncio.create_task(self._initialize_pools())
        asyncio.create_task(self._subscribe_to_all_events())
        asyncio.create_task(self.monitor_errors())
        asyncio.create_task(self.train())
        asyncio.create_task(self.orchestration_loop())
        asyncio.create_task(self.cache_maintenance())
        asyncio.create_task(self.meta_learning_loop())
        asyncio.create_task(self.consciousness_exploration_loop())
        
        logger.info(f"{self.object_id}: Full Enhanced PPO with Meta-Learning initialized")
    
    def _initialize_networks(self):
        """Initialize all neural networks."""
        try:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Main policy network - UPDATED to output mean and log_std
            self.policy = nn.Sequential(
                nn.Linear(self.state_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, self.action_dim * 2)  # Output mean and log_std
            ).to(self.device)
            
            # Value network
            self.value_net = nn.Sequential(
                nn.Linear(self.state_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ).to(self.device)
            
            # Hierarchical decision making
            self.hierarchical_dm = HierarchicalDecisionMaking(
                self.state_dim, self.action_dim, self.device
            )
            
            # Combined optimizer
            self.all_parameters = (
                list(self.policy.parameters()) +
                list(self.value_net.parameters()) +
                list(self.hierarchical_dm.strategic_network.parameters()) +
                list(self.hierarchical_dm.tactical_network.parameters()) +
                list(self.hierarchical_dm.operational_network.parameters())
            )
            
            self.optimizer = optim.Adam(self.all_parameters, lr=1e-3)
            
            logger.info(f"{self.object_id}: Networks initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to initialize networks: {e}")
            raise
    
    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions stochastically and compute log_probs and values."""
        # Use hierarchical_dm if available, else policy
        if self.hierarchical_dm:
            raw_output = self.hierarchical_dm.forward(state)
        else:
            raw_output = self.policy(state)
        
        mean = raw_output[:, :self.action_dim]
        log_std = raw_output[:, self.action_dim:]
        std = torch.exp(log_std.clamp(-20, 2))  # Clamp for numerical stability
        
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.value_net(state).squeeze(-1)
        
        return action, log_prob, value
    
    def _get_distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Helper method to get distribution from states."""
        if self.hierarchical_dm:
            raw_output = self.hierarchical_dm.forward(state)
        else:
            raw_output = self.policy(state)
        mean = raw_output[:, :self.action_dim]
        log_std = raw_output[:, self.action_dim:]
        std = torch.exp(log_std.clamp(-20, 2))
        return torch.distributions.Normal(mean, std)
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_value: torch.Tensor, 
                    dones: torch.Tensor, gamma: float = None, lam: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        gamma = gamma or self.gamma
        lam = lam or self.lam
        
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1.0 - dones[t]) * last_gae_lam
            last_gae_lam = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def ppo_update(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, 
                   returns: torch.Tensor, advantages: torch.Tensor):
        """Perform PPO update with clipped surrogate objective."""
        dataset_size = states.size(0)
        
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                idx = perm[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]
                
                # Get new log_probs and values
                _, new_log_probs, new_values = self.get_action_and_value(batch_states)
                
                # Policy loss with clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(new_values, batch_returns)
                
                # Entropy for exploration
                dist = self._get_distribution(batch_states)
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.all_parameters, 0.5)  # Gradient clipping on all parameters
                self.optimizer.step()
    
    async def _spawn_bubble_with_pool(self, bubble_type: str):
        """Spawn bubble using pool and overseer system."""
        # Try to get from pool first
        bubble_config = await self.bubble_pool.acquire(bubble_type, 
                                                      f"{bubble_type}_{uuid.uuid4().hex[:8]}", 
                                                      self.context)
        bubble_id = bubble_config['id']
        
        # Spawn through overseer
        spawn_event = Event(
            type=Actions.OVERSEER_CONTROL,
            data=UniversalCode(Tags.DICT, {
                "action_type": "SPAWN_BUBBLE",
                "payload": {
                    "bubble_type": bubble_type,
                    "bubble_id": bubble_id,
                    "kwargs": {"use_mock": False}
                }
            }),
            origin=self.object_id,
            priority=7
        )
        
        await self.context.dispatch_event(spawn_event)
        logger.info(f"{self.object_id}: Spawned {bubble_type} (id: {bubble_id})")
    
    async def _initialize_pools(self):
        """Initialize bubble pools."""
        await asyncio.sleep(0.1)
        
        # Warm up pools for common bubble types
        bubble_types = ["SimpleLLMBubble", "FeedbackBubble", "ErrorHandlerBubble"]
        await self.bubble_pool.warm_up(bubble_types, self.context)
        
        logger.info(f"{self.object_id}: Bubble pools warmed up")
    
    async def _subscribe_to_all_events(self):
        """Subscribe to all events for monitoring."""
        await asyncio.sleep(0.1)
        try:
            # Subscribe to all event types for error detection and meta-learning
            for action in Actions:
                await EventService.subscribe(action, self.handle_event)
            
            logger.info(f"{self.object_id}: Subscribed to all events")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe: {e}")
    
    async def process_single_event(self, event: Event):
        """Process events with error detection and meta-learning."""
        self.execution_count += 1
        
        # Detect errors
        error_detected, error_details = await self._detect_error(event)
        if error_detected:
            await self._track_error_pattern(event, error_details)
        
        # Extract learning experience
        if event.type == Actions.ACTION_TAKEN:
            experience = await self._create_learning_experience(event)
            if experience:
                self.experience_buffer.append(experience)
                self.total_experiences += 1
        
        # Handle meta-knowledge updates
        if event.type == Actions.META_KNOWLEDGE_UPDATE:
            await self._handle_meta_knowledge_update(event)
        
        # Handle consciousness emergence
        if event.type == Actions.CONSCIOUSNESS_EMERGENCE:
            await self._handle_consciousness_emergence(event)
        
        # Handle user help responses
        if event.type == Actions.USER_HELP_RESPONSE:
            await self._handle_user_help_response(event)
        
        await super().process_single_event(event)
    
    async def _handle_user_help_response(self, event: Event):
        """Handle user help response."""
        if hasattr(event.data, 'value') and isinstance(event.data.value, dict):
            response_data = event.data.value
            user_response = response_data.get("user_response", {})
            
            # Apply user suggestion
            suggestion = user_response.get("suggestion", "")
            if suggestion:
                logger.info(f"{self.object_id}: Received user help: {suggestion}")
                
                # Adjust strategy based on user input
                if "explore" in suggestion.lower() or "innovation" in suggestion.lower():
                    self.objective_weights["innovation"] = min(0.5, self.objective_weights["innovation"] * 1.2)
                    if hasattr(self, 'hierarchical_dm'):
                        self.hierarchical_dm.current_strategy = "aggressive"
                
                elif "stable" in suggestion.lower() or "conservative" in suggestion.lower():
                    self.objective_weights["stability"] = min(0.5, self.objective_weights["stability"] * 1.2)
                    if hasattr(self, 'hierarchical_dm'):
                        self.hierarchical_dm.current_strategy = "conservative"
                
                # Reset failed attempts counter
                self.user_interaction.failed_attempts = 0
    
    async def analyze_problem_and_spawn(self, problem_description: Dict) -> Optional[str]:
        """Analyze a problem and potentially spawn a specialized algorithm."""
        
        problem_type = self._classify_problem(problem_description)
        self.problem_patterns[problem_type] += 1
        
        # Check if we should spawn a new algorithm
        if self._should_spawn_algorithm(problem_type, problem_description):
            algorithm_type = self._select_algorithm_type(problem_type)
            config = self._generate_algorithm_config(problem_type, problem_description)
            
            # Spawn the algorithm
            algorithm_id = await self._spawn_algorithm(algorithm_type, config)
            
            logger.info(f"{self.object_id}: Spawned {algorithm_type} for {problem_type} problems")
            
            return algorithm_id
        
        return None
    
    def _classify_problem(self, problem: Dict) -> str:
        """Classify the type of optimization problem."""
        if problem.get("discrete_variables", False):
            return "combinatorial"
        elif problem.get("multiple_objectives", False):
            return "multi_objective"
        elif problem.get("noisy_function", False):
            return "stochastic"
        elif problem.get("high_dimension", False):
            return "high_dimensional"
        elif problem.get("non_convex", False):
            return "non_convex"
        else:
            return "general"
    
    def _should_spawn_algorithm(self, problem_type: str, problem: Dict) -> bool:
        """Decide whether to spawn a new algorithm."""
        # Don't spawn if we already have too many
        if len(self.spawned_algorithms) >= self.max_algorithms:
            return False
        
        # Check if existing algorithms are struggling
        if problem_type in self.algorithm_performance:
            recent_performance = self.algorithm_performance[problem_type][-10:]
            avg_performance = np.mean(recent_performance) if recent_performance else 0
            
            if avg_performance < self.spawn_threshold:
                return True
        
        # Spawn for new problem types
        if self.problem_patterns[problem_type] >= 3 and problem_type not in self.spawned_algorithms:
            return True
        
        return False
    
    def _select_algorithm_type(self, problem_type: str) -> str:
        """Select appropriate algorithm type for the problem."""
        algorithm_mapping = {
            "combinatorial": "genetic_algorithm",
            "multi_objective": "particle_swarm",
            "stochastic": "evolutionary_strategy",
            "high_dimensional": "curiosity_driven_rl",
            "non_convex": "genetic_algorithm",
            "general": "particle_swarm"
        }
        
        return algorithm_mapping.get(problem_type, "genetic_algorithm")
    
    def _generate_algorithm_config(self, problem_type: str, problem: Dict) -> Dict:
        """Generate configuration for the new algorithm."""
        base_config = {
            "problem_type": problem_type,
            "created_at": time.time(),
            "parent": self.object_id
        }
        
        # Add problem-specific configurations
        if problem_type == "combinatorial":
            base_config.update({
                "population_size": 100,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8
            })
        elif problem_type == "stochastic":
            base_config.update({
                "population": 50,
                "sigma": 0.1,
                "learning_rate": 0.01
            })
        elif problem_type == "high_dimensional":
            base_config.update({
                "curiosity_weight": 0.7,
                "exploration_bonus": 0.2
            })
        elif problem_type == "multi_objective":
            base_config.update({
                "swarm_size": 30,
                "inertia": 0.7,
                "cognitive_weight": 1.5,
                "social_weight": 1.5
            })
        
        return base_config
    
    async def _spawn_algorithm(self, algorithm_type: str, config: Dict) -> str:
        """Actually spawn a new algorithm bubble."""
        algorithm_id = f"{algorithm_type}_{len(self.spawned_algorithms)}"
        
        # Create algorithm instance
        algorithm = LearningAlgorithmFactory.create_algorithm(algorithm_type, config)
        
        # Track spawned algorithm
        self.spawned_algorithms[algorithm_id] = {
            "type": algorithm_type,
            "config": config,
            "algorithm": algorithm,
            "spawned_at": time.time(),
            "performance_history": []
        }
        
        self.algorithms_spawned += 1
        
        # Announce spawn event
        spawn_event = Event(
            type=Actions.SPAWN_ALGORITHM,
            data=UniversalCode(Tags.DICT, {
                "algorithm_type": algorithm_type,
                "algorithm_id": algorithm_id,
                "config": config,
                "purpose": f"Specialized for {config['problem_type']} problems"
            }),
            origin=self.object_id,
            priority=8
        )
        
        await self.context.dispatch_event(spawn_event)
        
        return algorithm_id
    
    async def _detect_error(self, event: Event) -> Tuple[bool, Dict]:
        """Detect if an event contains an error."""
        error_detected = False
        error_details = {}
        
        # Check metadata
        if hasattr(event, 'metadata') and event.metadata:
            if event.metadata.get('error') or event.metadata.get('exception'):
                error_detected = True
                error_details = {
                    'source': 'metadata',
                    'error': event.metadata.get('error', event.metadata.get('exception')),
                    'type': 'explicit_error'
                }
        
        # Check event data
        if not error_detected and hasattr(event, 'data') and isinstance(event.data, UniversalCode):
            data_str = str(event.data.value).lower() if event.data.value else ""
            
            for keyword in self.error_keywords:
                if keyword in data_str:
                    error_detected = True
                    error_details = {
                        'source': 'content',
                        'keyword': keyword,
                        'type': 'keyword_match',
                        'content_sample': data_str[:200]
                    }
                    break
        
        return error_detected, error_details
    
    async def _track_error_pattern(self, event: Event, error_details: Dict):
        """Track error patterns for handler creation."""
        error_type = error_details.get('type', 'unknown')
        event_type = event.type.name if hasattr(event.type, 'name') else str(event.type)
        origin = event.origin or 'unknown'
        
        pattern_id = f"{error_type}:{event_type}:{origin}"
        
        if pattern_id not in self.error_patterns:
            self.error_patterns[pattern_id] = ErrorPattern(error_type, event_type, origin)
        
        pattern = self.error_patterns[pattern_id]
        pattern.add_occurrence(event, error_details)
        
        logger.debug(f"{self.object_id}: Error pattern '{pattern_id}' count: {pattern.occurrence_count}")
        
        # Create handler if threshold reached
        if (pattern.occurrence_count >= self.error_threshold and 
            not pattern.handler_created and
            pattern_id not in self.created_handlers):
            await self._create_error_handler(pattern)
    
    async def _create_error_handler(self, pattern: ErrorPattern):
        """Create specialized error handler bubble."""
        logger.info(f"{self.object_id}: Creating handler for '{pattern.pattern_id}'")
        
        try:
            # Analyze with LLM
            if not pattern.analyzed:
                await self._analyze_error_with_llm(pattern)
            
            # Generate handler code
            if not pattern.solution_code:
                await self._generate_handler_code(pattern)
            
            # Create handler
            handler_id = f"errorhandler_{pattern.error_type}_{uuid.uuid4().hex[:8]}"
            
            spawn_event = Event(
                type=Actions.OVERSEER_CONTROL,
                data=UniversalCode(Tags.DICT, {
                    "action_type": "SPAWN_DYNAMIC_BUBBLE",
                    "payload": {
                        "bubble_code": pattern.solution_code,
                        "bubble_id": handler_id,
                        "metadata": {
                            "error_pattern": pattern.pattern_id,
                            "root_cause": pattern.root_cause,
                            "created_by": self.object_id
                        }
                    }
                }),
                origin=self.object_id,
                priority=9
            )
            
            await self.context.dispatch_event(spawn_event)
            
            pattern.handler_created = True
            pattern.handler_bubble_id = handler_id
            self.created_handlers.add(pattern.pattern_id)
            self.handlers_created += 1
            
            logger.info(f"{self.object_id}: Created handler '{handler_id}'")
            
            # Announce
            await self.add_chat_message(
                f"🛡️ Created error handler for: {pattern.error_type}\n"
                f"Root cause: {pattern.root_cause}"
            )
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to create handler: {e}")
    
    async def _analyze_error_with_llm(self, pattern: ErrorPattern):
        """Use LLM to analyze error pattern."""
        summary = pattern.get_error_summary()
        
        prompt = f"""
Analyze this error pattern:
- Type: {summary['error_type']}
- Event: {summary['event_type']}
- Origin: {summary['origin_bubble']}
- Count: {summary['occurrence_count']}

Provide:
1. Root cause
2. Solution approach
3. Handler type needed
"""
        
        llm_event = Event(
            type=Actions.LLM_QUERY,
            data=UniversalCode(Tags.STRING, prompt),
            origin=self.object_id,
            priority=8
        )
        
        response = await self.context.dispatch_event_and_wait(llm_event, timeout=30)
        
        if response:
            pattern.root_cause = response.get('analysis', 'Unknown')
            pattern.analyzed = True
    
    async def _generate_handler_code(self, pattern: ErrorPattern):
        """Generate handler bubble code with real detection logic."""
        # Determine detection logic based on error type
        detection_logic = ""
        prevention_logic = ""
        
        if pattern.error_type == "keyword_match":
            # Find the most common error keyword
            error_keywords = []
            for detail in pattern.error_details:
                if 'keyword' in detail:
                    error_keywords.append(detail['keyword'])
            
            if error_keywords:
                most_common_keyword = max(set(error_keywords), key=error_keywords.count)
                detection_logic = f"""
        # Check for error keyword in event data
        if hasattr(event, 'data') and hasattr(event.data, 'value'):
            data_str = str(event.data.value).lower()
            if '{most_common_keyword}' in data_str:
                return True
        return False"""
                
                prevention_logic = f"""
        # Log and potentially modify event to prevent error
        logger.warning(f"{{self.object_id}}: Preventing {pattern.error_type} error")
        
        # Create corrective event if needed
        correction_event = Event(
            type=Actions.ERROR_CORRECTION,
            data=UniversalCode(Tags.DICT, {{
                "original_event": event.type,
                "error_prevented": "{pattern.error_type}",
                "correction_applied": True
            }}),
            origin=self.object_id,
            priority=9
        )
        await self.context.dispatch_event(correction_event)"""
        
        else:
            # Default detection for other error types
            detection_logic = """
        # Implement specific detection logic based on error pattern
        # This is a placeholder - should be customized based on actual error
        raise NotImplementedError("Detection logic not implemented for this error type")"""
            
            prevention_logic = """
        # Implement prevention logic
        raise NotImplementedError("Prevention logic not implemented for this error type")"""
        
        handler_code = f'''
from bubbles_core import UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions, logger, EventService
import asyncio

class {pattern.error_type.replace("-", "_").title()}Handler(UniversalBubble):
    """Auto-generated handler for {pattern.pattern_id}"""
    
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.error_pattern = "{pattern.pattern_id}"
        self.root_cause = "{pattern.root_cause or 'Unknown'}"
        self.mitigations = 0
        
        # Register ERROR_CORRECTION action if not exists
        if not hasattr(Actions, 'ERROR_CORRECTION'):
            Actions.ERROR_CORRECTION = "ERROR_CORRECTION"
        
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{{self.object_id}}: Error handler initialized for pattern {{self.error_pattern}}")
    
    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        await EventService.subscribe(Actions.{pattern.event_type}, self.handle_event)
    
    async def process_single_event(self, event: Event):
        if event.origin == "{pattern.origin}":
            # Check for error condition
            if await self._detect_error_condition(event):
                await self._prevent_error(event)
                self.mitigations += 1
                logger.info(f"{{self.object_id}}: Prevented error #{{self.mitigations}}")
        
        await super().process_single_event(event)
    
    async def _detect_error_condition(self, event: Event) -> bool:{detection_logic}
    
    async def _prevent_error(self, event: Event):{prevention_logic}
    
    async def get_status(self) -> Dict[str, Any]:
        status = await super().get_status()
        status.update({{
            "error_pattern": self.error_pattern,
            "root_cause": self.root_cause,
            "mitigations": self.mitigations,
            "handler_type": "auto_generated"
        }})
        return status
'''
        
        pattern.solution_code = handler_code
    
    async def _create_learning_experience(self, event: Event) -> Optional[LearningExperience]:
        """Create a learning experience from an action event."""
        if not hasattr(event.data, 'value') or not isinstance(event.data.value, dict):
            return None
        
        action_data = event.data.value
        bubble_id = event.origin
        
        # Get initial state
        initial_state = self.resource_manager.get_current_system_state() if self.resource_manager else {}
        
        # Wait for outcome
        await asyncio.sleep(2.0)
        
        # Get final state
        final_state = self.resource_manager.get_current_system_state() if self.resource_manager else {}
        
        # Calculate reward
        reward = self._calculate_meta_reward(initial_state, final_state, action_data)
        
        # Extract insights
        insights = [
            f"Action type: {action_data.get('action_type', 'unknown')}",
            f"CPU change: {final_state.get('cpu_percent', 0) - initial_state.get('cpu_percent', 0):.1f}%",
            f"Reward: {reward:.3f}"
        ]
        
        return LearningExperience(
            experience_id=f"exp_{self.total_experiences}",
            timestamp=time.time(),
            bubble_id=bubble_id,
            action_taken=action_data,
            initial_state=initial_state,
            final_state=final_state,
            reward=reward,
            success=reward > 0.5,
            insights=insights
        )
    
    def _calculate_meta_reward(self, initial: Dict, final: Dict, action: Dict) -> float:
        """Calculate reward for meta-learning."""
        reward = 0.0
        
        # Performance improvements
        if final.get("cpu_percent", 100) < initial.get("cpu_percent", 100):
            reward += 0.2
        
        if final.get("avg_response_time", float('inf')) < initial.get("avg_response_time", float('inf')):
            reward += 0.3
        
        # Learning progress
        if final.get("patterns_learned", 0) > initial.get("patterns_learned", 0):
            reward += 0.5
        
        # Error reduction
        if final.get("error_rate", 1.0) < initial.get("error_rate", 1.0):
            reward += 0.4
        
        # Penalize resource waste
        energy_consumed = initial.get("energy", 0) - final.get("energy", 0)
        if energy_consumed > 100:
            reward -= 0.1
        
        return np.clip(reward, -1.0, 1.0)
    
    async def _handle_meta_knowledge_update(self, event: Event):
        """Handle meta-knowledge updates from other components."""
        if hasattr(event.data, 'value') and isinstance(event.data.value, dict):
            knowledge = event.data.value
            
            # Update strategy based on meta-insights
            if hasattr(self, 'hierarchical_dm'):
                if knowledge.get('meta_insights', {}).get('success_rate', 0) < 0.5:
                    self.hierarchical_dm.current_strategy = "conservative"
                elif knowledge.get('discovered_patterns', 0) > 10:
                    self.hierarchical_dm.current_strategy = "innovative"
            
            # Store high-confidence patterns
            if 'high_confidence_patterns' in knowledge:
                for pattern_data in knowledge['high_confidence_patterns']:
                    pattern = MetaKnowledge(
                        pattern_id=pattern_data.get('name', f'meta_{self.patterns_discovered}'),
                        pattern_type=pattern_data.get('type', 'general'),
                        context=pattern_data,
                        learned_at=time.time(),
                        confidence=pattern_data.get('confidence', 0.7)
                    )
                    self.meta_knowledge_base[pattern.pattern_id] = pattern
                    self.patterns_discovered += 1
            
            logger.info(f"{self.object_id}: Received {knowledge.get('discovered_patterns', 0)} meta-patterns")
    
    async def _handle_consciousness_emergence(self, event: Event):
        """Handle consciousness emergence events."""
        if hasattr(event.data, 'value') and isinstance(event.data.value, dict):
            emergence_data = event.data.value
            
            # Adjust weights based on consciousness insights
            if emergence_data.get('confidence', 0) > 0.8:
                # High confidence in consciousness pattern - increase innovation
                self.objective_weights["innovation"] = min(0.5, 
                    self.objective_weights["innovation"] * 1.2)
                
                logger.info(f"{self.object_id}: Consciousness emergence detected - increasing innovation weight")
    
    async def train(self):
        """Main training loop with proper PPO updates."""
        if not self.env:
            logger.warning(f"{self.object_id}: Cannot train without environment")
            return
        
        logger.info(f"{self.object_id}: Starting enhanced training with meta-learning")
        
        episodes = 0
        while not self.context.stop_event.is_set():
            try:
                # Get curriculum-appropriate state
                complexity = self.curriculum.get_current_complexity()
                state = self.env.reset(complexity)
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                episode_reward = 0
                episode_length = 0
                success_count = 0
                
                # Rollout buffers
                states = []
                actions = []
                log_probs = []
                values = []
                rewards = []
                dones = []
                
                for step in range(self.env.max_steps):
                    # Get action and value
                    with torch.no_grad():
                        action, log_prob, value = self.get_action_and_value(state)
                    
                    # Store rollout data
                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    
                    # Execute action
                    next_state, reward, done = await self.env.step(action.squeeze(0))
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    rewards.append(reward)
                    dones.append(done)
                    
                    # Track rewards
                    self.reward_history.append(reward)
                    
                    # Generate explanation periodically
                    if step % 10 == 0:
                        action_dict = self._tensor_to_action_dict(action.squeeze(0))
                        state_dict = self.resource_manager.get_current_system_state()
                        explanation = await self.explainer.explain_action(state_dict, action_dict)
                    
                    episode_reward += reward
                    episode_length += 1
                    if reward > 0.5:
                        success_count += 1
                    
                    # Check if we should spawn algorithms
                    if reward < 0.3:
                        problem_context = {
                            "reward": reward,
                            "state_features": state.squeeze(0).tolist(),
                            "high_dimension": self.state_dim > 20,
                            "noisy_function": len(self.reward_history) > 2 and 
                                abs(reward - self.reward_history[-2]) > 0.5
                        }
                        
                        spawned_id = await self.analyze_problem_and_spawn(problem_context)
                        
                        if spawned_id and spawned_id in self.spawned_algorithms:
                            # Use the spawned algorithm
                            algorithm = self.spawned_algorithms[spawned_id]["algorithm"]
                            
                            # Run optimization with the specialized algorithm
                            async def objective(params):
                                param_tensor = torch.tensor(params, dtype=torch.float32)
                                next_state, reward, _ = await self.env.step(param_tensor)
                                return reward
                            
                            # Optimize with spawned algorithm
                            best_params, best_score = await algorithm.optimize(
                                objective,
                                state.squeeze(0).cpu().numpy(),
                                iterations=50
                            )
                            
                            # Update algorithm performance
                            self.algorithm_performance[spawned_id].append(best_score)
                            
                            if best_score > reward:
                                logger.info(f"Spawned algorithm {spawned_id} improved score: {best_score:.3f}")
                    
                    state = next_state
                    
                    if done:
                        break
                
                # After episode, get last value for GAE
                with torch.no_grad():
                    _, _, next_value = self.get_action_and_value(state)
                
                # Stack rollout data
                states = torch.cat(states)
                actions = torch.cat(actions)
                log_probs = torch.cat(log_probs)
                values = torch.cat(values)
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
                
                # Compute advantages and returns using GAE
                advantages, returns = self.compute_gae(rewards, values, next_value.item(), dones)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Perform PPO update
                self.ppo_update(states, actions, log_probs, returns, advantages)
                
                episodes += 1
                self.decisions_made += episode_length
                
                # Update curriculum
                success_rate = success_count / max(1, episode_length)
                self.curriculum.update(
                    episode_complete=True,
                    metrics={"success_rate": success_rate, "avg_reward": episode_reward}
                )
                
                # Check if stuck
                await self.user_interaction.check_if_stuck(episode_reward, success_rate)
                
                # Extract meta-patterns periodically
                if episodes % 50 == 0 and len(self.experience_buffer) >= 10:
                    patterns = await self._extract_patterns_from_experiences()
                    for pattern in patterns:
                        self.meta_knowledge_base[pattern.pattern_id] = pattern
                        self.patterns_discovered += 1
                
                # Log progress
                if episodes % 10 == 0:
                    metrics = {
                        "episode": episodes,
                        "reward": episode_reward,
                        "length": episode_length,
                        "success_rate": success_rate,
                        "curriculum_stage": self.curriculum.stages[self.curriculum.current_stage].name,
                        "cache_hit_rate": self.prediction_cache.get_hit_rate(),
                        "pool_hit_rate": self.bubble_pool.get_hit_rate(),
                        "handlers_created": self.handlers_created,
                        "algorithms_spawned": self.algorithms_spawned,
                        "patterns_discovered": self.patterns_discovered,
                        "strategy": self.hierarchical_dm.current_strategy if self.hierarchical_dm else "none"
                    }
                    
                    logger.info(f"{self.object_id}: Episode {episodes} - " +
                              f"Reward: {episode_reward:.2f}, " +
                              f"Stage: {metrics['curriculum_stage']}, " +
                              f"Algorithms: {self.algorithms_spawned}, " +
                              f"Patterns: {self.patterns_discovered}")
                    
                    # Save model
                    self.save_model()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"{self.object_id}: Training error: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    def _tensor_to_action_dict(self, action: torch.Tensor) -> Dict:
        """Convert action tensor to dictionary."""
        action_np = action.detach().cpu().numpy()
        
        return {
            "spawn_bubble": action_np[0] > 0.5,
            "destroy_bubble": action_np[1] > 0.5 if len(action_np) > 1 else False,
            "spawn_algorithm": action_np[2] > 0.5 if len(action_np) > 2 else False,
            "bubble_type": self._get_bubble_type_from_action(action_np),
            "algorithm_type": self._get_algorithm_type_from_action(action_np),
            "resource_changes": {
                "energy": action_np[3] * 500 if len(action_np) > 3 else 0,
                "priority": action_np[4] * 5 if len(action_np) > 4 else 0
            }
        }
    
    def _get_bubble_type_from_action(self, action: np.ndarray) -> str:
        """Determine bubble type from action vector."""
        bubble_types = self.curriculum.get_allowed_bubbles()
        if "all" in bubble_types:
            bubble_types = ["SimpleLLMBubble", "FeedbackBubble", "ErrorHandlerBubble", 
                          "DreamerV3Bubble", "CreativeSynthesisBubble"]
        
        if len(action) > 5:
            idx = int(action[5] * len(bubble_types)) % len(bubble_types)
            return bubble_types[idx]
        
        return bubble_types[0]
    
    def _get_algorithm_type_from_action(self, action: np.ndarray) -> str:
        """Determine algorithm type from action vector."""
        algorithm_types = ["genetic_algorithm", "curiosity_driven_rl", "particle_swarm", "evolutionary_strategy"]
        
        if len(action) > 6:
            idx = int(action[6] * len(algorithm_types)) % len(algorithm_types)
            return algorithm_types[idx]
        
        return algorithm_types[0]
    
    async def dispatch_to_bubble(self, bubble_id: str, action: Actions, data: Dict) -> Any:
        """Dispatch event to specific bubble and await response."""
        event = Event(
            type=action,
            data=UniversalCode(Tags.DICT, data),
            origin=self.object_id,
            target=bubble_id,
            priority=7
        )
        
        return await self.context.dispatch_event_and_wait(event, timeout=10)
    
    async def meta_learning_loop(self):
        """Extract patterns and learn from experiences."""
        await asyncio.sleep(10)
        
        while not self.context.stop_event.is_set():
            try:
                if len(self.experience_buffer) >= 20:
                    # Extract patterns
                    patterns = await self._extract_patterns_from_experiences()
                    
                    # Share with other components
                    if patterns:
                        knowledge_event = Event(
                            type=Actions.META_KNOWLEDGE_UPDATE,
                            data=UniversalCode(Tags.DICT, {
                                "discovered_patterns": len(patterns),
                                "high_confidence_patterns": [
                                    {
                                        "name": p.pattern_id,
                                        "type": p.pattern_type,
                                        "confidence": p.confidence,
                                        "context": p.context
                                    }
                                    for p in patterns if p.confidence > 0.7
                                ],
                                "source": self.object_id
                            }),
                            origin=self.object_id,
                            priority=8
                        )
                        
                        await self.context.dispatch_event(knowledge_event)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"{self.object_id}: Meta-learning loop error: {e}")
                await asyncio.sleep(120)
    
    async def _extract_patterns_from_experiences(self) -> List[MetaKnowledge]:
        """Extract patterns from recent experiences."""
        patterns = []
        
        recent_experiences = list(self.experience_buffer)[-50:]
        
        # Group by success/failure
        successful = [e for e in recent_experiences if e.success]
        failed = [e for e in recent_experiences if not e.success]
        
        # Analyze successful patterns
        if len(successful) >= 5:
            # Find common factors in successful experiences
            common_actions = defaultdict(int)
            for exp in successful:
                if 'action_type' in exp.action_taken:
                    common_actions[exp.action_taken['action_type']] += 1
            
            # Create pattern for most common successful action
            if common_actions:
                most_common = max(common_actions, key=common_actions.get)
                pattern = MetaKnowledge(
                    pattern_id=f"success_pattern_{most_common}",
                    pattern_type="success",
                    context={
                        "action_type": most_common,
                        "success_count": common_actions[most_common],
                        "conditions": self._extract_common_conditions(successful),
                        "category": "performance_enhancement"
                    },
                    learned_at=time.time(),
                    confidence=common_actions[most_common] / len(successful)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_common_conditions(self, experiences: List[LearningExperience]) -> Dict:
        """Extract common conditions from experiences."""
        if not experiences:
            return {}
        
        # Find average conditions
        avg_cpu = np.mean([e.initial_state.get('cpu_percent', 0) for e in experiences])
        avg_energy = np.mean([e.initial_state.get('energy', 0) for e in experiences])
        
        return {
            "avg_cpu_percent": avg_cpu,
            "avg_energy": avg_energy,
            "sample_size": len(experiences)
        }
    
    async def consciousness_exploration_loop(self):
        """Explore consciousness patterns periodically."""
        await asyncio.sleep(30)
        
        while not self.context.stop_event.is_set():
            try:
                if self.consciousness.enabled:
                    await self.consciousness.explore_consciousness_patterns()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"{self.object_id}: Consciousness exploration error: {e}")
                await asyncio.sleep(600)
    
    async def orchestration_loop(self):
        """Run orchestration patterns periodically."""
        await asyncio.sleep(10)
        
        while not self.context.stop_event.is_set():
            try:
                # Check if any patterns should be activated
                state = self.resource_manager.get_current_system_state()
                
                # Example: Activate cognitive pipeline for complex queries
                if state.get("event_frequencies", {}).get("LLM_QUERY_freq_per_min", 0) > 50:
                    pattern = self.orchestration_patterns.get("cognitive_pipeline")
                    if pattern:
                        result = await pattern.execute(self, {"problem": "Handle high query load"})
                        self.orchestrations_completed += 1
                        logger.info(f"{self.object_id}: Completed cognitive pipeline")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"{self.object_id}: Orchestration error: {e}")
                await asyncio.sleep(60)
    
    async def monitor_errors(self):
        """Monitor and report on error patterns."""
        while not self.context.stop_event.is_set():
            try:
                await asyncio.sleep(30)
                
                # Report active patterns
                active_patterns = [
                    p for p in self.error_patterns.values()
                    if not p.handler_created and p.occurrence_count >= 2
                ]
                
                if active_patterns:
                    report = "📊 Error Pattern Report:\n"
                    for pattern in active_patterns[:5]:
                        report += f"- {pattern.pattern_id}: {pattern.occurrence_count} occurrences\n"
                    
                    report += f"\nTotal handlers created: {len(self.created_handlers)}"
                    await self.add_chat_message(report)
                
            except Exception as e:
                logger.error(f"{self.object_id}: Error monitoring failed: {e}")
    
    async def cache_maintenance(self):
        """Periodic cache maintenance."""
        while not self.context.stop_event.is_set():
            try:
                await asyncio.sleep(60)
                
                # Clear expired predictions
                self.prediction_cache.clear_expired()
                
                # Log cache stats
                cache_size = len(self.prediction_cache.cache)
                hit_rate = self.prediction_cache.get_hit_rate()
                
                if cache_size > 0:
                    logger.debug(f"{self.object_id}: Cache maintenance - "
                               f"Size: {cache_size}, Hit rate: {hit_rate:.1%}")
                
            except Exception as e:
                logger.error(f"{self.object_id}: Cache maintenance error: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        base_status = await super().get_status()
        
        # Add all enhancement metrics
        base_status.update({
            # Performance metrics
            "bubble_pool_size": sum(len(b) for b in self.bubble_pool.idle_bubbles.values()),
            "pool_hit_rate": f"{self.bubble_pool.get_hit_rate():.1%}",
            "cache_size": len(self.prediction_cache.cache),
            "cache_hit_rate": f"{self.prediction_cache.get_hit_rate():.1%}",
            
            # Advanced features
            "curriculum_stage": self.curriculum.stages[self.curriculum.current_stage].name,
            "stage_progress": f"{self.curriculum.episodes_in_stage}/{self.curriculum.stages[self.curriculum.current_stage].duration}",
            "current_strategy": self.hierarchical_dm.current_strategy if self.hierarchical_dm else "none",
            
            # Algorithm spawning
            "algorithms_spawned": self.algorithms_spawned,
            "active_algorithms": list(self.spawned_algorithms.keys()),
            "algorithm_performance": {
                alg_id: np.mean(perfs[-10:]) if perfs else 0 
                for alg_id, perfs in self.algorithm_performance.items()
            },
            
            # Meta-learning
            "total_experiences": self.total_experiences,
            "patterns_discovered": self.patterns_discovered,
            "meta_knowledge_size": len(self.meta_knowledge_base),
            "recent_reward_avg": np.mean(list(self.reward_history)) if self.reward_history else 0,
            
            # Consciousness
            "consciousness_enabled": self.consciousness.enabled,
            "consciousness_patterns": len(self.consciousness.consciousness_patterns) if self.consciousness.enabled else 0,
            
            # Decision metrics
            "decisions_made": self.decisions_made,
            "explanations_generated": self.explanations_generated,
            
            # Error handling
            "error_patterns_detected": len(self.error_patterns),
            "handlers_created": self.handlers_created,
            "active_error_patterns": sum(1 for p in self.error_patterns.values() 
                                        if not p.handler_created),
            
            # Orchestration
            "orchestrations_completed": self.orchestrations_completed,
            "active_patterns": list(self.orchestration_patterns.keys()),
            
            # Multi-objective weights
            "objective_weights": self.objective_weights
        })
        
        return base_status
    
    def save_model(self):
        """Save all models and state."""
        if not self.policy:
            return
        
        try:
            checkpoint = {
                # Models
                'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value_net.state_dict(),
                'hierarchical_strategic': self.hierarchical_dm.strategic_network.state_dict(),
                'hierarchical_tactical': self.hierarchical_dm.tactical_network.state_dict(),
                'hierarchical_operational': self.hierarchical_dm.operational_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                
                # Curriculum
                'curriculum_stage': self.curriculum.current_stage,
                'curriculum_episodes': self.curriculum.episodes_in_stage,
                
                # Metrics
                'decisions_made': self.decisions_made,
                'handlers_created': self.handlers_created,
                'orchestrations_completed': self.orchestrations_completed,
                'algorithms_spawned': self.algorithms_spawned,
                'patterns_discovered': self.patterns_discovered,
                
                # Meta-learning
                'meta_knowledge': {
                    pid: {
                        'pattern_type': p.pattern_type,
                        'context': p.context,
                        'confidence': p.confidence,
                        'applications': p.applications
                    }
                    for pid, p in self.meta_knowledge_base.items()
                },
                
                # Cache stats
                'cache_stats': self.prediction_cache.stats,
                'pool_stats': self.bubble_pool.stats
            }
            
            torch.save(checkpoint, self.model_filename)
            logger.debug(f"{self.object_id}: Saved model checkpoint")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to save model: {e}")
    
    def load_model(self):
        """Load all models and state."""
        if not os.path.exists(self.model_filename):
            return
        
        try:
            checkpoint = torch.load(self.model_filename, map_location=self.device)
            
            # Load models
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            self.hierarchical_dm.strategic_network.load_state_dict(checkpoint['hierarchical_strategic'])
            self.hierarchical_dm.tactical_network.load_state_dict(checkpoint['hierarchical_tactical'])
            self.hierarchical_dm.operational_network.load_state_dict(checkpoint['hierarchical_operational'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load curriculum
            self.curriculum.current_stage = checkpoint.get('curriculum_stage', 0)
            self.curriculum.episodes_in_stage = checkpoint.get('curriculum_episodes', 0)
            
            # Load metrics
            self.decisions_made = checkpoint.get('decisions_made', 0)
            self.handlers_created = checkpoint.get('handlers_created', 0)
            self.orchestrations_completed = checkpoint.get('orchestrations_completed', 0)
            self.algorithms_spawned = checkpoint.get('algorithms_spawned', 0)
            self.patterns_discovered = checkpoint.get('patterns_discovered', 0)
            
            # Load meta-knowledge
            if 'meta_knowledge' in checkpoint:
                for pid, pdata in checkpoint['meta_knowledge'].items():
                    self.meta_knowledge_base[pid] = MetaKnowledge(
                        pattern_id=pid,
                        pattern_type=pdata['pattern_type'],
                        context=pdata['context'],
                        learned_at=time.time(),
                        confidence=pdata['confidence'],
                        applications=pdata['applications']
                    )
            
            # Load stats
            if 'cache_stats' in checkpoint:
                self.prediction_cache.stats = checkpoint['cache_stats']
            if 'pool_stats' in checkpoint:
                self.bubble_pool.stats = checkpoint['pool_stats']
            
            logger.info(f"{self.object_id}: Loaded model from checkpoint")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to load model: {e}")
            raise

