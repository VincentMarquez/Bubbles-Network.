import asyncio
import time
import logging
import random
import sys
import re
import ast
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import deque
import numpy as np
import uuid
import traceback

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes remain the same as original
    class nn: Module = object; Sequential = object; Linear = object; ReLU = object; GRU = object; SiLU = object
    class torch: Tensor = object; float32 = None; no_grad = staticmethod(lambda: nullcontext()); zeros = staticmethod(lambda *a, **kw: None)
    class optim: Adam = object
    np = None
    print("WARNING: PyTorch not found. OverseerBubble will use heuristic mode.", file=sys.stderr)

from bubbles_core import (
    UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions,
    logger, EventService
)

logger = logging.getLogger(__name__)

# --- Self-Planning Templates for Code Generation ---
class SelfPlanningTemplates:
    """Templates from 'Self-Planning Code Generation with Large Language Models'"""
    
    @staticmethod
    def planning_only_template(intent: str) -> str:
        """Template for generating plan from intent"""
        return f"""Intent: {intent}

Plan:
1. [First step - setup/initialization]
2. [Core logic step 1]
3. [Core logic step 2]
4. [Error handling/validation]
5. [Return result]

Generate a concise 4-8 step plan for this intent."""

    @staticmethod
    def one_phase_template(intent: str, plan: str) -> str:
        """Template for generating plan + code together"""
        return f'''"""
Intent: {intent}

Plan:
{plan}
"""

# Write your code here
'''

    @staticmethod
    def plan_from_code_template(code: str) -> str:
        """Extract plan from existing code"""
        return f"""Given this code:

```python
{code}
```

Extract the plan as numbered steps (keep it concise, 4-8 steps):

Plan:"""

    @staticmethod
    def code_cot_template(intent: str) -> str:
        """Chain of Thought template for code generation"""
        return f"""Let's think step by step to implement: {intent}

1. What inputs/outputs are needed?
2. What are the main operations?
3. What edge cases should we handle?
4. How do we ensure reliability?

Now implement with these considerations:

```python"""

    @staticmethod
    def concise_plan_format(intent: str) -> str:
        """Extremely concise planning format"""
        return f"""Intent: {intent}

Plan (ultra-concise):
1. Setup
2. Process
3. Handle errors
4. Return"""

# --- Enhanced Recovery Layers for Code Healing ---
class CodeRecoveryLayers:
    """Recovery strategies specifically for code-related issues"""
    
    @staticmethod
    def fix_syntax_error(error_info: Dict) -> Dict:
        """Attempt to fix common syntax errors"""
        return {
            "action_type": "CODE_FIX",
            "payload": {
                "fix_type": "syntax",
                "error_info": error_info,
                "strategy": "ast_repair"
            }
        }
    
    @staticmethod
    def rollback_code(bubble_id: str, version: int) -> Dict:
        """Rollback to a previous known-good version"""
        return {
            "action_type": "CODE_ROLLBACK",
            "payload": {
                "bubble_id": bubble_id,
                "target_version": version
            }
        }
    
    @staticmethod
    def inject_error_handler(bubble_id: str, error_type: str) -> Dict:
        """Inject error handling code into problematic bubbles"""
        return {
            "action_type": "CODE_INJECT",
            "payload": {
                "bubble_id": bubble_id,
                "injection_type": "error_handler",
                "error_type": error_type
            }
        }
    
    @staticmethod
    def refactor_problematic_code(bubble_id: str, issue_type: str) -> Dict:
        """Request LLM to refactor problematic code patterns"""
        return {
            "action_type": "CODE_REFACTOR",
            "payload": {
                "bubble_id": bubble_id,
                "issue_type": issue_type,
                "use_llm": True
            }
        }

class RecoveryLayers:
    """Original recovery layers for system-level issues"""
    @staticmethod
    def boost_energy():
        return {"action_type": "ADJUST_ENERGY", "payload": {"amount": 1000}}

    @staticmethod
    def reroute_data():
        return {"action_type": "SPAWN_BUBBLE", "payload": {"type": "ALT_ROUTER"}}
    
    @staticmethod
    def restart_bubble(bubble_id: str):
        return {"action_type": "RESTART_BUBBLE", "payload": {"bubble_id": bubble_id}}

# --- Warning Pattern Analyzer ---
class WarningPatternAnalyzer:
    """Analyzes warning patterns to predict and prevent errors"""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=1000)
        self.error_correlations = {}
        self.predictive_models = {}
    
    def record_event(self, event_type: str, source: str, details: Dict):
        """Record an event (warning or error) for pattern analysis"""
        self.pattern_history.append({
            "type": event_type,
            "source": source,
            "timestamp": time.time(),
            "details": details
        })
    
    def analyze_pattern_to_error_correlation(self):
        """Find which warning patterns lead to errors"""
        warning_to_error_map = {}
        
        for i, event in enumerate(self.pattern_history):
            if event["type"] == "ERROR":
                # Look back for warnings from same source
                source = event["source"]
                lookback_window = 300  # 5 minutes
                
                preceding_warnings = []
                for j in range(i-1, -1, -1):
                    prev_event = self.pattern_history[j]
                    if (prev_event["source"] == source and 
                        prev_event["type"] == "WARNING" and
                        event["timestamp"] - prev_event["timestamp"] < lookback_window):
                        preceding_warnings.append(prev_event)
                
                if preceding_warnings:
                    pattern_key = self._create_pattern_key(preceding_warnings)
                    if pattern_key not in warning_to_error_map:
                        warning_to_error_map[pattern_key] = []
                    warning_to_error_map[pattern_key].append(event)
        
        # Calculate correlation strengths
        for pattern, errors in warning_to_error_map.items():
            self.error_correlations[pattern] = {
                "error_count": len(errors),
                "confidence": len(errors) / max(1, self._count_pattern_occurrences(pattern)),
                "average_time_to_error": self._calculate_avg_time_to_error(pattern, errors)
            }
    
    def predict_error_probability(self, current_warnings: List[Dict]) -> float:
        """Predict probability of error based on current warnings"""
        if not current_warnings:
            return 0.0
            
        pattern_key = self._create_pattern_key(current_warnings)
        
        if pattern_key in self.error_correlations:
            correlation = self.error_correlations[pattern_key]
            return correlation["confidence"]
        
        return 0.0
    
    def _create_pattern_key(self, warnings: List[Dict]) -> str:
        """Create a hashable key from warning pattern"""
        warning_types = sorted([w.get("warning_type", w.get("details", {}).get("warning_type", "unknown")) for w in warnings])
        return "|".join(warning_types)
    
    def _count_pattern_occurrences(self, pattern_key: str) -> int:
        """Count how many times a pattern occurred"""
        count = 0
        pattern_types = set(pattern_key.split("|"))
        
        for i in range(len(self.pattern_history)):
            window_warnings = []
            for j in range(i, min(i+10, len(self.pattern_history))):
                if self.pattern_history[j]["type"] == "WARNING":
                    window_warnings.append(self.pattern_history[j])
            
            if window_warnings:
                window_key = self._create_pattern_key(window_warnings)
                if set(window_key.split("|")) == pattern_types:
                    count += 1
        
        return count
    
    def _calculate_avg_time_to_error(self, pattern: str, errors: List[Dict]) -> float:
        """Calculate average time from warning pattern to error"""
        times = []
        for error in errors:
            # Find corresponding warning time
            for event in reversed(self.pattern_history):
                if event["type"] == "WARNING" and event["source"] == error["source"]:
                    times.append(error["timestamp"] - event["timestamp"])
                    break
        
        return sum(times) / len(times) if times else 0

# --- Enhanced Fault Monitor with Code Analysis ---
class CodeAwareFaultMonitor:
    """Fault monitor that can analyze code-related issues"""
    
    def __init__(self, condition_fn: Callable, recovery_fn: Callable, 
                 name: str = "Unnamed", analyze_code: bool = False,
                 use_self_planning: bool = True):
        self.condition = condition_fn
        self.recovery = recovery_fn
        self.name = name
        self.analyze_code = analyze_code
        self.use_self_planning = use_self_planning
        self.error_patterns = {
            "syntax": r"SyntaxError|IndentationError|TabError",
            "import": r"ImportError|ModuleNotFoundError",
            "attribute": r"AttributeError",
            "type": r"TypeError",
            "runtime": r"RuntimeError|RecursionError"
        }
        self.recovery_plans = {}  # Cache successful recovery plans

    async def check_and_recover(self, state: Dict, overseer):
        """Check condition and apply recovery with code analysis"""
        if self.condition(state):
            if self.analyze_code:
                # Analyze code-related metrics
                code_issues = self._analyze_code_issues(state)
                if code_issues:
                    action = self._select_code_recovery(code_issues)
                else:
                    action = self.recovery(state)
            else:
                action = self.recovery(state) if callable(self.recovery) else self.recovery()
            
            await overseer._execute_control_action(action, state)
            
            # Learn from this recovery
            if hasattr(overseer, 'record_recovery'):
                await overseer.record_recovery(self.name, state, action)
    
    def _analyze_code_issues(self, state: Dict) -> Optional[Dict]:
        """Analyze state for code-related issues"""
        metrics = state.get("metrics", {})
        
        # Check for error patterns in recent logs
        recent_errors = metrics.get("recent_errors", [])
        for error in recent_errors:
            error_str = str(error)
            for error_type, pattern in self.error_patterns.items():
                if re.search(pattern, error_str):
                    return {
                        "type": error_type,
                        "error": error,
                        "bubble_id": error.get("bubble_id"),
                        "traceback": error.get("traceback", "")
                    }
        
        # Check for performance degradation that might indicate code issues
        if metrics.get("avg_execution_time_ms", 0) > 5000:
            return {"type": "performance", "metric": "execution_time"}
        
        return None
    
    def _select_code_recovery(self, code_issues: Dict) -> Dict:
        """Select appropriate recovery action based on code issue type"""
        issue_type = code_issues["type"]
        
        # Check if we have a cached recovery plan for this issue type
        if self.use_self_planning and issue_type in self.recovery_plans:
            cached_plan = self.recovery_plans[issue_type]
            return {
                "action_type": "CODE_FIX_WITH_PLAN",
                "payload": {
                    "plan": cached_plan,
                    "issue_info": code_issues
                }
            }
        
        # Standard recovery selection
        if issue_type == "syntax":
            return CodeRecoveryLayers.fix_syntax_error(code_issues)
        elif issue_type == "import":
            return CodeRecoveryLayers.inject_error_handler(
                code_issues.get("bubble_id", "unknown"), "import"
            )
        elif issue_type == "performance":
            return CodeRecoveryLayers.refactor_problematic_code(
                code_issues.get("bubble_id", "unknown"), "performance"
            )
        else:
            # Default to rollback for unknown issues
            return CodeRecoveryLayers.rollback_code(
                code_issues.get("bubble_id", "unknown"), -1
            )

    async def learn_recovery_plan(self, issue_type: str, successful_plan: str):
        """Learn and cache successful recovery plans"""
        if self.use_self_planning:
            self.recovery_plans[issue_type] = successful_plan
            logger.info(f"Monitor {self.name}: Learned recovery plan for {issue_type}")

# --- Warning Monitor ---
class WarningMonitor:
    """Monitor for warning-level events that haven't become errors yet"""
    
    def __init__(self, analyzer: WarningPatternAnalyzer):
        self.name = "WarningMonitor"
        self.analyzer = analyzer
        self.warning_patterns = {}
        self.warning_thresholds = {
            "CORRELATION_WARNING": 5,  # Fix after 5 occurrences
            "PERFORMANCE_WARNING": 3,
            "MEMORY_WARNING": 2,
            "CODE_SMELL_WARNING": 10
        }
    
    async def check_and_recover(self, warnings: List[Dict], overseer):
        """Analyze warnings and trigger preventive action if needed"""
        if not warnings:
            return
        
        # Group warnings by type and source
        warning_groups = {}
        for warning in warnings:
            key = (warning.get("warning_type"), warning.get("source"))
            if key not in warning_groups:
                warning_groups[key] = []
            warning_groups[key].append(warning)
        
        # Find most critical warning pattern
        for (warning_type, source), instances in warning_groups.items():
            if len(instances) >= self.warning_thresholds.get(warning_type, 5):
                action = self._create_preventive_action(warning_type, source, instances)
                if action["action_type"] != "NO_OP":
                    await overseer._execute_control_action(action, {"warnings": instances})
    
    def _create_preventive_action(self, warning_type: str, source: str, 
                                  instances: List[Dict]) -> Dict:
        """Create preventive action based on warning pattern"""
        
        if warning_type == "CORRELATION_WARNING":
            return {
                "action_type": "CODE_INJECT",
                "payload": {
                    "bubble_id": source,
                    "injection_type": "correlation_id_handler",
                    "pattern": "missing_correlation_id"
                }
            }
        elif warning_type == "PERFORMANCE_WARNING":
            return {
                "action_type": "CODE_REFACTOR",
                "payload": {
                    "bubble_id": source,
                    "issue_type": "performance_optimization",
                    "metrics": {"warning_count": len(instances)}
                }
            }
        elif warning_type == "MEMORY_WARNING":
            return {
                "action_type": "CODE_INJECT",
                "payload": {
                    "bubble_id": source,
                    "injection_type": "memory_cleanup",
                    "usage_pattern": "high_memory"
                }
            }
        
        return {"action_type": "NO_OP", "payload": {}}

# --- Metrics Dashboard ---
class OverseerMetricsDashboard:
    """Aggregates system health metrics for overseer decisions"""
    
    def __init__(self):
        self.metrics = {
            "warnings_by_type": {},
            "warnings_by_bubble": {},
            "fixes_applied": 0,
            "fixes_successful": 0,
            "errors_prevented": 0,
            "system_health_score": 100.0
        }
    
    def update_warning_metrics(self, warning_type: str, source: str):
        """Update warning counters"""
        if warning_type not in self.metrics["warnings_by_type"]:
            self.metrics["warnings_by_type"][warning_type] = 0
        self.metrics["warnings_by_type"][warning_type] += 1
        
        if source not in self.metrics["warnings_by_bubble"]:
            self.metrics["warnings_by_bubble"][source] = {}
        if warning_type not in self.metrics["warnings_by_bubble"][source]:
            self.metrics["warnings_by_bubble"][source][warning_type] = 0
        self.metrics["warnings_by_bubble"][source][warning_type] += 1
        
        # Update health score
        self._recalculate_health_score()
    
    def record_fix_applied(self, fix_type: str, bubble_id: str):
        """Record that a fix was applied"""
        self.metrics["fixes_applied"] += 1
        logger.info(f"Applied {fix_type} fix to {bubble_id}. Total fixes: {self.metrics['fixes_applied']}")
    
    def record_fix_outcome(self, success: bool):
        """Record whether a fix was successful"""
        if success:
            self.metrics["fixes_successful"] += 1
            self.metrics["errors_prevented"] += 1
    
    def _recalculate_health_score(self):
        """Recalculate overall system health score"""
        score = 100.0
        
        # Deduct for warnings
        total_warnings = sum(self.metrics["warnings_by_type"].values())
        score -= min(50, total_warnings * 0.5)
        
        # Add for successful fixes
        if self.metrics["fixes_applied"] > 0:
            fix_success_rate = self.metrics["fixes_successful"] / self.metrics["fixes_applied"]
            score += fix_success_rate * 20
        
        # Bonus for prevented errors
        score += min(30, self.metrics["errors_prevented"] * 2)
        
        self.metrics["system_health_score"] = max(0, min(100, score))
    
    def get_problem_bubbles(self, threshold: int = 10) -> List[Dict]:
        """Get bubbles with high warning counts"""
        problem_bubbles = []
        
        for bubble, warnings in self.metrics["warnings_by_bubble"].items():
            total_warnings = sum(warnings.values())
            if total_warnings > threshold:
                problem_bubbles.append({
                    "bubble_id": bubble,
                    "warning_count": total_warnings,
                    "top_issues": sorted(warnings.items(), key=lambda x: x[1], reverse=True)[:3]
                })
        
        return sorted(problem_bubbles, key=lambda x: x["warning_count"], reverse=True)

# --- System Environment ---
class SystemEnv:
    """Environment for OverseerBubble meta-RL"""
    def __init__(self, resource_manager):
        self.resource_manager = resource_manager
        self.action_types = ["ADJUST_ENERGY", "PAUSE_BUBBLE", "SPAWN_BUBBLE", "NO_OP", 
                           "CODE_FIX", "CODE_ROLLBACK", "CODE_INJECT", "CODE_REFACTOR"]
        self.state_dim = 12  # Increased for code metrics
        self.action_dim = len(self.action_types)

    def get_state(self) -> np.ndarray:
        """Get current system state including code health metrics"""
        state = self.resource_manager.get_current_system_state()
        metrics = state.get("metrics", {})
        vector = [
            state.get("energy", 0) / 10000.0,
            state.get("cpu_percent", 0) / 100.0,
            state.get("memory_percent", 0) / 100.0,
            state.get("num_bubbles", 0) / 20.0,
            metrics.get("avg_llm_response_time_ms", 0) / 60000.0,
            metrics.get("code_update_count", 0) / 100.0,
            metrics.get("prediction_cache_hit_rate", 0),
            metrics.get("events_published_total", 0) / 1000.0,
            metrics.get("llm_error_count", 0) / 100.0,
            metrics.get("bubbles_spawned", 0) / 10.0,
            # New code health metrics
            metrics.get("code_error_rate", 0),
            metrics.get("code_fix_success_rate", 0)
        ]
        return np.array(vector, dtype=np.float32)

    def compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """Compute reward including code health"""
        cpu = next_state[1] * 100.0
        cache_hit = next_state[6]
        llm_errors = next_state[8] * 100.0
        code_error_rate = next_state[10]
        code_fix_success = next_state[11]
        
        # Base reward
        reward = -cpu / 100.0 + cache_hit - llm_errors / 50.0
        
        # Code health bonus/penalty
        reward -= code_error_rate * 2.0  # Penalty for code errors
        reward += code_fix_success * 1.5  # Bonus for successful fixes
        
        # Action-specific rewards
        if self.action_types[action] == "NO_OP":
            reward -= 0.1
        elif self.action_types[action] in ["CODE_FIX", "CODE_REFACTOR"]:
            reward += 0.2  # Encourage proactive code maintenance
            
        return max(-10.0, min(10.0, reward))

# --- Enhanced OverseerBubble with Code Self-Healing and Warning Monitoring ---
class OverseerBubble(UniversalBubble):
    """Enhanced meta-RL bubble with code self-healing and warning monitoring capabilities"""
    
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        # Extract neural network parameters
        state_dim = kwargs.pop('state_dim', 12)
        action_dim = kwargs.pop('action_dim', 8)
        hidden_dim = kwargs.pop('hidden_dim', 256)
        latent_dim = kwargs.pop('latent_dim', 32)
        num_categories = kwargs.pop('num_categories', 32)
        
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Initialize attributes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.batch_size = 16
        self.batch_length = 64
        self.learning_rate = 4e-5
        self.execution_count = 0
        self.replay_buffer = deque(maxlen=5000000)
        self.metrics_history = deque(maxlen=50)
        self.env = SystemEnv(context.resource_manager)
        
        # Code healing specific
        self.code_history = {}  # Track code versions for rollback
        self.recovery_history = deque(maxlen=1000)  # Track successful recoveries
        self.pending_explanations = deque(maxlen=10)
        
        # Warning monitoring
        self.warning_analyzer = WarningPatternAnalyzer()
        self.warning_monitor = WarningMonitor(self.warning_analyzer)
        self.metrics_dashboard = OverseerMetricsDashboard()
        self.recent_warnings = deque(maxlen=100)
        
        # Healer-style runtime error handling
        self.runtime_patches = {}  # Cache of runtime patches
        self.error_contexts = deque(maxlen=100)  # Track error contexts
        self.healer_enabled = True
        self.max_heal_attempts = 3
        
        # Initialize monitors with code-aware variants
        self.monitors = [
            CodeAwareFaultMonitor(
                lambda s: s.get("cpu_percent", 0) > 90, 
                RecoveryLayers.boost_energy, 
                name="HighCPU"
            ),
            CodeAwareFaultMonitor(
                lambda s: s.get("metrics", {}).get("packet_loss", 0) > 0.2, 
                RecoveryLayers.reroute_data, 
                name="PacketLoss"
            ),
            CodeAwareFaultMonitor(
                lambda s: s.get("metrics", {}).get("llm_error_count", 0) > 5,
                lambda s: CodeRecoveryLayers.refactor_problematic_code(
                    s.get("error_bubble_id", "unknown"), "llm_errors"
                ),
                name="HighLLMErrors",
                analyze_code=True
            ),
            CodeAwareFaultMonitor(
                lambda s: len(s.get("metrics", {}).get("recent_errors", [])) > 0,
                lambda s: None,  # Recovery selected by code analysis
                name="CodeErrors",
                analyze_code=True
            )
        ]
        
        # Initialize PyTorch models if available - WITH DIMENSION VALIDATION
        if TORCH_AVAILABLE:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # CRITICAL FIX: Validate state dimensions before creating models
            temp_state = self.env.get_state()
            actual_state_dim = len(temp_state)
            
            if actual_state_dim != self.state_dim:
                logger.warning(f"{self.object_id}: Adjusting state_dim from {self.state_dim} to {actual_state_dim} to match environment")
                self.state_dim = actual_state_dim
            
            # Now create models with correct dimensions
            self.world_model = DreamerV3WorldModel(self.state_dim, self.hidden_dim, self.latent_dim, self.num_categories).to(self.device)
            self.policy = GRUPolicy(self.latent_dim, self.hidden_dim, self.action_dim).to(self.device)
            self.optimizer = optim.Adam(
                list(self.world_model.parameters()) + list(self.policy.parameters()), 
                lr=self.learning_rate, 
                eps=1e-20
            )
            self.world_hidden_state = torch.zeros(1, self.latent_dim, device=self.device)
            self.policy_hidden_state = torch.zeros(1, self.hidden_dim, device=self.device)
            logger.info(f"{self.object_id}: Initialized enhanced DreamerV3 with state_dim={self.state_dim} on {self.device}")
        else:
            self.world_model = None
            self.policy = None
            self.optimizer = None
            self.world_hidden_state = None
            self.policy_hidden_state = None
            logger.warning(f"{self.object_id}: PyTorch unavailable, using heuristic mode")
        
        asyncio.create_task(self._subscribe_to_events())
        # FIX: Start periodic reporting
        asyncio.create_task(self.periodic_reporting())
        logger.info(f"{self.object_id}: Initialized with code self-healing and warning monitoring")

    async def _subscribe_to_events(self):
        """Subscribe to system monitoring, code-related, and warning events"""
        await asyncio.sleep(0.1)
        try:
            # Original subscriptions
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.TUNING_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.OVERSEER_REPORT, self.handle_event)
            
            # Code-related subscriptions
            await EventService.subscribe(Actions.CODE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.BUBBLE_ERROR, self.handle_event)
            
            # Warning subscriptions using Actions instead of ExtendedActions
            await EventService.subscribe(Actions.WARNING_EVENT, self.handle_event)
            await EventService.subscribe(Actions.CORRELATION_WARNING, self.handle_event)
            await EventService.subscribe(Actions.PERFORMANCE_WARNING, self.handle_event)
            await EventService.subscribe(Actions.MEMORY_WARNING, self.handle_event)
            
            logger.debug(f"{self.object_id}: Subscribed to all events including warnings")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    # FIX: Add the missing handle_event dispatcher
    async def handle_event(self, event: Event):
        """Main event dispatcher that routes to appropriate handlers"""
        try:
            if event.type == Actions.SYSTEM_STATE_UPDATE:
                await self.handle_system_state(event)
            elif event.type == Actions.TUNING_UPDATE:
                # Handle tuning updates if needed
                pass
            elif event.type == Actions.OVERSEER_REPORT:
                # Handle overseer reports if needed
                pass
            elif event.type == Actions.CODE_UPDATE:
                await self.handle_code_update(event)
            elif event.type == Actions.BUBBLE_ERROR:
                await self.handle_bubble_error(event)
            elif event.type == Actions.WARNING_EVENT:
                await self.handle_warning_event(event)
            elif event.type == Actions.CORRELATION_WARNING:
                await self.handle_correlation_warning(event)
            elif event.type == Actions.PERFORMANCE_WARNING:
                await self.handle_performance_warning(event)
            elif event.type == Actions.MEMORY_WARNING:
                await self.handle_memory_warning(event)
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event {event.type}: {e}", exc_info=True)

    async def handle_warning_event(self, event: Event):
        """Handle generic warning events"""
        if not isinstance(event.data, UniversalCode):
            return
        
        warning_data = event.data.value
        warning_type = warning_data.get("warning_type", "UNKNOWN")
        source = warning_data.get("source", "unknown")
        
        # Record warning
        self.recent_warnings.append(warning_data)
        self.warning_analyzer.record_event("WARNING", source, warning_data)
        self.metrics_dashboard.update_warning_metrics(warning_type, source)
        
        logger.info(f"{self.object_id}: Warning from {source}: {warning_type}")
        
        # Check if we should take preventive action
        await self._check_warning_patterns()

    async def handle_correlation_warning(self, event: Event):
        """Specifically handle correlation ID warnings"""
        if not isinstance(event.data, UniversalCode):
            return
        
        data = event.data.value
        source = data.get("source", "unknown")
        
        # Skip core components
        if source in ["bubbles_core", "unknown"] or not source:
            logger.info(f"{self.object_id}: Skipping correlation warning from core component: {source}")
            return
        
        # Add to general warnings
        warning_data = {
            "warning_type": "CORRELATION_WARNING",
            "source": source,
            "timestamp": time.time(),
            "details": data
        }
        
        self.recent_warnings.append(warning_data)
        self.warning_analyzer.record_event("WARNING", source, warning_data)
        self.metrics_dashboard.update_warning_metrics("CORRELATION_WARNING", source)
        
        logger.info(f"{self.object_id}: Correlation warning from {source}")
        
        # Only try to fix actual bubbles we have code for
        manageable_ids = self._get_manageable_bubble_ids()
        if source not in manageable_ids:
            logger.info(f"{self.object_id}: {source} is not a managed bubble, skipping fix")
            return
            
        # Check if we should fix this bubble
        source_warnings = [w for w in self.recent_warnings if w.get("source") == source and w.get("warning_type") == "CORRELATION_WARNING"]
        
        if len(source_warnings) >= self.warning_monitor.warning_thresholds.get("CORRELATION_WARNING", 5):
            logger.warning(f"{self.object_id}: {source} has {len(source_warnings)} correlation warnings - fixing")
            fix_action = await self._generate_correlation_fix(source)
            if fix_action:
                await self._execute_control_action(fix_action, {"warning": data})
                self.metrics_dashboard.record_fix_applied("correlation_id", source)

    async def handle_performance_warning(self, event: Event):
        """Handle performance-related warnings"""
        if not isinstance(event.data, UniversalCode):
            return
        
        data = event.data.value
        source = data.get("source", "unknown")
        
        warning_data = {
            "warning_type": "PERFORMANCE_WARNING",
            "source": source,
            "timestamp": time.time(),
            "details": data
        }
        
        self.recent_warnings.append(warning_data)
        self.warning_analyzer.record_event("WARNING", source, warning_data)
        self.metrics_dashboard.update_warning_metrics("PERFORMANCE_WARNING", source)
        
        await self._check_warning_patterns()

    async def handle_memory_warning(self, event: Event):
        """Handle memory-related warnings"""
        if not isinstance(event.data, UniversalCode):
            return
        
        data = event.data.value
        source = data.get("source", "unknown")
        
        warning_data = {
            "warning_type": "MEMORY_WARNING",
            "source": source,
            "timestamp": time.time(),
            "details": data
        }
        
        self.recent_warnings.append(warning_data)
        self.warning_analyzer.record_event("WARNING", source, warning_data)
        self.metrics_dashboard.update_warning_metrics("MEMORY_WARNING", source)
        
        await self._check_warning_patterns()

    async def _check_warning_patterns(self):
        """Check if warning patterns indicate need for preventive action"""
        # Let warning monitor check recent warnings
        await self.warning_monitor.check_and_recover(list(self.recent_warnings), self)
        
        # Check for predictive patterns
        self.warning_analyzer.analyze_pattern_to_error_correlation()
        error_probability = self.warning_analyzer.predict_error_probability(list(self.recent_warnings)[-10:])
        
        if error_probability > 0.7:
            logger.warning(f"{self.object_id}: High error probability detected: {error_probability:.2%}")
            # Take preemptive action
            problem_bubbles = self.metrics_dashboard.get_problem_bubbles(threshold=5)
            if problem_bubbles:
                worst_bubble = problem_bubbles[0]
                action = CodeRecoveryLayers.refactor_problematic_code(
                    worst_bubble["bubble_id"], 
                    "high_error_probability"
                )
                await self._execute_control_action(action, {"prediction": error_probability})

    async def _generate_correlation_fix(self, bubble_id: str) -> Optional[Dict]:
        """Generate fix for missing correlation IDs"""
        
        # Get current code
        if bubble_id not in self.code_history:
            logger.warning(f"{self.object_id}: No code history for {bubble_id}")
            return None
        
        current_code = self.code_history[bubble_id][-1]["code"]
        
        # Analyze where correlation_id should be added
        fix_prompt = f"""The following code is missing correlation_id in its events:

```python
{current_code}
```

Add correlation_id generation and propagation to all event publishing.
Use: correlation_id = str(uuid.uuid4()) if not provided.
Make sure to:
1. Accept correlation_id in function parameters where needed
2. Generate new correlation_id if not provided
3. Pass correlation_id to all dispatched events

Return the fixed code that properly handles correlation_ids."""
        
        try:
            # Use LLM to generate fix
            from flood_control import query_llm_with_flood_control
            
            result = await query_llm_with_flood_control(
                prompt=fix_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=4
            )
            
            if result.get("response"):
                fixed_code = self._extract_code_from_response(result["response"])
                
                if self._validate_code(fixed_code):
                    return {
                        "action_type": "CODE_UPDATE",
                        "payload": {
                            "bubble_id": bubble_id,
                            "code": fixed_code,
                            "reason": "auto_fix_correlation_id",
                            "warning_count": len([w for w in self.recent_warnings 
                                                if w.get("source") == bubble_id and 
                                                w.get("warning_type") == "CORRELATION_WARNING"])
                        }
                    }
                else:
                    logger.error(f"{self.object_id}: Generated code failed validation")
        except Exception as e:
            logger.error(f"{self.object_id}: Error generating correlation fix: {e}")
        
        return None

    async def handle_code_update(self, event: Event):
        """Track code updates for potential rollback"""
        if not isinstance(event.data, UniversalCode):
            return
            
        update_data = event.data.value
        bubble_id = update_data.get("bubble_id")
        code = update_data.get("code")
        
        if bubble_id and code:
            if bubble_id not in self.code_history:
                self.code_history[bubble_id] = deque(maxlen=10)
            
            self.code_history[bubble_id].append({
                "version": len(self.code_history[bubble_id]),
                "code": code,
                "timestamp": time.time(),
                "metrics_snapshot": self.env.get_state()
            })
            
            logger.info(f"{self.object_id}: Tracked code update for {bubble_id}")

    async def handle_bubble_error(self, event: Event):
        """Handle errors from bubbles and attempt recovery"""
        if not isinstance(event.data, UniversalCode):
            return
            
        error_data = event.data.value
        
        # Handle both old format (bubble_id, error_type) and new format (source, warning_type)
        bubble_id = error_data.get("bubble_id") or error_data.get("source")
        error_type = error_data.get("error_type") or error_data.get("warning_type", "UNKNOWN")
        traceback_str = error_data.get("traceback", "")
        
        # Skip if no bubble_id or if it's a core component we can't fix
        if not bubble_id or bubble_id == "bubbles_core":
            logger.info(f"{self.object_id}: Skipping error from core component: {bubble_id}")
            return
        
        # Record as error event
        self.warning_analyzer.record_event("ERROR", bubble_id, error_data)
        
        logger.warning(f"{self.object_id}: Bubble error from {bubble_id}: {error_type}")
        
        # Attempt immediate recovery only for actual bubbles
        manageable_ids = self._get_manageable_bubble_ids()
        if bubble_id in manageable_ids:
            recovery_action = await self._diagnose_and_fix_code(bubble_id, error_type, traceback_str)
            if recovery_action:
                await self._execute_control_action(recovery_action, {"error": error_data})

    async def _diagnose_and_fix_code(self, bubble_id: str, error_type: str, 
                                     traceback_str: str) -> Optional[Dict]:
        """Enhanced diagnosis with error contextualization and multiple fix strategies"""
        try:
            # Skip if bubble doesn't exist or we don't have code history
            if bubble_id not in self.code_history:
                logger.info(f"{self.object_id}: No code history for {bubble_id}, skipping diagnosis")
                return None
                
            # Gather comprehensive error context
            error_context = {
                "bubble_id": bubble_id,
                "error_type": error_type,
                "traceback": traceback_str,
                "timestamp": time.time(),
                "system_state": self._get_system_health_snapshot(),
                "recent_changes": self._get_recent_code_changes(bubble_id),
                "error_frequency": self._get_error_frequency(bubble_id, error_type)
            }
            
            # Strategy 1: Quick AST-based syntax fix
            if "SyntaxError" in error_type or "IndentationError" in error_type:
                if bubble_id in self.code_history and self.code_history[bubble_id]:
                    current_code = self.code_history[bubble_id][-1]["code"]
                    fixed_code = self._try_ast_fix(current_code, error_context)
                    if fixed_code:
                        logger.info(f"Successfully applied AST fix for {bubble_id}")
                        return CodeRecoveryLayers.fix_syntax_error(error_context)
            
            # Strategy 2: Pattern-based recovery from history
            similar_fix = await self._find_similar_successful_fix(error_context)
            if similar_fix:
                logger.info(f"Found similar successful fix pattern for {bubble_id}")
                return similar_fix
            
            # Strategy 3: Rollback to stable version
            if bubble_id in self.code_history and len(self.code_history[bubble_id]) > 1:
                stable_version = self._find_last_stable_version(bubble_id)
                if stable_version is not None:
                    logger.info(f"Rolling back {bubble_id} to stable version {stable_version}")
                    return CodeRecoveryLayers.rollback_code(bubble_id, stable_version)
            
            # Strategy 4: Context-aware refactoring
            if error_context["error_frequency"] > 3:
                # Persistent errors require more aggressive refactoring
                logger.warning(f"Persistent errors in {bubble_id}, requesting full refactor")
                return {
                    "action_type": "CODE_REFACTOR",
                    "payload": {
                        "bubble_id": bubble_id,
                        "issue_type": error_type,
                        "context": error_context,
                        "strategy": "comprehensive_refactor"
                    }
                }
            
            # Default: Intelligent refactoring with context
            return CodeRecoveryLayers.refactor_problematic_code(bubble_id, error_type)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error in enhanced diagnosis: {e}", exc_info=True)
            return None

    def _get_system_health_snapshot(self) -> Dict:
        """Get current system health metrics"""
        try:
            state = self.env.get_state()
            return {
                "cpu_usage": float(state[1]) if len(state) > 1 else 0,
                "memory_usage": float(state[2]) if len(state) > 2 else 0,
                "error_rate": float(state[10]) if len(state) > 10 else 0,
                "timestamp": time.time()
            }
        except Exception:
            return {}

    def _get_recent_code_changes(self, bubble_id: str, window: int = 5) -> List[Dict]:
        """Get recent code changes for a bubble"""
        if bubble_id not in self.code_history:
            return []
        
        history = list(self.code_history[bubble_id])
        return history[-window:] if len(history) > window else history

    def _get_error_frequency(self, bubble_id: str, error_type: str) -> int:
        """Count recent errors of the same type for a bubble"""
        if not bubble_id or not error_type:
            return 0
            
        count = 0
        for record in self.recovery_history:
            state_snapshot = record.get("state_snapshot", {})
            if state_snapshot:
                record_bubble = state_snapshot.get("error_bubble_id") or state_snapshot.get("bubble_id")
                record_error = state_snapshot.get("error_type") or state_snapshot.get("warning_type")
                
                if (record_bubble == bubble_id and
                    record_error == error_type and
                    time.time() - record.get("timestamp", 0) < 3600):  # Within last hour
                    count += 1
        return count

    def _get_manageable_bubble_ids(self) -> List[str]:
        """Get IDs of bubbles we can actually manage/fix"""
        manageable_ids = []
        
        # Get all registered bubbles
        try:
            all_bubbles = self.context.get_all_bubbles()
            manageable_ids.extend([b.object_id for b in all_bubbles])
        except:
            pass
        
        # Also include any bubbles we have code history for
        manageable_ids.extend(self.code_history.keys())
        
        # Remove duplicates and filter out core components
        manageable_ids = list(set(manageable_ids))
        manageable_ids = [bid for bid in manageable_ids if bid not in ["bubbles_core", "EventService", "ResourceManager"]]
        
        return manageable_ids

    async def _find_similar_successful_fix(self, error_context: Dict) -> Optional[Dict]:
        """Find similar successful fixes from recovery history"""
        error_type = error_context["error_type"]
        
        # Look for successful fixes for similar errors
        for record in reversed(self.recovery_history):
            if (record.get("success") and 
                record.get("monitor") == "CodeErrors" and
                record.get("state_snapshot", {}).get("error_type") == error_type):
                
                # Check if the fix is still relevant
                fix_age = time.time() - record.get("timestamp", 0)
                if fix_age < 86400:  # Within 24 hours
                    return record.get("action")
        
        return None

    def _find_last_stable_version(self, bubble_id: str) -> Optional[int]:
        """Find the last stable version of code for a bubble"""
        if bubble_id not in self.code_history:
            return None
        
        # Look for version with good metrics
        for version_info in reversed(list(self.code_history[bubble_id])[:-1]):
            metrics = version_info.get("metrics_snapshot", [])
            if len(metrics) > 10:
                error_rate = float(metrics[10])
                if error_rate < 0.05:  # Less than 5% error rate
                    return version_info["version"]
        
        return None

    async def handle_system_state(self, event: Event):
        """Enhanced system state handling with code health monitoring and shape validation"""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            return
            
        state = event.data.value
        
        # Run all monitors including code-aware ones
        for monitor in self.monitors:
            await monitor.check_and_recover(state, self)
        
        self.metrics_history.append(state)
        
        # Continue with RL-based action selection
        state_vector = self._vectorize_state(state)
        if state_vector is None:
            action = self._heuristic_action(state)
            await self._execute_control_action(action, state)
            return

        if TORCH_AVAILABLE and self.policy:
            try:
                # RL-based action selection with shape validation
                state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Verify tensor shape matches model expectations
                expected_input_size = self.world_model.encoder[0].in_features
                if state_tensor.shape[1] != expected_input_size:
                    logger.error(f"{self.object_id}: Tensor shape {state_tensor.shape} doesn't match model input size {expected_input_size}")
                    action = self._heuristic_action(state)
                    await self._execute_control_action(action, state)
                    return
                
                z, next_latent, _ = self.world_model(state_tensor, self.world_hidden_state)
                self.world_hidden_state = next_latent.detach()
                action_dist, self.policy_hidden_state, value = self.policy(z.float(), self.policy_hidden_state)
                action_idx = action_dist.sample().item()
                action = {"action_type": self.env.action_types[action_idx], "payload": {}}
                
                # Update for code-specific actions
                if self.env.action_types[action_idx].startswith("CODE_"):
                    action["payload"] = self._prepare_code_action_payload(
                        self.env.action_types[action_idx], state
                    )
                
                next_state = self.env.get_state()
                reward = self.env.compute_reward(state_vector, action_idx, next_state)
                self.replay_buffer.append((state_vector, action_idx, reward, next_state))
                await self._execute_control_action(action, state)
            except Exception as e:
                logger.error(f"{self.object_id}: Error in RL forward pass: {e}", exc_info=True)
                # Fall back to heuristic
                action = self._heuristic_action(state)
                await self._execute_control_action(action, state)
        else:
            action = self._heuristic_action(state)
            await self._execute_control_action(action, state)

    def _prepare_code_action_payload(self, action_type: str, state: Dict) -> Dict:
        """Prepare payload for code-specific actions"""
        metrics = state.get("metrics", {})
        
        if action_type == "CODE_FIX":
            recent_errors = metrics.get("recent_errors", [])
            if recent_errors:
                return {
                    "error_info": recent_errors[0],
                    "fix_type": "auto"
                }
        elif action_type == "CODE_ROLLBACK":
            # Find bubble with highest error rate
            error_bubbles = metrics.get("error_bubbles", {})
            if error_bubbles:
                worst_bubble = max(error_bubbles.items(), key=lambda x: x[1])[0]
                return {
                    "bubble_id": worst_bubble,
                    "target_version": -1  # Previous version
                }
        
        return {}

    async def record_recovery(self, monitor_name: str, state: Dict, action: Dict):
        """Enhanced recovery recording with pattern learning and success tracking"""
        recovery_record = {
            "timestamp": time.time(),
            "monitor": monitor_name,
            "state_snapshot": state,
            "action": action,
            "system_health_before": self._get_system_health_snapshot(),
            "code_versions_before": self._capture_code_versions(),
            "success": None,  # Will be determined asynchronously
            "outcome_metrics": {}
        }
        
        # Add to history immediately
        self.recovery_history.append(recovery_record)
        
        # Schedule outcome evaluation
        asyncio.create_task(self._evaluate_recovery_outcome(recovery_record))
        
        # Extract patterns for learning
        if len(self.recovery_history) >= 10:
            await self._learn_recovery_patterns()

    async def _evaluate_recovery_outcome(self, recovery_record: Dict):
        """Evaluate the success of a recovery action after some time"""
        # Wait for the action to take effect
        await asyncio.sleep(30)
        
        try:
            # Get current state
            current_health = self._get_system_health_snapshot()
            before_health = recovery_record["system_health_before"]
            
            # Calculate improvement metrics
            cpu_improved = current_health.get("cpu_usage", 100) < before_health.get("cpu_usage", 100)
            error_reduced = current_health.get("error_rate", 1) < before_health.get("error_rate", 1)
            
            # Determine success
            recovery_record["success"] = cpu_improved or error_reduced
            recovery_record["outcome_metrics"] = {
                "cpu_delta": current_health.get("cpu_usage", 0) - before_health.get("cpu_usage", 0),
                "error_delta": current_health.get("error_rate", 0) - before_health.get("error_rate", 0),
                "evaluation_time": time.time()
            }
            
            # Update dashboard
            self.metrics_dashboard.record_fix_outcome(recovery_record["success"])
            
            # If successful, learn from it
            if recovery_record["success"]:
                logger.info(f"Recovery action '{recovery_record['monitor']}' was successful")
                await self._propagate_successful_pattern(recovery_record)
            else:
                logger.warning(f"Recovery action '{recovery_record['monitor']}' did not improve system")
                
        except Exception as e:
            logger.error(f"Error evaluating recovery outcome: {e}")
            recovery_record["success"] = False

    async def _learn_recovery_patterns(self):
        """Analyze recovery history to learn effective patterns"""
        try:
            # Group by monitor and success
            successful_patterns = {}
            failed_patterns = {}
            
            for record in self.recovery_history:
                if record.get("success") is None:
                    continue
                    
                monitor = record["monitor"]
                pattern_key = f"{monitor}_{record['action'].get('action_type', 'unknown')}"
                
                if record["success"]:
                    if pattern_key not in successful_patterns:
                        successful_patterns[pattern_key] = []
                    successful_patterns[pattern_key].append(record)
                else:
                    if pattern_key not in failed_patterns:
                        failed_patterns[pattern_key] = []
                    failed_patterns[pattern_key].append(record)
            
            # Update monitor strategies based on success rates
            for pattern_key, successes in successful_patterns.items():
                failure_count = len(failed_patterns.get(pattern_key, []))
                success_rate = len(successes) / (len(successes) + failure_count)
                
                if success_rate > 0.7:  # High success rate
                    logger.info(f"Pattern '{pattern_key}' has {success_rate:.2%} success rate")
                    # Could adjust monitor thresholds or recovery strategies here
            
            # Identify problematic patterns
            for pattern_key, failures in failed_patterns.items():
                if len(failures) > 5:
                    logger.warning(f"Pattern '{pattern_key}' has failed {len(failures)} times")
                    # Could disable or modify ineffective strategies
                    
        except Exception as e:
            logger.error(f"Error learning recovery patterns: {e}")

    async def _propagate_successful_pattern(self, recovery_record: Dict):
        """Share successful recovery patterns with monitors"""
        try:
            monitor_name = recovery_record["monitor"]
            action_type = recovery_record["action"].get("action_type")
            
            # Find the monitor that generated this recovery
            for monitor in self.monitors:
                if monitor.name == monitor_name:
                    # If this monitor supports learning, update it
                    if hasattr(monitor, 'update_success_metrics'):
                        await monitor.update_success_metrics(
                            action_type, 
                            recovery_record["outcome_metrics"]
                        )
                    break
            
            # Also update RL model with this success
            if TORCH_AVAILABLE and self.policy:
                # Convert to reward signal for RL
                reward = self._calculate_recovery_reward(recovery_record)
                if reward > 0:
                    state_vector = self._vectorize_state(recovery_record["state_snapshot"])
                    if state_vector is not None:
                        # Add to replay buffer with bonus reward
                        action_idx = self.env.action_types.index(action_type) if action_type in self.env.action_types else 0
                        self.replay_buffer.append((state_vector, action_idx, reward * 1.5, state_vector))
                        
        except Exception as e:
            logger.error(f"Error propagating successful pattern: {e}")

    def _calculate_recovery_reward(self, recovery_record: Dict) -> float:
        """Calculate reward value from recovery outcome"""
        metrics = recovery_record.get("outcome_metrics", {})
        
        # Positive for improvements, negative for degradation
        cpu_reward = -metrics.get("cpu_delta", 0) / 50.0  # CPU reduction is good
        error_reward = -metrics.get("error_delta", 0) * 5.0  # Error reduction is very good
        
        return max(-1.0, min(1.0, cpu_reward + error_reward))

    def _capture_code_versions(self) -> Dict[str, int]:
        """Capture current code versions for all bubbles"""
        versions = {}
        for bubble_id, history in self.code_history.items():
            if history:
                versions[bubble_id] = history[-1]["version"]
        return versions

    def _heuristic_action(self, state: Dict) -> Dict:
        """Enhanced heuristic with code-aware decisions"""
        cpu_percent = state.get("cpu_percent", 0)
        energy = state.get("energy", 0)
        metrics = state.get("metrics", {})
        
        # Check for code issues first
        if metrics.get("code_error_rate", 0) > 0.2:
            return {"action_type": "CODE_REFACTOR", "payload": {"reason": "High error rate"}}
        elif len(metrics.get("recent_errors", [])) > 3:
            return {"action_type": "CODE_FIX", "payload": {"reason": "Multiple recent errors"}}
        
        # Check warning patterns
        problem_bubbles = self.metrics_dashboard.get_problem_bubbles(threshold=5)
        if problem_bubbles:
            worst_bubble = problem_bubbles[0]
            return {
                "action_type": "CODE_INJECT",
                "payload": {
                    "bubble_id": worst_bubble["bubble_id"],
                    "reason": f"{worst_bubble['warning_count']} warnings",
                    "top_issues": worst_bubble["top_issues"]
                }
            }
        
        # Original heuristics
        if cpu_percent > 90:
            return {"action_type": "PAUSE_BUBBLE", "payload": {"reason": "High CPU"}}
        elif energy < 2000:
            return {"action_type": "ADJUST_ENERGY", "payload": {"amount": 1000}}
            
        return {"action_type": "NO_OP", "payload": {"reason": "Stable state"}}

    async def _execute_control_action(self, action: Dict, state: Dict):
        """Execute control action with code-specific handling"""
        action_type = action.get("action_type")
        payload = action.get("payload", {})
        
        logger.info(f"{self.object_id}: Executing control action: {action_type}")
        
        # Special handling for code actions
        if action_type.startswith("CODE_"):
            await self._execute_code_action(action_type, payload, state)
        
        # Dispatch control event
        control_uc = UniversalCode(
            Tags.DICT, 
            {"action_type": action_type, "payload": payload},
            description=f"Overseer control: {action_type}"
        )
        control_event = Event(
            type=Actions.OVERSEER_CONTROL, 
            data=control_uc, 
            origin=self.object_id, 
            priority=6
        )
        await self.context.dispatch_event(control_event)
        
        if action_type != "NO_OP":
            await self._generate_explanation(action, state)

    async def _execute_code_action(self, action_type: str, payload: Dict, state: Dict):
        """Execute code-specific recovery actions"""
        try:
            if action_type == "CODE_FIX":
                await self._attempt_code_fix(payload, state)
            elif action_type == "CODE_ROLLBACK":
                await self._rollback_code(payload)
            elif action_type == "CODE_INJECT":
                await self._inject_error_handler(payload)
            elif action_type == "CODE_REFACTOR":
                await self._request_code_refactor(payload, state)
        except Exception as e:
            logger.error(f"{self.object_id}: Error executing code action {action_type}: {e}")

    async def _attempt_code_fix(self, payload: Dict, state: Dict):
        """Attempt to fix code using AST manipulation or LLM"""
        error_info = payload.get("error_info", {})
        bubble_id = error_info.get("bubble_id")
        
        if not bubble_id or bubble_id not in self.code_history:
            return
            
        current_code = self.code_history[bubble_id][-1]["code"]
        
        # Try simple AST-based fixes first
        fixed_code = self._try_ast_fix(current_code, error_info)
        
        if fixed_code:
            fix_event = Event(
                type=Actions.CODE_UPDATE,
                data=UniversalCode(Tags.DICT, {
                    "bubble_id": bubble_id,
                    "code": fixed_code,
                    "fix_type": "ast_auto_fix",
                    "original_error": error_info
                }),
                origin=self.object_id,
                priority=7
            )
            await self.context.dispatch_event(fix_event)

    def _try_ast_fix(self, code: str, error_info: Dict) -> Optional[str]:
        """Enhanced AST-based code fixes with multiple repair strategies"""
        try:
            # Try to parse the code
            ast.parse(code)
            return None  # No syntax error
        except SyntaxError as e:
            lines = code.split('\n')
            error_msg = str(e)
            
            # Strategy 1: Fix missing colons
            if "expected ':'" in error_msg or "invalid syntax" in error_msg:
                line_num = e.lineno - 1
                if line_num < len(lines):
                    line = lines[line_num]
                    # Check for common patterns that need colons
                    if any(line.strip().startswith(kw) for kw in ['if', 'elif', 'else', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with']):
                        if not line.rstrip().endswith(':'):
                            lines[line_num] = line.rstrip() + ':'
                            fixed_code = '\n'.join(lines)
                            if self._validate_code(fixed_code):
                                return fixed_code
            
            # Strategy 2: Fix mismatched parentheses/brackets
            if "unexpected EOF" in error_msg or "invalid syntax" in error_msg:
                fixed_code = self._fix_mismatched_brackets(code)
                if fixed_code and self._validate_code(fixed_code):
                    return fixed_code
            
            # Strategy 3: Fix indentation errors
            if "IndentationError" in error_info.get("error_type", "") or "unexpected indent" in error_msg:
                fixed_code = self._fix_indentation(lines)
                if fixed_code and self._validate_code(fixed_code):
                    return fixed_code
            
            # Strategy 4: Fix missing imports
            if "NameError" in error_info.get("error_type", ""):
                fixed_code = self._fix_missing_imports(code, error_info)
                if fixed_code and self._validate_code(fixed_code):
                    return fixed_code
            
            # Strategy 5: Fix common typos
            fixed_code = self._fix_common_typos(code, e)
            if fixed_code and self._validate_code(fixed_code):
                return fixed_code
        
        except Exception as fix_error:
            logger.error(f"Error in AST fix attempt: {fix_error}")
        
        return None

    def _fix_mismatched_brackets(self, code: str) -> Optional[str]:
        """Fix mismatched parentheses, brackets, and braces"""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        reverse_pairs = {v: k for k, v in pairs.items()}
        
        lines = code.split('\n')
        
        # Count brackets
        for line in lines:
            for char in line:
                if char in pairs:
                    stack.append(char)
                elif char in reverse_pairs:
                    if stack and stack[-1] == reverse_pairs[char]:
                        stack.pop()
                    else:
                        # Mismatched closing bracket
                        return None
        
        # Add missing closing brackets
        if stack:
            closing_brackets = ''.join(pairs[bracket] for bracket in reversed(stack))
            lines.append(closing_brackets)
            return '\n'.join(lines)
        
        return None

    def _fix_indentation(self, lines: List[str]) -> Optional[str]:
        """Advanced indentation fixing with context awareness"""
        fixed_lines = []
        indent_stack = [0]  # Stack to track indentation levels
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped:  # Empty line
                fixed_lines.append('')
                continue
            
            # Determine expected indentation
            if stripped.startswith(('elif', 'else', 'except', 'finally')):
                # These should match the indentation of their corresponding if/try
                if len(indent_stack) > 1:
                    indent_stack.pop()
                indent_level = indent_stack[-1]
            elif stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')):
                # These typically appear inside blocks
                indent_level = indent_stack[-1]
            elif any(stripped.startswith(kw + ' ') or stripped.startswith(kw + ':') 
                    for kw in ['if', 'for', 'while', 'with', 'def', 'class', 'try']):
                # Block starters
                indent_level = indent_stack[-1]
                if stripped.endswith(':'):
                    indent_stack.append(indent_level + 1)
            else:
                # Regular statements
                indent_level = indent_stack[-1]
            
            # Apply indentation
            fixed_lines.append('    ' * indent_level + stripped)
            
            # Adjust stack for dedentation
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped and not any(next_stripped.startswith(kw) 
                                           for kw in ['elif', 'else', 'except', 'finally']):
                    # Check if we need to dedent
                    if indent_level > 0 and len(indent_stack) > 1:
                        # Simple heuristic: dedent after return/pass/break
                        if stripped.startswith(('return', 'pass', 'break', 'continue')):
                            indent_stack.pop()
        
        return '\n'.join(fixed_lines)

    def _fix_missing_imports(self, code: str, error_info: Dict) -> Optional[str]:
        """Attempt to fix missing import errors"""
        error_msg = str(error_info.get("error", ""))
        
        # Common module mappings
        common_imports = {
            'np': 'import numpy as np',
            'pd': 'import pandas as pd',
            'torch': 'import torch',
            'nn': 'from torch import nn',
            'plt': 'import matplotlib.pyplot as plt',
            'asyncio': 'import asyncio',
            'logging': 'import logging',
            'json': 'import json',
            'os': 'import os',
            'sys': 'import sys',
            'time': 'import time',
            're': 'import re',
            'random': 'import random',
            'deque': 'from collections import deque',
            'Dict': 'from typing import Dict',
            'List': 'from typing import List',
            'Optional': 'from typing import Optional',
            'Any': 'from typing import Any',
            'Tuple': 'from typing import Tuple',
        }
        
        # Extract the missing name from error
        import re
        match = re.search(r"name '(\w+)' is not defined", error_msg)
        if match:
            missing_name = match.group(1)
            
            # Check if we have a known import for this
            if missing_name in common_imports:
                import_line = common_imports[missing_name]
                lines = code.split('\n')
                
                # Add import at the beginning
                import_index = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(('import', 'from')):
                        import_index = i
                        break
                
                lines.insert(import_index, import_line)
                return '\n'.join(lines)
        
        return None

    def _fix_common_typos(self, code: str, error: SyntaxError) -> Optional[str]:
        """Fix common typos and syntax mistakes"""
        lines = code.split('\n')
        line_num = error.lineno - 1 if error.lineno else -1
        
        if 0 <= line_num < len(lines):
            line = lines[line_num]
            
            # Common typo patterns
            typo_fixes = [
                (r'\bture\b', 'True'),
                (r'\bfalse\b', 'False'),
                (r'\bnone\b', 'None'),
                (r'\bexept\b', 'except'),
                (r'\bfinaly\b', 'finally'),
                (r'\belif\s*\(', 'elif '),  # elif( -> elif 
                (r'\bif\s*\(', 'if '),      # if( -> if 
                (r'===', '=='),             # JS equality to Python
                (r'!==', '!='),             # JS inequality to Python
            ]
            
            fixed_line = line
            for pattern, replacement in typo_fixes:
                fixed_line = re.sub(pattern, replacement, fixed_line)
            
            if fixed_line != line:
                lines[line_num] = fixed_line
                return '\n'.join(lines)
        
        return None

    async def _rollback_code(self, payload: Dict):
        """Rollback code to a previous version"""
        bubble_id = payload.get("bubble_id")
        target_version = payload.get("target_version", -1)
        
        if not bubble_id or bubble_id not in self.code_history:
            return
        
        history = self.code_history[bubble_id]
        if not history:
            return
        
        # Get target version
        if target_version == -1:  # Previous version
            target_version = len(history) - 2
        
        if 0 <= target_version < len(history):
            version_info = history[target_version]
            rollback_event = Event(
                type=Actions.CODE_UPDATE,
                data=UniversalCode(Tags.DICT, {
                    "bubble_id": bubble_id,
                    "code": version_info["code"],
                    "fix_type": "rollback",
                    "rollback_version": target_version,
                    "reason": payload.get("reason", "Error recovery")
                }),
                origin=self.object_id,
                priority=7
            )
            await self.context.dispatch_event(rollback_event)
            logger.info(f"{self.object_id}: Rolled back {bubble_id} to version {target_version}")

    async def _inject_error_handler(self, payload: Dict):
        """Inject error handling code into a bubble"""
        bubble_id = payload.get("bubble_id")
        injection_type = payload.get("injection_type")
        
        if not bubble_id or bubble_id not in self.code_history:
            return
        
        current_code = self.code_history[bubble_id][-1]["code"]
        
        # Generate appropriate injection based on type
        if injection_type == "error_handler":
            injected_code = await self._generate_error_handler_injection(current_code, payload)
        elif injection_type == "correlation_id_handler":
            injected_code = await self._generate_correlation_handler_injection(current_code)
        elif injection_type == "memory_cleanup":
            injected_code = await self._generate_memory_cleanup_injection(current_code)
        else:
            return
        
        if injected_code and self._validate_code(injected_code):
            inject_event = Event(
                type=Actions.CODE_UPDATE,
                data=UniversalCode(Tags.DICT, {
                    "bubble_id": bubble_id,
                    "code": injected_code,
                    "fix_type": "injection",
                    "injection_type": injection_type
                }),
                origin=self.object_id,
                priority=7
            )
            await self.context.dispatch_event(inject_event)
            self.metrics_dashboard.record_fix_applied(injection_type, bubble_id)

    async def _generate_error_handler_injection(self, code: str, payload: Dict) -> Optional[str]:
        """Generate code with error handling injected"""
        error_type = payload.get("error_type", "general")
        
        # Analyze code structure
        try:
            tree = ast.parse(code)
            
            # Find functions that need error handling
            functions_to_wrap = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if function already has try-except
                    has_error_handling = any(
                        isinstance(child, ast.Try) 
                        for child in node.body
                    )
                    if not has_error_handling:
                        functions_to_wrap.append(node.name)
            
            if not functions_to_wrap:
                return code  # Already has error handling
            
            # Generate prompt for LLM
            from flood_control import query_llm_with_flood_control
            
            inject_prompt = f"""Add comprehensive error handling to the following code:

```python
{code}
```

Focus on these functions that lack error handling: {', '.join(functions_to_wrap)}

Requirements:
1. Wrap function bodies in try-except blocks
2. Log errors with proper context
3. Return appropriate default values on error
4. Maintain original functionality
5. Add error type: {error_type} specific handling if applicable

Return the enhanced code with error handling."""

            result = await query_llm_with_flood_control(
                prompt=inject_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=4
            )
            
            if result.get("response"):
                return self._extract_code_from_response(result["response"])
                
        except Exception as e:
            logger.error(f"Error generating error handler injection: {e}")
        
        return None

    async def _generate_correlation_handler_injection(self, code: str) -> Optional[str]:
        """Generate code with correlation ID handling"""
        # This is handled by _generate_correlation_fix
        return await self._generate_correlation_fix_code(code)

    async def _generate_correlation_fix_code(self, code: str) -> Optional[str]:
        """Helper to generate correlation ID fix"""
        try:
            from flood_control import query_llm_with_flood_control
            
            fix_prompt = f"""Fix the following code to properly handle correlation IDs:

```python
{code}
```

Requirements:
1. Import uuid at the top
2. Accept correlation_id as parameter in relevant functions
3. Generate correlation_id = str(uuid.uuid4()) if not provided
4. Pass correlation_id to all event dispatches
5. Maintain backward compatibility

Return the fixed code."""

            result = await query_llm_with_flood_control(
                prompt=fix_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=4
            )
            
            if result.get("response"):
                return self._extract_code_from_response(result["response"])
                
        except Exception as e:
            logger.error(f"Error generating correlation fix: {e}")
        
        return None

    async def _generate_memory_cleanup_injection(self, code: str) -> Optional[str]:
        """Generate code with memory cleanup improvements"""
        try:
            from flood_control import query_llm_with_flood_control
            
            cleanup_prompt = f"""Add memory management improvements to the following code:

```python
{code}
```

Requirements:
1. Clear large objects when no longer needed
2. Use generators instead of lists where appropriate
3. Limit collection sizes (deque with maxlen)
4. Add periodic cleanup routines
5. Implement __del__ methods if needed

Return the memory-optimized code."""

            result = await query_llm_with_flood_control(
                prompt=cleanup_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=4
            )
            
            if result.get("response"):
                return self._extract_code_from_response(result["response"])
                
        except Exception as e:
            logger.error(f"Error generating memory cleanup: {e}")
        
        return None

    async def _request_code_refactor(self, payload: Dict, state: Dict):
        """Request comprehensive code refactoring"""
        bubble_id = payload.get("bubble_id")
        issue_type = payload.get("issue_type")
        
        if not bubble_id or bubble_id not in self.code_history:
            return
        
        current_code = self.code_history[bubble_id][-1]["code"]
        
        # Use self-planning approach for complex refactoring
        refactored_code = await self._self_planning_code_fix(
            current_code, issue_type, bubble_id, state
        )
        
        if refactored_code and self._validate_code(refactored_code):
            refactor_event = Event(
                type=Actions.CODE_UPDATE,
                data=UniversalCode(Tags.DICT, {
                    "bubble_id": bubble_id,
                    "code": refactored_code,
                    "fix_type": "refactor",
                    "issue_type": issue_type,
                    "strategy": payload.get("strategy", "self_planning")
                }),
                origin=self.object_id,
                priority=7
            )
            await self.context.dispatch_event(refactor_event)
            self.metrics_dashboard.record_fix_applied("refactor", bubble_id)

    async def _self_planning_code_fix(self, current_code: str, issue_type: str, 
                                      bubble_id: str, state: Dict) -> Optional[str]:
        """Enhanced two-phase self-planning with validation checkpoints"""
        try:
            from flood_control import query_llm_with_flood_control
            
            # Phase 1: Deep analysis and plan extraction
            analysis_result = await self._analyze_code_deeply(
                current_code, issue_type, bubble_id, state
            )
            
            if not analysis_result:
                return await self._direct_code_fix(current_code, issue_type, state)
            
            # Phase 2: Multi-step planning with validation
            current_plan = analysis_result.get("plan", "")
            root_causes = analysis_result.get("root_causes", [])
            
            # Generate improved plan addressing root causes
            improved_plan_prompt = self._create_root_cause_aware_plan_prompt(
                current_plan, issue_type, root_causes, state
            )
            
            improved_plan_result = await query_llm_with_flood_control(
                prompt=improved_plan_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=3
            )
            
            if improved_plan_result.get("blocked"):
                return None
                
            improved_plan = improved_plan_result.get("response", "")
            
            # Validation checkpoint 1: Ensure plan addresses all root causes
            if not self._validate_plan_coverage(improved_plan, root_causes):
                logger.warning(f"Plan doesn't address all root causes, enhancing...")
                improved_plan = await self._enhance_plan_for_coverage(
                    improved_plan, root_causes
                )
            
            # Generate code from validated plan
            code_gen_prompt = self._create_robust_code_gen_prompt(
                improved_plan, current_code, issue_type, root_causes
            )
            
            code_result = await query_llm_with_flood_control(
                prompt=code_gen_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=2
            )
            
            if code_result.get("blocked") or not code_result.get("response"):
                return None
                
            refactored_code = self._extract_code_from_response(code_result["response"])
            
            # Validation checkpoint 2: Comprehensive code validation
            validation_result = self._comprehensive_code_validation(
                refactored_code, current_code, issue_type
            )
            
            if validation_result["is_valid"]:
                logger.info(f"{self.object_id}: Successfully refactored {bubble_id} with confidence {validation_result['confidence']}")
                
                # Record successful plan for future use
                await self._record_successful_plan(
                    bubble_id, issue_type, improved_plan, validation_result
                )
                
                return refactored_code
            else:
                logger.warning(f"{self.object_id}: Validation failed: {validation_result['issues']}")
                
                # Attempt targeted fixes for validation issues
                if validation_result["fixable"]:
                    return await self._apply_targeted_fixes(
                        refactored_code, validation_result["issues"]
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"{self.object_id}: Error in enhanced self-planning: {e}")
            return None

    async def _analyze_code_deeply(self, code: str, issue_type: str, 
                                   bubble_id: str, state: Dict) -> Optional[Dict]:
        """Perform deep analysis of code to identify root causes"""
        try:
            from flood_control import query_llm_with_flood_control
            
            analysis_prompt = f"""Analyze this code that has {issue_type} issues:

```python
{code}
```

Recent errors: {state.get("metrics", {}).get("recent_errors", [])}

Provide:
1. Current plan (what the code is trying to do)
2. Root causes of the {issue_type} issues (be specific)
3. Risk areas that might cause future problems
4. Dependencies and assumptions

Format response as:
PLAN: [numbered steps]
ROOT_CAUSES: [numbered list]
RISKS: [bullet points]
DEPENDENCIES: [list]"""

            result = await query_llm_with_flood_control(
                prompt=analysis_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=3
            )
            
            if result.get("response"):
                return self._parse_analysis_response(result["response"])
            
            return None
            
        except Exception as e:
            logger.error(f"Deep analysis error: {e}")
            return None

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse structured analysis response"""
        analysis = {
            "plan": "",
            "root_causes": [],
            "risks": [],
            "dependencies": []
        }
        
        current_section = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            if line.startswith("PLAN:"):
                current_section = "plan"
                analysis["plan"] = line[5:].strip()
            elif line.startswith("ROOT_CAUSES:"):
                current_section = "root_causes"
            elif line.startswith("RISKS:"):
                current_section = "risks"
            elif line.startswith("DEPENDENCIES:"):
                current_section = "dependencies"
            elif line and current_section:
                if current_section == "plan":
                    analysis["plan"] += "\n" + line
                else:
                    analysis[current_section].append(line.lstrip("- •123456789."))
        
        return analysis

    def _create_root_cause_aware_plan_prompt(self, current_plan: str, issue_type: str,
                                            root_causes: List[str], state: Dict) -> str:
        """Create prompt that addresses specific root causes"""
        return f"""Current plan that's causing {issue_type} issues:
{current_plan}

Root causes identified:
{chr(10).join(f"- {cause}" for cause in root_causes)}

System context:
- Error rate: {state.get('metrics', {}).get('code_error_rate', 'unknown')}
- Recent similar errors: {state.get('metrics', {}).get('llm_error_count', 0)}

Generate an improved plan that:
1. Maintains the original functionality
2. Specifically addresses each root cause
3. Adds comprehensive error handling
4. Includes validation steps
5. Optimizes for reliability over performance

Improved Plan (be specific and detailed):"""

    def _validate_plan_coverage(self, plan: str, root_causes: List[str]) -> bool:
        """Check if plan addresses all identified root causes"""
        plan_lower = plan.lower()
        
        coverage_keywords = {
            "import": ["import", "module", "dependency"],
            "error": ["error", "exception", "try", "except", "handle"],
            "validation": ["validate", "check", "verify", "ensure"],
            "performance": ["optimize", "efficient", "cache", "limit"],
            "syntax": ["syntax", "format", "structure", "parse"]
        }
        
        for cause in root_causes:
            cause_lower = cause.lower()
            covered = False
            
            # Check for relevant keywords
            for category, keywords in coverage_keywords.items():
                if any(kw in cause_lower for kw in keywords):
                    if any(kw in plan_lower for kw in keywords):
                        covered = True
                        break
            
            if not covered:
                return False
        
        return True

    async def _enhance_plan_for_coverage(self, plan: str, root_causes: List[str]) -> str:
        """Enhance plan to ensure it covers all root causes"""
        uncovered_causes = []
        plan_lower = plan.lower()
        
        for cause in root_causes:
            if cause.lower() not in plan_lower:
                uncovered_causes.append(cause)
        
        if not uncovered_causes:
            return plan
        
        # Add steps for uncovered causes
        additional_steps = "\n\nAdditional steps to address remaining issues:"
        for i, cause in enumerate(uncovered_causes, 1):
            additional_steps += f"\n{len(plan.split('\n')) + i}. Address {cause}"
        
        return plan + additional_steps

    def _create_robust_code_gen_prompt(self, plan: str, original_code: str, 
                                      issue_type: str, root_causes: List[str]) -> str:
        """Create prompt for robust code generation"""
        return f"""Generate Python code based on this comprehensive plan:

Plan:
{plan}

Root causes to address:
{chr(10).join(f"- {cause}" for cause in root_causes)}

Context: This fixes {issue_type} issues in this code:
```python
{original_code[:500]}...  # truncated
```

Requirements:
1. Follow the plan exactly
2. Include comprehensive error handling
3. Add logging at key points
4. Ensure all imports are included
5. Add type hints where beneficial
6. Include docstrings
7. Make the code maintainable and testable

Generate the complete, working code:

```python"""

    def _comprehensive_code_validation(self, new_code: str, old_code: str, 
                                     issue_type: str) -> Dict:
        """Perform comprehensive validation of generated code"""
        result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "fixable": True
        }
        
        # Basic syntax validation
        if not self._validate_code(new_code):
            result["is_valid"] = False
            result["issues"].append("Syntax validation failed")
            result["confidence"] *= 0.5
        
        # Check for improvement over original
        if new_code.strip() == old_code.strip():
            result["is_valid"] = False
            result["issues"].append("No changes made to code")
            result["confidence"] = 0.0
            result["fixable"] = False
        
        # Specific issue type validation
        if issue_type == "import" and "import" not in new_code:
            result["confidence"] *= 0.7
            result["issues"].append("Missing import statements")
        
        if issue_type == "performance" and len(new_code) > len(old_code) * 2:
            result["confidence"] *= 0.8
            result["issues"].append("Code complexity increased significantly")
        
        # Error handling validation
        if "try:" not in new_code and "except" not in new_code:
            result["confidence"] *= 0.9
            result["issues"].append("No error handling added")
        
        # Calculate final validity
        result["is_valid"] = result["confidence"] > 0.6
        
        return result

    async def _apply_targeted_fixes(self, code: str, issues: List[str]) -> Optional[str]:
        """Apply targeted fixes for specific validation issues"""
        fixed_code = code
        
        for issue in issues:
            if "Missing import" in issue:
                fixed_code = self._add_missing_imports(fixed_code)
            elif "No error handling" in issue:
                fixed_code = await self._add_basic_error_handling(fixed_code)
            elif "Syntax" in issue:
                ast_fixed = self._try_ast_fix(fixed_code, {"error_type": "SyntaxError"})
                if ast_fixed:
                    fixed_code = ast_fixed
        
        if self._validate_code(fixed_code):
            return fixed_code
        
        return None

    def _add_missing_imports(self, code: str) -> str:
        """Add commonly missing imports"""
        lines = code.split('\n')
        
        # Common imports that might be missing
        standard_imports = [
            'import logging',
            'import asyncio',
            'import time',
            'import uuid',
            'from typing import Dict, List, Optional, Any'
        ]
        
        # Find where to insert imports
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(('import', 'from', '#')):
                insert_index = i
                break
        
        # Add imports that aren't already present
        for imp in standard_imports:
            if imp not in code:
                lines.insert(insert_index, imp)
                insert_index += 1
        
        return '\n'.join(lines)

    async def _add_basic_error_handling(self, code: str) -> str:
        """Add basic error handling to functions"""
        try:
            tree = ast.parse(code)
            
            # This would require AST transformation
            # For now, return original code
            return code
        except:
            return code

    async def _record_successful_plan(self, bubble_id: str, issue_type: str,
                                    plan: str, validation_result: Dict):
        """Record successful recovery plan for future use"""
        success_record = {
            "timestamp": time.time(),
            "bubble_id": bubble_id,
            "issue_type": issue_type,
            "plan": plan,
            "confidence": validation_result["confidence"],
            "validation": validation_result
        }
        
        # Update recovery history
        self.recovery_history.append({
            "timestamp": time.time(),
            "monitor": "SelfPlanningFix",
            "state_snapshot": {"issue_type": issue_type},
            "action": {"plan": plan},
            "success": True,
            "metadata": success_record
        })
        
        # Learn pattern for similar issues
        for monitor in self.monitors:
            if hasattr(monitor, 'learn_recovery_plan'):
                await monitor.learn_recovery_plan(issue_type, plan)
        
        logger.info(f"Recorded successful {issue_type} fix plan with confidence {validation_result['confidence']}")

    async def _direct_code_fix(self, code: str, issue_type: str, state: Dict) -> Optional[str]:
        """Direct code fix without self-planning (fallback)"""
        try:
            from flood_control import query_llm_with_flood_control
            
            fix_prompt = f"""Fix the following code that has {issue_type} issues:

```python
{code}
```

Provide the corrected code with:
1. Proper error handling
2. Clear comments
3. All necessary imports
4. Performance optimizations if needed

Fixed code:
```python"""

            result = await query_llm_with_flood_control(
                prompt=fix_prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=3
            )
            
            if result.get("response"):
                return self._extract_code_from_response(result["response"])
            return None
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error in direct code fix: {e}")
            return None

    def _extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response"""
        # Look for code blocks
        import re
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to extract from response
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                in_code = True
            if in_code:
                code_lines.append(line)
                
        return '\n'.join(code_lines) if code_lines else response

    def _validate_code(self, code: str) -> bool:
        """Comprehensive code validation including syntax, safety, and compatibility checks"""
        if not code or not code.strip():
            return False
            
        # Level 1: Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        # Level 2: Safety validation - check for dangerous operations
        dangerous_patterns = [
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'open\s*\([^,)]*["\']w["\']',  # Writing files
            r'os\.system\s*\(',
            r'subprocess\.',
            r'compile\s*\(',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                logger.warning(f"Code validation failed: dangerous pattern '{pattern}' detected")
                return False
        
        # Level 3: Structure validation - ensure required components
        try:
            # Check for basic structure integrity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Ensure functions have at least a pass or return
                    if not node.body:
                        return False
                elif isinstance(node, ast.ClassDef):
                    # Ensure classes have at least a pass
                    if not node.body:
                        return False
        except Exception as e:
            logger.error(f"Code structure validation error: {e}")
            return False
        
        # Level 4: Import validation - check that imports are sensible
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Reject imports of clearly dangerous modules
                    if alias.name in ['os', 'subprocess', 'shutil'] and not self._is_safe_import_context(tree, alias.name):
                        logger.warning(f"Code validation failed: unsafe import of '{alias.name}'")
                        return False
        
        return True

    def _is_safe_import_context(self, tree: ast.AST, module_name: str) -> bool:
        """Check if an import is used safely in the code"""
        safe_usages = {
            'os': ['path.join', 'path.exists', 'environ.get'],
            'subprocess': [],  # Generally unsafe
            'shutil': ['copy', 'copytree'],  # Limited safe operations
        }
        
        # For now, implement basic check - can be extended
        return module_name in ['os'] and any(
            hasattr(node, 'attr') and node.attr in safe_usages.get(module_name, [])
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        )

    def _vectorize_state(self, state: Dict) -> Optional[np.ndarray]:
        """Convert state dict to vector for RL"""
        try:
            return self.env.get_state()
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing state: {e}")
            return None

    async def _generate_explanation(self, action: Dict, state: Dict):
        """Generate explanation for control action"""
        explanation = {
            "timestamp": time.time(),
            "action": action,
            "system_state": {
                "cpu": state.get("cpu_percent", 0),
                "memory": state.get("memory_percent", 0),
                "energy": state.get("energy", 0),
                "health_score": self.metrics_dashboard.metrics["system_health_score"]
            },
            "reasoning": self._get_action_reasoning(action, state)
        }
        
        self.pending_explanations.append(explanation)
        
        # Periodically publish explanations
        if len(self.pending_explanations) >= 5:
            await self._publish_explanations()

    def _get_action_reasoning(self, action: Dict, state: Dict) -> str:
        """Get reasoning for why action was taken"""
        action_type = action.get("action_type")
        payload = action.get("payload", {})
        
        if action_type == "CODE_FIX":
            return f"Fixing code errors: {payload.get('reason', 'Unknown')}"
        elif action_type == "CODE_ROLLBACK":
            return f"Rolling back unstable code: {payload.get('reason', 'Error recovery')}"
        elif action_type == "CODE_INJECT":
            return f"Injecting handlers for: {payload.get('injection_type', 'Unknown')}"
        elif action_type == "CODE_REFACTOR":
            return f"Refactoring for: {payload.get('issue_type', 'Unknown')}"
        elif action_type == "ADJUST_ENERGY":
            return f"Adjusting energy by {payload.get('amount', 0)}"
        elif action_type == "PAUSE_BUBBLE":
            return f"Pausing bubble: {payload.get('reason', 'Resource management')}"
        elif action_type == "NO_OP":
            return "System stable, no action needed"
        else:
            return f"Action: {action_type}"

    async def _publish_explanations(self):
        """Publish accumulated explanations"""
        if not self.pending_explanations:
            return
        
        report = {
            "timestamp": time.time(),
            "explanations": list(self.pending_explanations),
            "metrics": self.metrics_dashboard.metrics,
            "problem_bubbles": self.metrics_dashboard.get_problem_bubbles(threshold=5)
        }
        
        report_event = Event(
            type=Actions.OVERSEER_REPORT,
            data=UniversalCode(Tags.DICT, report),
            origin=self.object_id,
            priority=3
        )
        
        await self.context.dispatch_event(report_event)
        self.pending_explanations.clear()

    async def periodic_reporting(self):
        """Periodic system health reporting"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            try:
                # Analyze warning patterns
                self.warning_analyzer.analyze_pattern_to_error_correlation()
                
                # Generate health report
                health_report = {
                    "timestamp": time.time(),
                    "system_health_score": self.metrics_dashboard.metrics["system_health_score"],
                    "warnings_summary": self.metrics_dashboard.metrics["warnings_by_type"],
                    "fixes_summary": {
                        "applied": self.metrics_dashboard.metrics["fixes_applied"],
                        "successful": self.metrics_dashboard.metrics["fixes_successful"],
                        "success_rate": (self.metrics_dashboard.metrics["fixes_successful"] / 
                                       max(1, self.metrics_dashboard.metrics["fixes_applied"]))
                    },
                    "problem_bubbles": self.metrics_dashboard.get_problem_bubbles(threshold=10),
                    "error_predictions": self._generate_error_predictions()
                }
                
                # Publish report
                health_event = Event(
                    type=Actions.OVERSEER_REPORT,
                    data=UniversalCode(Tags.DICT, health_report),
                    origin=self.object_id,
                    priority=3
                )
                
                await self.context.dispatch_event(health_event)
                
                logger.info(f"{self.object_id}: Published health report - Score: {health_report['system_health_score']:.1f}")
                
            except Exception as e:
                logger.error(f"{self.object_id}: Error in periodic reporting: {e}")

    def _generate_error_predictions(self) -> List[Dict]:
        """Generate predictions about potential errors"""
        predictions = []
        
        # Check each bubble's warning patterns
        for bubble_id, warnings in self.metrics_dashboard.metrics["warnings_by_bubble"].items():
            bubble_warnings = [
                w for w in self.recent_warnings 
                if w.get("source") == bubble_id
            ][-10:]  # Last 10 warnings
            
            if bubble_warnings:
                error_prob = self.warning_analyzer.predict_error_probability(bubble_warnings)
                
                if error_prob > 0.5:
                    predictions.append({
                        "bubble_id": bubble_id,
                        "error_probability": error_prob,
                        "warning_types": list(warnings.keys()),
                        "recommendation": self._get_prevention_recommendation(bubble_id, warnings)
                    })
        
        return sorted(predictions, key=lambda x: x["error_probability"], reverse=True)[:5]

    def _get_prevention_recommendation(self, bubble_id: str, warnings: Dict) -> str:
        """Get recommendation for preventing errors"""
        top_warning = max(warnings.items(), key=lambda x: x[1])[0] if warnings else None
        
        recommendations = {
            "CORRELATION_WARNING": "Add correlation ID handling to all events",
            "PERFORMANCE_WARNING": "Optimize algorithms and add caching",
            "MEMORY_WARNING": "Implement memory cleanup and use generators",
            "CODE_SMELL_WARNING": "Refactor code for better maintainability"
        }
        
        return recommendations.get(top_warning, "Review and refactor problematic code")

    async def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info(f"{self.object_id}: Shutting down - Final health score: {self.metrics_dashboard.metrics['system_health_score']:.1f}")
        
        # Publish final report
        await self._publish_explanations()
        
        # Save learned patterns if needed
        # This could be extended to persist learning across restarts
        
        await super().cleanup()


# --- Neural Network Models ---
class DreamerV3WorldModel(nn.Module):
    """Discrete world model for encoding system states"""
    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int, num_categories: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * num_categories)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.dynamics = nn.GRUCell(latent_dim, latent_dim)
        self.latent_dim = latent_dim
        self.num_categories = num_categories

    def forward(self, state: torch.Tensor, prev_latent: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state.size(0)
        logits = self.encoder(state).view(batch_size, self.latent_dim, self.num_categories)
        z = Categorical(logits=logits).sample()
        if prev_latent is None:
            prev_latent = torch.zeros(batch_size, self.latent_dim, device=state.device)
        next_latent = self.dynamics(z.float(), prev_latent)
        recon = self.decoder(next_latent)
        return z, next_latent, recon


class GRUPolicy(nn.Module):
    """GRU-augmented policy network for action selection"""
    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor, torch.Tensor]:
        hidden = self.gru(latent, hidden)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return Categorical(logits=logits), hidden, value
