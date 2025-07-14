# bubbles_Agent9.py
# Complete implementation with real M4 hardware monitoring
# All simulated CPU/energy replaced with real hardware data

import asyncio
import json
import time
import uuid
import logging
import importlib
import aiohttp
import hashlib
import os
import sys
import fastapi
import psutil
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Union, Tuple, Type
import random
from contextlib import nullcontext
import traceback
import re
from io import StringIO

# Import M4 Hardware Monitoring
try:
    from m4_hardware_bubble import (
        M4HardwareBubble, EnhancedResourceManager, create_m4_hardware_bubble,
        HardwareActions, HealthStatus
    )
    M4_HARDWARE_AVAILABLE = True
    print("✅ M4 Hardware Monitoring: AVAILABLE")
except ImportError as e:
    M4_HARDWARE_AVAILABLE = False
    print(f"⚠️  M4 Hardware Monitoring: NOT AVAILABLE ({e})")

# Add this import
try:
    from consciousness_bubble_integration import ConsciousnessDetector, AtEngineV3, LLMAnalyzer
    CONSCIOUSNESS_AVAILABLE = True
    print("✅ Consciousness Engine: AVAILABLE")
except ImportError as e:
    CONSCIOUSNESS_AVAILABLE = False
    print(f"⚠️  Consciousness Engine: NOT AVAILABLE ({e})")

# --- PyTorch/NN Imports (Optional) ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn: Module = object; Sequential = object; Linear = object; ReLU = object; MSELoss = object; GRUCell = object; Tanh = object
    class torch: Tensor = object; cat = staticmethod(lambda *a, **kw: None); stack = staticmethod(lambda *a, **kw: None); tensor = staticmethod(lambda *a, **kw: None); float32 = None; no_grad = staticmethod(lambda: nullcontext()); zeros = staticmethod(lambda *a, **kw: None); zeros_like = staticmethod(lambda *a, **kw: None); unsqueeze = staticmethod(lambda *a, **kw: None); squeeze = staticmethod(lambda *a, **kw: None); detach = staticmethod(lambda *a, **kw: None); cpu = staticmethod(lambda *a, **kw: a[0]); numpy = staticmethod(lambda *a, **kw: None)
    class optim: Adam = object
    np = None
    print("WARNING: PyTorch not found. DreamerV3Bubble will be disabled.", file=sys.stderr)

# --- SciPy Import (Optional) ---
try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    def ttest_ind(*args, **kwargs):
        return (0.0, 1.0)
    print("WARNING: SciPy not found. FeedbackBubble statistical significance test disabled.", file=sys.stderr)

# --- Pylint Import (Optional) ---
try:
    import pylint.lint
    import pylint.reporters
    from pylint.exceptions import InvalidMessageError
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False
    print("WARNING: Pylint not found. DynamicBubbleManager code validation disabled.", file=sys.stderr)

# --- Core Imports ---
try:
    from bubbles_core import (
        OLLAMA_HOST_URL, MODEL_NAME, FALLBACK_MODEL, API_ENDPOINT, REQUEST_TIMEOUT, RETRY_ATTEMPTS, RETRY_DELAY,
        Tags, Actions,
        InvalidTagError, LLMCallError, CodeExecutionError, CodeValidationError, PredictionError,
        extract_code, robust_json_parse, execute_python_code,
        UniversalCode, Event, EventService,
        SystemContext, ResourceManager, EventDispatcher, ChatBox,
        UniversalBubble
    )
    
    # Add STRATEGIC_ANALYSIS to Actions if not present
    if not hasattr(Actions, 'STRATEGIC_ANALYSIS'):
        Actions.STRATEGIC_ANALYSIS = "STRATEGIC_ANALYSIS"
        
except ImportError as e:
    msg = f"CRITICAL ERROR: Failed to import from bubbles_core.py. Ensure it's in the Python path. Error: {e}"
    print(msg, file=sys.stderr)
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)

# Define timeout for MetaReasoningBubble
ACTION_GENERATION_TIMEOUT = 2000.0
MAX_ACTION_GEN_RETRIES = 3

# ============================================================================
# Enhanced ResourceManager with Real M4 Hardware
# ============================================================================

class RealHardwareResourceManager(ResourceManager):
    """ResourceManager that uses only real M4 hardware data"""
    
    def __init__(self, context: SystemContext):
        super().__init__(context)
        self.m4_bubble = None
        self._using_real_hardware = False
        logger.info("RealHardwareResourceManager initialized - waiting for M4 hardware bubble")
    
    def set_m4_bubble(self, m4_bubble: 'M4HardwareBubble'):
        """Set the M4 hardware bubble for real metrics"""
        self.m4_bubble = m4_bubble
        self._using_real_hardware = True
        logger.info("✅ ResourceManager now using REAL M4 hardware metrics")
    
    def get_current_system_state(self) -> Dict[str, Any]:
        """Get system state - ONLY real hardware data"""
        if self.m4_bubble and self.m4_bubble.metrics_history:
            # Return the latest real hardware metrics
            latest_metrics = self.m4_bubble.metrics_history[-1]
            
            # Calculate energy based on power consumption
            power_watts = latest_metrics.get('hardware', {}).get('power', {}).get('estimated_total_watts', 0)
            energy = 10000 - (power_watts * 100)  # Higher power = lower energy
            
            # Ensure backward compatibility with expected fields
            state = {
                'timestamp': latest_metrics.get('timestamp', time.time()),
                'energy': energy,
                'cpu_percent': latest_metrics.get('cpu_percent', 0),
                'memory_percent': latest_metrics.get('memory_percent', 0),
                'num_bubbles': len(self.context.get_all_bubbles()),
                'metrics': self.metrics.copy(),
                'event_frequencies': self._calculate_event_frequencies(),
                'source': 'M4HardwareBubble',
                
                # Include real hardware details
                'hardware': latest_metrics.get('hardware', {}),
                'performance_profile': latest_metrics.get('performance_profile', 'balanced'),
                'hardware_health': latest_metrics.get('hardware_health', {}),
                
                # Add default values for DreamerV3 compatibility
                'gravity_force': 0.0,
                'gravity_direction': 0.0,
                'bubble_pos_x': 0.0,
                'bubble_pos_y': 0.0,
                'cluster_id': 0,
                'cluster_strength': 0.0,
                'response_time_perturbation': 0.0,
                
                # Add any custom fields from other metrics
                **{k: v for k, v in latest_metrics.items() 
                   if k not in ['timestamp', 'energy', 'cpu_percent', 'memory_percent', 
                               'hardware', 'performance_profile', 'hardware_health']}
            }
            
            return state
        else:
            # Fallback to basic psutil if M4 not available yet
            logger.warning("M4 hardware not available, using psutil fallback")
            return {
                'timestamp': time.time(),
                'energy': self.resources.get('energy', 10000),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'num_bubbles': len(self.context.get_all_bubbles()),
                'metrics': self.metrics.copy(),
                'event_frequencies': self._calculate_event_frequencies(),
                'source': 'psutil_fallback',
                'hardware': {},
                'warning': 'Real M4 hardware metrics not available',
                
                # Add default values for DreamerV3 compatibility
                'gravity_force': 0.0,
                'gravity_direction': 0.0,
                'bubble_pos_x': 0.0,
                'bubble_pos_y': 0.0,
                'cluster_id': 0,
                'cluster_strength': 0.0,
                'response_time_perturbation': 0.0
            }
    
    def get_resource_level(self, resource_type: str) -> Union[int, float]:
        """Get resource level from real hardware"""
        if self.m4_bubble and self.m4_bubble.metrics_history:
            latest = self.m4_bubble.metrics_history[-1]
            
            if resource_type == 'cpu_percent':
                return latest.get('cpu_percent', 0)
            elif resource_type == 'memory_percent':
                return latest.get('memory_percent', 0)
            elif resource_type == 'energy':
                return latest.get('energy', self.resources.get('energy', 10000))
            elif resource_type == 'gpu_percent':
                return latest.get('hardware', {}).get('gpu', {}).get('usage_percent', 0)
            elif resource_type == 'neural_engine_percent':
                return latest.get('hardware', {}).get('neural_engine', {}).get('usage_percent', 0)
            elif resource_type == 'power_watts':
                return latest.get('hardware', {}).get('power', {}).get('estimated_total_watts', 0)
            elif resource_type == 'performance_cores_percent':
                return latest.get('hardware', {}).get('cpu', {}).get('performance_cores_percent', 0)
            elif resource_type == 'efficiency_cores_percent':
                return latest.get('hardware', {}).get('cpu', {}).get('efficiency_cores_percent', 0)
        
        # Fallback to stored resources
        return super().get_resource_level(resource_type)
    
    async def trigger_state_update(self):
        """Trigger state update with real hardware data"""
        if not self._using_real_hardware:
            logger.warning("Triggering state update without real hardware data")
        
        state = self.get_current_system_state()
        state_uc = UniversalCode(
            Tags.DICT, 
            state, 
            description="System state update (real hardware)",
            metadata={"timestamp": time.time(), "real_hardware": self._using_real_hardware}
        )
        
        state_event = Event(
            type=Actions.SYSTEM_STATE_UPDATE,
            data=state_uc,
            origin="ResourceManager",
            priority=10
        )
        
        await self.context.dispatch_event(state_event)
        logger.debug(f"Triggered SYSTEM_STATE_UPDATE with {'REAL' if self._using_real_hardware else 'FALLBACK'} data")

# ============================================================================
# SimpleLLMBubble (unchanged from original)
# ============================================================================

# SimpleLLMBubble - Complete Implementation
# Multi-provider LLM service bubble with Gemini, Grok, and Ollama support
import asyncio
import aiohttp
import json
import time
import logging
import hashlib
import os
import uuid
import ssl
from typing import Optional, Dict, Any, List
from collections import defaultdict, deque
from enum import Enum
import random

from bubbles_core import (
    UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions, 
    logger, EventService, LLMCallError
)

# API Configuration
OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST", "http://10.0.0.XXXXX")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AXXXXXX")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "xXXXXX")

# Model names - using Gemini Pro Preview as requested
GEMINI_MODEL = "gemini-2.5-pro-preview-06-05"
GROK_MODEL = "grok-4-latest"
OLLAMA_MODEL = "gemma:7b"

class LLMProvider(Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"
    GROK = "grok"

class ProviderConfig:
    def __init__(self, provider: LLMProvider, endpoint: str, headers: Dict[str, str], 
                 model: str, max_tokens: int = 2048, rate_limit: float = 1.0):
        self.provider = provider
        self.endpoint = endpoint
        self.headers = headers
        self.model = model
        self.max_tokens = max_tokens
        self.rate_limit = rate_limit
        self.last_call_time = 0
        self.failure_count = 0
        self.max_failures = 3

class CostController:
    """Manages API cost tracking and limits."""
    
    def __init__(self, hourly_limit: float = 10.0, daily_limit: float = 100.0):
        self.hourly_limit = hourly_limit
        self.daily_limit = daily_limit
        self.hourly_costs = deque()
        self.daily_costs = deque()
        self.cost_per_token = {
            LLMProvider.GEMINI: 0.000001,  # $0.000001 per token
            LLMProvider.GROK: 0.000002,    # $0.000002 per token
            LLMProvider.OLLAMA: 0.0        # Free for local
        }
        self._lock = asyncio.Lock()
    
    async def check_cost_limits(self, provider: LLMProvider, estimated_tokens: int) -> bool:
        """Check if request would exceed cost limits."""
        async with self._lock:
            estimated_cost = self.cost_per_token.get(provider, 0) * estimated_tokens
            current_time = time.time()
            
            self._cleanup_old_entries(current_time)
            
            hourly_total = sum(cost for timestamp, cost in self.hourly_costs 
                             if current_time - timestamp < 3600)
            daily_total = sum(cost for timestamp, cost in self.daily_costs 
                            if current_time - timestamp < 86400)
            
            if hourly_total + estimated_cost > self.hourly_limit:
                raise LLMCallError(
                    f"Hourly cost limit would be exceeded: ${hourly_total:.4f} + ${estimated_cost:.4f} > ${self.hourly_limit}"
                )
            
            if daily_total + estimated_cost > self.daily_limit:
                raise LLMCallError(
                    f"Daily cost limit would be exceeded: ${daily_total:.4f} + ${estimated_cost:.4f} > ${self.daily_limit}"
                )
            
            return True
    
    async def record_cost(self, provider: LLMProvider, actual_tokens: int):
        """Record actual cost after successful API call."""
        async with self._lock:
            cost = self.cost_per_token.get(provider, 0) * actual_tokens
            current_time = time.time()
            self.hourly_costs.append((current_time, cost))
            self.daily_costs.append((current_time, cost))
    
    def _cleanup_old_entries(self, current_time: float):
        """Remove entries older than 24 hours."""
        cutoff_time = current_time - 86400
        
        while self.hourly_costs and self.hourly_costs[0][0] < cutoff_time:
            self.hourly_costs.popleft()
        
        while self.daily_costs and self.daily_costs[0][0] < cutoff_time:
            self.daily_costs.popleft()
    
    async def get_current_usage(self) -> Dict[str, float]:
        """Get current usage statistics."""
        async with self._lock:
            current_time = time.time()
            self._cleanup_old_entries(current_time)
            
            hourly_total = sum(cost for timestamp, cost in self.hourly_costs 
                             if current_time - timestamp < 3600)
            daily_total = sum(cost for timestamp, cost in self.daily_costs 
                            if current_time - timestamp < 86400)
            
            return {
                "hourly_spent": hourly_total,
                "hourly_limit": self.hourly_limit,
                "daily_spent": daily_total,
                "daily_limit": self.daily_limit
            }

class CircuitBreaker:
    """Implements circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    logger.info(f"Circuit breaker entering half-open state")
                else:
                    raise LLMCallError(f"Circuit breaker is open (will retry in {self.recovery_timeout - (time.time() - self.last_failure_time):.0f}s)")
        
        try:
            result = await func(*args, **kwargs)
            
            async with self._lock:
                if self.state == "half-open":
                    self.state = "closed"
                    logger.info(f"Circuit breaker closed after successful call")
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

class GlobalThrottler:
    """Global request throttling to prevent system overload."""
    
    def __init__(self, max_concurrent: int = 5, max_per_minute: int = 30):
        self.max_concurrent = max_concurrent
        self.max_per_minute = max_per_minute
        self.concurrent_requests = 0
        self.request_times = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            cutoff = current_time - 60
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()
            
            if self.concurrent_requests >= self.max_concurrent:
                raise LLMCallError(f"Maximum concurrent requests reached ({self.max_concurrent})")
            
            if len(self.request_times) >= self.max_per_minute:
                wait_time = 60 - (current_time - self.request_times[0])
                raise LLMCallError(f"Rate limit exceeded (max {self.max_per_minute}/min, retry in {wait_time:.0f}s)")
            
            self.concurrent_requests += 1
            self.request_times.append(current_time)
    
    async def release(self):
        """Release request slot."""
        async with self._lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)

class SimpleLLMBubble(UniversalBubble):
    """Multi-provider LLM service with equal-tier provider management."""
    
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Create SSL context that doesn't verify certificates
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        logger.warning(f"{self.object_id}: SSL certificate verification disabled - USE ONLY FOR DEVELOPMENT/TESTING")
        
        # Initialize safety systems
        self.cost_controller = CostController(
            hourly_limit=kwargs.get("hourly_cost_limit", 10.0),
            daily_limit=kwargs.get("daily_cost_limit", 100.0)
        )
        self.circuit_breakers = {}
        self.global_throttler = GlobalThrottler(
            max_concurrent=kwargs.get("max_concurrent_requests", 5),
            max_per_minute=kwargs.get("max_requests_per_minute", 30)
        )
        
        # Initialize providers
        self.providers = self._initialize_providers()
        
        # Initialize circuit breakers for each provider
        for provider in self.providers:
            self.circuit_breakers[provider] = CircuitBreaker()
        
        # Queue and optimization initialization
        self.response_cache = context.response_cache
        self.priority_queue = asyncio.PriorityQueue(maxsize=100)
        self.batch_accumulator = []
        self.last_batch_time = time.time()
        self.provider_metrics = defaultdict(lambda: {"success": 0, "total": 0, "avg_duration": 0})
        
        # Load balancing configuration
        self.load_balance_strategy = kwargs.get("load_balance_strategy", "weighted_round_robin")
        self.last_provider_index = 0
        
        # Bubble-specific optimizations
        self.bubble_priorities = {
            "creativesynthesis_bubble": 1,
            "metareasoning_bubble": 2,
            "dreamerv3_bubble": 3,
            "feedback_bubble": 4,
            "dynamicmanager_bubble": 5,
            "user_chat": 1,
        }
        
        # Bubble-specific energy costs
        self.bubble_energy_costs = {
            "creativesynthesis_bubble": 0.5,
            "metareasoning_bubble": 2.0,
            "dreamerv3_bubble": 1.5,
            "feedback_bubble": 0.8,
            "default": 1.0
        }
        
        # Statistics tracking
        self.query_success_count = 0
        self.query_total_count = 0
        self.query_durations = deque(maxlen=100)
        
        asyncio.create_task(self._subscribe_to_events())
        asyncio.create_task(self._process_priority_queue())
        asyncio.create_task(self._batch_processor())
        asyncio.create_task(self.log_initial_status())
        logger.info(f"{self.object_id}: Initialized with {len(self.providers)} equal-tier LLM providers")
    
    def _initialize_providers(self) -> Dict[LLMProvider, ProviderConfig]:
        """Initialize configurations for all LLM providers with validation."""
        providers = {}
        
        # Ollama configuration with validation
        if OLLAMA_HOST_URL:
            # Validate URL format
            if not OLLAMA_HOST_URL.startswith(('http://', 'https://')):
                logger.warning(f"{self.object_id}: Invalid OLLAMA_HOST_URL format: {OLLAMA_HOST_URL}")
            else:
                # Test connection before adding
                try:
                    import requests
                    # Disable SSL verification for requests as well
                    response = requests.get(f"{OLLAMA_HOST_URL}/api/tags", timeout=2, verify=False)
                    if response.status_code == 200:
                        providers[LLMProvider.OLLAMA] = ProviderConfig(
                            provider=LLMProvider.OLLAMA,
                            endpoint=f"{OLLAMA_HOST_URL}/api/generate",
                            headers={"Content-Type": "application/json"},
                            model=OLLAMA_MODEL,
                            max_tokens=4096,
                            rate_limit=0.1
                        )
                        logger.info(f"{self.object_id}: Ollama provider initialized at {OLLAMA_HOST_URL}")
                    else:
                        logger.warning(f"{self.object_id}: Ollama API returned status {response.status_code}")
                except Exception as e:
                    logger.warning(f"{self.object_id}: Could not connect to Ollama at {OLLAMA_HOST_URL}: {e}")
        
        # Gemini configuration
        if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
            providers[LLMProvider.GEMINI] = ProviderConfig(
                provider=LLMProvider.GEMINI,
                endpoint=f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers={"Content-Type": "application/json"},
                model=GEMINI_MODEL,
                max_tokens=32768,  # Pro Preview model supports higher token count
                rate_limit=2.0
            )
            logger.info(f"{self.object_id}: Gemini provider initialized with model {GEMINI_MODEL}")
        
        # Grok configuration
        if GROK_API_KEY and GROK_API_KEY != "YOUR_API_KEY_HERE":
            providers[LLMProvider.GROK] = ProviderConfig(
                provider=LLMProvider.GROK,
                endpoint="https://api.x.ai/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROK_API_KEY}"
                },
                model=GROK_MODEL,
                max_tokens=8192,
                rate_limit=2.0
            )
            logger.info(f"{self.object_id}: Grok provider initialized with model {GROK_MODEL}")
        
        if not providers:
            logger.error(f"{self.object_id}: No LLM providers could be initialized!")
        
        return providers
    
    async def _create_timeout_config(self, provider: LLMProvider) -> aiohttp.ClientTimeout:
        """Create provider-specific timeout configuration."""
        if provider == LLMProvider.OLLAMA:
            return aiohttp.ClientTimeout(total=30, connect=5, sock_read=25)
        else:  # Cloud providers need longer timeouts
            return aiohttp.ClientTimeout(total=60, connect=10, sock_read=50)
    
    def _create_client_session(self, timeout: aiohttp.ClientTimeout) -> aiohttp.ClientSession:
        """Create aiohttp client session with SSL verification disabled."""
        connector = aiohttp.TCPConnector(ssl=self.ssl_context)
        return aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        words = len(text.split())
        chars = len(text)
        word_estimate = words * 1.3
        char_estimate = chars / 4
        return int((word_estimate + char_estimate) / 2)
    
    async def _validate_token_limits(self, provider: LLMProvider, prompt: str, 
                                   requested_tokens: int) -> int:
        """Validate and adjust token limits."""
        config = self.providers[provider]
        prompt_tokens = self._estimate_tokens(prompt)
        prompt_with_margin = int(prompt_tokens * 1.2)
        available_tokens = config.max_tokens - prompt_with_margin
        
        if available_tokens < 100:
            raise LLMCallError(
                f"Prompt too long for {provider.value}: ~{prompt_tokens} tokens "
                f"(limit: {config.max_tokens} with response)"
            )
        
        safe_limit = min(requested_tokens, available_tokens)
        
        logger.debug(
            f"{self.object_id}: Token validation for {provider.value} - "
            f"Prompt: ~{prompt_tokens}, Available: {available_tokens}, Using: {safe_limit}"
        )
        
        return safe_limit
    
    async def check_provider_health(self) -> Dict[str, Any]:
        """Check health status of all configured providers."""
        health_status = {}
        
        for provider, config in self.providers.items():
            try:
                if provider == LLMProvider.OLLAMA:
                    timeout = aiohttp.ClientTimeout(total=5)
                    async with self._create_client_session(timeout) as session:
                        async with session.get(f"{OLLAMA_HOST_URL}/api/tags") as response:
                            if response.status == 200:
                                data = await response.json()
                                models = [m['name'] for m in data.get('models', [])]
                                health_status[provider.value] = {
                                    "status": "healthy",
                                    "available_models": models,
                                    "configured_model": config.model,
                                    "model_available": config.model in models
                                }
                            else:
                                health_status[provider.value] = {
                                    "status": "unhealthy",
                                    "error": f"HTTP {response.status}"
                                }
                
                elif provider == LLMProvider.GEMINI:
                    # Quick validation without making actual API call
                    health_status[provider.value] = {
                        "status": "configured",
                        "model": config.model,
                        "rate_limit": f"{60/config.rate_limit:.0f} requests/min"
                    }
                
                elif provider == LLMProvider.GROK:
                    # Quick validation without making actual API call
                    health_status[provider.value] = {
                        "status": "configured",
                        "model": config.model,
                        "rate_limit": f"{60/config.rate_limit:.0f} requests/min"
                    }
                    
            except Exception as e:
                health_status[provider.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    async def log_initial_status(self):
        """Log initial provider status after initialization."""
        health_status = await self.check_provider_health()
        
        logger.info(f"{self.object_id}: Provider Health Check Results:")
        for provider, status in health_status.items():
            if status.get("status") == "healthy":
                logger.info(f"  ✅ {provider}: Healthy - Model: {status.get('configured_model', 'N/A')} (Available: {status.get('model_available', False)})")
            elif status.get("status") == "configured":
                logger.info(f"  ✅ {provider}: Configured - Model: {status.get('model', 'N/A')} ({status.get('rate_limit', 'N/A')})")
            elif status.get("status") == "unhealthy":
                logger.warning(f"  ❌ {provider}: Unhealthy - {status.get('error', 'Unknown error')}")
            else:
                logger.warning(f"  ⚠️  {provider}: {status.get('status', 'Unknown')} - {status.get('error', 'Not available')}")
        
        # Update metrics
        healthy_count = sum(1 for s in health_status.values() if s.get("status") in ["healthy", "configured"])
        if self.context.resource_manager:
            self.context.resource_manager.metrics["llm_providers_available"] = healthy_count
    
    async def _subscribe_to_events(self):
        """Subscribe to LLM query events."""
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.LLM_QUERY, self.handle_event)
            await EventService.subscribe(Actions.MULTI_LLM_QUERY, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to LLM_QUERY and MULTI_LLM_QUERY")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)
    
    async def handle_event(self, event: Event):
        """Handle incoming LLM query events with priority routing."""
        if event.type in [Actions.LLM_QUERY, Actions.MULTI_LLM_QUERY]:
            if event.type == Actions.LLM_QUERY and isinstance(event.data, UniversalCode) and event.data.tag == Tags.STRING:
                correlation_id = event.data.metadata.get("correlation_id", "unknown")
                if not correlation_id or correlation_id == "unknown":
                    logger.warning(f"{self.object_id}: LLM_QUERY missing correlation_id, skipping")
                    return
                
                origin = event.origin or "unknown"
                priority = self.bubble_priorities.get(origin, 5)
                
                if self._is_batchable(event):
                    self.batch_accumulator.append(event)
                    logger.debug(f"{self.object_id}: Added to batch queue (cid: {correlation_id[:8]})")
                else:
                    try:
                        await self.priority_queue.put((priority, time.time(), event))
                        logger.debug(f"{self.object_id}: Queued LLM_QUERY with priority {priority} (cid: {correlation_id[:8]})")
                    except asyncio.QueueFull:
                        logger.warning(f"{self.object_id}: Priority queue full, rejecting query")
                        await self._send_error_response(event, "Query queue full")
            
            elif event.type == Actions.MULTI_LLM_QUERY:
                priority = event.priority if hasattr(event, 'priority') else 3
                try:
                    await self.priority_queue.put((priority, time.time(), event))
                except asyncio.QueueFull:
                    logger.warning(f"{self.object_id}: Query queue full, rejecting multi-query")
                    await self._send_error_response(event, "Query queue full")
        else:
            await super().handle_event(event)
    
    def _is_batchable(self, event: Event) -> bool:
        """Check if a query can be batched."""
        metadata = event.data.metadata or {}
        origin = event.origin
        
        if origin == "creativesynthesis_bubble" and "proposal" in event.data.value.lower():
            return True
        
        if origin in ["user_chat", "metareasoning_bubble"]:
            return False
        
        return False
    
    async def _batch_processor(self):
        """Process accumulated batch queries periodically."""
        BATCH_WAIT_TIME = 0.5
        while not self.context.stop_event.is_set():
            try:
                await asyncio.sleep(BATCH_WAIT_TIME)
                
                if self.batch_accumulator and (time.time() - self.last_batch_time > BATCH_WAIT_TIME):
                    batch = self.batch_accumulator[:5]
                    self.batch_accumulator = self.batch_accumulator[5:]
                    
                    if batch:
                        await self._process_batch(batch)
                        self.last_batch_time = time.time()
                        
            except Exception as e:
                logger.error(f"{self.object_id}: Error in batch processor: {e}", exc_info=True)
    
    async def _process_batch(self, batch: List[Event]):
        """Process a batch of similar queries efficiently."""
        if not batch:
            return
        
        combined_prompt = "Answer the following queries concisely, separating responses with '---RESPONSE---':\n\n"
        correlation_ids = []
        
        for event in batch:
            prompt = self._truncate_prompt(event.data.value)
            combined_prompt += f"Query: {prompt}\n\n"
            correlation_ids.append(event.data.metadata.get("correlation_id", "unknown"))
        
        provider = self._select_provider(combined_prompt, {"batch": True})
        
        try:
            response_text = await self._call_llm_provider(provider, combined_prompt, {"batch": True})
            
            responses = response_text.split("---RESPONSE---")
            for i, event in enumerate(batch):
                individual_response = responses[i].strip() if i < len(responses) else "Batch processing error"
                await self._send_llm_response(
                    individual_response, 
                    event.origin, 
                    event.data.metadata or {}, 
                    provider=provider.value
                )
                
            logger.info(f"{self.object_id}: Processed batch of {len(batch)} queries with {provider.value}")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Batch processing error: {e}", exc_info=True)
            for event in batch:
                await self._send_error_response(event, f"Batch error: {str(e)}")
    
    def _truncate_prompt(self, prompt: str, max_length: int = 4096) -> str:
        """Truncate prompt to fit within token limits."""
        if len(prompt) > max_length:
            return prompt[:max_length-100] + "\n[TRUNCATED]"
        return prompt
    
    async def _process_priority_queue(self):
        """Process priority-ordered LLM queries."""
        while not self.context.stop_event.is_set():
            try:
                priority, timestamp, event = await asyncio.wait_for(
                    self.priority_queue.get(), 
                    timeout=1.0
                )
                
                if event.type == Actions.MULTI_LLM_QUERY:
                    await self._handle_multi_llm_query(event)
                else:
                    await self._handle_single_llm_query(event)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"{self.object_id}: Error processing priority queue: {e}", exc_info=True)
    
    def _get_energy_cost(self, origin: str) -> float:
        """Get energy cost for a query based on origin bubble."""
        return self.bubble_energy_costs.get(origin, self.bubble_energy_costs["default"])
    
    def _select_provider(self, prompt: str, metadata: Dict[str, Any]) -> LLMProvider:
        """Select optimal provider using load balancing strategy."""
        # Check if specific provider requested
        requested_provider = metadata.get("provider")
        if requested_provider:
            try:
                provider = LLMProvider(requested_provider)
                if provider in self.providers and self._is_provider_available(provider):
                    return provider
            except ValueError:
                pass
        
        # Get available providers
        available_providers = [p for p in self.providers.keys() if self._is_provider_available(p)]
        
        if not available_providers:
            # If no providers available, return the least failed one
            return min(self.providers.keys(), 
                      key=lambda p: self.providers[p].failure_count)
        
        # Apply load balancing strategy
        if self.load_balance_strategy == "round_robin":
            return self._round_robin_select(available_providers)
        elif self.load_balance_strategy == "weighted_round_robin":
            return self._weighted_round_robin_select(available_providers)
        elif self.load_balance_strategy == "least_connections":
            return self._least_connections_select(available_providers)
        elif self.load_balance_strategy == "performance_based":
            return self._performance_based_select(available_providers)
        else:
            # Default to random selection
            return random.choice(available_providers)
    
    def _is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available for use."""
        config = self.providers.get(provider)
        if not config:
            return False
        
        # Check if provider has not exceeded failure threshold
        if config.failure_count >= config.max_failures:
            return False
        
        # Check circuit breaker state
        circuit_breaker = self.circuit_breakers.get(provider)
        if circuit_breaker and circuit_breaker.state == "open":
            # Check if it's time to try half-open
            if time.time() - circuit_breaker.last_failure_time > circuit_breaker.recovery_timeout:
                return True
            return False
        
        return True
    
    def _round_robin_select(self, providers: List[LLMProvider]) -> LLMProvider:
        """Simple round-robin selection."""
        self.last_provider_index = (self.last_provider_index + 1) % len(providers)
        return providers[self.last_provider_index]
    
    def _weighted_round_robin_select(self, providers: List[LLMProvider]) -> LLMProvider:
        """Weighted round-robin based on success rates."""
        weights = []
        for provider in providers:
            metrics = self.provider_metrics[provider]
            if metrics["total"] > 0:
                success_rate = metrics["success"] / metrics["total"]
                weights.append(max(0.1, success_rate))  # Minimum weight of 0.1
            else:
                weights.append(0.5)  # Default weight for new providers
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(providers)] * len(providers)
        
        # Select based on weights
        return random.choices(providers, weights=weights)[0]
    
    def _least_connections_select(self, providers: List[LLMProvider]) -> LLMProvider:
        """Select provider with least active connections."""
        # This is a simplified version - in production, you'd track actual connections
        return min(providers, key=lambda p: self.provider_metrics[p]["total"] % 10)
    
    def _performance_based_select(self, providers: List[LLMProvider]) -> LLMProvider:
        """Select based on response time performance."""
        best_provider = providers[0]
        best_score = float('inf')
        
        for provider in providers:
            metrics = self.provider_metrics[provider]
            if metrics["total"] > 0:
                # Score based on average duration and success rate
                success_rate = metrics["success"] / metrics["total"]
                score = metrics["avg_duration"] / max(0.1, success_rate)
            else:
                score = 1.0  # Default score for new providers
            
            if score < best_score:
                best_score = score
                best_provider = provider
        
        return best_provider
    
    async def _handle_single_llm_query(self, event: Event):
        """Handle a single LLM query without fallback mechanism."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.STRING:
            return
        
        prompt = event.data.value
        metadata = event.data.metadata or {}
        correlation_id = metadata.get("correlation_id", str(uuid.uuid4()))
        origin = event.origin or "unknown"
        
        # Update statistics
        self.query_total_count += 1
        
        # Get energy cost
        energy_cost = self._get_energy_cost(origin)
        if self.resource_manager and not await self.resource_manager.consume_resource('energy', energy_cost):
            logger.warning(f"{self.object_id}: Insufficient energy for query from {origin} (cost: {energy_cost})")
            await self._send_error_response(event, "Insufficient energy")
            return
        
        # Check cache
        cache_key = self._generate_cache_key(prompt, origin)
        cached_response = await self.response_cache.get(cache_key)
        if cached_response:
            self.query_success_count += 1
            await self._send_llm_response(cached_response, origin, metadata, 
                                        cached=True, provider="cache")
            return
        
        # Select provider
        provider = self._select_provider(prompt, metadata)
        
        # Execute query
        start_time = time.time()
        try:
            response = await self._call_llm_provider(provider, prompt, metadata)
            duration = time.time() - start_time
            
            # Update metrics
            self._update_provider_metrics(provider, True, duration)
            self.query_success_count += 1
            self.query_durations.append(duration)
            
            # Cache response
            await self.response_cache.put(cache_key, response)
            
            # Send response
            await self._send_llm_response(response, origin, metadata, 
                                        provider=provider.value, duration=duration)
            
            logger.info(f"{self.object_id}: Completed query from {origin} (cid: {correlation_id[:8]}, "
                       f"provider: {provider.value}, duration: {duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._update_provider_metrics(provider, False, duration)
            self.query_durations.append(duration)
            
            # No fallback - just report the error
            logger.error(f"{self.object_id}: Provider {provider.value} failed: {e}")
            await self._send_error_response(event, f"Provider {provider.value} error: {str(e)}")
    
    async def _handle_multi_llm_query(self, event: Event):
        """Handle a query that should be sent to multiple LLM providers."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.STRING:
            return
        
        prompt = event.data.value
        metadata = event.data.metadata or {}
        correlation_id = metadata.get("correlation_id", str(uuid.uuid4()))
        
        # Get list of providers to query
        target_providers = metadata.get("providers", list(self.providers.keys()))
        
        # Query all providers in parallel
        tasks = []
        for provider in target_providers:
            if provider in self.providers:
                task = asyncio.create_task(
                    self._call_llm_provider_safe(provider, prompt, metadata)
                )
                tasks.append((provider, task))
        
        # Wait for all responses
        responses = {}
        for provider, task in tasks:
            try:
                response = await task
                if response:
                    responses[provider.value] = response
            except Exception as e:
                logger.error(f"{self.object_id}: Provider {provider.value} failed: {e}")
                responses[provider.value] = f"Error: {str(e)}"
        
        # Send multi-response
        response_data = {
            "responses": responses,
            "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "timestamp": time.time()
        }
        
        response_uc = UniversalCode(
            Tags.DICT,
            response_data,
            description="Multi-LLM responses",
            metadata={**metadata, "response_count": len(responses)}
        )
        
        response_event = Event(
            type=Actions.MULTI_LLM_RESPONSE,
            data=response_uc,
            origin=self.object_id,
            priority=3
        )
        
        await self.context.dispatch_event(response_event)
    
    async def _call_llm_provider(self, provider: LLMProvider, prompt: str, 
                                metadata: Dict[str, Any]) -> str:
        """Call LLM provider with all safety measures."""
        config = self.providers.get(provider)
        if not config:
            raise LLMCallError(f"Provider {provider.value} not configured")
        
        # Global throttling
        await self.global_throttler.acquire()
        
        try:
            # Token validation
            requested_tokens = metadata.get("max_tokens", config.max_tokens)
            safe_token_limit = await self._validate_token_limits(
                provider, prompt, requested_tokens
            )
            
            # Cost checking
            estimated_total_tokens = self._estimate_tokens(prompt) + safe_token_limit
            await self.cost_controller.check_cost_limits(provider, estimated_total_tokens)
            
            # Circuit breaker protection
            circuit_breaker = self.circuit_breakers[provider]
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - config.last_call_time
            if time_since_last < config.rate_limit:
                await asyncio.sleep(config.rate_limit - time_since_last)
            config.last_call_time = time.time()
            
            # Execute with circuit breaker
            result = await circuit_breaker.call(
                self._execute_provider_call, 
                provider, prompt, metadata, safe_token_limit
            )
            
            # Record actual cost
            await self.cost_controller.record_cost(provider, estimated_total_tokens)
            
            return result
            
        finally:
            await self.global_throttler.release()
    
    async def _execute_provider_call(self, provider: LLMProvider, prompt: str,
                                   metadata: Dict[str, Any], token_limit: int) -> str:
        """Execute the actual provider API call."""
        if provider == LLMProvider.OLLAMA:
            return await self._call_ollama(self.providers[provider], prompt, metadata, token_limit)
        elif provider == LLMProvider.GEMINI:
            return await self._call_gemini(self.providers[provider], prompt, metadata, token_limit)
        elif provider == LLMProvider.GROK:
            return await self._call_grok(self.providers[provider], prompt, metadata, token_limit)
        else:
            raise LLMCallError(f"Unknown provider: {provider}")
    
    async def _call_ollama(self, config: ProviderConfig, prompt: str, 
                          metadata: Dict[str, Any], token_limit: int) -> str:
        """Call Ollama API with timeout protection and SSL verification disabled."""
        timeout = await self._create_timeout_config(LLMProvider.OLLAMA)
        
        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": token_limit,
                "temperature": metadata.get("temperature", 0.7)
            }
        }
        
        async with self._create_client_session(timeout) as session:
            try:
                async with session.post(config.endpoint, json=payload, 
                                      headers=config.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data.get("response", "")
            except asyncio.TimeoutError:
                raise LLMCallError(f"Ollama request timed out after {timeout.total}s")
            except aiohttp.ClientError as e:
                raise LLMCallError(f"Ollama API error: {str(e)}")
    
    async def _call_gemini(self, config: ProviderConfig, prompt: str,
                          metadata: Dict[str, Any], token_limit: int) -> str:
        """Call Gemini API with timeout protection and SSL verification disabled."""
        timeout = await self._create_timeout_config(LLMProvider.GEMINI)
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": metadata.get("temperature", 0.7),
                "maxOutputTokens": token_limit,
                "topP": metadata.get("top_p", 0.95),
                "topK": metadata.get("top_k", 40)
            }
        }
        
        url = f"{config.endpoint}?key={GEMINI_API_KEY}"
        
        async with self._create_client_session(timeout) as session:
            try:
                async with session.post(url, json=payload, 
                                      headers=config.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if "candidates" in data and data["candidates"]:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if parts and "text" in parts[0]:
                                return parts[0]["text"]
                    
                    raise LLMCallError("Invalid Gemini response format")
            except asyncio.TimeoutError:
                raise LLMCallError(f"Gemini request timed out after {timeout.total}s")
            except aiohttp.ClientError as e:
                raise LLMCallError(f"Gemini API error: {str(e)}")
    
    async def _call_grok(self, config: ProviderConfig, prompt: str,
                        metadata: Dict[str, Any], token_limit: int) -> str:
        """Call Grok API with timeout protection and SSL verification disabled."""
        timeout = await self._create_timeout_config(LLMProvider.GROK)
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": metadata.get("system_prompt", 
                             "You are a helpful AI assistant participating in a multi-agent system.")
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": config.model,
            "stream": False,
            "temperature": metadata.get("temperature", 0.7),
            "max_tokens": token_limit
        }
        
        async with self._create_client_session(timeout) as session:
            try:
                async with session.post(config.endpoint, json=payload,
                                      headers=config.headers) as response:
                    if response.status == 429:
                        raise LLMCallError("Grok rate limit exceeded")
                    response.raise_for_status()
                    data = await response.json()
                    
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"]
                    
                    raise LLMCallError("Invalid Grok response format")
            except asyncio.TimeoutError:
                raise LLMCallError(f"Grok request timed out after {timeout.total}s")
            except aiohttp.ClientError as e:
                raise LLMCallError(f"Grok API error: {str(e)}")
    
    async def _call_llm_provider_safe(self, provider: LLMProvider, prompt: str,
                                     metadata: Dict[str, Any]) -> Optional[str]:
        """Safely call an LLM provider with error handling."""
        try:
            return await self._call_llm_provider(provider, prompt, metadata)
        except Exception as e:
            logger.error(f"{self.object_id}: Provider {provider.value} error: {e}")
            config = self.providers.get(provider)
            if config:
                config.failure_count += 1
            return None
    
    def _update_provider_metrics(self, provider: LLMProvider, success: bool, duration: float):
        """Update metrics for a provider."""
        metrics = self.provider_metrics[provider]
        metrics["total"] += 1
        if success:
            metrics["success"] += 1
        
        # Update rolling average duration
        old_avg = metrics["avg_duration"]
        old_count = metrics["total"] - 1
        if old_count > 0:
            metrics["avg_duration"] = (old_avg * old_count + duration) / metrics["total"]
        else:
            metrics["avg_duration"] = duration
    
    def _update_metrics(self, duration_s: float = 0.0):
        """Update resource manager metrics with enhanced tracking."""
        if self.context.resource_manager:
            success_rate = self.query_success_count / self.query_total_count if self.query_total_count > 0 else 0.0
            avg_duration = sum(self.query_durations) / len(self.query_durations) if self.query_durations else 0.0
            
            self.context.resource_manager.metrics.update({
                "llm_query_count": self.query_total_count,
                "llm_query_success_rate": success_rate,
                "llm_query_duration": duration_s,
                "llm_query_avg_duration": avg_duration,
                "llm_queue_size": self.priority_queue.qsize(),
                "llm_batch_queue_size": len(self.batch_accumulator)
            })
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue and system status for monitoring."""
        provider_status = await self.get_provider_status()
        cost_status = await self.get_cost_status()
        
        return {
            "priority_queue_size": self.priority_queue.qsize(),
            "batch_queue_size": len(self.batch_accumulator),
            "success_rate": self.query_success_count / self.query_total_count if self.query_total_count > 0 else 0.0,
            "avg_duration": sum(self.query_durations) / len(self.query_durations) if self.query_durations else 0.0,
            "total_queries": self.query_total_count,
            "providers": provider_status,
            "cost": cost_status,
            "load_balance_strategy": self.load_balance_strategy
        }
    
    def _generate_cache_key(self, prompt: str, origin: str) -> str:
        """Generate cache key for prompt."""
        combined = f"{origin}:{prompt}:{int(time.time() // 60)}"  # Cache keys change every minute
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _send_llm_response(self, text: str, origin: str, metadata: Dict[str, Any],
                               cached: bool = False, provider: str = "unknown",
                               duration: float = 0):
        """Send LLM response event."""
        response_metadata = {
            **metadata,
            "cached_response": cached,
            "llm_provider": provider,
            "response_timestamp": time.time(),
            "duration_s": round(duration, 3)
        }
        
        response_uc = UniversalCode(
            Tags.STRING,
            text,
            description="LLM response",
            metadata=response_metadata
        )
        
        response_event = Event(
            type=Actions.LLM_RESPONSE,
            data=response_uc,
            origin=self.object_id,
            priority=3
        )
        
        await self.context.dispatch_event(response_event)
    
    async def _send_error_response(self, event: Event, error_message: str):
        """Send error response for failed query."""
        metadata = event.data.metadata or {}
        metadata["is_error"] = True
        
        await self._send_llm_response(
            f"Error: {error_message}",
            event.origin,
            metadata,
            provider="error"
        )
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        status = {}
        for provider, config in self.providers.items():
            metrics = self.provider_metrics[provider]
            circuit_state = self.circuit_breakers[provider].state
            
            status[provider.value] = {
                "available": self._is_provider_available(provider),
                "circuit_breaker": circuit_state,
                "failure_count": config.failure_count,
                "success_rate": metrics["success"] / metrics["total"] if metrics["total"] > 0 else 0,
                "avg_response_time": round(metrics["avg_duration"], 3),
                "total_queries": metrics["total"]
            }
        return status
    
    async def get_cost_status(self) -> Dict[str, Any]:
        """Get current cost usage status."""
        return await self.cost_controller.get_current_usage()
    
    async def autonomous_step(self):
        """Periodic maintenance and monitoring."""
        await super().autonomous_step()
        
        # Gradually reduce failure counts for recovery
        if self.execution_count % 100 == 0:
            for config in self.providers.values():
                if config.failure_count > 0:
                    config.failure_count = max(0, config.failure_count - 1)
                    logger.debug(f"{self.object_id}: Reduced failure count for {config.provider.value}")
        
        # Log provider status periodically
        if self.execution_count % 50 == 0:
            status = await self.get_provider_status()
            cost_status = await self.get_cost_status()
            
            logger.info(f"{self.object_id}: Provider status: {status}")
            logger.info(f"{self.object_id}: Cost status: ${cost_status['hourly_spent']:.4f}/${cost_status['hourly_limit']} (hourly), ${cost_status['daily_spent']:.4f}/${cost_status['daily_limit']} (daily)")
        
        # Check cost limits
        cost_status = await self.get_cost_status()
        hourly_percentage = (cost_status['hourly_spent'] / cost_status['hourly_limit']) * 100
        daily_percentage = (cost_status['daily_spent'] / cost_status['daily_limit']) * 100
        
        if hourly_percentage > 80 or daily_percentage > 80:
            logger.warning(
                f"{self.object_id}: Approaching cost limits - "
                f"Hourly: {hourly_percentage:.1f}%, Daily: {daily_percentage:.1f}%"
            )
        
        await asyncio.sleep(0.1)

# ============================================================================
# FeedbackBubble (unchanged from original)
# ============================================================================

class FeedbackBubble(UniversalBubble):
    """Observes system metrics before/after actions to provide feedback."""
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.expected_impacts: Dict[str, Dict[str, float]] = {}
        self.observation_delay = 45
        self.observation_tasks: Dict[str, asyncio.Task] = {}
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized. Ready to observe action impacts (Delay: {self.observation_delay}s).")
        if not SCIPY_AVAILABLE:
            logger.warning(f"{self.object_id}: SciPy not found, statistical significance testing disabled.")

    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to ACTION_TAKEN.")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to ACTION_TAKEN: {e}", exc_info=True)

    async def process_single_event(self, event: Event):
        if event.type == Actions.ACTION_TAKEN:
            action_data = event.data.value if isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT else {}
            action_type = action_data.get("action_type")
            payload = action_data.get("payload", {})
            if action_type == Actions.ACTION_TYPE_CODE_UPDATE.name:
                await self.handle_code_update_action(event, payload)
        else:
            await super().process_single_event(event)

    async def handle_code_update_action(self, event: Event, payload: Dict):
        if not self.resource_manager:
            logger.error(f"{self.object_id}: ResourceManager unavailable, cannot capture metrics.")
            return

        update_desc = payload.get("description", f"Update_{event.event_id[:8]}")
        update_id = f"{update_desc}_{int(time.time())}"
        if update_id in self.observations:
            logger.warning(f"{self.object_id}: Already observing update '{update_id}'. Ignoring duplicate.")
            return

        current_state = self.resource_manager.get_current_system_state()
        self.observations[update_id] = current_state
        self.expected_impacts[update_id] = payload.get("expected_impact", {})
        logger.info(f"{self.object_id}: Observing code update '{update_id}'. Captured pre-update state.")
        if update_id in self.observation_tasks:
            self.observation_tasks[update_id].cancel()
        self.observation_tasks[update_id] = asyncio.create_task(self.check_post_update_metrics(update_id))

    async def check_post_update_metrics(self, update_id: str):
        try:
            await asyncio.sleep(self.observation_delay)
            if update_id not in self.observations:
                logger.warning(f"{self.object_id}: Observation '{update_id}' was cancelled or removed before check.")
                return
            if not self.resource_manager:
                logger.error(f"{self.object_id}: ResourceManager unavailable for post-update check '{update_id}'.")
                self.observations.pop(update_id, None)
                self.expected_impacts.pop(update_id, None)
                return

            pre_update_state = self.observations.pop(update_id)
            expected_impact = self.expected_impacts.pop(update_id, {})
            pre_update_metrics = pre_update_state.get("metrics", {})
            post_update_state = self.resource_manager.get_current_system_state()
            post_update_metrics = post_update_state.get("metrics", {})

            logger.info(f"{self.object_id}: Checking post-update metrics for '{update_id}'.")
            comparison_results = []
            significant_changes = {}
            pre_time = pre_update_metrics.get("avg_llm_response_time_ms", 0)
            post_time = post_update_metrics.get("avg_llm_response_time_ms", 0)
            delta_time = post_time - pre_time
            comparison_results.append(f"Avg LLM Resp Time: {pre_time:.1f}ms -> {post_time:.1f}ms (Delta: {delta_time:+.1f}ms)")
            pre_cache = pre_update_metrics.get("prediction_cache_hit_rate", 0)
            post_cache = post_update_metrics.get("prediction_cache_hit_rate", 0)
            delta_cache = post_cache - pre_cache
            comparison_results.append(f"Pred Cache Hit Rate: {pre_cache:.3f} -> {post_cache:.3f} (Delta: {delta_cache:+.3f})")

            impact_summary = []
            for metric, target_delta in expected_impact.items():
                pre_val = pre_update_metrics.get(metric)
                post_val = post_update_metrics.get(metric)
                if pre_val is not None and post_val is not None:
                    actual_delta = post_val - pre_val
                    achieved = False
                    if target_delta < 0 and actual_delta <= target_delta: achieved = True
                    elif target_delta > 0 and actual_delta >= target_delta: achieved = True
                    elif target_delta == 0 and actual_delta == 0: achieved = True
                    impact_summary.append(f"  - Expected {metric}: {target_delta:+.2f}, Actual: {actual_delta:+.2f} ({'Achieved' if achieved else 'Not Achieved'})")
                else:
                    impact_summary.append(f"  - Expected {metric}: {target_delta:+.2f} (Cannot compare, metric missing pre/post)")

            feedback_msg = f"Feedback on Update '{update_id}':\n"
            feedback_msg += "\n".join(comparison_results) + "\n"
            if significant_changes:
                feedback_msg += "Statistically Significant Changes (p<0.05):\n"
                for metric, pval in significant_changes.items():
                    feedback_msg += f"  - {metric} (p={pval:.4f})\n"
            if expected_impact:
                feedback_msg += "Expected Impact Check:\n" + "\n".join(impact_summary)

            await self.add_chat_message(f"System Update Feedback:\n{feedback_msg}")
            logger.info(f"Feedback generated for '{update_id}'.")

        except asyncio.CancelledError:
            logger.info(f"{self.object_id}: Observation check for '{update_id}' cancelled.")
            self.observations.pop(update_id, None)
            self.expected_impacts.pop(update_id, None)
        except Exception as e:
            logger.error(f"{self.object_id}: Error checking post-update metrics for '{update_id}': {e}", exc_info=True)
            self.observations.pop(update_id, None)
            self.expected_impacts.pop(update_id, None)
        finally:
            self.observation_tasks.pop(update_id, None)

    async def autonomous_step(self):
        await super().autonomous_step()
        await asyncio.sleep(5)

# ============================================================================
# Other Bubble Classes (unchanged from original)
# ============================================================================

# --- DreamerV3Bubble Class ---
class DreamerV3Bubble(UniversalBubble):
    """Implements DreamerV3 with fixed training, gravity, clustering, and perturbation support."""
    def __init__(self, object_id: str, context: SystemContext, state_dim: int = 16, action_dim: int = 5, hidden_dim: int = 256, num_categories: int = 32, horizon: int = 15, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        self.horizon = horizon
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 32
        self.sequence_length = 10
        self.learning_rate = 1e-4
        self.execution_count = 0
        self.nan_inf_count = 0  # Track consecutive NaN/Inf occurrences
        self.training_metrics = {
            "state_loss": 0.0, "reward_loss": 0.0, "kl_loss": 0.0,
            "continuation_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
            "entropy": 0.0, "disagreement_loss": 0.0, "recon_loss": 0.0
        }
        self.return_range = None
        self.ema_alpha = 0.99
        self.current_known_state: Optional[Dict[str, Any]] = None
        self.state_action_history: Dict[str, Tuple[Dict, Dict]] = {}

        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: PyTorch unavailable, switching to placeholder mode.")
            self.world_model = None
            self.world_model_ensemble = None
            self.actor = None
            self.critic = None
            self.critic_ema = None
            self.device = None
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            if self.device.type == "mps":
                logger.info(f"{self.object_id}: Using MPS device for M4 chip.")
            else:
                logger.warning(f"{self.object_id}: MPS unavailable, falling back to CPU.")
            self.world_model = self._build_world_model()
            self.world_model_ensemble = [self._build_world_model() for _ in range(2)]
            self.actor = self._build_actor()
            self.critic = self._build_critic()
            self.critic_ema = self._build_critic()
            self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.learning_rate)
            self.ensemble_optimizers = [optim.Adam(m.parameters(), lr=self.learning_rate) for m in self.world_model_ensemble]
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
            self.world_model.to(self.device)
            for m in self.world_model_ensemble:
                m.to(self.device)
            self.actor.to(self.device)
            self.critic.to(self.device)
            self.critic_ema.to(self.device)
            nn.init.zeros_(self.world_model.reward_predictor[-1].weight)
            nn.init.zeros_(self.critic.net[-1].weight)
            self._update_critic_ema(1.0)
            self.replay_buffer.clear()

        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized DreamerV3 (Mode: {'NN' if TORCH_AVAILABLE else 'Placeholder'}, Device: {self.device if TORCH_AVAILABLE else 'None'}).")

    def _symlog(self, x):
        if not TORCH_AVAILABLE:
            return x
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def _symexp(self, x):
        if not TORCH_AVAILABLE:
            return x
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def _twohot_encode(self, y, bins):
        if not TORCH_AVAILABLE:
            return torch.zeros(y.size(0), len(bins))
        if y.dim() > 1:
            y = y.squeeze()
        y = torch.clamp(y, bins[0], bins[-1])  # Ensure values are within bin range
        idx = torch.searchsorted(bins, y)
        idx = idx.clamp(0, len(bins) - 2)
        lower = bins[idx]
        upper = bins[idx + 1]
        weight = (y - lower) / (upper - lower + 1e-8)
        twohot = torch.zeros(y.size(0), len(bins), device=self.device)
        twohot.scatter_(1, idx.unsqueeze(1), 1.0 - weight.unsqueeze(1))
        twohot.scatter_(1, (idx + 1).unsqueeze(1), weight.unsqueeze(1))
        return twohot

    def _build_world_model(self):
        if not TORCH_AVAILABLE:
            return None
        class WorldModel(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim, num_categories):
                super().__init__()
                self.num_categories = num_categories
                self.encoder = nn.Sequential(
                    nn.Linear(state_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.categorical_encoder = nn.Linear(hidden_dim, num_categories * 32)
                self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, state_dim)
                )
                self.reward_predictor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 41)
                )
                self.continuation_predictor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid()
                )

            def forward(self, state, action, hidden, device=torch.device("cpu")):
                x = torch.cat([self._symlog(state), action], dim=-1).to(device)
                z = self.encoder(x)
                logits = self.categorical_encoder(z).view(-1, 32, self.num_categories)
                logits = 0.99 * logits + 0.01 * torch.ones_like(logits) / self.num_categories
                dist = torch.distributions.Categorical(logits=logits)
                sample = dist.sample()
                h = self.rnn(z, hidden)
                next_state = self.decoder(h)
                reward_logits = self.reward_predictor(h)
                continuation = self.continuation_predictor(h)
                kl_loss = torch.mean(torch.distributions.kl_divergence(
                    dist, torch.distributions.Categorical(probs=torch.ones_like(logits) / self.num_categories)
                ))
                return next_state, reward_logits, continuation, h, kl_loss, sample

            def _symlog(self, x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return WorldModel(self.state_dim, self.action_dim, self.hidden_dim, self.num_categories)

    def _build_actor(self):
        if not TORCH_AVAILABLE:
            return None
        class Actor(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim, device):
                super().__init__()
                self.device = device
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, action_dim)
                )

            def forward(self, state):
                logits = self.net(self._symlog(state).to(self.device))
                return torch.distributions.Categorical(logits=logits)

            def _symlog(self, x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return Actor(self.state_dim, self.action_dim, self.hidden_dim, self.device)

    def _build_critic(self):
        if not TORCH_AVAILABLE:
            return None
        class Critic(nn.Module):
            def __init__(self, state_dim, hidden_dim, device):
                super().__init__()
                self.device = device
                self.net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 41)
                )

            def forward(self, state):
                logits = self.net(self._symlog(state).to(self.device))
                return torch.distributions.Categorical(logits=logits)

            def _symlog(self, x):
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return Critic(self.state_dim, self.hidden_dim, self.device)

    def _update_critic_ema(self, alpha=0.01):
        if not TORCH_AVAILABLE:
            return
        for param, ema_param in zip(self.critic.parameters(), self.critic_ema.parameters()):
            ema_param.data.mul_(1.0 - alpha).add_(param.data, alpha=alpha)

    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            await EventService.subscribe(Actions.PREDICT_STATE_QUERY, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, ACTION_TAKEN, PREDICT_STATE_QUERY")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    def _vectorize_state(self, state: Dict) -> Optional[torch.Tensor]:
        """Converts state dictionary to a symlog-transformed tensor with gravity and clustering."""
        if not TORCH_AVAILABLE:
            logger.warning(f"{self.object_id}: Cannot vectorize state, PyTorch unavailable.")
            return None
        try:
            metrics = state.get("metrics", {})
            event_frequencies = state.get("event_frequencies", {})
            perturbation = state.get("response_time_perturbation", 0.0)
            vector = [
                state.get("energy", 0) / 10000.0,
                state.get("cpu_percent", 0) / 100.0,
                state.get("memory_percent", 0) / 100.0,
                state.get("num_bubbles", 0) / 20.0,
                metrics.get("avg_llm_response_time_ms", 0) / 60000.0 * (1 + perturbation),
                metrics.get("code_update_count", 0) / 100.0,
                metrics.get("prediction_cache_hit_rate", 0),
                event_frequencies.get("LLM_QUERY_freq_per_min", 0) / 60.0,
                event_frequencies.get("CODE_UPDATE_freq_per_min", 0) / 10.0,
                event_frequencies.get("ACTION_TAKEN_freq_per_min", 0) / 60.0,
                state.get("gravity_force", 0.0) / 10.0,
                state.get("gravity_direction", 0.0) / 360.0,
                state.get("bubble_pos_x", 0.0) / 100.0,
                state.get("bubble_pos_y", 0.0) / 100.0,
                state.get("cluster_id", 0) / 10.0,
                state.get("cluster_strength", 0.0) / 1.0
            ]
            if len(vector) != self.state_dim:
                logger.error(f"{self.object_id}: State vector dim mismatch! Expected {self.state_dim}, got {len(vector)}.")
                return None
            tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
            return self._symlog(tensor)
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing state: {e}", exc_info=True)
            return None

    def _vectorize_action(self, action: Dict) -> Optional[torch.Tensor]:
        if not TORCH_AVAILABLE:
            logger.warning(f"{self.object_id}: Cannot vectorize action, PyTorch unavailable.")
            return None
        try:
            action_type_str = action.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
            action_types_ordered = [
                Actions.ACTION_TYPE_CODE_UPDATE.name, Actions.ACTION_TYPE_SELF_QUESTION.name,
                Actions.ACTION_TYPE_SPAWN_BUBBLE.name, Actions.ACTION_TYPE_DESTROY_BUBBLE.name,
                Actions.ACTION_TYPE_NO_OP.name
            ]
            if len(action_types_ordered) != self.action_dim:
                raise ValueError(f"Action dimension mismatch: Code expects {len(action_types_ordered)}, configured {self.action_dim}")
            vector = [1.0 if action_type_str == at else 0.0 for at in action_types_ordered]
            return torch.tensor(vector, dtype=torch.float32).to(self.device)
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing action: {e}", exc_info=True)
            return None

    def _devectorize_state(self, state_vector: torch.Tensor) -> Dict:
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        try:
            vec = self._symexp(state_vector).detach().cpu().numpy()
            if len(vec) != self.state_dim:
                return {"error": f"State vector dim mismatch: got {len(vec)}, expected {self.state_dim}"}
            state = {
                "energy": vec[0] * 10000.0,
                "cpu_percent": max(0.0, min(100.0, vec[1] * 100.0)),
                "memory_percent": max(0.0, min(100.0, vec[2] * 100.0)),
                "num_bubbles": max(0, int(round(vec[3] * 20.0))),
                "metrics": {
                    "avg_llm_response_time_ms": max(0.0, vec[4] * 60000.0),
                    "code_update_count": max(0, int(round(vec[5] * 100.0))),
                    "prediction_cache_hit_rate": max(0.0, min(1.0, vec[6])),
                },
                "event_frequencies": {
                    "LLM_QUERY_freq_per_min": max(0.0, vec[7] * 60.0),
                    "CODE_UPDATE_freq_per_min": max(0.0, vec[8] * 10.0),
                    "ACTION_TAKEN_freq_per_min": max(0.0, vec[9] * 60.0),
                },
                "gravity_force": max(0.0, vec[10] * 10.0),
                "gravity_direction": max(0.0, min(360.0, vec[11] * 360.0)),
                "bubble_pos_x": vec[12] * 100.0,
                "bubble_pos_y": vec[13] * 100.0,
                "cluster_id": max(0, int(round(vec[14] * 10.0))),
                "cluster_strength": max(0.0, min(1.0, vec[15])),
                "timestamp": time.time(),
                "categorical_confidence": 0.7,
                "continuation_probability": 0.9,
                "response_time_perturbation": 0.1
            }
            return state
        except Exception as e:
            logger.error(f"{self.object_id}: Error devectorizing state: {e}", exc_info=True)
            return {"error": f"State devectorization failed: {e}"}

    def _devectorize_action(self, action_vector: torch.Tensor) -> str:
        if not TORCH_AVAILABLE:
            return Actions.ACTION_TYPE_NO_OP.name
        try:
            action_types_ordered = [
                Actions.ACTION_TYPE_CODE_UPDATE.name, Actions.ACTION_TYPE_SELF_QUESTION.name,
                Actions.ACTION_TYPE_SPAWN_BUBBLE.name, Actions.ACTION_TYPE_DESTROY_BUBBLE.name,
                Actions.ACTION_TYPE_NO_OP.name
            ]
            action_idx = torch.argmax(action_vector).item()
            return action_types_ordered[action_idx]
        except Exception as e:
            logger.error(f"{self.object_id}: Error devectorizing action: {e}", exc_info=True)
            return Actions.ACTION_TYPE_NO_OP.name

    def _compute_reward(self, prev_state: Dict, next_state: Dict) -> torch.Tensor:
        if not TORCH_AVAILABLE:
            return torch.tensor([0.0])
        prev_energy = prev_state.get("energy", 0)
        next_energy = next_state.get("energy", 0)
        reward = (next_energy - prev_energy) / 1000.0
        reward = max(-10.0, min(10.0, reward))  # Normalize reward
        logger.debug(f"{self.object_id}: Computed reward: {reward:.4f}")
        return torch.tensor([reward], dtype=torch.float32).to(self.device)

    def _rebuild_models(self):
        if not TORCH_AVAILABLE:
            return
        self.replay_buffer.clear()
        self.world_model = self._build_world_model()
        self.world_model_ensemble = [self._build_world_model() for _ in range(2)]
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.critic_ema = self._build_critic()
        self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.learning_rate)
        self.ensemble_optimizers = [optim.Adam(m.parameters(), lr=self.learning_rate) for m in self.world_model_ensemble]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.world_model.to(self.device)
        for m in self.world_model_ensemble:
            m.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_ema.to(self.device)
        nn.init.zeros_(self.world_model.reward_predictor[-1].weight)
        nn.init.zeros_(self.critic.net[-1].weight)
        logger.info(f"{self.object_id}: Rebuilt models with state_dim={self.state_dim}")

    async def load_external_data(self, file_path: str):
        """Loads external data into the replay buffer, using next item's state as next_state if available."""
        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: Cannot load external data, PyTorch unavailable.")
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            for i, item in enumerate(data):
                state_vec = self._vectorize_state(item["state"])
                action_vec = self._vectorize_action(item["action"])
                reward = torch.tensor([item.get("reward", 0.0)], dtype=torch.float32).to(self.device)
                next_state = data[i + 1]["state"] if i + 1 < len(data) else item["state"]
                next_state_vec = self._vectorize_state(next_state)
                if state_vec is None or action_vec is None or next_state_vec is None:
                    logger.warning(f"{self.object_id}: Skipping invalid transition at index {i} due to vectorization failure.")
                    continue
                tensors = {"state_vec": state_vec, "action_vec": action_vec, "reward": reward, "next_state_vec": next_state_vec}
                for name, tensor in tensors.items():
                    if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                        logger.warning(f"{self.object_id}: Skipping transition due to NaN/Inf in {name} at index {i}")
                        break
                else:
                    self.replay_buffer.append((state_vec, action_vec, reward, next_state_vec))
            logger.info(f"{self.object_id}: Loaded {len(self.replay_buffer)} valid transitions from {file_path}.")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to load external data from {file_path}: {e}", exc_info=True)

    def check_training_stability(self, loss: torch.Tensor) -> bool:
        if not TORCH_AVAILABLE:
            return False
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_inf_count += 1
            logger.error(f"{self.object_id}: Detected unstable training (NaN/Inf loss). Consecutive count: {self.nan_inf_count}")
            if self.nan_inf_count > 2:
                self._rebuild_models()
                self.nan_inf_count = 0
                logger.info(f"{self.object_id}: Rebuilt models due to repeated NaN/Inf losses.")
            else:
                self.learning_rate *= 0.5
                self.world_optimizer = optim.Adam(self.world_model.parameters(), lr=self.learning_rate)
                for i, m in enumerate(self.world_model_ensemble):
                    self.ensemble_optimizers[i] = optim.Adam(m.parameters(), lr=self.learning_rate)
                self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
                self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
                logger.info(f"{self.object_id}: Reduced learning rate to {self.learning_rate}")
            return False
        self.nan_inf_count = 0
        return True

    def get_training_metrics(self) -> Dict:
        return self.training_metrics

    async def process_single_event(self, event: Event):
        self.execution_count += 1
        if event.type == Actions.SYSTEM_STATE_UPDATE:
            await self.handle_state_update(event)
        elif event.type == Actions.ACTION_TAKEN:
            await self.handle_action_taken(event)
        elif event.type == Actions.PREDICT_STATE_QUERY:
            await self.handle_predict_query(event)
        else:
            await super().process_single_event(event)

    async def handle_state_update(self, event: Event):
        """Handles SYSTEM_STATE_UPDATE to form transitions with stored actions."""
        if event.origin != "ResourceManager" or not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.debug(f"{self.object_id}: Invalid SYSTEM_STATE_UPDATE event: origin={event.origin}, data_type={type(event.data)}")
            return
        new_state = event.data.value
        new_ts = new_state.get("timestamp")
        if not new_ts:
            logger.warning(f"{self.object_id}: SYSTEM_STATE_UPDATE missing timestamp")
            return
        logger.debug(f"{self.object_id}: Received SYSTEM_STATE_UPDATE (ts: {new_ts:.2f})")

        if self.current_known_state is not None and TORCH_AVAILABLE:
            prev_ts = self.current_known_state.get("timestamp", 0)
            action_to_link, action_ts_to_del, latest_action_ts = None, None, -1
            history_keys = list(self.state_action_history.keys())
            for act_ts_str in history_keys:
                if act_ts_str not in self.state_action_history:
                    continue
                act_ts = float(act_ts_str)
                state_before_act, action_data = self.state_action_history[act_ts_str]
                if prev_ts <= act_ts <= new_ts or (new_ts - 60 <= act_ts <= new_ts):
                    if act_ts > latest_action_ts:
                        latest_action_ts = act_ts
                        action_to_link = action_data
                        action_ts_to_del = act_ts_str
                elif act_ts < prev_ts - 300:
                    self.state_action_history.pop(act_ts_str, None)
            if action_to_link:
                state_vec = self._vectorize_state(state_before_act)
                action_vec = self._vectorize_action(action_to_link)
                next_state_vec = self._vectorize_state(new_state)
                if state_vec is not None and action_vec is not None and next_state_vec is not None:
                    reward = self._compute_reward(state_before_act, new_state)
                    tensors = {"state_vec": state_vec, "action_vec": action_vec, "reward": reward, "next_state_vec": next_state_vec}
                    for name, tensor in tensors.items():
                        if torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor)):
                            logger.warning(f"{self.object_id}: Skipping transition due to NaN/Inf in {name} at ts {new_ts}")
                            break
                    else:
                        self.replay_buffer.append((state_vec, action_vec, reward, next_state_vec))
                        act_type = action_to_link.get("action_type", "UNKNOWN")
                        logger.info(f"{self.object_id}: Stored transition (Action '{act_type}') in replay buffer. Size: {len(self.replay_buffer)}")
                else:
                    logger.warning(f"{self.object_id}: Failed to vectorize state/action for transition at ts {new_ts}")
                if action_ts_to_del:
                    self.state_action_history.pop(action_ts_to_del, None)
            else:
                logger.debug(f"{self.object_id}: No action found between ts {prev_ts} and {new_ts} or within 60s.")

        self.current_known_state = new_state

    async def handle_action_taken(self, event: Event):
        """Stores actions with current state for later transition formation."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.debug(f"{self.object_id}: Invalid ACTION_TAKEN event: data_type={type(event.data)}")
            return
        action_data = event.data.value
        timestamp = event.data.metadata.get("timestamp", time.time())
        if self.current_known_state is not None and TORCH_AVAILABLE:
            self.state_action_history[str(timestamp)] = (self.current_known_state, action_data)
            logger.debug(f"{self.object_id}: Stored action '{action_data.get('action_type', 'UNKNOWN')}' at ts {timestamp:.2f}. History size: {len(self.state_action_history)}")
        else:
            logger.warning(f"{self.object_id}: ACTION_TAKEN at {timestamp} but no current state or PyTorch unavailable.")

    async def handle_predict_query(self, event: Event):
        if not TORCH_AVAILABLE:
            await self._send_prediction_response(event.origin, event.data.metadata.get("correlation_id"), error="PyTorch unavailable")
            return
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            await self._send_prediction_response(event.origin, event.data.metadata.get("correlation_id"), error="Invalid query format")
            return
        query_data = event.data.value
        origin_bubble_id = event.origin
        correlation_id = event.data.metadata.get("correlation_id")
        if not correlation_id:
            return
        current_state = query_data.get("current_state")
        action_to_simulate = query_data.get("action")
        if not current_state or not action_to_simulate or not isinstance(action_to_simulate, dict):
            await self._send_prediction_response(origin_bubble_id, correlation_id, error="Missing state or valid action")
            return
        act_type = action_to_simulate.get('action_type', 'UNKNOWN')
        logger.info(f"{self.object_id}: Received PREDICT_STATE_QUERY {correlation_id[:8]} from {origin_bubble_id} for action: {act_type}")

        if not self.world_model:
            await self._send_prediction_response(origin_bubble_id, correlation_id, error="DreamerV3 not available")
            return

        try:
            state_vector = self._vectorize_state(current_state)
            action_vector = self._vectorize_action(action_to_simulate)
            if state_vector is None or action_vector is None:
                raise ValueError("Vectorization failed")

            self.world_model.eval()
            hidden = torch.zeros(1, self.hidden_dim, dtype=torch.float32).to(self.device)
            predicted_states, predicted_continuations = [], []
            with torch.no_grad():
                current_state_vec = state_vector.unsqueeze(0)
                for _ in range(self.horizon):
                    action_vec = action_vector.unsqueeze(0)
                    next_state, _, continuation, hidden, _, _ = self.world_model(current_state_vec, action_vec, hidden, device=self.device)
                    predicted_states.append(next_state)
                    predicted_continuations.append(continuation)
                    current_state_vec = next_state

            predicted_state = self._devectorize_state(predicted_states[-1].squeeze(0))
            predicted_state["continuation_probability"] = predicted_continuations[-1].item()
            if "error" in predicted_state:
                raise ValueError(predicted_state["error"])

            await self._send_prediction_response(origin_bubble_id, correlation_id, prediction=predicted_state)
        except Exception as e:
            logger.error(f"{self.object_id}: Prediction error for {correlation_id[:8]}: {e}", exc_info=True)
            await self._send_prediction_response(origin_bubble_id, correlation_id, error=f"Prediction failed: {e}")

    async def _send_prediction_response(self, requester_id: Optional[str], correlation_id: Optional[str], prediction: Optional[Dict] = None, error: Optional[str] = None):
        if not requester_id or not correlation_id:
            logger.error(f"{self.object_id}: Cannot send prediction response - missing requester_id or correlation_id.")
            return
        if not self.dispatcher:
            logger.error(f"{self.object_id}: Cannot send prediction response, dispatcher unavailable.")
            return

        response_payload = {"correlation_id": correlation_id}
        if prediction and not error:
            response_payload["predicted_state"] = prediction
            response_payload["error"] = None
            status = "SUCCESS"
        else:
            response_payload["predicted_state"] = None
            response_payload["error"] = error if error else "Unknown prediction error"
            status = "ERROR"

        response_uc = UniversalCode(Tags.DICT, response_payload, description=f"Predicted state response ({status})")
        response_event = Event(type=Actions.PREDICT_STATE_RESPONSE, data=response_uc, origin=self.object_id, priority=2)
        await self.context.dispatch_event(response_event)
        logger.info(f"{self.object_id}: Sent PREDICT_STATE_RESPONSE ({status}) for {correlation_id[:8]} to {requester_id}")

    async def train_world_model(self):
        """Trains the world model with available transitions, using dynamic batch sizing."""
        if not TORCH_AVAILABLE or not self.world_model or len(self.replay_buffer) == 0:
            logger.debug(f"{self.object_id}: Skipping world model training (Torch: {TORCH_AVAILABLE}, Buffer: {len(self.replay_buffer)})")
            return
        try:
            batch_size = min(self.batch_size, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)
            states, actions, rewards, next_states = zip(*batch)
            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.stack(actions).to(self.device)
            rewards_tensor = torch.stack(rewards).to(self.device)
            next_states_tensor = torch.stack(next_states).to(self.device)
            continuation_tensor = torch.ones(batch_size, 1, dtype=torch.float32).to(self.device)

            self.world_model.train()
            for m in self.world_model_ensemble:
                m.train()
            self.world_optimizer.zero_grad()
            for opt in self.ensemble_optimizers:
                opt.zero_grad()

            hidden = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)
            predicted_next_states, reward_logits, predicted_continuations, _, kl_loss, _ = self.world_model(states_tensor, actions_tensor, hidden, device=self.device)

            state_loss = torch.mean((predicted_next_states - next_states_tensor) ** 2)
            bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
            rewards_tensor = torch.clamp(rewards_tensor, min=-20.0, max=20.0)  # Ensure rewards match bin range
            reward_targets = self._twohot_encode(rewards_tensor, bins)
            reward_loss = torch.nn.functional.cross_entropy(reward_logits, reward_targets, reduction='mean')
            continuation_loss = nn.BCELoss()(predicted_continuations, continuation_tensor)
            kl_loss = torch.max(kl_loss, torch.tensor(1.0, device=self.device))
            total_loss = 1.0 * (state_loss + reward_loss + continuation_loss) + 1.0 * kl_loss

            if not self.check_training_stability(total_loss):
                return

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1000)
            self.world_optimizer.step()

            for m, opt in zip(self.world_model_ensemble, self.ensemble_optimizers):
                m.train()
                opt.zero_grad()
                pred_next_states, _, _, _, _, _ = m(states_tensor, actions_tensor, hidden, device=self.device)
                ensemble_loss = torch.mean((pred_next_states - next_states_tensor) ** 2)
                ensemble_loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1000)
                opt.step()

            disagreement_loss = torch.mean(torch.var(torch.stack([m(states_tensor, actions_tensor, hidden, device=self.device)[0] for m in self.world_model_ensemble], dim=0)))

            self.training_metrics.update({
                "state_loss": state_loss.item(),
                "reward_loss": reward_loss.item(),
                "continuation_loss": continuation_loss.item(),
                "kl_loss": kl_loss.item(),
                "disagreement_loss": disagreement_loss.item(),
                "recon_loss": 0.0
            })

            logger.info(f"{self.object_id}: Trained world model with {batch_size} samples, loss: {total_loss.item():.6f}, state: {state_loss.item():.6f}, reward: {reward_loss.item():.6f}")
        except Exception as e:
            logger.error(f"{self.object_id}: World model training error: {e}", exc_info=True)

    async def train_actor_critic(self):
        """Trains the actor and critic with available transitions, using dynamic batch sizing."""
        if not TORCH_AVAILABLE or not self.actor or not self.critic or len(self.replay_buffer) == 0:
            logger.debug(f"{self.object_id}: Skipping actor-critic training (Torch: {TORCH_AVAILABLE}, Buffer: {len(self.replay_buffer)})")
            return
        try:
            batch_size = min(self.batch_size, len(self.replay_buffer))
            batch = random.sample(self.replay_buffer, batch_size)
            states, actions = zip(*[(s, a) for s, a, _, _ in batch])
            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.stack(actions).to(self.device)
            self.world_model.eval()
            self.actor.train()
            self.critic.train()
            hidden = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32).to(self.device)

            imagined_states, imagined_rewards, imagined_continuations, actions_taken = [], [], [], []
            current_state = states_tensor
            for step in range(self.horizon):
                action_dist = self.actor(current_state)
                action = action_dist.sample().unsqueeze(-1)
                action_one_hot = torch.zeros(action.size(0), self.action_dim, device=self.device)
                action_one_hot.scatter_(1, action, 1.0)
                with torch.no_grad():
                    ensemble_next_states = [m(current_state, action_one_hot, hidden, device=self.device)[0] for m in self.world_model_ensemble]
                next_state_var = torch.var(torch.stack(ensemble_next_states, dim=0), dim=0).mean(dim=-1)
                next_state_var = torch.clamp(next_state_var, min=0.0, max=100.0)  # Cap variance
                del ensemble_next_states
                next_state, reward_logits, continuation, hidden, _, _ = self.world_model(current_state, action_one_hot, hidden, device=self.device)
                bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
                reward = torch.sum(torch.softmax(reward_logits, dim=-1) * bins, dim=-1) + 0.01 * next_state_var
                reward = torch.clamp(reward, min=-10.0, max=10.0)  # Clip rewards
                imagined_states.append(next_state)
                imagined_rewards.append(reward)
                imagined_continuations.append(continuation)
                actions_taken.append(action)
                current_state = next_state

            bins = self._symexp(torch.linspace(-20, 20, 41).to(self.device))
            values = [torch.sum(torch.softmax(self.critic(s).logits, dim=-1) * bins, dim=-1) for s in imagined_states]
            values_tensor = torch.stack(values[:-1], dim=1)

            returns = []
            lambda_return = values[-1]
            for r, c in zip(reversed(imagined_rewards), reversed(imagined_continuations)):
                c = c.squeeze(-1)
                lambda_return = r + 0.997 * c * lambda_return
                returns.append(lambda_return)
            returns = torch.stack(list(reversed(returns)), dim=1)
            returns = torch.clamp(returns, min=-100.0, max=100.0)  # Clip returns

            if self.return_range is None:
                self.return_range = torch.tensor(1.0, device=self.device)
            else:
                percentiles = torch.quantile(returns, torch.tensor([0.05, 0.95], device=self.device))
                range_estimate = percentiles[1] - percentiles[0]
                self.return_range = self.ema_alpha * self.return_range + (1 - self.ema_alpha) * range_estimate
            norm_factor = torch.max(torch.tensor(1.0, device=self.device), self.return_range)

            with torch.no_grad():
                ensemble_outputs = [m(states_tensor, actions_tensor, hidden, device=self.device)[0] for m in self.world_model_ensemble]
                disagreement_loss = torch.mean(torch.var(torch.stack(ensemble_outputs, dim=0)))

            self.actor_optimizer.zero_grad()
            advantages = (returns[:, :-1] - values_tensor) / norm_factor
            advantages = torch.clamp(advantages, min=-10.0, max=10.0)  # Clip advantages
            log_probs = torch.stack([self.actor(s).log_prob(a.squeeze(-1)) for s, a in zip(imagined_states[:-1], actions_taken[:-1])], dim=1)
            log_probs = torch.clamp(log_probs, min=-10.0, max=10.0)  # Clip log probabilities
            actor_loss = -(log_probs * advantages.detach()).mean()
            entropy = self.actor(states_tensor).entropy().mean()
            total_actor_loss = actor_loss - 3e-4 * entropy + 0.005 * disagreement_loss
            total_actor_loss = torch.clamp(total_actor_loss, min=-1000.0, max=1000.0)  # Clip total_actor_loss
            if self.check_training_stability(total_actor_loss):
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # Tighter clipping
                self.actor_optimizer.step()

            returns_detached = returns.detach()
            imagined_states_detached = [s.detach() for s in imagined_states]
            self.critic_optimizer.zero_grad()
            critic_losses = [nn.MSELoss()(torch.sum(torch.softmax(self.critic(s).logits, dim=-1) * bins, dim=-1), r) for s, r in zip(imagined_states_detached[:-1], returns_detached[:, :-1].unbind(dim=1))]
            critic_loss = torch.stack(critic_losses).mean()
            if self.check_training_stability(critic_loss):
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()

            self._update_critic_ema()
            self.training_metrics.update({
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropy.item(),
                "disagreement_loss": disagreement_loss.item()
            })

            logger.info(f"{self.object_id}: Trained actor-critic with {batch_size} samples, actor_loss: {actor_loss.item():.6f}, critic_loss: {critic_loss.item():.6f}")
        except Exception as e:
            logger.error(f"{self.object_id}: Actor-critic training error: {e}", exc_info=True)

    async def autonomous_step(self):
        """Trains the world model and actor-critic periodically."""
        await super().autonomous_step()
        if self.execution_count % 10 == 0:
            await self.train_world_model()
            await self.train_actor_critic()
        await asyncio.sleep(0.5)

# --- MetaReasoningBubble Class ---
# --- Complete Modified MetaReasoningBubble Class ---


class MetaReasoningBubble(UniversalBubble):
    """Manages high-level reasoning and action selection with explicit triggering."""
    
    def __init__(self, object_id: str, context: SystemContext, cycle_interval: float = 180.0, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.cycle_interval = cycle_interval  # Kept for backwards compatibility but not used
        self.last_action_cycle_time = 0
        self.action_history: List[Tuple[float, Dict]] = []
        self.execution_count = 0
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized MetaReasoningBubble (Explicit Triggering Mode).")

    async def _subscribe_to_events(self):
        """Subscribe to relevant events including the explicit STRATEGIC_ANALYSIS trigger."""
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            await EventService.subscribe(Actions.LLM_RESPONSE, self.handle_event)
            # Subscribe to explicit strategic analysis trigger
            await EventService.subscribe(Actions.STRATEGIC_ANALYSIS, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, ACTION_TAKEN, LLM_RESPONSE, STRATEGIC_ANALYSIS")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    async def process_single_event(self, event: Event):
        """Process incoming events with explicit handling for STRATEGIC_ANALYSIS."""
        self.execution_count += 1
        if event.type == Actions.STRATEGIC_ANALYSIS:
            # Explicit trigger for strategic LLM query
            logger.info(f"{self.object_id}: Received explicit STRATEGIC_ANALYSIS event, triggering strategic LLM query.")
            await self.generate_action()
        elif event.type == Actions.SYSTEM_STATE_UPDATE:
            await self.handle_state_update(event)
        elif event.type == Actions.ACTION_TAKEN:
            await self.handle_action_taken(event)
        elif event.type == Actions.LLM_RESPONSE:
            await self.handle_llm_response(event)
        else:
            await super().process_single_event(event)

    async def handle_state_update(self, event: Event):
        """Handle system state update events."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.warning(f"{self.object_id}: Invalid SYSTEM_STATE_UPDATE data: {event.data}")
            return
        state = event.data.value
        logger.debug(f"{self.object_id}: Received SYSTEM_STATE_UPDATE at ts {state.get('timestamp', 0):.2f}")
        self.action_history.append((time.time(), state))

    async def handle_action_taken(self, event: Event):
        """Handle action taken events."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.warning(f"{self.object_id}: Invalid ACTION_TAKEN data: {event.data}")
            return
        action_data = event.data.value
        logger.debug(f"{self.object_id}: Recorded action {action_data.get('action_type', 'UNKNOWN')}")
        self.action_history.append((time.time(), action_data))

    async def handle_llm_response(self, event: Event):
        """Handle LLM response events."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.STRING:
            logger.warning(f"{self.object_id}: Invalid LLM_RESPONSE data: {event.data}")
            return
        response_text = event.data.value
        metadata = event.data.metadata or {}
        correlation_id = metadata.get("correlation_id")
        logger.debug(f"{self.object_id}: Received LLM_RESPONSE {correlation_id[:8]}: {response_text[:5000]}...")
        self.action_history.append((time.time(), {"llm_response": response_text}))

    async def autonomous_step(self):
        """No automatic periodic triggering - only sleeps."""
        await super().autonomous_step()
        await asyncio.sleep(1)  # idle sleep without periodic triggering

    async def generate_action(self):
        """Generates and dispatches a high-level action - called only on explicit trigger."""
        if not self.resource_manager or not await self.resource_manager.consume_resource('energy', 2.0):
            logger.warning(f"{self.object_id}: Insufficient energy for action generation")
            await self.add_chat_message("Paused action planning due to low energy...")
            return

        state = self.resource_manager.get_current_system_state() if self.resource_manager else {}
        metrics = state.get("metrics", {})
        
        # Include real hardware details in the prompt
        hardware = state.get("hardware", {})
        cpu_info = hardware.get("cpu", {})
        gpu_info = hardware.get("gpu", {})
        power_info = hardware.get("power", {})
        
        prompt = (
            f"Current system state:\n"
            f"- Energy: {state.get('energy', 0):.0f}\n"
            f"- CPU Total: {state.get('cpu_percent', 0):.1f}%\n"
            f"  - P-Cores: {cpu_info.get('performance_cores_percent', 0):.1f}%\n"
            f"  - E-Cores: {cpu_info.get('efficiency_cores_percent', 0):.1f}%\n"
            f"- GPU: {gpu_info.get('usage_percent', 0):.1f}%\n"
            f"- Memory: {state.get('memory_percent', 0):.1f}%\n"
            f"- Power: {power_info.get('estimated_total_watts', 0):.1f}W\n"
            f"- LLM Resp Time: {metrics.get('avg_llm_response_time_ms', 0):.0f}ms\n"
            f"- Cache Hit Rate: {metrics.get('prediction_cache_hit_rate', 0):.3f}\n"
            f"Suggest an action to optimize system performance (e.g., code update, bubble spawn, no-op)."
        )
        
        correlation_id = str(uuid.uuid4())
        query_uc = UniversalCode(
            Tags.STRING, 
            prompt, 
            description="Action suggestion", 
            metadata={"correlation_id": correlation_id, "response_to": self.object_id}
        )
        query_event = Event(
            type=Actions.LLM_QUERY, 
            data=query_uc, 
            origin=self.object_id, 
            priority=2
        )

        await self.context.dispatch_event(query_event)
        logger.info(f"{self.object_id}: Requested action suggestion {correlation_id[:8]}")
        await self.add_chat_message("Planning next action based on system state...")


# --- Helper function to trigger strategic analysis ---
async def trigger_strategic_analysis(system_context: SystemContext, reason: str = "manual_trigger", **kwargs):
    """
    Explicitly trigger strategic analysis in the MetaReasoningBubble.
    
    Args:
        system_context: The system context to dispatch events
        reason: Reason for triggering the analysis
        **kwargs: Additional data to include in the event
    
    Example:
        await trigger_strategic_analysis(
            system_context, 
            reason="performance_threshold",
            cpu_usage=85.5,
            memory_pressure="high"
        )
    """
    event_data = {
        "reason": reason,
        "timestamp": time.time(),
        "details": f"Explicit strategic analysis requested: {reason}"
    }
    event_data.update(kwargs)
    
    strategic_event = Event(
        type=Actions.STRATEGIC_ANALYSIS,
        data=UniversalCode(
            tag=Tags.DICT,
            value=event_data
        ),
        origin="admin_control" if reason == "manual_trigger" else reason,
        priority=10
    )
    
    await system_context.dispatch_event(strategic_event)
    logger.info(f"Dispatched STRATEGIC_ANALYSIS event with reason: {reason}")

# ============================================================================
# CreativeSynthesisBubble (unchanged from original)
# ============================================================================

class CreativeSynthesisBubble(UniversalBubble):
    """Generates novel proposals and manages creative processes."""
    def __init__(self, object_id: str, context: SystemContext, proposal_interval: float = 300.0, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.proposal_interval = proposal_interval
        self.last_proposal_time = 0
        self.last_reflection_time = 0
        self.self_reflection_interval = 3600.0
        self.novelty_weight = 0.7
        self.confidence_threshold = 0.5
        self.insight_memory: List[Tuple[float, Optional[Dict], Optional[Dict], Optional[str]]] = []
        self.pattern_detector: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.proposal_history: List[Tuple[Dict, float, str]] = []
        self.creativity_journal: List[Tuple[str, float]] = []
        self.analogy_domains = ["orchestra", "ecosystem", "galaxy", "city", "neural network"]
        self._pending_proposal_corr_id: Optional[str] = None
        self._proposal_response: Optional[str] = None
        self._proposal_complete_event = asyncio.Event()
        self._pending_predictions: Dict[str, str] = {}
        self._prediction_results: Dict[str, float] = {}
        self._prediction_complete_event = asyncio.Event()
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized as core component with enhanced visibility.")
        asyncio.create_task(self.add_chat_message("Listening to the system's pulse for inspiration..."))

    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            await EventService.subscribe(Actions.LLM_RESPONSE, self.handle_event)
            await EventService.subscribe(Actions.USER_RESPONSE, self.handle_event)
            await EventService.subscribe(Actions.PREDICT_STATE_RESPONSE, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to multiple event types")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)
            await self.add_chat_message(f"Error: Could not connect to system events: {e}")

    async def process_single_event(self, event: Event):
        self.execution_count += 1
        if event.type == Actions.SYSTEM_STATE_UPDATE:
            await self.handle_state_update(event)
        elif event.type == Actions.ACTION_TAKEN:
            await self.handle_action_taken(event)
        elif event.type in (Actions.LLM_RESPONSE, Actions.USER_RESPONSE):
            await self.handle_llm_response(event)
        elif event.type == Actions.PREDICT_STATE_RESPONSE:
            await self.handle_predict_response(event)
        else:
            await super().process_single_event(event)

    async def handle_state_update(self, event: Event):
        if not isinstance(event.data, UniversalCode):
            logger.warning(f"{self.object_id}: Invalid SYSTEM_STATE_UPDATE data type: {type(event.data)}")
            return
        state = event.data.value if event.data.tag == Tags.DICT else {}
        timestamp = state.get("timestamp", time.time())
        metrics = state.get("metrics", {})
        self.insight_memory.append((timestamp, state, None, None))

        for metric in ["avg_llm_response_time_ms", "prediction_cache_hit_rate"]:
            value = metrics.get(metric, 0)
            self.pattern_detector[metric].append((timestamp, value))
            if len(self.pattern_detector[metric]) > 10:
                values = [v for _, v in self.pattern_detector[metric][-10:]]
                if max(values) - min(values) > 0.2 * max(values):
                    logger.info(f"{self.object_id}: Detected oscillation in {metric}: {min(values):.3f} to {max(values):.3f}")
                    self.insight_memory.append((timestamp, None, None, f"Oscillation in {metric}"))
                    await self.add_chat_message(f"Pattern detected: {metric} is oscillating, inspiring a new idea...")

        logger.debug(f"{self.object_id}: Stored state update at {timestamp:.2f}")

    async def handle_action_taken(self, event: Event):
        if not isinstance(event.data, UniversalCode):
            logger.warning(f"{self.object_id}: Invalid ACTION_TAKEN data type: {type(event.data)}")
            return
        action_data = event.data.value if event.data.tag == Tags.DICT else {}
        timestamp = event.data.metadata.get("timestamp", time.time())
        self.insight_memory.append((timestamp, None, action_data, None))
        logger.debug(f"{self.object_id}: Stored action {action_data.get('action_type', 'UNKNOWN')} at {timestamp:.2f}")

    async def handle_llm_response(self, event: Event):
        if not isinstance(event.data, UniversalCode):
            logger.warning(f"{self.object_id}: Invalid LLM_RESPONSE/USER_RESPONSE data type: {type(event.data)}")
            return
        metadata = event.data.metadata or {}
        response_text = event.data.value if event.data.tag == Tags.STRING else ""
        timestamp = metadata.get("response_timestamp", time.time())
        correlation_id = metadata.get("correlation_id")
        event_type = event.type

        if event_type == Actions.LLM_RESPONSE and correlation_id == self._pending_proposal_corr_id:
            self._proposal_response = response_text
            self._proposal_complete_event.set()
            logger.info(f"{self.object_id}: Received proposal response {correlation_id[:8]}")
            await self.add_chat_message(f"Crafted a new proposal from LLM wisdom: {response_text[:5000]}...")
        elif event_type == Actions.USER_RESPONSE and correlation_id == self._pending_proposal_corr_id:
            self._proposal_response = response_text
            self._proposal_complete_event.set()
            logger.info(f"{self.object_id}: Received user response {correlation_id[:8]}")
            await self.add_chat_message(f"User responded to my question: {response_text[:5000]}...")
        else:
            self.insight_memory.append((timestamp, None, None, response_text))
            logger.debug(f"{self.object_id}: Stored {'LLM' if event_type == Actions.LLM_RESPONSE else 'user'} response at {timestamp:.2f}")

    async def handle_predict_response(self, event: Event):
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            return
        response_data = event.data.value
        correlation_id = response_data.get("correlation_id")
        if correlation_id in self._pending_predictions:
            predicted_state = response_data.get("predicted_state")
            error = response_data.get("error")
            proposal_id = self._pending_predictions[correlation_id]
            if error:
                logger.error(f"{self.object_id}: Prediction error for proposal {proposal_id}: {error}")
                self._prediction_results[proposal_id] = 0.0
            else:
                score = self._evaluate_predicted_state(predicted_state)
                self._prediction_results[proposal_id] = score
                logger.debug(f"{self.object_id}: Prediction score for proposal {proposal_id}: {score:.3f}")
            self._pending_predictions.pop(correlation_id)
            if not self._pending_predictions:
                self._prediction_complete_event.set()

    def _evaluate_predicted_state(self, predicted_state: Dict) -> float:
        """Evaluate predicted state for proposal scoring with categorical states and continuation."""
        metrics = predicted_state.get("metrics", {})
        energy = predicted_state.get("energy", 0)
        cpu_percent = predicted_state.get("cpu_percent", 0)
        categorical_confidence = predicted_state.get("categorical_confidence", 0.5)
        continuation_prob = predicted_state.get("continuation_probability", 0.5)
        score = (
            metrics.get("prediction_cache_hit_rate", 0) * 0.25 +
            (10000 - energy) / 10000.0 * 0.2 +
            (100 - cpu_percent) / 100.0 * 0.2 +
            categorical_confidence * 0.25 +
            continuation_prob * 0.1
        )
        return score

    async def synthesize_insights(self) -> str:
        """Generates a rich summary of system trends, patterns, and analogies."""
        if not self.insight_memory:
            logger.info(f"{self.object_id}: No insights yet, using initial system state for synthesis")
            return "Initial system state: Ready to explore creative possibilities..."

        states, actions, feedback = [], [], []
        for entry in list(self.insight_memory)[-20:]:
            if not isinstance(entry, tuple) or len(entry) != 4:
                logger.warning(f"{self.object_id}: Malformed insight entry: {entry}")
                continue
            timestamp, state, action, fb = entry
            if state: states.append(state)
            if action: actions.append(action)
            if fb: feedback.append(fb)

        summary = ["Creative Synthesis Insights:"]
        if states:
            latest_state = states[-1]
            metrics = latest_state.get("metrics", {})
            hardware = latest_state.get("hardware", {})
            cpu_info = hardware.get("cpu", {})
            
            summary.append(f"- System State (t={latest_state.get('timestamp', 0):.2f}):")
            summary.append(f"  Energy: {latest_state.get('energy', 0):.0f}, CPU: {latest_state.get('cpu_percent', 0):.1f}%")
            if cpu_info:
                summary.append(f"  P-Cores: {cpu_info.get('performance_cores_percent', 0):.1f}%, E-Cores: {cpu_info.get('efficiency_cores_percent', 0):.1f}%")
            summary.append(f"  LLM Response: {metrics.get('avg_llm_response_time_ms', 0):.0f}ms, Cache: {metrics.get('prediction_cache_hit_rate', 0):.3f}")
            for metric, values in self.pattern_detector.items():
                if values: summary.append(f"  Pattern in {metric}: {values[-1][1]:.3f} (recent)")
        if actions:
            action_types = [a.get("action_type", "UNKNOWN") for a in actions[-5:]]
            summary.append(f"- Recent Actions: {', '.join(action_types)}")
        if feedback:
            feedback_snippet = feedback[-1][:100] + ("..." if len(feedback[-1]) > 100 else "")
            summary.append(f"- Recent Feedback: {feedback_snippet}")

        speculative = f"- What-If Scenario: What if {random.choice(['bubble count doubled', 'energy dropped to 100', 'LLM response time halved'])}?"
        summary.append(speculative)

        analogy_domain = random.choice(self.analogy_domains)
        analogy = f"- Analogy: System as a {analogy_domain} (e.g., bubbles as {analogy_domain} entities collaborating)"
        summary.append(analogy)

        return "\n".join(summary)

    async def ask_user_question(self, question: str) -> Optional[str]:
        """Poses a question to the user and awaits response."""
        correlation_id = str(uuid.uuid4())
        query_uc = UniversalCode(
            Tags.STRING,
            question,
            description="Question to user",
            metadata={"correlation_id": correlation_id, "response_to": self.object_id}
        )
        query_event = Event(
            type=Actions.USER_QUERY,
            data=query_uc,
            origin=self.object_id,
            priority=3
        )
        self._pending_proposal_corr_id = correlation_id
        self._proposal_complete_event.clear()
        self._proposal_response = None
        await self.context.dispatch_event(query_event)
        logger.info(f"{self.object_id}: Sent USER_QUERY {correlation_id[:8]}: {question[:5000]}...")
        try:
            await asyncio.wait_for(self._proposal_complete_event.wait(), timeout=60.0)
            response = self._proposal_response
            logger.info(f"{self.object_id}: Received USER_RESPONSE {correlation_id[:8]}: {response[:5000]}...")
            return response
        except asyncio.TimeoutError:
            logger.warning(f"{self.object_id}: Timeout waiting for USER_RESPONSE {correlation_id[:8]}")
            return None
        finally:
            self._pending_proposal_corr_id = None

    async def generate_creative_proposal(self) -> Optional[Dict]:
        """Queries the LLM for a highly novel proposal using analogies and speculation."""
        if not self.resource_manager or not await self.resource_manager.consume_resource('energy', 5.0):
            logger.warning(f"{self.object_id}: Insufficient energy for novel proposal generation")
            await self.add_chat_message("Pausing creativity due to low energy...")
            return None
        if self.resource_manager.get_resource_level("cpu_percent") > 90:
            logger.warning(f"{self.object_id}: High CPU usage, skipping proposal generation")
            await self.add_chat_message("System too busy, holding off on new ideas...")
            return None

        await self.add_chat_message("Crafting a novel idea inspired by the system's pulse...")
        insights = await self.synthesize_insights()
        analogy_domain = random.choice(self.analogy_domains)
        prompt = (
            f"You are a creative genius in the 'Bubbles' AI system, striving for AGI-like novelty.\n"
            f"Based on the following insights, propose a *highly novel* idea to enhance system intelligence, emergence, or adaptability.\n"
            f"Use an analogy to the domain of {analogy_domain} to inspire your idea.\n"
            f"Consider speculative scenarios (e.g., extreme states) and combine insights in unexpected ways.\n"
            f"Proposal types: ACTION (new action), QUESTION (exploratory query), EXPERIMENT (test hypothesis), RECONFIGURE (system tweak).\n"
            f"Respond with a JSON object: {{'proposal_type': str, 'payload': {{'description': str, 'narrative': str, 'action_type': str (if ACTION), 'question_text': str (if QUESTION), 'experiment_details': dict (if EXPERIMENT), 'reconfig_params': dict (if RECONFIGURE), 'expected_impact': {{metric: value}}}}}}\n"
            f"Include 'novelty_score' (0.0-1.0), 'confidence' (0.0-1.0), and a 'narrative' describing the vision.\n\n"
            f"Insights:\n{insights}\n\n"
            f"Proposal (JSON only):"
        )

        correlation_id = str(uuid.uuid4())
        metadata = {"correlation_id": correlation_id, "response_to": self.object_id, "target_bubble": "simplellm_bubble"}
        query_uc = UniversalCode(Tags.STRING, prompt, description="Novel proposal", metadata=metadata)
        query_event = Event(type=Actions.LLM_QUERY, data=query_uc, origin=self.object_id, priority=2)

        self._pending_proposal_corr_id = correlation_id
        self._proposal_complete_event.clear()
        self._proposal_response = None

        models = ["gemma:7b", "gemma:2b"]
        for model in models:
            query_uc.metadata["model"] = model
            try:
                logger.debug(f"{self.object_id}: Attempting LLM query with model {model}")
                await self.context.dispatch_event(query_event)
                await asyncio.wait_for(self._proposal_complete_event.wait(), timeout=180.0)
                break
            except asyncio.TimeoutError:
                logger.warning(f"{self.object_id}: Timed out waiting for novel proposal {correlation_id[:8]} with model {model}")
                if model == models[-1]:
                    await self.add_chat_message("Proposal timed out with all models, will try again soon...")
                    return None
                continue
            finally:
                self._pending_proposal_corr_id = None

        if not self._proposal_response:
            logger.error(f"{self.object_id}: No valid proposal response received")
            await self.add_chat_message("Failed to craft a proposal, retrying later...")
            return None

        proposal = robust_json_parse(self._proposal_response)
        if not isinstance(proposal, dict) or 'proposal_type' not in proposal:
            logger.error(f"{self.object_id}: Invalid proposal format: {self._proposal_response[:200]}")
            await self.add_chat_message("Invalid proposal format, will refine next attempt...")
            return None

        strategy = f"Analogy: {analogy_domain}, Novelty Weight: {self.novelty_weight}"
        self.creativity_journal.append((strategy, 0.5))
        await self.add_chat_message(f"New proposal ready: {proposal['proposal_type']} - {proposal['payload']['narrative'][:5000]}...")
        return proposal

    async def score_proposal(self, proposal: Dict) -> float:
        """Scores a proposal using DreamerV3Bubble predictions."""
        novelty = proposal.get("novelty_score", 0.7)
        confidence = proposal.get("confidence", 0.5)
        impact = sum(abs(v) for v in proposal.get("payload", {}).get("expected_impact", {}).values())
        feasibility = 1.0 if confidence > self.confidence_threshold else confidence
        type_bonus = 0.2 if proposal.get("proposal_type", "") in ["EXPERIMENT", "RECONFIGURE"] else 0.0

        correlation_id = str(uuid.uuid4())
        proposal_id = str(uuid.uuid4())
        current_state = self.resource_manager.get_current_system_state() if self.resource_manager else {}
        query_data = {
            "current_state": current_state,
            "action": {"action_type": proposal.get("proposal_type", "ACTION"), "payload": proposal.get("payload", {})}
        }
        query_uc = UniversalCode(Tags.DICT, query_data, description="Proposal prediction", metadata={"correlation_id": correlation_id})
        query_event = Event(type=Actions.PREDICT_STATE_QUERY, data=query_uc, origin=self.object_id, priority=2)

        self._pending_predictions = {correlation_id: proposal_id}
        self._prediction_results = {}
        self._prediction_complete_event = asyncio.Event()

        await self.context.dispatch_event(query_event)
        try:
            await asyncio.wait_for(self._prediction_complete_event.wait(), timeout=130.0)
            simulation_score = self._prediction_results.get(proposal_id, 0.0)
        except asyncio.TimeoutError:
            logger.warning(f"{self.object_id}: Timeout waiting for proposal prediction {correlation_id[:8]}")
            simulation_score = 0.0
        finally:
            self._pending_predictions.clear()

        score = (
            self.novelty_weight * (novelty + type_bonus + simulation_score) +
            (1 - self.novelty_weight) * feasibility
        ) * (1 + impact / 10)
        logger.debug(f"{self.object_id}: Scored proposal {proposal.get('proposal_type', 'UNKNOWN')}: {score:.3f} (Simulation: {simulation_score:.3f})")
        return score

    async def execute_proposal(self, proposal: Dict):
        """Executes or dispatches the novel proposal."""
        self.execution_count += 1
        proposal_type = proposal.get("proposal_type")
        payload = proposal.get("payload", {})
        narrative = payload.get("narrative", "A bold step toward emergence")
        logger.info(f"{self.object_id}: Executing proposal: {narrative[:100]}...")
        await self.add_chat_message(f"Bringing to life: {narrative[:100]}...")

        if proposal_type == "ACTION":
            action_type = payload.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
            try:
                action_enum = Actions[action_type]
                update_uc = UniversalCode(Tags.DICT, {"action_type": action_type, "payload": payload}, description=narrative)
                update_event = Event(type=Actions.ACTION_TAKEN, data=update_uc, origin=self.object_id, priority=4)
                await self.context.dispatch_event(update_event)
                logger.info(f"{self.object_id}: Executed novel action {action_type}")
                await self.add_chat_message(f"Action executed: {action_type}")
            except KeyError:
                update_uc = UniversalCode(Tags.STRING, payload.get("code_snippet", ""), description=payload.get("description", narrative))
                update_event = Event(type=Actions.CODE_UPDATE, data=update_uc, origin=self.object_id, priority=5)
                await self.context.dispatch_event(update_event)
                logger.info(f"{self.object_id}: Dispatched novel code update")
                await self.add_chat_message("Dispatched code update for creativity...")
        elif proposal_type == "QUESTION":
            question_text = payload.get("question_text", "")
            if question_text:
                query_uc = UniversalCode(Tags.STRING, question_text, description=narrative, metadata={"response_to": self.object_id, "target_bubble": "simplellm_bubble"})
                query_event = Event(type=Actions.LLM_QUERY, data=query_uc, origin=self.object_id, priority=2)
                await self.context.dispatch_event(query_event)
                logger.info(f"{self.object_id}: Posed novel question: {question_text[:100]}...")
                await self.add_chat_message(f"Posed question: {question_text[:100]}...")
        elif proposal_type == "EXPERIMENT":
            exp_details = payload.get("experiment_details", {})
            update_uc = UniversalCode(Tags.DICT, {"action_type": "EXPERIMENT", "payload": payload}, description=narrative)
            update_event = Event(type=Actions.ACTION_TAKEN, data=update_uc, origin=self.object_id, priority=4)
            await self.context.dispatch_event(update_event)
            logger.info(f"{self.object_id}: Initiated experiment: {exp_details.get('hypothesis', 'UNKNOWN')[:100]}...")
            await self.add_chat_message(f"Experiment launched: {exp_details.get('hypothesis', 'UNKNOWN')[:100]}...")
        elif proposal_type == "RECONFIGURE":
            reconfig_params = payload.get("reconfig_params", {})
            update_uc = UniversalCode(Tags.DICT, {"action_type": "RECONFIGURE", "payload": payload}, description=narrative)
            update_event = Event(type=Actions.ACTION_TAKEN, data=update_uc, origin=self.object_id, priority=4)
            await self.context.dispatch_event(update_event)
            logger.info(f"{self.object_id}: Reconfigured system with params: {list(reconfig_params.keys())}")
            await self.add_chat_message(f"System reconfigured with: {list(reconfig_params.keys())}")

        score = await self.score_proposal(proposal)
        self.proposal_history.append((proposal, score, "PENDING"))

    async def reflect_on_creativity(self):
        """Reflects on proposal frequency and resource impact, adjusting strategy, and asks user for input if needed."""
        current_time = time.time()
        if current_time - self.last_reflection_time < self.self_reflection_interval:
            return

        self.last_reflection_time = current_time
        proposal_count = len(self.proposal_history)
        window_hours = self.self_reflection_interval / 3600
        proposal_rate = proposal_count / window_hours if window_hours > 0 else 0
        energy_used = proposal_count * 5.0
        success_rate = sum(1 for _, _, outcome in self.proposal_history if outcome == "SUCCESS") / proposal_count if proposal_count > 0 else 0

        reflection = [f"Reflecting on my creativity (t={current_time:.2f}):"]
        reflection.append(f"- Proposals made: {proposal_count} ({proposal_rate:.2f}/hour)")
        reflection.append(f"- Energy used: {energy_used:.0f} units")
        reflection.append(f"- Success rate: {success_rate:.2f}")

        if proposal_rate > 10:
            self.proposal_interval = min(self.proposal_interval * 1.5, 600)
            reflection.append(f"- Too frequent, slowing to {self.proposal_interval:.0f}s intervals")
        elif proposal_rate < 2 and success_rate > 0.5:
            self.proposal_interval = max(self.proposal_interval / 1.5, 120)
            reflection.append(f"- Underactive, speeding to {self.proposal_interval:.0f}s intervals")

        if self.resource_manager:
            cpu_percent = self.resource_manager.get_resource_level("cpu_percent")
            if cpu_percent > 90:
                self.proposal_interval = min(self.proposal_interval * 1.2, 600)
                reflection.append(f"- High CPU ({cpu_percent:.1f}%), slowing proposals")

        user_input = None
        if success_rate < 0.3 or proposal_rate > 10:
            question = "My proposals aren't landing well or I'm generating too many. Any suggestions for improving my creativity?"
            user_input = await self.ask_user_question(question)
            if user_input:
                reflection.append(f"- User suggestion: {user_input[:100]}...")
                if "more novel" in user_input.lower():
                    self.novelty_weight = min(self.novelty_weight + 0.1, 0.9)
                    reflection.append(f"- Increased novelty_weight to {self.novelty_weight:.2f}")

        reflection_msg = "\n".join(reflection)
        logger.info(f"{self.object_id}: {reflection_msg}")
        await self.add_chat_message(f"As the system's creative spark:\n{reflection_msg}")

    async def autonomous_step(self):
        """Periodically generates and executes novel proposals, with reflection."""
        await super().autonomous_step()
        current_time = time.time()
        if current_time - self.last_proposal_time < self.proposal_interval:
            await asyncio.sleep(10)
            return

        self.last_proposal_time = current_time
        self.execution_count += 1
        logger.info(f"{self.object_id}: Initiating novel proposal cycle")
        await self.add_chat_message("Initiating a burst of creativity...")

        proposal = await self.generate_creative_proposal()
        if proposal and await self.score_proposal(proposal) > 0.5:
            await self.execute_proposal(proposal)
        else:
            logger.warning(f"{self.object_id}: No viable proposal generated or score too low")
            await self.add_chat_message("No viable idea this time, but inspiration will strike again...")

        await self.reflect_on_creativity()
        await asyncio.sleep(10)

# ============================================================================
# DynamicManagerBubble (unchanged from original)
# ============================================================================

class DynamicManagerBubble(UniversalBubble):
    """Manages dynamic spawning and destruction of bubbles."""
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.known_types = [
            'SimpleLLMBubble', 'FeedbackBubble', 'DreamerV3Bubble',
            'DynamicManagerBubble', 'MetaReasoningBubble',
            'CreativeSynthesisBubble', 'CompositeBubble', 'M4HardwareBubble'
        ]
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized. Known types: {self.known_types}")

    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, ACTION_TAKEN")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    async def process_single_event(self, event: Event):
        self.execution_count += 1
        if event.type == Actions.SYSTEM_STATE_UPDATE:
            await self.handle_state_update(event)
        elif event.type == Actions.ACTION_TAKEN:
            await self.handle_action_taken(event)
        else:
            await super().process_single_event(event)

    async def handle_state_update(self, event: Event):
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.warning(f"{self.object_id}: Invalid SYSTEM_STATE_UPDATE data: {event.data}")
            return
        state = event.data.value
        logger.debug(f"{self.object_id}: Processing SYSTEM_STATE_UPDATE")
        await self.evaluate_system_state(state)

    async def handle_action_taken(self, event: Event):
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            logger.warning(f"{self.object_id}: Invalid ACTION_TAKEN data: {event.data}")
            return
        action_data = event.data.value
        action_type = action_data.get("action_type")
        if action_type in [Actions.ACTION_TYPE_SPAWN_BUBBLE.name, Actions.ACTION_TYPE_DESTROY_BUBBLE.name]:
            logger.info(f"{self.object_id}: Handling bubble lifecycle action: {action_type}")
            await self.handle_bubble_lifecycle_action(action_data)

    async def evaluate_system_state(self, state: Dict):
        cpu_percent = state.get("cpu_percent", 0)
        num_bubbles = state.get("num_bubbles", 0)
        if cpu_percent > 85 and num_bubbles > 5:
            logger.info(f"{self.object_id}: High CPU ({cpu_percent:.1f}%) with {num_bubbles} bubbles. Considering bubble destruction.")
            await self.propose_bubble_destruction()
        elif cpu_percent < 50 and num_bubbles < 10:
            logger.info(f"{self.object_id}: Low CPU ({cpu_percent:.1f}%) with {num_bubbles} bubbles. Considering bubble spawning.")
            await self.propose_bubble_spawn()

    async def propose_bubble_spawn(self):
        bubble_type = random.choice(['SimpleLLMBubble', 'FeedbackBubble'])
        action_data = {
            "action_type": Actions.ACTION_TYPE_SPAWN_BUBBLE.name,
            "payload": {"bubble_type": bubble_type, "object_id": f"{bubble_type.lower()}_{uuid.uuid4().hex[:8]}"}
        }
        await self.publish_action_taken(Actions.ACTION_TYPE_SPAWN_BUBBLE, action_data)

    async def propose_bubble_destruction(self):
        bubbles = self.context.get_all_bubbles()
        if bubbles:
            bubble = random.choice(bubbles)
            action_data = {
                "action_type": Actions.ACTION_TYPE_DESTROY_BUBBLE.name,
                "payload": {"bubble_id": bubble.object_id}
            }
            await self.publish_action_taken(Actions.ACTION_TYPE_DESTROY_BUBBLE, action_data)

    async def handle_bubble_lifecycle_action(self, action_data: Dict):
        action_type = action_data.get("action_type")
        payload = action_data.get("payload", {})
        if action_type == Actions.ACTION_TYPE_SPAWN_BUBBLE.name:
            bubble_type = payload.get("bubble_type")
            object_id = payload.get("object_id")
            if bubble_type in self.known_types:
                try:
                    bubble_class = globals()[bubble_type]
                    bubble_instance = bubble_class(object_id=object_id, context=self.context)
                    logger.info(f"{self.object_id}: Spawned {bubble_type} with ID {object_id}")
                    await self.add_chat_message(f"Spawned new bubble: {object_id}")
                except Exception as e:
                    logger.error(f"{self.object_id}: Failed to spawn {bubble_type}: {e}", exc_info=True)
        elif action_type == Actions.ACTION_TYPE_DESTROY_BUBBLE.name:
            bubble_id = payload.get("bubble_id")
            bubble = self.context.get_bubble(bubble_id)
            if bubble:
                try:
                    await bubble.self_destruct()
                    logger.info(f"{self.object_id}: Destroyed bubble {bubble_id}")
                    await self.add_chat_message(f"Destroyed bubble: {bubble_id}")
                except Exception as e:
                    logger.error(f"{self.object_id}: Failed to destroy {bubble_id}: {e}", exc_info=True)

    async def autonomous_step(self):
        await super().autonomous_step()
        await asyncio.sleep(60)

# ============================================================================
# CompositeBubble (unchanged from original)
# ============================================================================

class CompositeBubble(UniversalBubble):
    """A container bubble that manages the lifecycle of other bubbles."""
    def __init__(self, object_id: str, context: SystemContext, sub_bubble_list: List[UniversalBubble], **kwargs):
        if not isinstance(sub_bubble_list, list) or not all(isinstance(b, UniversalBubble) for b in sub_bubble_list):
            raise TypeError("CompositeBubble requires a list of UniversalBubble instances.")
        self._sub_bubbles = sub_bubble_list
        super().__init__(object_id=object_id, context=context, **kwargs)
        logger.info(f"{self.object_id}: Initialized, managing {len(self._sub_bubbles)} sub-bubbles.")

    async def start_autonomous_loop(self):
        """Starts its own loop and the loops of all managed sub-bubbles."""
        logger.info(f"{self.object_id}: Starting composite loop and sub-bubble loops...")
        await super().start_autonomous_loop()
        start_tasks = []
        for bubble in self._sub_bubbles:
            if hasattr(bubble, 'start_autonomous_loop') and callable(bubble.start_autonomous_loop):
                start_tasks.append(asyncio.create_task(bubble.start_autonomous_loop()))
            else:
                logger.warning(f"{self.object_id}: Sub-bubble {bubble.object_id} missing start_autonomous_loop method.")
        results = await asyncio.gather(*start_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                bubble_id = self._sub_bubbles[i].object_id if i < len(self._sub_bubbles) else "unknown"
                logger.error(f"{self.object_id}: Error starting loop for sub-bubble {bubble_id}: {result}", exc_info=result)
        logger.info(f"{self.object_id}: All sub-bubble start routines initiated.")

    async def stop_autonomous_loop(self):
        """Stops its own loop and the loops of all managed sub-bubbles."""
        logger.info(f"{self.object_id}: Stopping composite loop and sub-bubble loops...")
        stop_tasks = []
        for bubble in self._sub_bubbles:
            if hasattr(bubble, 'stop_autonomous_loop') and callable(bubble.stop_autonomous_loop):
                stop_tasks.append(asyncio.create_task(bubble.stop_autonomous_loop()))
            else:
                logger.warning(f"{self.object_id}: Sub-bubble {bubble.object_id} missing stop_autonomous_loop method.")
        results = await asyncio.gather(*stop_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                bubble_id = self._sub_bubbles[i].object_id if i < len(self._sub_bubbles) else "unknown"
                logger.error(f"{self.object_id}: Error stopping loop for sub-bubble {bubble_id}: {result}", exc_info=result)
        await super().stop_autonomous_loop()
        logger.info(f"{self.object_id}: Composite loop and sub-bubble stop routines completed.")

    async def self_destruct(self):
        """Initiates self-destruct for itself and all managed sub-bubbles."""
        logger.info(f"{self.object_id}: Initiating composite self-destruct...")
        destruct_tasks = []
        sub_bubbles_copy = list(self._sub_bubbles)
        self._sub_bubbles.clear()
        for bubble in sub_bubbles_copy:
            if hasattr(bubble, 'self_destruct') and callable(bubble.self_destruct):
                destruct_tasks.append(asyncio.create_task(bubble.self_destruct()))
            else:
                logger.warning(f"{self.object_id}: Sub-bubble {bubble.object_id} missing self_destruct method.")
                if hasattr(bubble, 'stop_autonomous_loop'): await bubble.stop_autonomous_loop()
                self.context.unregister_bubble(bubble.object_id)
        results = await asyncio.gather(*destruct_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                bubble_id = sub_bubbles_copy[i].object_id if i < len(sub_bubbles_copy) else "unknown"
                logger.error(f"{self.object_id}: Error during self-destruct for sub-bubble {bubble_id}: {result}", exc_info=result)
        await super().self_destruct()
        logger.info(f"{self.object_id}: Composite self-destruct complete.")

    async def autonomous_step(self):
        """Composite bubble's own autonomous actions."""
        await super().autonomous_step()
        if self.execution_count % 100 == 0:
            active_subs = sum(1 for b in self._sub_bubbles if getattr(b, '_process_task', None) and not b._process_task.done())
            logger.debug(f"{self.object_id}: Managing {len(self._sub_bubbles)} sub-bubbles ({active_subs} active).")
        await asyncio.sleep(1)

# ============================================================================
# Helper Functions
# ============================================================================

def setup_llm_response_handler(context: SystemContext):
    """Sets up the event handler for displaying LLM responses and handling code execution prompts."""
    chat_box = context.chat_box
    handler_logger = logging.getLogger("ResponseHandler")

    async def display_llm_response(event: Event):
        """Handles LLM_RESPONSE events for display and potential code execution."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.STRING:
            return
        metadata = event.data.metadata or {}
        response_text = event.data.value
        origin_bubble = event.origin or "unknown"
        is_error = metadata.get("is_error", False)
        correlation_id = metadata.get("correlation_id")
        is_cached = metadata.get("cached_response", False)
        prefix = f"LLM Error ({origin_bubble})" if is_error else f"LLM ({origin_bubble})"
        if is_cached: prefix += " [Cached]"
        if correlation_id: prefix += f" [cid:{correlation_id[:6]}]"
        message = f"{prefix}: {response_text}"
        await chat_box.add_message(message)
        if not is_error:
            try:
                extracted_code = extract_code(response_text)
                if extracted_code:
                    await chat_box.add_message("--- Code Blocks Detected ---")
                    if os.environ.get("BUBBLES_ENABLE_CODE_EXEC", "0") != "1":
                        await chat_box.add_message("System: Code execution is disabled by configuration (BUBBLES_ENABLE_CODE_EXEC not set to 1).")
                        return
                    for i, code_block in enumerate(extracted_code, 1):
                        code_block_clean = code_block.strip()
                        await chat_box.add_message(f"\n--- Block {i} ---\n{code_block_clean}\n---------------")
                        confirm_prompt = f"\n### WARNING ### Execute Code Block {i}? Review carefully!\nType 'EXECUTE {i}' to run, anything else to skip: "
                        try:
                            loop = asyncio.get_running_loop()
                            confirm = await loop.run_in_executor(None, input, confirm_prompt)
                            if context.stop_event.is_set():
                                await chat_box.add_message("System: Shutdown signal received, skipping execution.")
                                break
                            if confirm.strip().upper() == f'EXECUTE {i}':
                                handler_logger.info(f"User confirmed execution for Block {i}.")
                                await chat_box.add_message(f"System: Executing Block {i}...")
                                execution_result = await loop.run_in_executor(None, execute_python_code, code_block_clean)
                                await chat_box.add_message(f"System: Block {i} Result:\n{execution_result}")
                            else:
                                handler_logger.info(f"User declined execution for Block {i}.")
                                await chat_box.add_message(f"System: Skipped Block {i}.")
                        except EOFError:
                            handler_logger.warning("EOF received during code execution confirmation.")
                            await chat_box.add_message("System: EOF received, skipping remaining code blocks.")
                            break
                        except Exception as e:
                            handler_logger.error(f"Error during code exec confirmation/execution for Block {i}: {e}", exc_info=True)
                            await chat_box.add_message(f"SysErr: Failed during Block {i} confirmation/execution: {e}")
                    await chat_box.add_message("--- End Code Block Check ---")
            except Exception as e:
                handler_logger.error(f"Error processing code blocks in LLM response: {e}", exc_info=True)
                await chat_box.add_message(f"SysErr: Failed processing code blocks in response: {e}")

    async def register_handler():
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.LLM_RESPONSE, display_llm_response)
            logger.info("LLM Response handler registered.")
        except Exception as e:
            logger.error(f"Failed to register LLM Response handler: {e}", exc_info=True)

    asyncio.create_task(register_handler())

async def handle_simple_chat(context: SystemContext):
    """Handles user input from the console, routing to CreativeSynthesisBubble or SimpleLLMBubble."""
    if not context.chat_box or not context.stop_event:
        logger.critical("Chat handler cannot start: ChatBox or StopEvent missing from context.")
        return
    chat_box = context.chat_box
    stop_event = context.stop_event
    loop = asyncio.get_running_loop()
    await chat_box.add_message("System: Chat handler started. Type 'quit' to exit.")
    await chat_box.add_message("Commands: quit, status, explain_code, trigger_action_cycle, trigger_state, add_transition <JSON>, llm <query>, or query for CreativeSynthesisBubble.")

    async def handle_llm_response(event: Event):
        if event.type == Actions.LLM_RESPONSE and isinstance(event.data, UniversalCode) and event.data.tag == Tags.STRING:
            try:
                response = event.data.value
                correlation_id = event.data.metadata.get("correlation_id", "unknown")
                origin = event.data.metadata.get("response_to", "unknown")
                await chat_box.add_message(f"LLM Response (cid: {correlation_id[:6]}, origin: {origin}): {response}")
                logger.debug(f"Displayed LLM response (cid: {correlation_id[:6]}): {response[:5000]}...")
            except Exception as e:
                logger.error(f"Error displaying LLM response: {e}", exc_info=True)

    async def handle_user_query(event: Event):
        if event.type == Actions.USER_QUERY and isinstance(event.data, UniversalCode) and event.data.tag == Tags.STRING:
            try:
                question = event.data.value
                correlation_id = event.data.metadata.get("correlation_id", "unknown")
                origin = event.origin
                await chat_box.add_message(f"Question from {origin} (cid: {correlation_id[:6]}): {question}")
                response = await loop.run_in_executor(None, input, "Your response: ")
                if stop_event.is_set():
                    await chat_box.add_message("System: Shutdown signal received, response discarded.")
                    return
                response_uc = UniversalCode(
                    Tags.STRING,
                    response,
                    description="User response",
                    metadata={"correlation_id": correlation_id, "response_to": origin, "response_timestamp": time.time()}
                )
                response_event = Event(
                    type=Actions.USER_RESPONSE,
                    data=response_uc,
                    origin="user_chat",
                    priority=3
                )
                await context.dispatch_event(response_event)
                logger.debug(f"Chat: Dispatched USER_RESPONSE (cid: {correlation_id[:6]})")
            except Exception as e:
                logger.error(f"Error handling USER_QUERY: {e}", exc_info=True)

    try:
        await EventService.subscribe(Actions.LLM_RESPONSE, handle_llm_response)
        await EventService.subscribe(Actions.USER_QUERY, handle_user_query)
        logger.debug("Chat handler subscribed to LLM_RESPONSE and USER_QUERY events")
    except Exception as e:
        logger.error(f"Failed to subscribe to events: {e}", exc_info=True)
        await chat_box.add_message(f"System Error: Failed to initialize chat handler: {e}")
        return

    last_query_time = 0
    query_interval = 2.0
    last_trigger_time = 0
    trigger_cooldown = 30.0

    while not stop_event.is_set():
        try:
            current_time = time.time()
            if current_time - last_query_time < query_interval:
                await asyncio.sleep(query_interval - (current_time - last_query_time))
                continue
            try:
                user_input = await loop.run_in_executor(None, input, f"You (time: {time.strftime('%H:%M:%S')}): ")
            except EOFError:
                logger.warning("Chat: EOF detected during input, retrying...")
                await asyncio.sleep(1)
                continue
            logger.debug(f"Chat: Received input: {user_input}")
            last_query_time = current_time
            user_input = user_input.strip()
            if stop_event.is_set():
                break
            if not user_input:
                continue
            input_lower = user_input.lower()
            origin = "user_chat"
            dispatch_llm_query = False
            llm_prompt = None
            metadata = {"correlation_id": str(uuid.uuid4()), "response_to": "user_chat"}

            if input_lower == 'quit':
                await chat_box.add_message("System: Quit command received. Initiating shutdown...")
                stop_event.set()
                break
            elif input_lower == 'trigger_action_cycle':
                if current_time - last_trigger_time < trigger_cooldown:
                    await chat_box.add_message(f"System: Trigger action cycle on cooldown (wait {trigger_cooldown - (current_time - last_trigger_time):.1f}s).")
                    continue
                last_trigger_time = current_time
                await chat_box.add_message(f"System: Manually triggering MetaReasoningBubble action cycle...")
                await trigger_strategic_analysis(context, reason="manual_trigger")
                await chat_box.add_message(f"System: Strategic analysis triggered.")
            elif input_lower == 'trigger_state':
                logger.debug(f"Chat: Processing trigger_state command")
                if context.resource_manager:
                    try:
                        await context.resource_manager.trigger_state_update()
                        await chat_box.add_message("System: Manually triggered SYSTEM_STATE_UPDATE")
                        logger.info("Chat: Successfully triggered SYSTEM_STATE_UPDATE")
                    except Exception as e:
                        await chat_box.add_message(f"System Error: Failed to trigger state update: {e}")
                        logger.error(f"Chat: Failed to trigger state update: {e}", exc_info=True)
                else:
                    await chat_box.add_message("System Error: ResourceManager unavailable")
                    logger.error("Chat: ResourceManager unavailable for trigger_state")
            elif input_lower == 'hw_status':
                # Show hardware status
                m4_bubble = context.get_bubble("m4_hardware_bubble")
                if m4_bubble and hasattr(m4_bubble, 'get_hardware_status'):
                    hw_status = m4_bubble.get_hardware_status()
                    await chat_box.add_message(json.dumps(hw_status, indent=2))
                else:
                    await chat_box.add_message("System Error: M4 Hardware bubble not found")
            elif input_lower in ['report', 'status']:
                await chat_box.add_message("System: Generating system status report...")
                if context.resource_manager:
                    state = context.resource_manager.get_current_system_state()
                    status_report = json.dumps(state, indent=2, default=str)
                    await chat_box.add_message(f"Current State:\n{status_report}")
                else:
                    await chat_box.add_message("System Error: ResourceManager unavailable.")
            elif input_lower == 'explain_code':
                script_path = os.path.abspath(__file__)
                await chat_box.add_message(f"System: Reading code from {os.path.basename(script_path)}...")
                try:
                    with open(script_path, 'r', encoding='utf-8') as f:
                        code_content = await loop.run_in_executor(None, f.read)
                    logger.info(f"Read {len(code_content)} chars from {script_path}")
                    MAX_CODE_LEN = 15000
                    truncated_code = code_content[:MAX_CODE_LEN] if len(code_content) > MAX_CODE_LEN else code_content
                    msg = f"Sending {'TRUNCATED ' if len(code_content) > MAX_CODE_LEN else ''}code ({len(truncated_code)} chars) to CreativeSynthesisBubble..."
                    await chat_box.add_message(f"System: {msg}")
                    llm_prompt = (
                        f"You are the CreativeSynthesisBubble in the 'Bubbles' AI system. "
                        f"Explain the high-level architecture, components (like Context, Bubbles - DreamerV3, MetaReasoning, etc.), "
                        f"event flow, and purpose of this Python 'Bubbles' system code:\n```python\n{truncated_code}\n```\nExplanation:"
                    )
                    dispatch_llm_query = True
                    metadata["target_bubble"] = "creativesynthesis_bubble"
                except Exception as e:
                    logger.error(f"Failed to read source code for explanation: {e}", exc_info=True)
                    await chat_box.add_message(f"SysErr: Failed to read code file: {e}")
            elif input_lower.startswith('add_transition '):
                try:
                    transition_json = user_input[len('add_transition '):]
                    transition_data = json.loads(transition_json)
                    required_keys = ['state', 'action', 'reward', 'next_state']
                    if not all(k in transition_data for k in required_keys):
                        await chat_box.add_message("System: Invalid transition format. Must include state, action, reward, next_state.")
                        continue
                    dreamer_bubble = context.get_bubble("dreamerv3_bubble")
                    if dreamer_bubble and hasattr(dreamer_bubble, 'add_ad_hoc_transition'):
                        await dreamer_bubble.add_ad_hoc_transition(transition_data)
                        await chat_box.add_message("System: Ad hoc transition added to DreamerV3Bubble replay buffer.")
                    else:
                        await chat_box.add_message("System Error: DreamerV3Bubble not found or does not support ad hoc transitions.")
                except json.JSONDecodeError as e:
                    await chat_box.add_message(f"System: Invalid JSON format for transition: {e}")
                except Exception as e:
                    await chat_box.add_message(f"System Error: Failed to add transition: {e}")
            elif input_lower.startswith('llm '):
                query_text = user_input[4:].strip()
                if not query_text:
                    await chat_box.add_message("System: LLM query cannot be empty. Use 'llm <query>'.")
                    continue
                await chat_box.add_message(f"System: Sending query to SimpleLLMBubble: {query_text[:5000]}...")
                llm_prompt = f"User query: {query_text}\nRespond directly to the user's query."
                dispatch_llm_query = True
                metadata["target_bubble"] = "simplellm_bubble"
            else:
                creative_bubble = context.get_bubble("creativesynthesis_bubble")
                if not creative_bubble:
                    await chat_box.add_message("System Error: CreativeSynthesisBubble not found.")
                    continue
                await chat_box.add_message(f"System: Sending query to CreativeSynthesisBubble: {user_input[:5000]}...")
                sys_ctx = (
                    "You are the CreativeSynthesisBubble in the 'Bubbles' AI system, acting as the overseer. "
                    "You weave novel ideas, propose actions, and interact directly with the user. "
                    "Respond to the user's query with a creative or insightful answer, possibly suggesting a proposal or action."
                )
                llm_prompt = f"{sys_ctx}\n\nUser query: {user_input}"
                if await chat_box.is_duplicate_query(llm_prompt):
                    await chat_box.add_message("System: Ignoring duplicate query within cooldown period.")
                    continue
                dispatch_llm_query = True
                metadata["target_bubble"] = "creativesynthesis_bubble"

            if dispatch_llm_query and llm_prompt:
                query_uc = UniversalCode(Tags.STRING, llm_prompt, description=f"Query from {origin}", metadata=metadata)
                query_event = Event(type=Actions.LLM_QUERY, data=query_uc, origin=origin, priority=3)
                await context.dispatch_event(query_event)
                logger.debug(f"Chat: Dispatched LLM_QUERY (cid:{metadata['correlation_id'][:6]}, target:{metadata['target_bubble']})")

        except asyncio.CancelledError:
            logger.info("Chat handler task cancelled.")
            break
        except Exception as e:
            logger.error(f"Chat Handler Error: {e}", exc_info=True)
            try:
                await chat_box.add_message(f"System Error in Chat Handler: {e}")
            except Exception as chat_err:
                print(f"CRITICAL CHATBOX ERROR: {chat_err}", file=sys.stderr)
            await asyncio.sleep(1)

    logger.info("Chat handler loop finished.")

# ============================================================================
# Main Function with M4 Hardware Integration
# ============================================================================

async def main_test():
    """Main function with real M4 hardware monitoring integration."""
    logger.info("Starting Bubbles system with REAL M4 Hardware Monitoring...")
    try:
        # Initialize SystemContext and core components
        context = SystemContext()
        context.chat_box = ChatBox()  # Initialize chat_box early
        chat_box = context.chat_box
        await chat_box.add_message("System: Initializing Bubbles Network with M4 Hardware...")
        await chat_box.add_message("============================================================")
        await chat_box.add_message("✅ REAL M4 HARDWARE MONITORING ENABLED")
        await chat_box.add_message("============================================================")

        # Initialize EventDispatcher before ResourceManager
        context.event_dispatcher = EventDispatcher(context)
        logger.info("EventDispatcher initialized successfully")

        # Initialize ResourceManager with real hardware support
        context.resource_manager = RealHardwareResourceManager(context)
        logger.info("RealHardwareResourceManager initialized successfully")

        # Initialize web server
        context.initialize_web_server()
        logger.info("Web server started successfully")

        # Create M4 Hardware Bubble FIRST
        logger.info("Creating M4 Hardware Bubble...")
        m4_config = {
            'sudo_password': os.environ.get('SUDO_PASSWORD', 'SChool123!'),  # Set via environment variable
            'monitoring': {
                'interval_seconds': 2.0,
                'adaptive_sampling': True
            },
            'features': {
                'enable_real_metrics': True
            }
        }
        
        if M4_HARDWARE_AVAILABLE:
            m4_bubble = M4HardwareBubble(
                object_id="m4_hardware_bubble",
                context=context,
                hardware_config=m4_config
            )
            
            # Connect M4 bubble to ResourceManager
            context.resource_manager.set_m4_bubble(m4_bubble)
            
            # Start M4 monitoring
            asyncio.create_task(m4_bubble.start_autonomous_loop())
            
            await chat_box.add_message("✅ M4 Hardware Monitoring: ACTIVE")
            await chat_box.add_message(f"   - Apple Silicon: {m4_bubble.m4_monitor.monitor.is_apple_silicon}")
            await chat_box.add_message(f"   - Sudo Access: {m4_bubble.m4_monitor.monitor.has_sudo}")
            
            # Wait for first hardware metrics
            await asyncio.sleep(3)
            
            # Show initial hardware status
            hw_status = m4_bubble.get_hardware_status()
            if 'current_metrics' in hw_status:
                metrics = hw_status['current_metrics']
                await chat_box.add_message(f"📊 Initial M4 Status:")
                await chat_box.add_message(f"   CPU: {metrics['cpu']['total_usage_percent']:.1f}% (P:{metrics['cpu']['performance_cores_percent']:.1f}% E:{metrics['cpu']['efficiency_cores_percent']:.1f}%)")
                await chat_box.add_message(f"   Memory: {metrics['memory']['usage_percent']:.1f}%")
                await chat_box.add_message(f"   Power: {metrics['power']['estimated_total_watts']:.1f}W")
        else:
            await chat_box.add_message("⚠️  M4 Hardware Monitoring: NOT AVAILABLE (falling back to psutil)")
            m4_bubble = None

        # Initialize other bubbles
        bubbles = [
            ("simplellm_bubble", SimpleLLMBubble, {}),
            ("feedback_bubble", FeedbackBubble, {}),
            ("creativesynthesis_bubble", CreativeSynthesisBubble, {"proposal_interval": 900.0}),
            ("dreamerv3_bubble", DreamerV3Bubble, {}),
            ("metareasoning_bubble", MetaReasoningBubble, {"cycle_interval": 1800.0}),
            ("dynamicmanager_bubble", DynamicManagerBubble, {}),
        ]
        
        sub_bubble_list = []
        for bubble_id, bubble_class, kwargs in bubbles:
            try:
                bubble = bubble_class(object_id=bubble_id, context=context, **kwargs)
                sub_bubble_list.append(bubble)
                logger.info(f"Instantiated and registered bubble {bubble_class.__name__} with ID {bubble_id}")
            except Exception as e:
                logger.error(f"Failed to initialize bubble {bubble_id}: {e}", exc_info=True)
                await chat_box.add_message(f"Error: Failed to initialize {bubble_id}: {e}")

        # Initialize CompositeBubble with sub_bubble_list
        composite_bubble = CompositeBubble(
            object_id="composite_bubble",
            context=context,
            sub_bubble_list=sub_bubble_list
        )
        await composite_bubble.start_autonomous_loop()  # Starts sub-bubble loops
        logger.info("System: Initialized CompositeBubble with sub-bubbles")

        # Setup LLM response handler
        setup_llm_response_handler(context)

        # Load external data for DreamerV3Bubble
        dreamer_bubble = context.get_bubble("dreamerv3_bubble")
        if dreamer_bubble and hasattr(dreamer_bubble, 'load_external_data'):
            try:
                await dreamer_bubble.load_external_data("external_data.json")
                logger.info("Successfully loaded external data for DreamerV3Bubble")
            except Exception as e:
                logger.error(f"Failed to load external data for DreamerV3Bubble: {e}", exc_info=True)

        # Start chat handler
        await chat_box.add_message("System: Bubbles Network Initialized. Starting autonomous loops...")
        await chat_box.add_message("Commands: quit, status, hw_status, explain_code, trigger_action_cycle, trigger_state, add_transition <JSON>, llm <query>, <query>")
        chat_task = asyncio.create_task(handle_simple_chat(context), name="ChatHandlerTask")
        logger.info("Chat handler task started successfully")

        # Verify we're using real hardware
        if M4_HARDWARE_AVAILABLE and m4_bubble:
            await asyncio.sleep(2)  # Wait for some metrics
            state = context.resource_manager.get_current_system_state()
            if state.get('source') == 'M4HardwareBubble':
                await chat_box.add_message("✅ Verified: Using REAL M4 hardware metrics")
            else:
                await chat_box.add_message(f"⚠️  Warning: Using {state.get('source', 'unknown')} for metrics")

        # Wait for tasks
        tasks_to_wait = [
            chat_task,
            context.web_server_task,
            composite_bubble._process_task,
        ]
        
        if M4_HARDWARE_AVAILABLE and m4_bubble:
            tasks_to_wait.append(m4_bubble._process_task)
        
        await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    except Exception as e:
        logger.critical(f"Failed to run main_test: {e}", exc_info=True)
        if 'chat_box' in locals():
            await chat_box.add_message(f"CRITICAL: Failed to run Bubbles system: {e}")
        raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Show startup message
    print("\n" + "="*60)
    print("🚀 BUBBLES AGENT 9 - M4 HARDWARE EDITION")
    print("="*60)
    print("This version includes:")
    print("✅ Real M4 hardware metrics (CPU, GPU, Neural Engine)")
    print("✅ Complete removal of simulated CPU/energy")
    print("✅ Power consumption monitoring")
    print("✅ Thermal state tracking")
    print("\nStarting system...")
    print("="*60 + "\n")
    

