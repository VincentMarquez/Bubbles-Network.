# bubbles_core.py - PHASE 1 COMPLETE: Circuit Breaker + Specialized Analysis System
# Contains critical fixes for memory management, LLM flooding, and system stability
# INTEGRATED: Circuit breaker, cognitive bloat analysis, specialized roles, flood control
# FIXED: Removed ALL fake psutil CPU data - using REAL M4 hardware metrics only!

import asyncio
import aiohttp
import json
import time
import logging
import os
import sys
import struct
import ast
from enum import Enum
from collections import deque, defaultdict
from typing import Dict, Any, Callable, List, Optional, Union, Tuple, Coroutine, Type
import re
import hashlib
from io import StringIO
from contextlib import redirect_stdout, nullcontext
import importlib.util
import types
import uuid
import difflib
import logging.handlers
import uuid
import weakref
import math
import random
import psutil
import statistics # For median calculation
from fastapi import FastAPI
import uvicorn
import collections # Added explicitly for allowed_modules
from dataclasses import dataclass

# --- EMERGENCY MEMORY FIX ---
# Set MPS memory environment variable to prevent out-of-memory errors
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
print(f"🔧 EMERGENCY FIX: Set PYTORCH_MPS_HIGH_WATERMARK_RATIO={os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO')}")

# --- BUBBLE MANAGEMENT CONSTANTS ---
MAX_CONCURRENT_BUBBLES = 30  # Reduced from 25 for stability
MEMORY_THRESHOLD_PAUSE = 95  # Pause bubbles at 95% memory
MEMORY_THRESHOLD_CRITICAL = 98  # Force cleanup at 98%

# --- Logging Configuration ---
log_formatter = logging.Formatter(
    '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "filename": "%(filename)s", "lineno": "%(lineno)d", "message": "%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Use a unique name for the log file to avoid conflicts if running multiple instances
log_file_name = f'bubbles_log_{int(time.time())}.json'
log_file_handler = logging.handlers.RotatingFileHandler(log_file_name, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
log_file_handler.setFormatter(log_formatter)
log_console_handler = logging.StreamHandler(sys.stdout)
log_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'))

root_logger = logging.getLogger()
root_logger.handlers.clear() # Clear existing handlers if any
root_logger.setLevel(logging.INFO) # Set root level (can be overridden by specific loggers)
root_logger.addHandler(log_file_handler)
root_logger.addHandler(log_console_handler)

# Ensure logs are flushed on exit
def shutdown_logging():
    logging.shutdown()
import atexit
atexit.register(shutdown_logging)

# Get a logger for this specific module
logger = logging.getLogger(__name__) # Use module name
logger.info(f"Logging initialized. Log file: {log_file_name}")
logger.info(f"🔧 PHASE 1 COMPLETE: Circuit Breaker + Specialized Analysis + Emergency fixes")

# --- LLM Configuration ---
OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST", "http://10.0.0.XXXX") # Default to localhost
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma:7b")
FALLBACK_MODEL = os.environ.get("OLLAMA_FALLBACK_MODEL", "gemma:2b") # Allow fallback config
API_ENDPOINT = f"{OLLAMA_HOST_URL}/api/generate"
REQUEST_TIMEOUT = 108545 # Total timeout for the request in seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5 # Base delay in seconds for retries

# --- Regexes ---
CODE_BLOCK_REGEX = re.compile(r"```(?:[a-zA-Z0-9_.-]*)?\s*([\s\S]*?)\s*```", re.IGNORECASE) # Handles optional language tag and whitespace
JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", re.IGNORECASE) # Handles JSON objects or arrays in blocks
FALLBACK_JSON_REGEX = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])") # Fallback for bare JSON object or array

# --- Custom Exceptions ---
class InvalidTagError(ValueError): pass
class LLMCallError(Exception): pass
class CodeExecutionError(Exception): pass
class CodeValidationError(Exception): pass
class PredictionError(Exception): pass

# --- Enums ---
class Tags(Enum):
    """Defines data types for UniversalCode values."""
    INTEGER = 0x01
    FLOAT = 0x02
    STRING = 0x03
    BOOLEAN = 0x04
    BINARY = 0x05
    NULL = 0x06
    DICT = 0x17
    LIST = 0x18

class Actions(Enum):
    """Defines event types and potential system actions."""
    # Core Events
    LLM_QUERY = "LLM_QUERY"                 # Request to LLM
    LLM_RESPONSE = "LLM_RESPONSE"           # Response from LLM
    API_CALL = "API_CALL"                   # Generic external API call (future use)
    SYSTEM_STATE_UPDATE = "SYSTEM_STATE_UPDATE" # Periodic state broadcast
    CODE_UPDATE = "CODE_UPDATE"             # Proposal/notification of code change
    EVENT_PUBLISHED = "EVENT_PUBLISHED"     # Meta-event (future use)
    HA_CONTROL = "HA_CONTROL"
    OVERSEER_CONTROL = "OVERSEER_CONTROL"
    OVERSEER_REPORT = "OVERSEER_REPORT"
    TUNING_UPDATE = "TUNING_UPDATE"
    USER_QUERY = "USER_QUERY"
    USER_RESPONSE = "USER_RESPONSE"
    USER_PROMPT = "USER_PROMPT"
    API_RESPONSE = "API_RESPONSE"
    BUBBLE_SCRIPT_EXECUTED = "BUBBLE_SCRIPT_EXECUTED"
    SCRIPT_EXECUTED = "SCRIPT_EXECUTED"
    SCRIPT_FAILED = "SCRIPT_FAILED"
    SCRIPT_SHARED = "SCRIPT_SHARED"
    CREATE_BUBBLE = "CREATE_BUBBLE"
    # ATENGINE # Consciousness analysis events
    ATENGINE_STATE_UPDATE = "ATENGINE_STATE_UPDATE"
    CONSCIOUSNESS_OBSERVATION = "CONSCIOUSNESS_OBSERVATION"
    CONSCIOUSNESS_REFLECTION = "CONSCIOUSNESS_REFLECTION"
    META_REFLECTION_REQUEST = "META_REFLECTION_REQUEST"
    RECURSIVE_SYNTHESIS = "RECURSIVE_SYNTHESIS"
    CONSCIOUSNESS_EMERGENCE = "CONSCIOUSNESS_EMERGENCE"
    PAWN_ALGORITHM = "SPAWN_ALGORITHM"
    USER_HELP_REQUEST = "USER_HELP_REQUEST"
    USER_HELP_RESPONSE = "USER_HELP_RESPONSE"
    RESOURCE_UPDATE = "RESOURCE_UPDATE"
    PATTERN_DISCOVERED = "PATTERN_DISCOVERED"
    ERROR_PATTERN_DETECTED = "ERROR_PATTERN_DETECTED"
    ALGORITHM_PERFORMANCE = "ALGORITHM_PERFORMANCE"
    META_LEARNING_UPDATE = "META_LEARNING_UPDATE"
    CURRICULUM_STAGE_CHANGE = "CURRICULUM_STAGE_CHANGE"
    ERROR_REPORT = "ERROR_REPORT"
    QML_PREDICTION = "QML_PREDICTION"
    SERIALIZE_OBJECT = "SERIALIZE_OBJECT"
    DESERIALIZE_OBJECT = "DESERIALIZE_OBJECT"
    SERIALIZATION_RESULT = "SERIALIZATION_RESULT"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"
    SPAWN_ALGORITHM = "SPAWN_ALGORITHM"
    QML_RESULT = "QML_RESULT"
    HARDWARE_ALERT = "HARDWARE_ALERT"
    HARDWARE_HEALTH_CHECK = "HARDWARE_HEALTH_CHECK"
    APEP_REFINEMENT_COMPLETE = "APEP_REFINEMENT_COMPLETE"
    APEP_STATUS_REQUEST = "APEP_STATUS_REQUEST"
    PERFORMANCE_METRIC = "PERFORMANCE_METRIC"
    META_KNOWLEDGE_UPDATE = "META_KNOWLEDGE_UPDATE"
    PATTERN_EXPLORATION = "PATTERN_EXPLORATION"
    KNOWLEDGE_TRANSFER = "KNOWLEDGE_TRANSFER"
    WORLD_MODEL_PREDICTION = "WORLD_MODEL_PREDICTION"
    GET_STATUS = "GET_STATUS"
    STRATEGIC_ANALYSIS = "STRATEGIC_ANALYSIS"
    BUBBLE_ERROR = "BUBBLE_ERROR"
    WARNING_EVENT = "WARNING_EVENT"
    CORRELATION_WARNING = "CORRELATION_WARNING"
    PERFORMANCE_WARNING = "PERFORMANCE_WARNING"
    MEMORY_WARNING = "MEMORY_WARNING"
    PATTERN_WARNING = "PATTERN_WARNING"
    CODE_SMELL_WARNING = "CODE_SMELL_WARNING"
    
    
    
    

    # Multi-LLM events
    MULTI_LLM_QUERY = "MULTI_LLM_QUERY"
    MULTI_LLM_RESPONSE = "MULTI_LLM_RESPONSE"

    # New events for QiskitBubble
    QUANTUM_RESULT = "QUANTUM_RESULT"
    QML_REQUEST = "QML_REQUEST"
    BATCH_REQUEST = "BATCH_REQUEST"

    # New events for LongTermMemoryBubble
    MEMORY_STORE = "MEMORY_STORE"
    MEMORY_RETRIEVE = "MEMORY_RETRIEVE"

    # New events for LangGraphBubble
    LANGGRAPH_STATE_UPDATED = "LANGGRAPH_STATE_UPDATED"

    # New events for AutoGenBubble
    AUTOGEN_RESULT = "AUTOGEN_RESULT"

    # Resource/Data Management
    SET_DATA = "SET_DATA"                   # Request to store data (future use)
    GET_DATA = "GET_DATA"                   # Request to retrieve data (future use)
    ADD_ENERGY = "ADD_ENERGY"               # Add energy resource
    REDUCE_ENERGY = "REDUCE_ENERGY"         # Consume energy resource
    SHARE_RESOURCE = "SHARE_RESOURCE"       # Transfer resource (future use)
    SENSOR_DATA = "SENSOR_DATA"

    # Bubble Lifecycle / Meta Actions
    SELFDESTRUCT = "SELFDESTRUCT"           # Request for a bubble to terminate
    ACTION_TAKEN = "ACTION_TAKEN"           # Notification that a bubble performed an action

    # World Model / Prediction
    PREDICT_STATE_QUERY = "PREDICT_STATE_QUERY"     # Request state prediction
    PREDICT_STATE_RESPONSE = "PREDICT_STATE_RESPONSE" # Result of state prediction

    # Meta-Reasoning Action Types (used within ACTION_TAKEN payload)
    ACTION_TYPE_CODE_UPDATE = "ACTION_CODE_UPDATE"         # Intention to update code
    ACTION_TYPE_SELF_QUESTION = "ACTION_SELF_QUESTION"     # Intention to ask LLM a question
    ACTION_TYPE_SPAWN_BUBBLE = "ACTION_SPAWN_BUBBLE"       # Intention/action of spawning
    ACTION_TYPE_DESTROY_BUBBLE = "ACTION_DESTROY_BUBBLE"   # Intention/action of destroying
    ACTION_TYPE_NO_OP = "ACTION_NO_OP"                     # Intention/action to do nothing







# =============================================================================
# PHASE 1 ENHANCEMENT 1: Circuit Breaker for LLM Requests
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, rejecting requests  
    HALF_OPEN = "half_open"  # Testing if service recovered

class LLMCircuitBreaker:
    """Phase 1: Circuit breaker for LLM requests integrated into query_llm."""
    
    def __init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = 5
        self.recovery_timeout = 30.0
        self.half_open_max_calls = 3
        self.half_open_call_count = 0
        self.last_failure_time = 0
        self.response_times = deque(maxlen=100)
        
    def can_execute(self) -> bool:
        """Check if LLM requests can be executed."""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if current_time - self.last_failure_time >= self.recovery_timeout:
                logger.info("🔧 LLM Circuit breaker moving to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_call_count = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_call_count < self.half_open_max_calls
        return False
    
    def record_success(self, duration: float):
        """Record successful LLM call."""
        self.success_count += 1
        self.response_times.append(duration)
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_call_count += 1
            if self.half_open_call_count >= self.half_open_max_calls:
                logger.info("🔧 LLM Circuit breaker moving to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
    
    def record_failure(self):
        """Record failed LLM call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(f"🔧 LLM Circuit breaker OPENING ({self.failure_count} failures)")
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning("🔧 LLM Circuit breaker returning to OPEN")
            self.state = CircuitState.OPEN
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0
        )
        total_requests = self.success_count + self.failure_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        
        return {
            'state': self.state.value,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time * 1000,
            'recovery_timeout': self.recovery_timeout
        }

class LLMRequestController:
    """
    AGGRESSIVE FIX: Controls LLM request flooding from multiple bubbles.
    
    Features:
    - Rate limiting (max requests per minute)
    - Request deduplication (same prompt within time window)
    - Priority queuing (important requests first)
    - Bubble throttling (limit per-bubble request rate)
    """
    
    def __init__(self):
        # Rate limiting
        self.max_requests_per_minute = 1  # Adjust based on your LLM service capacity
        self.request_timestamps = deque(maxlen=100)
        
        # Deduplication
        self.recent_prompts: Dict[str, float] = {}  # prompt_hash -> timestamp
        self.dedup_window = 30.0  # seconds
        
        # Per-bubble throttling
        self.bubble_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.max_requests_per_bubble_per_minute = 5
        
        # Statistics
        self.requests_blocked = 0
        self.requests_deduplicated = 0
        self.requests_processed = 0
        
        logger.info("🚨 LLM Request Controller: AGGRESSIVE flood control active")
    
    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate hash for prompt deduplication."""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
    
    def _clean_old_data(self):
        """Clean old timestamps and prompts."""
        current_time = time.time()
        
        # Clean rate limit timestamps (older than 1 minute)
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        
        # Clean dedup prompts (older than dedup window)
        expired_prompts = [
            prompt_hash for prompt_hash, timestamp in self.recent_prompts.items()
            if current_time - timestamp > self.dedup_window
        ]
        for prompt_hash in expired_prompts:
            del self.recent_prompts[prompt_hash]
        
        # Clean bubble request timestamps
        for bubble_id in list(self.bubble_requests.keys()):
            bubble_queue = self.bubble_requests[bubble_id]
            while bubble_queue and current_time - bubble_queue[0] > 60:
                bubble_queue.popleft()
    
    def should_allow_request(self, prompt: str, origin_bubble: str, priority: int = 5) -> Dict[str, Any]:
        """
        Check if request should be allowed.
        
        Returns:
        - {"allowed": True} if request can proceed
        - {"allowed": False, "reason": "..."} if blocked
        """
        current_time = time.time()
        self._clean_old_data()
        
        # Check 1: Rate limiting (global)
        recent_requests = len([t for t in self.request_timestamps if current_time - t < 60])
        if recent_requests >= self.max_requests_per_minute:
            self.requests_blocked += 1
            return {
                "allowed": False, 
                "reason": f"Rate limit exceeded ({recent_requests}/{self.max_requests_per_minute} per minute)",
                "retry_after": 60
            }
        
        # Check 2: Per-bubble throttling
        bubble_recent = len([t for t in self.bubble_requests[origin_bubble] if current_time - t < 60])
        if bubble_recent >= self.max_requests_per_bubble_per_minute:
            self.requests_blocked += 1
            return {
                "allowed": False,
                "reason": f"Bubble {origin_bubble} throttled ({bubble_recent}/{self.max_requests_per_bubble_per_minute} per minute)",
                "retry_after": 60
            }
        
        # Check 3: Deduplication
        prompt_hash = self._get_prompt_hash(prompt)
        if prompt_hash in self.recent_prompts:
            time_since = current_time - self.recent_prompts[prompt_hash]
            if time_since < self.dedup_window:
                self.requests_deduplicated += 1
                return {
                    "allowed": False,
                    "reason": f"Duplicate prompt within {self.dedup_window}s window",
                    "cached_result": True,
                    "retry_after": max(0, self.dedup_window - time_since)
                }
        
        # Allow request
        self.request_timestamps.append(current_time)
        self.bubble_requests[origin_bubble].append(current_time)
        self.recent_prompts[prompt_hash] = current_time
        self.requests_processed += 1
        
        return {"allowed": True}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get request controller statistics."""
        total_requests = self.requests_processed + self.requests_blocked + self.requests_deduplicated
        
        return {
            "total_requests": total_requests,
            "requests_processed": self.requests_processed,
            "requests_blocked": self.requests_blocked,
            "requests_deduplicated": self.requests_deduplicated,
            "block_rate": self.requests_blocked / total_requests if total_requests > 0 else 0,
            "dedup_rate": self.requests_deduplicated / total_requests if total_requests > 0 else 0,
            "current_rate_limit": len(self.request_timestamps),
            "max_rate_limit": self.max_requests_per_minute,
            "active_bubbles": len(self.bubble_requests)
        }

# Global instances
_llm_circuit_breaker = LLMCircuitBreaker()
_llm_request_controller = LLMRequestController()

# =============================================================================
# PHASE 1 ENHANCEMENT 2: Specialized Analysis Roles System
# =============================================================================

class AnalysisRole(Enum):
    """Specialized roles for different types of analysis."""
    PRIMARY_COORDINATOR = "primary_coordinator"      # OverseerBubble - high-level decisions
    HARDWARE_SPECIALIST = "hardware_specialist"      # M4HardwareBubble - hardware issues
    PREDICTION_SPECIALIST = "prediction_specialist"  # QuantumOracleBubble - future implications
    USER_INTERFACE = "user_interface"               # SimpleLLMBubble - user interactions
    CREATIVE_ANALYST = "creative_analyst"           # CreativeSynthesisBubble - novel solutions
    META_REASONER = "meta_reasoner"                 # MetaReasoningBubble - system reasoning
    SECURITY_SPECIALIST = "security_specialist"     # SecurityBubble - safety/security
    PERFORMANCE_SPECIALIST = "performance_specialist" # PPOBubble - optimization
    KNOWLEDGE_SPECIALIST = "knowledge_specialist"   # RAGBubble - information retrieval

class EventCategory(Enum):
    """Categories of events for specialized routing."""
    HARDWARE_EVENT = "hardware"           # CPU, memory, thermal, power issues
    SYSTEM_CRISIS = "crisis"              # Critical system states requiring immediate action
    USER_INTERACTION = "user"             # User queries, requests, interactions
    PERFORMANCE_EVENT = "performance"     # Optimization, learning, adaptation
    PREDICTION_REQUEST = "prediction"     # Future state analysis, forecasting
    CREATIVE_TASK = "creative"            # Novel problem solving, innovation
    SECURITY_EVENT = "security"           # Safety, security, error conditions
    META_ANALYSIS = "meta"                # System reasoning, self-reflection
    ROUTINE_UPDATE = "routine"            # Regular status updates, low-priority
    KNOWLEDGE_QUERY = "knowledge"         # Information retrieval, fact-finding

@dataclass
class AnalysisRequest:
    """Structured request for specialized analysis."""
    event: 'Event'
    category: EventCategory
    urgency: str  # "low", "medium", "high", "critical"
    specialist_roles: List[AnalysisRole]
    requires_consensus: bool = False
    confidence_threshold: float = 0.7
    max_analysts: int = 1
    context_data: Dict[str, Any] = None

class EventAnalysisRouter:
    """
    Routes events to specialized analyst bubbles instead of broadcasting to all.
    
    This is the core system that eliminates your LLM request flooding.
    """
    
    def __init__(self, context: 'SystemContext'):
        self.context = context
        
        # Registry of available specialist bubbles
        self.specialists: Dict[AnalysisRole, List[str]] = defaultdict(list)
        
        # Analysis history and performance tracking
        self.analysis_history: Dict[str, Dict] = {}
        self.specialist_performance: Dict[AnalysisRole, Dict] = defaultdict(lambda: {
            'requests': 0, 'successes': 0, 'avg_confidence': 0.0, 'avg_response_time': 0.0
        })
        
        # Event classification patterns
        self.classification_patterns = self._build_classification_patterns()
        
        # Statistics
        self.requests_routed = 0
        self.redundancy_eliminated = 0
        self.consensus_requests = 0
        
        logger.info("🎯 EventAnalysisRouter: Specialized analysis roles active")
    
    def register_specialist(self, bubble_id: str, role: AnalysisRole):
        """Register a bubble as a specialist for specific types of analysis."""
        if bubble_id not in self.specialists[role]:
            self.specialists[role].append(bubble_id)
            logger.info(f"🎯 Registered {bubble_id} as {role.value} specialist")
    
    def auto_register_bubbles(self):
        """Automatically register bubbles based on their type."""
        bubble_role_mapping = {
            'OverseerBubble': AnalysisRole.PRIMARY_COORDINATOR,
            'M4HardwareBubble': AnalysisRole.HARDWARE_SPECIALIST,
            'QuantumOracleBubble': AnalysisRole.PREDICTION_SPECIALIST,
            'SimpleLLMBubble': AnalysisRole.USER_INTERFACE,
            'CreativeSynthesisBubble': AnalysisRole.CREATIVE_ANALYST,
            'MetaReasoningBubble': AnalysisRole.META_REASONER,
            'PPOBubble': AnalysisRole.PERFORMANCE_SPECIALIST,
            'FullEnhancedPPO': AnalysisRole.PERFORMANCE_SPECIALIST,
            'RAGBubble': AnalysisRole.KNOWLEDGE_SPECIALIST,
            'SecurityBubble': AnalysisRole.SECURITY_SPECIALIST
        }
        
        for bubble in self.context.get_all_bubbles():
            bubble_type = type(bubble).__name__
            if bubble_type in bubble_role_mapping:
                role = bubble_role_mapping[bubble_type]
                self.register_specialist(bubble.object_id, role)
    
    def _build_classification_patterns(self) -> Dict[EventCategory, List[str]]:
        """Build patterns for automatic event classification."""
        return {
            EventCategory.HARDWARE_EVENT: [
                'memory', 'cpu', 'thermal', 'temperature', 'power', 'battery',
                'neural_engine', 'gpu', 'throttle', 'hardware', 'performance_profile',
                'mps', 'utilization', 'constraint', 'swap', 'pressure'
            ],
            EventCategory.SYSTEM_CRISIS: [
                'critical', 'emergency', 'crisis', 'failure', 'error', 'crash',
                'timeout', 'overload', 'unavailable', 'threshold_exceeded'
            ],
            EventCategory.USER_INTERACTION: [
                'user', 'query', 'request', 'chat', 'input', 'command', 'help',
                'question', 'llm_query', 'user_prompt'
            ],
            EventCategory.PERFORMANCE_EVENT: [
                'optimization', 'training', 'learning', 'ppo', 'reward', 'policy',
                'algorithm', 'performance', 'efficiency', 'benchmark'
            ],
            EventCategory.PREDICTION_REQUEST: [
                'predict', 'forecast', 'future', 'trend', 'prophecy', 'oracle',
                'implications', 'consequences', 'outcome'
            ],
            EventCategory.CREATIVE_TASK: [
                'creative', 'novel', 'innovative', 'synthesis', 'brainstorm',
                'solution', 'idea', 'proposal', 'design'
            ],
            EventCategory.SECURITY_EVENT: [
                'security', 'safety', 'risk', 'threat', 'vulnerability', 'breach',
                'unauthorized', 'suspicious', 'malicious'
            ],
            EventCategory.META_ANALYSIS: [
                'reasoning', 'meta', 'reflection', 'analysis', 'introspection',
                'self', 'consciousness', 'thinking'
            ],
            EventCategory.KNOWLEDGE_QUERY: [
                'information', 'knowledge', 'fact', 'data', 'search', 'retrieval',
                'rag', 'documentation', 'reference'
            ]
        }
    
    def classify_event(self, event: 'Event') -> Tuple[EventCategory, str]:
        """
        Classify event into category and determine urgency.
        
        Returns: (category, urgency_level)
        """
        # Extract text content from event
        event_text = ""
        try:
            if hasattr(event, 'data') and hasattr(event.data, 'value'):
                if isinstance(event.data.value, str):
                    event_text = event.data.value.lower()
                elif isinstance(event.data.value, dict):
                    event_text = str(event.data.value).lower()
            
            # Add event type to text
            event_text += f" {event.type.name.lower()}"
            
        except Exception as e:
            logger.warning(f"Error extracting event text for classification: {e}")
            event_text = event.type.name.lower()
        
        # Classify by pattern matching
        category_scores = defaultdict(int)
        
        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                if pattern in event_text:
                    category_scores[category] += 1
        
        # Determine category (highest score wins)
        if category_scores:
            category = max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            category = EventCategory.ROUTINE_UPDATE
        
        # Determine urgency based on keywords and event type
        urgency = "medium"  # default
        
        if any(critical in event_text for critical in ['critical', 'emergency', 'crisis', 'failure']):
            urgency = "critical"
        elif any(high in event_text for high in ['high', 'urgent', 'important', 'warning']):
            urgency = "high"
        elif any(low in event_text for low in ['routine', 'normal', 'info', 'debug']):
            urgency = "low"
        
        # Event type-based urgency adjustments
        if event.type.name in ['SYSTEM_STATE_UPDATE', 'SENSOR_DATA']:
            urgency = "low"  # Usually routine
        elif event.type.name in ['USER_QUERY', 'LLM_QUERY']:
            urgency = max(urgency, "medium")  # User requests are at least medium
        
        return category, urgency
    
    def determine_specialists(self, category: EventCategory, urgency: str) -> AnalysisRequest:
        """
        Determine which specialists should analyze this event.
        
        This is where we eliminate redundancy - instead of 6 bubbles analyzing
        the same event, we select 1-2 specialists based on the event type.
        """
        
        # Base specialist assignment by category
        specialist_mapping = {
            EventCategory.HARDWARE_EVENT: [AnalysisRole.HARDWARE_SPECIALIST],
            EventCategory.SYSTEM_CRISIS: [AnalysisRole.PRIMARY_COORDINATOR, AnalysisRole.HARDWARE_SPECIALIST],
            EventCategory.USER_INTERACTION: [AnalysisRole.USER_INTERFACE],
            EventCategory.PERFORMANCE_EVENT: [AnalysisRole.PERFORMANCE_SPECIALIST],
            EventCategory.PREDICTION_REQUEST: [AnalysisRole.PREDICTION_SPECIALIST],
            EventCategory.CREATIVE_TASK: [AnalysisRole.CREATIVE_ANALYST],
            EventCategory.SECURITY_EVENT: [AnalysisRole.SECURITY_SPECIALIST, AnalysisRole.PRIMARY_COORDINATOR],
            EventCategory.META_ANALYSIS: [AnalysisRole.META_REASONER],
            EventCategory.KNOWLEDGE_QUERY: [AnalysisRole.KNOWLEDGE_SPECIALIST],
            EventCategory.ROUTINE_UPDATE: []  # No specialized analysis needed
        }
        
        base_specialists = specialist_mapping.get(category, [AnalysisRole.PRIMARY_COORDINATOR])
        
        # Adjust based on urgency
        if urgency == "critical":
            # Critical events get primary coordinator + specialist
            if AnalysisRole.PRIMARY_COORDINATOR not in base_specialists:
                base_specialists = [AnalysisRole.PRIMARY_COORDINATOR] + base_specialists
            requires_consensus = len(base_specialists) > 1
            max_analysts = min(len(base_specialists), 2)
        elif urgency == "high":
            # High urgency gets specialist + possible second opinion
            requires_consensus = len(base_specialists) > 1
            max_analysts = min(len(base_specialists), 2)
        else:
            # Medium/low urgency gets single specialist
            requires_consensus = False
            max_analysts = 1
            if len(base_specialists) > 1:
                base_specialists = base_specialists[:1]  # Just take the first specialist
        
        return AnalysisRequest(
            event=None,  # Will be set by caller
            category=category,
            urgency=urgency,
            specialist_roles=base_specialists,
            requires_consensus=requires_consensus,
            confidence_threshold=0.7 if urgency in ["high", "critical"] else 0.5,
            max_analysts=max_analysts
        )
    
    async def route_for_analysis(self, event: 'Event') -> Dict[str, Any]:
        """
        Main routing function - replaces broadcasting to all bubbles.
        
        Instead of 6 bubbles analyzing the same event, this intelligently
        routes to 1-2 specialists based on event type and urgency.
        """
        start_time = time.time()
        
        # Classify the event
        category, urgency = self.classify_event(event)
        
        # Determine specialist requirements
        analysis_req = self.determine_specialists(category, urgency)
        analysis_req.event = event
        
        # Check if any analysis is needed
        if not analysis_req.specialist_roles:
            logger.debug(f"🎯 No analysis needed for routine {category.value} event")
            self.redundancy_eliminated += 5  # Estimate of bubbles that would have analyzed
            return {
                "analysis_performed": False,
                "reason": "Routine event requires no specialized analysis",
                "category": category.value,
                "urgency": urgency
            }
        
        # Find available specialists
        available_specialists = []
        for role in analysis_req.specialist_roles:
            role_bubbles = self.specialists.get(role, [])
            active_bubbles = [bid for bid in role_bubbles if self.context.get_bubble(bid) is not None]
            if active_bubbles:
                # Select best performer or first available
                specialist_id = active_bubbles[0]  # Simplified selection
                available_specialists.append((role, specialist_id))
        
        if not available_specialists:
            logger.warning(f"🎯 No specialists available for {category.value} event")
            # Fallback to primary coordinator if available
            primary_bubbles = self.specialists.get(AnalysisRole.PRIMARY_COORDINATOR, [])
            if primary_bubbles:
                specialist_id = primary_bubbles[0]
                available_specialists = [(AnalysisRole.PRIMARY_COORDINATOR, specialist_id)]
        
        if not available_specialists:
            logger.error(f"🎯 No bubbles available for analysis of {category.value} event")
            return {
                "analysis_performed": False,
                "reason": "No specialist bubbles available",
                "category": category.value,
                "urgency": urgency
            }
        
        # Perform specialized analysis
        analysis_results = await self._execute_specialized_analysis(analysis_req, available_specialists)
        
        # Track statistics
        self.requests_routed += 1
        redundancy_saved = 6 - len(available_specialists)  # Estimate based on your current system
        self.redundancy_eliminated += redundancy_saved
        
        duration = time.time() - start_time
        
        logger.info(f"🎯 Routed {category.value} event to {len(available_specialists)} specialists "
                   f"(saved {redundancy_saved} redundant LLM calls) in {duration:.2f}s")
        
        return {
            "analysis_performed": True,
            "category": category.value,
            "urgency": urgency,
            "specialists_used": [role.value for role, _ in available_specialists],
            "redundancy_eliminated": redundancy_saved,
            "analysis_results": analysis_results,
            "duration": duration
        }
    
    async def _execute_specialized_analysis(self, analysis_req: AnalysisRequest, 
                                           specialists: List[Tuple[AnalysisRole, str]]) -> List[Dict]:
        """Execute the specialized analysis with selected bubbles."""
        results = []
        
        for role, bubble_id in specialists:
            try:
                bubble = self.context.get_bubble(bubble_id)
                if not bubble:
                    continue
                
                # Create specialized prompt based on role
                specialized_prompt = self._create_specialized_prompt(analysis_req, role)
                
                # Execute analysis with flood control
                result = await query_llm_with_flood_control(
                    prompt=specialized_prompt,
                    system_context=self.context,
                    origin_bubble=bubble_id,
                    priority=self._get_role_priority(role, analysis_req.urgency)
                )
                
                if not result.get("blocked") and result.get("response"):
                    # Track performance
                    perf = self.specialist_performance[role]
                    perf['requests'] += 1
                    if not result.get("error"):
                        perf['successes'] += 1
                    
                    results.append({
                        "role": role.value,
                        "bubble_id": bubble_id,
                        "analysis": result["response"],
                        "confidence": self._extract_confidence(result["response"]),
                        "duration": result.get("duration_s", 0)
                    })
                    
                    logger.debug(f"🎯 {role.value} analysis complete from {bubble_id}")
                else:
                    logger.warning(f"🎯 {role.value} analysis blocked/failed from {bubble_id}")
                    
            except Exception as e:
                logger.error(f"🎯 Error in specialized analysis for {role.value}: {e}")
        
        return results
    
    def _create_specialized_prompt(self, analysis_req: AnalysisRequest, role: AnalysisRole) -> str:
        """Create role-specific prompts instead of generic analysis."""
        
        event = analysis_req.event
        event_summary = self._summarize_event(event)
        
        # Role-specific prompt templates
        role_prompts = {
            AnalysisRole.HARDWARE_SPECIALIST: f"""
As the Hardware Specialist, analyze this system event for hardware-related implications:

Event: {event_summary}

Focus on:
- Hardware resource utilization (CPU, memory, thermal, power)
- Performance bottlenecks and constraints
- Hardware optimization recommendations
- Thermal/power management needs

Provide specific hardware-focused insights and actionable recommendations.
Response format: {{
    "hardware_status": "healthy|degraded|critical",
    "bottlenecks": ["list of identified bottlenecks"],
    "recommendations": ["specific hardware actions"],
    "confidence": 0.8
}}
""",
            
            AnalysisRole.PREDICTION_SPECIALIST: f"""
As the Prediction Specialist, forecast the implications and future outcomes of this event:

Event: {event_summary}

Focus on:
- Short-term and long-term system implications
- Trend analysis and pattern recognition
- Risk assessment and probability estimation
- Preventive measures and early warnings

Provide predictive insights about system evolution.
Response format: {{
    "prediction": "specific prediction about what will happen",
    "timeframe": "immediate|short_term|long_term",
    "probability": 0.8,
    "risk_level": "low|medium|high|critical",
    "confidence": 0.8
}}
""",
            
            AnalysisRole.PRIMARY_COORDINATOR: f"""
As the Primary Coordinator, determine the appropriate system response to this event:

Event: {event_summary}

Focus on:
- Overall system impact assessment
- Coordination of system-wide responses
- Priority and urgency evaluation
- Resource allocation decisions

Provide strategic guidance for system management.
Response format: {{
    "action_required": "none|monitor|intervene|emergency",
    "priority": "low|medium|high|critical",
    "system_impact": "description of overall impact",
    "recommended_actions": ["high-level actions"],
    "confidence": 0.8
}}
""",
            
            AnalysisRole.USER_INTERFACE: f"""
As the User Interface Specialist, handle this user interaction:

Event: {event_summary}

Focus on:
- Understanding user intent and needs
- Providing helpful and relevant responses
- User experience optimization
- Clear communication and explanation

Provide user-focused response and assistance.
Response format: {{
    "user_response": "direct response to user",
    "intent_understood": true/false,
    "additional_help_needed": true/false,
    "confidence": 0.8
}}
"""
        }
        
        # Get role-specific prompt or fallback to generic
        if role in role_prompts:
            return role_prompts[role]
        else:
            return f"""
As a {role.value.replace('_', ' ').title()} specialist, analyze this event:

Event: {event_summary}

Provide analysis specific to your area of expertise with confidence score.
"""
    
    def _summarize_event(self, event: 'Event') -> str:
        """Create a concise summary of the event for analysis."""
        try:
            summary = f"Type: {event.type.name}\n"
            summary += f"Origin: {event.origin}\n"
            
            if hasattr(event.data, 'value'):
                value = event.data.value
                if isinstance(value, dict):
                    # Summarize key fields
                    key_fields = ['action', 'status', 'error', 'message', 'severity']
                    relevant_data = {k: v for k, v in value.items() if k in key_fields}
                    if relevant_data:
                        summary += f"Data: {relevant_data}\n"
                    else:
                        # Include first few fields
                        sample_data = dict(list(value.items())[:3])
                        summary += f"Data: {sample_data}...\n"
                elif isinstance(value, str):
                    summary += f"Message: {value[:200]}...\n" if len(value) > 200 else f"Message: {value}\n"
                else:
                    summary += f"Data: {str(value)[:100]}...\n"
            
            return summary
        except Exception as e:
            return f"Type: {event.type.name}, Origin: {event.origin} (summary error: {e})"
    
    def _get_role_priority(self, role: AnalysisRole, urgency: str) -> int:
        """Get LLM request priority based on role and urgency."""
        base_priorities = {
            AnalysisRole.PRIMARY_COORDINATOR: 2,
            AnalysisRole.HARDWARE_SPECIALIST: 3,
            AnalysisRole.SECURITY_SPECIALIST: 2,
            AnalysisRole.PREDICTION_SPECIALIST: 4,
            AnalysisRole.USER_INTERFACE: 3,
            AnalysisRole.CREATIVE_ANALYST: 6,
            AnalysisRole.META_REASONER: 5,
            AnalysisRole.PERFORMANCE_SPECIALIST: 4,
            AnalysisRole.KNOWLEDGE_SPECIALIST: 5
        }
        
        base_priority = base_priorities.get(role, 5)
        
        # Adjust for urgency
        if urgency == "critical":
            return max(1, base_priority - 2)
        elif urgency == "high":
            return max(1, base_priority - 1)
        else:
            return base_priority
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from analysis response."""
        try:
            # Look for confidence in response
            confidence_match = re.search(r'"confidence":\s*([0-9.]+)', response)
            if confidence_match:
                return float(confidence_match.group(1))
            
            # Look for percentage confidence
            percent_match = re.search(r'confidence[:\s]*([0-9]+)%', response, re.IGNORECASE)
            if percent_match:
                return float(percent_match.group(1)) / 100
            
        except:
            pass
        
        return 0.7  # Default confidence
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about analysis routing."""
        total_specialists = sum(len(bubbles) for bubbles in self.specialists.values())
        
        return {
            "requests_routed": self.requests_routed,
            "redundancy_eliminated": self.redundancy_eliminated,
            "efficiency_gain": f"{self.redundancy_eliminated / max(1, self.requests_routed):.1f}x",
            "registered_specialists": dict(self.specialists),
            "total_specialists": total_specialists,
            "specialist_performance": dict(self.specialist_performance)
        }

# =============================================================================
# PHASE 1 ENHANCEMENT 3: Cognitive Bloat Analysis
# =============================================================================

@dataclass
class BubbleMetrics:
    """Metrics for cognitive bloat calculation."""
    bubble_id: str
    memory_usage_mb: float
    recent_event_count: int
    execution_count: int
    last_activity_time: float
    queue_size: int
    response_time_avg: float
    error_count: int
    creation_time: float
    bubble_type: str

class CognitiveBloatAnalyzer:
    """PHASE 1: Cognitive bloat analysis integrated into ResourceManager."""
    
    def __init__(self):
        self.bubble_metrics: Dict[str, BubbleMetrics] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.bloat_threshold = 2.0
        self.paused_for_bloat = 0
        self.memory_savings_mb = 0.0
        self.last_analysis_time = 0
        
    def calculate_cognitive_bloat(self, bubble_metrics: BubbleMetrics) -> float:
        """Calculate cognitive bloat score (higher = more bloated)."""
        current_time = time.time()
        
        # Core metric: memory per activity ratio
        if bubble_metrics.recent_event_count == 0:
            activity_ratio = bubble_metrics.memory_usage_mb * 10  # Heavy penalty for zero activity
        else:
            activity_ratio = bubble_metrics.memory_usage_mb / bubble_metrics.recent_event_count
        
        # Time-based penalty
        time_since_activity = current_time - bubble_metrics.last_activity_time
        activity_penalty = min(time_since_activity / 300, 2.0)  # Max 2x penalty after 5 minutes
        
        # Queue efficiency
        if bubble_metrics.recent_event_count > 0:
            queue_efficiency = bubble_metrics.queue_size / bubble_metrics.recent_event_count
        else:
            queue_efficiency = bubble_metrics.queue_size * 2
        
        # Error penalty
        error_penalty = 1.0 + (bubble_metrics.error_count * 0.1)
        
        # Age penalty
        bubble_age_hours = (current_time - bubble_metrics.creation_time) / 3600
        age_penalty = 1.0 + min(bubble_age_hours / 24, 0.5)
        
        # Response time penalty
        response_penalty = 1.0 + min(bubble_metrics.response_time_avg / 5.0, 1.0)
        
        # Type-based adjustment
        type_multiplier = self._get_type_bloat_multiplier(bubble_metrics.bubble_type)
        
        cognitive_bloat = (
            activity_ratio * 
            activity_penalty * 
            (1.0 + queue_efficiency * 0.2) * 
            error_penalty * 
            age_penalty * 
            response_penalty * 
            type_multiplier
        )
        
        return cognitive_bloat
    
    def _get_type_bloat_multiplier(self, bubble_type: str) -> float:
        """Get type-specific multiplier for cognitive bloat."""
        # Core infrastructure (protect from pausing)
        if any(core in bubble_type for core in ['ResourceManager', 'EventDispatcher', 'LLMManager', 'M4Hardware']):
            return 0.3
        
        # High-compute bubbles (some protection)
        if any(compute in bubble_type for compute in ['DreamerV3', 'QML', 'PPO', 'Oracle']):
            return 0.8
        
        # Standard bubbles
        return 1.0
    
    def _estimate_bubble_memory(self, bubble) -> float:
        """Estimate memory usage for a bubble using real M4 hardware data if available."""
        try:
            # Check if we have M4 hardware metrics
            if hasattr(self, '_m4_hardware_bubble') and self._m4_hardware_bubble:
                # Use real memory pressure data
                hw_metrics = self._m4_hardware_bubble.get_hardware_status()
                if hw_metrics and 'current_metrics' in hw_metrics:
                    # Scale memory estimate based on real memory pressure
                    memory_percent = hw_metrics['current_metrics'].get('memory', {}).get('usage_percent', 50)
                    scale_factor = memory_percent / 100.0
                else:
                    scale_factor = 0.5
            else:
                scale_factor = 0.5
            
            base_memory = 10.0 * scale_factor
            queue_memory = getattr(bubble, 'event_queue', object()).qsize() * 0.01 if hasattr(getattr(bubble, 'event_queue', None), 'qsize') else 0
            execution_memory = min(getattr(bubble, 'execution_count', 0) * 0.001, 5.0)
            
            # Type-specific estimates
            type_estimates = {
                'DreamerV3': 50.0, 'QML': 40.0, 'PPO': 45.0, 'Oracle': 25.0,
                'RAG': 30.0, 'LLM': 15.0, 'Hardware': 10.0, 'Overseer': 20.0
            }
            
            type_memory = 10.0
            for type_name, memory in type_estimates.items():
                if type_name in type(bubble).__name__:
                    type_memory = memory * scale_factor
                    break
            
            return base_memory + queue_memory + execution_memory + type_memory
        except:
            return 20.0
    
    async def collect_bubble_metrics(self, context: 'SystemContext') -> Dict[str, BubbleMetrics]:
        """Collect metrics for all active bubbles."""
        metrics = {}
        current_time = time.time()
        
        # Try to get M4 hardware bubble reference
        for bubble in context.get_all_bubbles():
            if 'M4Hardware' in type(bubble).__name__:
                self._m4_hardware_bubble = bubble
                break
        
        for bubble in context.get_all_bubbles():
            try:
                bubble_id = bubble.object_id
                bubble_type = type(bubble).__name__
                
                # Collect metrics
                memory_usage_mb = self._estimate_bubble_memory(bubble)
                recent_event_count = getattr(bubble, 'recent_event_count', 0)
                execution_count = getattr(bubble, 'execution_count', 0)
                last_activity_time = getattr(bubble, 'last_activity_time', current_time)
                queue_size = getattr(bubble.event_queue, 'qsize', lambda: 0)()
                error_count = getattr(bubble, 'error_count', 0)
                creation_time = getattr(bubble, 'creation_time', current_time)
                
                bubble_metrics = BubbleMetrics(
                    bubble_id=bubble_id,
                    memory_usage_mb=memory_usage_mb,
                    recent_event_count=recent_event_count,
                    execution_count=execution_count,
                    last_activity_time=last_activity_time,
                    queue_size=queue_size,
                    response_time_avg=0.1,  # Simplified
                    error_count=error_count,
                    creation_time=creation_time,
                    bubble_type=bubble_type
                )
                
                metrics[bubble_id] = bubble_metrics
                
                # Update history
                bloat_score = self.calculate_cognitive_bloat(bubble_metrics)
                self.performance_history[bubble_id].append({
                    'timestamp': current_time,
                    'bloat_score': bloat_score,
                    'memory_mb': memory_usage_mb
                })
                
            except Exception as e:
                logger.warning(f"Failed to collect metrics for {bubble.object_id}: {e}")
        
        self.bubble_metrics = metrics
        self.last_analysis_time = current_time
        return metrics
    
    async def identify_bloated_bubbles(self, context: 'SystemContext', target_count: int = 1) -> List[Tuple[str, float]]:
        """Identify most bloated bubbles for pausing."""
        metrics = await self.collect_bubble_metrics(context)
        
        if not metrics:
            return []
        
        # Calculate bloat scores
        bubble_scores = []
        for bubble_id, bubble_metrics in metrics.items():
            bloat_score = self.calculate_cognitive_bloat(bubble_metrics)
            bubble_scores.append((bubble_id, bloat_score, bubble_metrics))
        
        # Sort by bloat score (highest first)
        bubble_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out core infrastructure if possible
        filtered = []
        core_types = ['ResourceManager', 'EventDispatcher', 'LLMManager']
        
        for bubble_id, score, metrics in bubble_scores:
            is_core = any(core_type in metrics.bubble_type for core_type in core_types)
            if not is_core or len(filtered) < target_count:
                filtered.append((bubble_id, score))
        
        if filtered:
            logger.info(f"🔧 Cognitive bloat: Top candidate {filtered[0][0]} (score: {filtered[0][1]:.2f})")
        
        return filtered[:target_count]
    
    def get_bloat_report(self) -> Dict[str, Any]:
        """Generate bloat analysis report."""
        if not self.bubble_metrics:
            return {'error': 'No metrics available'}
        
        bloat_scores = []
        memory_usage = []
        
        for metrics in self.bubble_metrics.values():
            bloat_score = self.calculate_cognitive_bloat(metrics)
            bloat_scores.append(bloat_score)
            memory_usage.append(metrics.memory_usage_mb)
        
        if not bloat_scores:
            return {'error': 'No data available'}
        
        avg_bloat = sum(bloat_scores) / len(bloat_scores)
        total_memory = sum(memory_usage)
        
        bloated_bubbles = [
            (bubble_id, self.calculate_cognitive_bloat(metrics))
            for bubble_id, metrics in self.bubble_metrics.items()
            if self.calculate_cognitive_bloat(metrics) > self.bloat_threshold
        ]
        bloated_bubbles.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_bubbles': len(self.bubble_metrics),
            'average_bloat': avg_bloat,
            'total_estimated_memory_mb': total_memory,
            'bloated_bubbles': bloated_bubbles[:5],
            'paused_for_bloat': self.paused_for_bloat,
            'memory_savings_mb': self.memory_savings_mb
        }

# --- Forward Declarations ---
# These help type hinting when classes reference each other before full definition
class EventDispatcher: pass
class ResourceManager: pass
class UniversalBubble: pass
class ChatBox: pass
class SystemContext: pass

# --- Helper Functions ---
def extract_code(text: str) -> List[str]:
    """Extracts code blocks (```...```) from text."""
    if not isinstance(text, str): return []
    # Findall returns only the capturing group (the code inside)
    return CODE_BLOCK_REGEX.findall(text)

def robust_json_parse(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """Attempts to parse JSON from text, handling markdown blocks and potential surrounding text."""
    if not isinstance(text, str): return None
    text = text.strip()
    if not text: return None

    try:
        # 1. Prioritize JSON within markdown blocks
        match = JSON_BLOCK_REGEX.search(text)
        if match:
            json_str = match.group(1) # Group 1 contains the object or array
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e_inner:
                 logger.warning(f"JSON block found but failed to parse: {e_inner}. Content: {json_str[:100]}...")
                 # Fall through to try parsing the whole text or other methods

        # 2. Try parsing the whole string if it looks like JSON
        if (text.startswith('{') and text.endswith('}')) or \
           (text.startswith('[') and text.endswith(']')):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                 # Don't log here, fallback is expected
                 pass # Fall through to fallback regex

        # 3. Fallback: Find the first potential JSON object/array using regex
        match_fallback = FALLBACK_JSON_REGEX.search(text)
        if match_fallback:
            potential_json = match_fallback.group(1)
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError as e_fallback:
                logger.warning(f"Fallback JSON regex matched but failed to parse: {e_fallback}. Content: {potential_json[:100]}...")

        # 4. If no JSON structure is clearly identifiable
        logger.debug(f"Robust JSON parse: No valid JSON structure found in text: {text[:100]}...")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during robust JSON parse: {e}", exc_info=True)
        return None

def fuzzy_match(str1: str, str2: str, threshold: float = 0.9) -> bool:
    """Performs fuzzy string matching using SequenceMatcher."""
    if not str1 or not str2 or not isinstance(str1, str) or not isinstance(str2, str):
        return False
    # Quick length check to avoid unnecessary computation
    len1, len2 = len(str1), len(str2)
    # Allow slightly more difference based on threshold
    if abs(len1 - len2) > max(len1, len2) * (1 - threshold) * 1.5:
        return False
    return difflib.SequenceMatcher(None, str1, str2).ratio() >= threshold

# --- Binary Data Handling ---
class BinaryConverter:
    """Handles encoding and decoding of values based on Tags."""
    @staticmethod
    def encode_value(value: Any, tag: Tags) -> bytes:
        """Encodes a Python value into bytes based on the specified tag."""
        try:
            if tag == Tags.INTEGER:
                return struct.pack(">q", int(value)) # Use 64-bit signed integer
            elif tag == Tags.FLOAT:
                return struct.pack(">d", float(value)) # Use 64-bit float (double)
            elif tag == Tags.STRING:
                encoded = str(value).encode("utf-8")
                return struct.pack(">I", len(encoded)) + encoded # Use unsigned int for length
            elif tag == Tags.BOOLEAN:
                return bytes([1]) if value else bytes([0])
            elif tag == Tags.BINARY:
                # Ensure input is bytes, encode if not (e.g., memoryview)
                data = value if isinstance(value, bytes) else bytes(value)
                return struct.pack(">I", len(data)) + data
            elif tag == Tags.NULL:
                return b''
            elif tag in (Tags.DICT, Tags.LIST):
                # Use JSON, ensure UTF-8 and compact separators
                json_str = json.dumps(value, default=str, ensure_ascii=False, separators=(',', ':'))
                encoded = json_str.encode("utf-8")
                return struct.pack(">I", len(encoded)) + encoded
            else:
                raise InvalidTagError(f"Unsupported tag for encoding: {tag}")
        except (TypeError, ValueError, struct.error) as e:
            logger.error(f"Encoding error (tag: {tag.name}, value type: {type(value)}): {e}. Falling back to string.", exc_info=False)
            try:
                encoded = str(value).encode("utf-8")
                return struct.pack(">I", len(encoded)) + encoded
            except Exception as fallback_e:
                 logger.critical(f"CRITICAL: Fallback encoding failed for {tag.name} / {type(value)}: {fallback_e}")
                 return b''
        except Exception as e:
             logger.error(f"Unexpected encoding error (tag: {tag.name}): {e}", exc_info=True)
             return b''

    @staticmethod
    def decode_value(binary_data: bytes, tag: Tags) -> Any:
        """Decodes bytes into a Python value based on the specified tag."""
        if not isinstance(binary_data, bytes):
             logger.error(f"Decode error: Input data is not bytes (type: {type(binary_data)})")
             return f"DECODE_ERROR: Input not bytes"

        try:
            if not binary_data and tag != Tags.NULL:
                return { # Default values for empty data
                    Tags.INTEGER: 0, Tags.FLOAT: 0.0, Tags.STRING: "",
                    Tags.BOOLEAN: False, Tags.BINARY: b'', Tags.DICT: {}, Tags.LIST: []
                }.get(tag)

            if tag == Tags.INTEGER:
                if len(binary_data) < 8: raise ValueError("Insufficient data for INTEGER (q)")
                return struct.unpack(">q", binary_data[:8])[0]
            elif tag == Tags.FLOAT:
                if len(binary_data) < 8: raise ValueError("Insufficient data for FLOAT (d)")
                return struct.unpack(">d", binary_data[:8])[0]
            elif tag == Tags.STRING:
                if len(binary_data) < 4: raise ValueError("Insufficient data for STRING length")
                length = struct.unpack(">I", binary_data[:4])[0]
                if len(binary_data) < 4 + length: raise ValueError(f"Insufficient data for STRING content (expected {length}, got {len(binary_data)-4})")
                return binary_data[4:4 + length].decode("utf-8")
            elif tag == Tags.BOOLEAN:
                if not binary_data: raise ValueError("Insufficient data for BOOLEAN")
                return bool(binary_data[0])
            elif tag == Tags.BINARY:
                 if len(binary_data) < 4: raise ValueError("Insufficient data for BINARY length")
                 length = struct.unpack(">I", binary_data[:4])[0]
                 if len(binary_data) < 4 + length: raise ValueError(f"Insufficient data for BINARY content (expected {length}, got {len(binary_data)-4})")
                 return binary_data[4:4 + length]
            elif tag == Tags.NULL:
                return None
            elif tag in (Tags.DICT, Tags.LIST):
                if len(binary_data) < 4: raise ValueError("Insufficient data for DICT/LIST length")
                length = struct.unpack(">I", binary_data[:4])[0]
                if len(binary_data) < 4 + length: raise ValueError(f"Insufficient data for DICT/LIST content (expected {length}, got {len(binary_data)-4})")
                return json.loads(binary_data[4:4 + length].decode("utf-8"))
            else:
                raise InvalidTagError(f"Unsupported tag for decoding: {tag}")
        except (struct.error, json.JSONDecodeError, UnicodeDecodeError, IndexError, ValueError) as e:
            logger.error(f"Decoding error (tag: {tag.name}, data length: {len(binary_data)}): {e}", exc_info=False)
            return f"DECODE_ERROR: {type(e).__name__} - {e}"
        except Exception as e:
            logger.error(f"Unexpected decoding error (tag: {tag.name}): {e}", exc_info=True)
            return f"UNEXPECTED_DECODE_ERROR: {e}"

# --- Core Data Structure ---
class UniversalCode:
    """A standardized wrapper for data passed within events."""
    def __init__(self,
                 tag: Tags,
                 value: Any,
                 binary_data: Optional[bytes] = None,
                 description: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a UniversalCode object.

        Args:
            tag: The data type tag from the Tags enum.
            value: The actual Python value.
            binary_data: Optional pre-computed binary representation.
            description: Optional human-readable description.
            metadata: Optional dictionary for extra information.

        Raises:
            InvalidTagError: If tag is not a Tags enum member.
            ValueError: If value type is incompatible with tag or value/metadata is not JSON-serializable.
        """
        # Validate that tag is a member of the Tags enum
        if not isinstance(tag, Tags):
            raise InvalidTagError(f"Invalid tag type: {type(tag)}. Must be a Tags enum member.")

        # Validate value type against tag
        expected_types = {
            Tags.INTEGER: int,
            Tags.FLOAT: float,
            Tags.STRING: str,
            Tags.BOOLEAN: bool,
            Tags.BINARY: (bytes, bytearray, memoryview),
            Tags.NULL: type(None),
            Tags.DICT: dict,
            Tags.LIST: list
        }
        if tag in expected_types and not isinstance(value, expected_types[tag]):
            raise ValueError(f"Value {value} of type {type(value)} is incompatible with tag {tag.name}")

        # Validate JSON-serializability for DICT and LIST to ensure stable hashing
        if tag in (Tags.DICT, Tags.LIST):
            try:
                json.dumps(value, sort_keys=True)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Value for {tag.name} is not JSON-serializable: {e}")

        # Validate metadata is JSON-serializable
        if metadata is not None:
            try:
                json.dumps(metadata)
            except (TypeError, ValueError) as e:
                logger.warning(f"Metadata is not JSON-serializable: {e}")
                metadata = {}

        # Assign core attributes with defaults for optional fields
        self.tag = tag
        self.value = value
        self.description = description if description is not None else ""
        self.metadata = metadata if metadata is not None else {}

        # Validate binary_data type
        if binary_data is not None and not isinstance(binary_data, bytes):
            logger.warning("Invalid binary_data type; using empty bytes")
            binary_data = b''

        # Generate or use binary data
        try:
            self.binary_data = BinaryConverter.encode_value(value, tag)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to generate binary data for UC (tag={tag.name}, value={value}, type={type(value)}): {e}", exc_info=True)
            self.binary_data = binary_data if binary_data is not None else b''

    def __repr__(self) -> str:
        """Provides a concise string representation."""
        try:
            v_repr = repr(self.value)
            v_truncated = v_repr[:47] + '...' if len(v_repr) > 50 else v_repr
            meta_repr = f", meta={self.metadata}" if self.metadata else ""
            desc_repr = f", desc='{self.description}'" if self.description else ""
            return f"UC(tag={self.tag.name}, val={v_truncated}{desc_repr}{meta_repr})"
        except (TypeError, ValueError) as e:
            logger.debug(f"Error in UC.__repr__ for tag={self.tag.name}: {e}")
            return f"UC(tag={self.tag.name}, ERROR_IN_REPR)"

    def __eq__(self, other):
        """Checks equality based on tag, value, and potentially metadata."""
        if not isinstance(other, UniversalCode):
            return NotImplemented
        try:
            return (self.tag == other.tag and
                    self.value == other.value and
                    self.description == other.description and
                    self.metadata == other.metadata)
        except TypeError as e:
            logger.debug(f"Equality check failed for UC(tag={self.tag.name}): {e}")
            return False

    def __hash__(self):
        """Allows UC objects to be used in sets/dicts with stable hashing."""
        try:
            meta_hashable = tuple(sorted(self.metadata.items())) if self.metadata else ""
            if self.tag in (Tags.DICT, Tags.LIST):
                value_hash = hash(json.dumps(self.value, sort_keys=True))
            else:
                value_hash = hash(self.value)
            return hash((self.tag, value_hash, self.description, meta_hashable))
        except (TypeError, ValueError) as e:
            logger.error(f"Unexpected error in hash for UC(tag={self.tag.name}): {e}", exc_info=True)
            raise

# --- Event Class ---
class Event:
    """Represents an event to be published and handled within the system."""
    def __init__(self,
                 type: Union[Tags, Actions],
                 data: UniversalCode,
                 origin: Optional[str] = None,
                 priority: int = 0,
                 metadata: Optional[Dict[str, Any]] = None,
                 correlation_id: Optional[str] = None):
        """
        Initializes an Event.

        Args:
            type: The type of the event (from Tags or Actions enums).
            data: The data payload wrapped in a UniversalCode object.
            origin: Optional identifier of the bubble/component publishing the event.
            priority: Integer priority (higher value means higher priority).
            metadata: Optional dictionary for extra information.
            correlation_id: Optional ID for request-response correlation.
        """
        if not isinstance(type, (Tags, Actions)):
            raise TypeError(f"Event type must be Tags or Actions enum member, got {type(type)}")
        if not isinstance(data, UniversalCode):
            logger.warning(f"Event created with non-UniversalCode data (type: {type(data)}). Auto-wrapping as STRING UC.")
            data = UniversalCode(Tags.STRING, str(data), description="Auto-wrapped data")

        self.type = type
        self.data = data
        self.origin = origin if origin is not None else "unknown"
        self.event_id = str(uuid.uuid4())
        self.timestamp = time.time()
        self.priority = priority
        self.metadata = metadata if metadata is not None else {}
        self.correlation_id = correlation_id

    def __repr__(self) -> str:
        """Provides a concise string representation of the event."""
        corr_str = f", corr_id={self.correlation_id[:8]}" if self.correlation_id else ""
        return (f"Event(id={self.event_id[:8]}, type={self.type.name}, "
                f"origin={self.origin}, prio={self.priority}{corr_str}, data={repr(self.data)})")

    def __lt__(self, other):
        """Comparison for priority queues (higher priority comes first)."""
        if not isinstance(other, Event):
            return NotImplemented
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority value means "less than" for min-heap
        return self.timestamp < other.timestamp

# --- EventService Class ---
class EventService:
    """
    A singleton-like class managing event subscriptions and publishing.
    Uses class methods for global access.
    """
    _listeners: Dict[Union[Tags, Actions], List[Callable[[Event], Coroutine[Any, Any, None]]]] = defaultdict(list)
    _lock = asyncio.Lock()
    _batch_queue: List[Event] = []
    _batch_interval: float = 0.02 # Process events every 20ms
    _batch_task: Optional[asyncio.Task] = None
    _is_processing = False # Prevent re-entrant processing

    @classmethod
    async def subscribe(cls, event_type: Union[Tags, Actions], listener: Callable[[Event], Coroutine[Any, Any, None]]):
        """Subscribes an asynchronous listener function to an event type."""
        if not asyncio.iscoroutinefunction(listener):
            logger.error(f"EventService: Listener {getattr(listener, '__qualname__', 'unknown')} for {event_type.name} is not a coroutine. Subscription ignored.")
            return

        async with cls._lock:
            listeners_list = cls._listeners[event_type]
            if listener not in listeners_list:
                listeners_list.append(listener)
                logger.debug(f"EventService: Listener {getattr(listener, '__qualname__', 'unknown')} subscribed to {event_type.name}")
            else:
                 logger.warning(f"EventService: Listener {getattr(listener, '__qualname__', 'unknown')} already subscribed to {event_type.name}.")

    @classmethod
    async def publish(cls, event: Event):
        """Adds an event to the batch queue for processing."""
        if not isinstance(event, Event):
             logger.error(f"EventService: Attempted to publish non-Event object: {type(event)}")
             return
        try:
            async with cls._lock:
                cls._batch_queue.append(event)

            # Start processing task if not already running or finished
            # Use a flag to prevent multiple concurrent processing tasks
            if not cls._is_processing and (cls._batch_task is None or cls._batch_task.done()):
                cls._batch_task = asyncio.create_task(cls._process_batch())

        except Exception as e:
            logger.error(f"EventService: Error queuing event {event}: {e}", exc_info=True)

    @classmethod
    async def _process_batch(cls):
        """Processes the queued events."""
        if cls._is_processing:
            logger.debug("EventService: Batch processing already running, skipping re-entrant call.")
            return
        cls._is_processing = True

        try:
            await asyncio.sleep(cls._batch_interval) # Allow more events to batch

            async with cls._lock:
                if not cls._batch_queue:
                    cls._is_processing = False
                    return # Nothing to process
                # Sort by priority (desc) then timestamp (asc)
                events_to_process = sorted(cls._batch_queue, key=lambda e: (-e.priority, e.timestamp))
                cls._batch_queue.clear()

            logger.debug(f"EventService: Processing batch of {len(events_to_process)} events.")

            for event in events_to_process:
                try:
                    listeners_to_call = cls._listeners.get(event.type, [])
                    if listeners_to_call:
                        tasks = [asyncio.create_task(cls._safe_call_listener(listener, event)) for listener in listeners_to_call]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for i, result in enumerate(results):
                            if isinstance(result, Exception):
                                listener_name = getattr(listeners_to_call[i], '__qualname__', 'unknown')
                                logger.error(f"EventService: Exception in listener '{listener_name}' handling {event}: {result}", exc_info=result)
                except Exception as e:
                    logger.error(f"EventService: Error processing event {event}: {e}", exc_info=True)
        finally:
            cls._is_processing = False # Release the processing flag
            # Check if more events arrived while processing and schedule again if needed
            async with cls._lock:
                 if cls._batch_queue and (cls._batch_task is None or cls._batch_task.done()):
                      if not cls._is_processing: # Check flag again before scheduling
                           cls._batch_task = asyncio.create_task(cls._process_batch())

    @staticmethod
    async def _safe_call_listener(listener: Callable[[Event], Coroutine[Any, Any, None]], event: Event):
        """Executes a listener coroutine, handling potential errors."""
        listener_name = getattr(listener, '__qualname__', 'unknown')
        try:
            # logger.debug(f"EventService: Calling listener '{listener_name}' for event {event.event_id[:8]}")
            await listener(event)
        except asyncio.CancelledError:
            logger.warning(f"EventService: Listener '{listener_name}' cancelled while handling {event}.")
            raise
        except Exception as e:
            logger.error(f"EventService: Unhandled error in listener '{listener_name}' for event {event}: {e}", exc_info=True)

# --- Enhanced Response Cache with Emergency Fixes ---
class ResponseCache:
    """Enhanced caches LLM prompts and responses with emergency fixes for 0.000 hit rate."""
    def __init__(self, max_size: int = 1000, fuzzy_threshold: float = 0.85):
        self.max_size = max_size
        # Emergency cache implementation with simple dict fallback
        self.cache_dict: Dict[str, Tuple[str, str, float]] = {}  # hash -> (prompt, response, timestamp)
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_enabled = True
        self._lock = asyncio.Lock()
        self.hit_count = 0
        self.miss_count = 0
        logger.info(f"🔧 EMERGENCY: Enhanced ResponseCache (max_size={max_size}, fuzzy={fuzzy_threshold})")

    def _hash_prompt(self, prompt: str) -> str:
        """Generates a SHA256 hash for a prompt."""
        return hashlib.sha256(prompt.encode('utf-8', errors='ignore')).hexdigest()

    async def get(self, prompt: str) -> Optional[str]:
        """Retrieves a cached response with improved hit detection."""
        if not isinstance(prompt, str) or not prompt.strip():
            return None

        prompt_hash = self._hash_prompt(prompt.strip())

        async with self._lock:
            # 1. Exact match check
            if prompt_hash in self.cache_dict:
                _, response, _ = self.cache_dict[prompt_hash]
                self.hit_count += 1
                logger.debug(f"ResponseCache: EXACT HIT for {prompt_hash[:10]}... (hits: {self.hit_count})")
                return response

            # 2. Fuzzy match check (if enabled)
            if self.fuzzy_enabled and len(self.cache_dict) > 0:
                prompt_lower = prompt.lower().strip()
                for cached_hash, (cached_prompt, cached_response, _) in self.cache_dict.items():
                    if fuzzy_match(prompt_lower, cached_prompt.lower().strip(), self.fuzzy_threshold):
                        self.hit_count += 1
                        logger.info(f"ResponseCache: FUZZY HIT for {cached_hash[:10]}... (hits: {self.hit_count})")
                        return cached_response

            # 3. No match
            self.miss_count += 1
            # logger.debug(f"ResponseCache: MISS for {prompt_hash[:10]}... (misses: {self.miss_count})")
            return None

    async def put(self, prompt: str, response: str):
        """Stores a prompt-response pair with LRU eviction."""
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str):
            logger.warning("ResponseCache: Invalid prompt or response for put.")
            return

        prompt_hash = self._hash_prompt(prompt.strip())
        current_time = time.time()

        async with self._lock:
            # Add/update entry
            self.cache_dict[prompt_hash] = (prompt.strip(), response, current_time)
            
            # LRU eviction if needed
            if len(self.cache_dict) > self.max_size:
                # Remove oldest entries (by timestamp)
                sorted_items = sorted(self.cache_dict.items(), key=lambda x: x[1][2])
                entries_to_remove = len(self.cache_dict) - self.max_size + 1
                for hash_to_remove, _ in sorted_items[:entries_to_remove]:
                    del self.cache_dict[hash_to_remove]
                logger.debug(f"ResponseCache: Evicted {entries_to_remove} old entries")

            logger.debug(f"ResponseCache: Stored {prompt_hash[:10]}... (cache size: {len(self.cache_dict)})")

    async def get_stats(self) -> Dict[str, Any]:
        """Returns cache performance statistics."""
        async with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0.0
            return {
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate": hit_rate,
                "cache_size": len(self.cache_dict),
                "max_size": self.max_size
            }

    async def clear(self):
        """Clears the entire cache."""
        async with self._lock:
            self.cache_dict.clear()
            self.hit_count = 0
            self.miss_count = 0
            logger.info("ResponseCache: Cleared.")

# --- Code Execution Function ---
def execute_python_code(code_to_execute: str) -> str:
    """Executes Python code with restricted globals for safety."""
    exec_logger = logging.getLogger("CodeExecutor")
    exec_logger.warning("--- !!! DANGER ZONE: EXECUTING CODE VIA exec() !!! ---")
    stdout_capture = StringIO()
    execution_output = ""
    result_msg = ""

    # Define restricted builtins and allowed modules
    allowed_builtins = {
        B: getattr(__builtins__, B) for B in [
            'print', 'range', 'len', 'list', 'dict', 'str', 'int', 'float', 'bool',
            'None', 'True', 'False', 'abs', 'round', 'max', 'min', 'sum', 'zip',
            'enumerate', 'isinstance', 'issubclass', 'Exception', 'ValueError', 'TypeError',
            'sorted', 'map', 'filter', 'bytes', 'bytearray', 'memoryview', 'complex', 'set',
            'frozenset', 'slice', 'object', 'type', 'repr', 'ascii', 'format', 'pow', 'divmod',
            'hash', 'id', 'any', 'all',
        ] if hasattr(__builtins__, B)
    }
    allowed_modules = {
        "math": math,
        "random": random,
        "re": re,
        "json": json,
        "time": time, # Allows time.sleep
        "collections": collections,
        "itertools": importlib.import_module("itertools"),
        "functools": importlib.import_module("functools"),
        "operator": importlib.import_module("operator"),
        "statistics": statistics,
    }

    restricted_globals = {
        "__builtins__": allowed_builtins,
        **allowed_modules
    }
    restricted_locals = {}

    try:
        with redirect_stdout(stdout_capture):
            exec(code_to_execute, restricted_globals, restricted_locals)
        execution_output = stdout_capture.getvalue().strip()
        result_msg = "Code execution successful."
        if execution_output: result_msg += f"\nCaptured Output:\n---\n{execution_output}\n---"
        else: result_msg += " (No output captured)."
        exec_logger.info(result_msg)
    except Exception as e:
        execution_output = stdout_capture.getvalue().strip()
        result_msg = f"EXECUTION FAILED: {type(e).__name__}: {e}"
        if execution_output: result_msg += f"\nCaptured Output:\n---\n{execution_output}\n---"
        exec_logger.error(result_msg)
    finally:
        stdout_capture.close()

    return result_msg

# =============================================================================
# PHASE 1 ENHANCED LLM FUNCTIONS
# =============================================================================

async def query_llm(prompt: str, system_context: Optional['SystemContext'] = None, 
                   temperature: float = 0.7, max_tokens: int = 8192) -> Dict[str, Any]:
    """
    PHASE 1 ENHANCED: LLM query function with circuit breaker and improved error handling.
    """
    global _llm_circuit_breaker
    
    if not isinstance(prompt, str) or not prompt.strip():
        return {"response": "", "error": "Invalid or empty prompt", "cached": False}
    
    prompt = prompt.strip()
    start_time = time.time()
    
    # PHASE 1: Check circuit breaker
    if not _llm_circuit_breaker.can_execute():
        logger.warning("🔧 LLM request blocked by circuit breaker")
        return {
            "response": "",
            "error": f"LLM service unavailable (circuit {_llm_circuit_breaker.state.value})",
            "cached": False,
            "circuit_state": _llm_circuit_breaker.state.value,
            "blocked": True
        }
    
    # Try cache first if available
    cached_response = None
    if system_context and system_context.response_cache:
        try:
            cached_response = await system_context.response_cache.get(prompt)
            if cached_response:
                duration = time.time() - start_time
                _llm_circuit_breaker.record_success(duration)
                logger.debug(f"🔧 LLM: Cache HIT for prompt {hashlib.sha256(prompt.encode()).hexdigest()[:10]}")
                return {
                    "response": cached_response,
                    "cached": True,
                    "duration_s": duration,
                    "model": "cached",
                    "error": None,
                    "circuit_state": _llm_circuit_breaker.state.value
                }
        except Exception as e:
            logger.warning(f"🔧 LLM: Cache lookup failed: {e}")
    
    # Prepare request payload
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }
    
    response_text = ""
    error_msg = None
    model_used = MODEL_NAME
    
    # Try main model first, then fallback
    for attempt in range(RETRY_ATTEMPTS):
        current_model = MODEL_NAME if attempt < 2 else FALLBACK_MODEL
        payload["model"] = current_model
        
        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(API_ENDPOINT, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "").strip()
                        model_used = current_model
                        
                        if response_text:
                            # PHASE 1: Record success in circuit breaker
                            duration = time.time() - start_time
                            _llm_circuit_breaker.record_success(duration)
                            
                            # Cache successful response
                            if system_context and system_context.response_cache:
                                try:
                                    await system_context.response_cache.put(prompt, response_text)
                                except Exception as cache_error:
                                    logger.warning(f"🔧 LLM: Failed to cache response: {cache_error}")
                            break
                        else:
                            error_msg = "Empty response from LLM"
                            logger.warning(f"🔧 LLM: Empty response from {current_model} (attempt {attempt + 1})")
                    else:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        logger.warning(f"🔧 LLM: HTTP error {response.status} from {current_model} (attempt {attempt + 1})")
                        
        except asyncio.TimeoutError:
            error_msg = f"Timeout after {REQUEST_TIMEOUT}s"
            logger.error(f"🔧 LLM: Timeout with {current_model} (attempt {attempt + 1})")
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"🔧 LLM: Exception with {current_model} (attempt {attempt + 1}): {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < RETRY_ATTEMPTS - 1:
            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    
    duration = time.time() - start_time
    
    # PHASE 1: Record result in circuit breaker
    if response_text:
        _llm_circuit_breaker.record_success(duration)
        logger.info(f"🔧 LLM: Success with {model_used} in {duration:.2f}s (length: {len(response_text)})")
    else:
        _llm_circuit_breaker.record_failure()
        logger.error(f"🔧 LLM: All attempts failed. Final error: {error_msg}")
    
    return {
        "response": response_text,
        "cached": False,
        "duration_s": duration,
        "model": model_used,
        "error": error_msg,
        "attempts": min(attempt + 1, RETRY_ATTEMPTS),
        "circuit_state": _llm_circuit_breaker.state.value
    }

async def query_llm_with_flood_control(prompt: str, system_context: Optional['SystemContext'] = None, 
                                      temperature: float = 0.7, max_tokens: int = 8192,
                                      origin_bubble: str = "unknown", priority: int = 5) -> Dict[str, Any]:
    """
    FLOOD-CONTROLLED LLM query function.
    
    Replace your existing query_llm calls with this version.
    """
    global _llm_request_controller
    
    if not isinstance(prompt, str) or not prompt.strip():
        return {"response": "", "error": "Invalid or empty prompt", "cached": False}
    
    prompt = prompt.strip()
    
    # FLOOD CONTROL: Check if request should be allowed
    permission = _llm_request_controller.should_allow_request(prompt, origin_bubble, priority)
    
    if not permission["allowed"]:
        reason = permission["reason"]
        retry_after = permission.get("retry_after", 0)
        
        logger.warning(f"🚨 LLM request BLOCKED from {origin_bubble}: {reason}")
        
        # If it's a duplicate, try to return cached result
        if permission.get("cached_result") and system_context and system_context.response_cache:
            try:
                cached_response = await system_context.response_cache.get(prompt)
                if cached_response:
                    logger.info(f"🚨 Returning cached result for blocked duplicate request")
                    return {
                        "response": cached_response,
                        "cached": True,
                        "blocked": True,
                        "block_reason": reason,
                        "error": None
                    }
            except Exception as e:
                logger.warning(f"Failed to get cached result for duplicate: {e}")
        
        return {
            "response": "",
            "error": f"Request blocked: {reason}",
            "blocked": True,
            "block_reason": reason,
            "retry_after": retry_after,
            "cached": False
        }
    
    # Request allowed - proceed with normal LLM query
    logger.debug(f"🚨 LLM request ALLOWED from {origin_bubble} (priority: {priority})")
    
    # Call the original enhanced query_llm function
    result = await query_llm(prompt, system_context, temperature, max_tokens)
    
    # Add flood control metadata
    result["flood_controlled"] = True
    result["origin_bubble"] = origin_bubble
    result["priority"] = priority
    
    return result

# --- Enhanced SystemContext with Emergency Fixes and Multi-Port Support ---
class SystemContext:
    """Enhanced SystemContext with critical missing methods and bubble management."""
    def __init__(self):
        self.event_dispatcher: Optional[EventDispatcher] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.chat_box: Optional[ChatBox] = None
        self.bubbles: weakref.WeakValueDictionary[str, 'UniversalBubble'] = weakref.WeakValueDictionary()
        self.stop_event: Optional[asyncio.Event] = asyncio.Event()
        self.response_cache: ResponseCache = ResponseCache()
        self.web_server: Optional['BubblesWebServer'] = None  # Reference to web server
        
        # CRITICAL FIX: Add missing attributes for dispatch_event_and_wait
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._response_handlers: Dict[str, Any] = {}
        
        # EMERGENCY: Bubble management
        self._bubble_count = 0
        self._paused_bubbles: Dict[str, 'UniversalBubble'] = {}
        
        # PHASE 1: Initialize enhancements
        self.cognitive_bloat_analyzer = CognitiveBloatAnalyzer()
        self.analysis_router: Optional[EventAnalysisRouter] = None
        
        logger.info("🔧 PHASE 1: SystemContext enhanced with circuit breaker, cognitive bloat, and specialized analysis")

    async def initialize_web_server(self, host="0.0.0.0", port=8080):
        """Initialize the web server using the standalone module."""
        try:
            # Import here to avoid circular dependencies
            from web_server import BubblesWebServer
            
            if not self.web_server:
                self.web_server = BubblesWebServer(self)
            
            success = await self.web_server.initialize(host, port)
            if success:
                logger.info(f"✅ Web server initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize web server")
                self.web_server = None
                return False
                
        except ImportError:
            logger.error("❌ Could not import BubblesWebServer. Make sure web_server.py is available.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize web server: {e}", exc_info=True)
            return False

    def initialize_web_server_sync(self, host="0.0.0.0", port=8080):
        """Synchronous version that creates async task - for backward compatibility"""
        try:
            # Create the async task
            task = asyncio.create_task(self.initialize_web_server(host, port))
            logger.info(f"🚀 Web server initialization started as background task")
            return task
        except Exception as e:
            logger.error(f"Failed to start web server task: {e}")
            return None

    def start_web_server(self, host="0.0.0.0", port=8080):
        """Simple non-async wrapper for initialize_web_server"""
        return self.initialize_web_server_sync(host, port)

    async def stop_web_server(self):
        """Stop the web server if running."""
        if self.web_server:
            await self.web_server.stop()
            self.web_server = None
            logger.info("Web server stopped")

    def get_server_info(self):
        """Get current server information."""
        if self.web_server:
            return self.web_server.get_server_info()
        else:
            return {
                "status": "not initialized",
                "port": None,
                "web_server": None
            }

    async def restart_web_server_new_port(self, new_port=8080):
        """Restart web server on a new port"""
        try:
            logger.info(f"🔄 Restarting web server on port {new_port}...")
            
            if self.web_server:
                success = await self.web_server.restart_on_port(new_port)
                if success:
                    logger.info(f"✅ Successfully restarted on port {new_port}")
                else:
                    logger.error("❌ Failed to restart web server")
                return success
            else:
                # No existing server, create new one
                return await self.initialize_web_server(port=new_port)
                
        except Exception as e:
            logger.error(f"Failed to restart on port {new_port}: {e}")
            return False

    def test_ports(self):
        """Test which ports are available"""
        if self.web_server:
            return self.web_server.test_ports()
        else:
            # Fallback implementation
            import socket
            test_ports = [8080, 8081, 3000, 5000, 8000, 9000, 4000, 7000, 8888, 9999]
            available_ports = []
            
            for port in test_ports:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(('0.0.0.0', port))
                        available_ports.append(port)
                        logger.info(f"✅ Port {port} is available")
                except OSError:
                    logger.warning(f"❌ Port {port} is in use or blocked")
            
            logger.info(f"🔍 Available ports: {available_ports}")
            return available_ports

    async def graceful_restart(self):
        """Gracefully restart the system"""
        try:
            logger.info("Initiating graceful restart...")
            
            # Stop all active bubbles
            stopped_bubbles = []
            for bubble_id, bubble in list(self.bubbles.items()):
                if hasattr(bubble, 'stop'):
                    await bubble.stop()
                    stopped_bubbles.append(bubble_id)
            
            # Clear the bubbles dictionary
            self.bubbles.clear()
            
            # Reset system state
            if self.stop_event:
                self.stop_event.set()
                await asyncio.sleep(0.1)  # Brief pause
                self.stop_event.clear()
            
            # Clear caches and reset counters
            self.response_cache.clear()
            self._bubble_count = len(self._paused_bubbles)  # Only count paused bubbles
            
            # Reset cognitive bloat analyzer
            self.cognitive_bloat_analyzer = CognitiveBloatAnalyzer()
            
            logger.info(f"Graceful restart completed. Stopped {len(stopped_bubbles)} bubbles.")
            return {"stopped_bubbles": stopped_bubbles, "paused_bubbles": len(self._paused_bubbles)}
            
        except Exception as e:
            logger.error(f"Error during graceful restart: {e}", exc_info=True)
            raise

    def register_bubble(self, bubble: 'UniversalBubble'):
        """Enhanced bubble registration with limits and memory checks."""
        if not hasattr(bubble, 'object_id'):
             logger.error(f"Attempted to register object without object_id: {bubble}")
             return

        bubble_id = bubble.object_id
        current_count = len(self.bubbles)

        # EMERGENCY: Check bubble limit
        if current_count >= MAX_CONCURRENT_BUBBLES:
            logger.warning(f"🔧 EMERGENCY: Bubble limit reached ({current_count}/{MAX_CONCURRENT_BUBBLES}). Pausing oldest bubble.")
            asyncio.create_task(self._enhanced_emergency_pause_oldest_bubble())

        # EMERGENCY: Check memory pressure
        if self.resource_manager:
            memory_percent = self.resource_manager.get_resource_level('memory_percent')
            if memory_percent > MEMORY_THRESHOLD_CRITICAL:
                logger.warning(f"🔧 EMERGENCY: Critical memory ({memory_percent:.1f}%). Force pausing bubbles.")
                asyncio.create_task(self._enhanced_emergency_pause_bubbles(2))  # Pause 2 bubbles
            elif memory_percent > MEMORY_THRESHOLD_PAUSE:
                logger.warning(f"🔧 EMERGENCY: High memory ({memory_percent:.1f}%). Pausing 1 bubble.")
                asyncio.create_task(self._enhanced_emergency_pause_bubbles(1))

        if bubble_id in self.bubbles:
            logger.warning(f"SystemContext: Overwriting existing registration for Bubble ID '{bubble_id}'.")
        
        self.bubbles[bubble_id] = bubble
        self._bubble_count += 1
        logger.info(f"🔧 SystemContext: Registered Bubble '{bubble_id}' ({self._bubble_count}/{MAX_CONCURRENT_BUBBLES}) (Type: {type(bubble).__name__})")

    async def _enhanced_emergency_pause_oldest_bubble(self):
        """PHASE 1 ENHANCED: Use cognitive bloat analysis for bubble selection."""
        try:
            logger.info("🔧 Enhanced emergency pause: Using cognitive bloat analysis")
            
            bloated_bubbles = await self.cognitive_bloat_analyzer.identify_bloated_bubbles(self, 1)
            
            if bloated_bubbles:
                bubble_id, bloat_score = bloated_bubbles[0]
                await self.pause_bubble(bubble_id)
                self.cognitive_bloat_analyzer.paused_for_bloat += 1
                
                # Estimate memory savings
                if hasattr(self.cognitive_bloat_analyzer, 'bubble_metrics'):
                    metrics = self.cognitive_bloat_analyzer.bubble_metrics.get(bubble_id)
                    if metrics:
                        self.cognitive_bloat_analyzer.memory_savings_mb += metrics.memory_usage_mb
                
                logger.info(f"🔧 Enhanced pause: Paused bloated bubble {bubble_id} (score: {bloat_score:.2f})")
            else:
                # Fallback to original logic
                if self.bubbles:
                    oldest_id = min(self.bubbles.keys(), 
                                   key=lambda bid: getattr(self.bubbles[bid], 'execution_count', 0))
                    await self.pause_bubble(oldest_id)
                    logger.warning(f"🔧 Enhanced pause: Fallback to oldest bubble {oldest_id}")
                
        except Exception as e:
            logger.error(f"Enhanced emergency pause failed: {e}")
            # Final fallback - pause any bubble
            if self.bubbles:
                bubble_id = next(iter(self.bubbles.keys()))
                await self.pause_bubble(bubble_id)

    async def _enhanced_emergency_pause_bubbles(self, count: int):
        """PHASE 1 ENHANCED: Use cognitive bloat analysis for multiple bubble pausing."""
        try:
            logger.info(f"🔧 Enhanced emergency pause: Analyzing {count} bubbles for cognitive bloat")
            
            bloated_bubbles = await self.cognitive_bloat_analyzer.identify_bloated_bubbles(self, count)
            
            if bloated_bubbles:
                for bubble_id, bloat_score in bloated_bubbles:
                    await self.pause_bubble(bubble_id)
                    self.cognitive_bloat_analyzer.paused_for_bloat += 1
                    
                    # Estimate memory savings
                    if hasattr(self.cognitive_bloat_analyzer, 'bubble_metrics'):
                        metrics = self.cognitive_bloat_analyzer.bubble_metrics.get(bubble_id)
                        if metrics:
                            self.cognitive_bloat_analyzer.memory_savings_mb += metrics.memory_usage_mb
                    
                    logger.info(f"🔧 Enhanced pause: Paused bloated bubble {bubble_id} (score: {bloat_score:.2f})")
            else:
                # Fallback to original logic
                active_bubbles = list(self.bubbles.keys())
                bubbles_to_pause = sorted(active_bubbles, 
                                        key=lambda bid: getattr(self.bubbles[bid], 'execution_count', 0),
                                        reverse=True)[:count]
                
                for bubble_id in bubbles_to_pause:
                    await self.pause_bubble(bubble_id)
                    logger.warning(f"🔧 Enhanced pause: Fallback pause {bubble_id}")
                    
        except Exception as e:
            logger.error(f"Enhanced emergency pause bubbles failed: {e}")

    async def pause_bubble(self, bubble_id: str):
        """Pause a bubble (move to paused state)."""
        try:
            if bubble_id in self.bubbles:
                bubble = self.bubbles[bubble_id]
                await bubble.stop_autonomous_loop()
                self._paused_bubbles[bubble_id] = bubble
                del self.bubbles[bubble_id]
                logger.info(f"🔧 SystemContext: Paused bubble '{bubble_id}'")
                
        except Exception as e:
            logger.error(f"Failed to pause bubble {bubble_id}: {e}")

    async def resume_bubble(self, bubble_id: str):
        """Resume a paused bubble."""
        try:
            if bubble_id in self._paused_bubbles and len(self.bubbles) < MAX_CONCURRENT_BUBBLES:
                bubble = self._paused_bubbles[bubble_id]
                await bubble.start_autonomous_loop()
                self.bubbles[bubble_id] = bubble
                del self._paused_bubbles[bubble_id]
                logger.info(f"🔧 SystemContext: Resumed bubble '{bubble_id}'")
                
        except Exception as e:
            logger.error(f"Failed to resume bubble {bubble_id}: {e}")

    def unregister_bubble(self, bubble_id: str):
        """Enhanced bubble unregistration."""
        if bubble_id in self.bubbles:
            self.bubbles.pop(bubble_id, None)
            self._bubble_count = max(0, self._bubble_count - 1)
            logger.info(f"SystemContext: Unregistered Bubble '{bubble_id}' ({self._bubble_count}/{MAX_CONCURRENT_BUBBLES}).")
        elif bubble_id in self._paused_bubbles:
            self._paused_bubbles.pop(bubble_id, None)
            logger.info(f"SystemContext: Unregistered paused Bubble '{bubble_id}'.")
        else:
            logger.debug(f"SystemContext: Attempted to unregister non-existent Bubble ID '{bubble_id}'.")

    def get_bubble(self, bubble_id: str) -> Optional['UniversalBubble']:
        """Retrieves a bubble instance by its ID (active or paused)."""
        return self.bubbles.get(bubble_id) or self._paused_bubbles.get(bubble_id)

    def get_all_bubble_ids(self) -> List[str]:
        """Returns a list of IDs of all currently registered bubbles."""
        return list(self.bubbles.keys()) + list(self._paused_bubbles.keys())

    def get_all_bubbles(self) -> List['UniversalBubble']:
        """Returns a list of all currently registered bubble instances."""
        return list(self.bubbles.values()) + list(self._paused_bubbles.values())

    async def dispatch_event(self, event: Event):
        """Dispatches an event through the registered EventDispatcher."""
        if self.event_dispatcher:
            await self.event_dispatcher.publish(event)
        else:
            logger.error("SystemContext: EventDispatcher not set, cannot dispatch event.")

    # CRITICAL FIX: The missing dispatch_event_and_wait method
    async def dispatch_event_and_wait(self, event: Event, timeout: float = 30.0) -> Optional[Any]:
        """
        CRITICAL FIX: Dispatch an event and wait for a response.
        
        This was the missing method breaking the entire autonomous system!
        """
        correlation_id = None
        try:
            # Generate unique correlation ID
            correlation_id = str(uuid.uuid4())
            
            # Add correlation ID to event
            event.correlation_id = correlation_id
            
            # Create future for response
            response_future = asyncio.Future()
            self._pending_responses[correlation_id] = response_future
            
            logger.debug(f"🔧 SystemContext: Dispatching event {event.type} with correlation_id {correlation_id[:8]}")
            
            # Dispatch the event normally
            await self.dispatch_event(event)
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                logger.debug(f"🔧 SystemContext: Received response for {correlation_id[:8]}")
                return response
                
            except asyncio.TimeoutError:
                logger.warning(f"🔧 SystemContext: Timeout waiting for response to {correlation_id[:8]}")
                return None
                
        except Exception as e:
            logger.error(f"🔧 SystemContext: dispatch_event_and_wait failed: {e}")
            return None
            
        finally:
            # Cleanup
            if correlation_id and correlation_id in self._pending_responses:
                del self._pending_responses[correlation_id]
    
    # CRITICAL FIX: send_response method
    async def send_response(self, correlation_id: str, response_data: Any):
        """
        CRITICAL FIX: Send a response to a waiting dispatch_event_and_wait call.
        """
        if correlation_id in self._pending_responses:
            future = self._pending_responses[correlation_id]
            if not future.done():
                future.set_result(response_data)
                logger.debug(f"🔧 SystemContext: Sent response for {correlation_id[:8]}")
            else:
                logger.warning(f"🔧 SystemContext: Future already done for {correlation_id[:8]}")
        else:
            logger.warning(f"🔧 SystemContext: No pending response for {correlation_id[:8]}")
    
    # CRITICAL FIX: handle_response_event method
    async def handle_response_event(self, event: Event):
        """
        CRITICAL FIX: Handle incoming response events and route them to waiting callers.
        """
        if hasattr(event, 'correlation_id') and event.correlation_id:
            correlation_id = event.correlation_id
            
            # Extract response data
            response_data = None
            if hasattr(event, 'data') and hasattr(event.data, 'value'):
                response_data = event.data.value
            else:
                response_data = getattr(event, 'response_data', None)
            
            # Send response to waiting caller
            await self.send_response(correlation_id, response_data)

# --- EventDispatcher Class ---
class EventDispatcher:
    """Handles publishing events via the EventService."""
    def __init__(self, context: SystemContext):
        self.context = context
        if context.event_dispatcher is None:
             context.event_dispatcher = self
             logger.info("EventDispatcher initialized and registered with context.")
        else:
             logger.warning("EventDispatcher initialized, but one was already registered in the context.")

    async def publish(self, event: Event):
        """Publishes an event using the global EventService."""
        try:
            await EventService.publish(event)
        except Exception as e:
            logger.error(f"EventDispatcher: Error publishing event {event}: {e}", exc_info=True)

# =============================================================================
# REAL M4 HARDWARE RESOURCE MANAGER (REPLACES ORIGINAL)
# =============================================================================

class ResourceManager:
    """
    Real M4 Hardware ResourceManager that integrates with M4HardwareBubble.
    This completely replaces the original ResourceManager - NO MORE FAKE CPU DATA!
    """
    def __init__(self, context: SystemContext, initial_energy=10000.0):
        self.context = context
        if context.resource_manager is None:
            context.resource_manager = self
            logger.info("🔧 Real M4 ResourceManager initialized - NO FAKE CPU DATA!")
        else:
            logger.warning("ResourceManager initialized, but one was already registered in the context.")

        self.resources: Dict[str, Union[int, float]] = {"energy": float(initial_energy)}
        self._resource_lock = asyncio.Lock()

        self.metrics: Dict[str, Any] = defaultdict(lambda: 0)
        self.metrics["llm_response_times"] = deque(maxlen=100)
        self.metrics["events_published_by_type"] = defaultdict(int)
        self._metrics_lock = asyncio.Lock()

        self.event_frequency_window = 60
        self.recent_event_timestamps: Dict[Union[Tags, Actions], deque] = defaultdict(lambda: deque(maxlen=200))
        self.tracked_event_types = list(Actions)

        # M4 Hardware reference - will be set when M4HardwareBubble is created
        self._m4_hardware_bubble = None
        self._last_hardware_fetch = 0
        self._hardware_cache = {}
        self._hardware_cache_duration = 2.0  # Cache hardware data for 2 seconds

        # EMERGENCY: Memory monitoring
        self._last_memory_check = 0
        self._memory_check_interval = 5  # Check every 5 seconds

        # Validate EventService availability
        if not hasattr(context, 'event_dispatcher') or not context.event_dispatcher:
            logger.error("ResourceManager: EventDispatcher not available in context.")
            raise RuntimeError("EventDispatcher not initialized in SystemContext")

        asyncio.create_task(self._subscribe_to_tracked_events())
        asyncio.create_task(self.periodic_state_publisher())
        asyncio.create_task(self._enhanced_emergency_memory_monitor())
        logger.info(f"🔧 Real M4 ResourceManager initialized (Initial Energy: {initial_energy}).")

    def set_m4_hardware_bubble(self, m4_bubble):
        """Set reference to M4 hardware bubble for real metrics."""
        self._m4_hardware_bubble = m4_bubble
        logger.info("🔧 ResourceManager: Connected to M4 hardware bubble for REAL metrics!")

    def _get_m4_hardware_metrics(self) -> Dict[str, Any]:
        """Get real hardware metrics from M4 bubble or fallback."""
        current_time = time.time()
        
        # Use cache if recent
        if current_time - self._last_hardware_fetch < self._hardware_cache_duration and self._hardware_cache:
            return self._hardware_cache
        
        # Try to get real M4 metrics
        if self._m4_hardware_bubble:
            try:
                hw_status = self._m4_hardware_bubble.get_hardware_status()
                if hw_status and 'current_metrics' in hw_status:
                    metrics = hw_status['current_metrics']
                    self._hardware_cache = {
                        'cpu_percent': metrics.get('cpu', {}).get('total_usage_percent', 0),
                        'memory_percent': metrics.get('memory', {}).get('usage_percent', 0),
                        'gpu_percent': metrics.get('gpu', {}).get('usage_percent', 0),
                        'neural_engine_percent': metrics.get('neural_engine', {}).get('usage_percent', 0),
                        'power_watts': metrics.get('power', {}).get('estimated_total_watts', 0),
                        'thermal_pressure': metrics.get('thermal', {}).get('thermal_pressure', 'nominal'),
                        'performance_cores_percent': metrics.get('cpu', {}).get('performance_cores_percent', 0),
                        'efficiency_cores_percent': metrics.get('cpu', {}).get('efficiency_cores_percent', 0),
                        'source': 'M4_hardware_bubble'
                    }
                    self._last_hardware_fetch = current_time
                    return self._hardware_cache
            except Exception as e:
                logger.warning(f"Failed to get M4 hardware metrics: {e}")
        
        # Fallback to basic memory-only metrics
        mem = psutil.virtual_memory()
        self._hardware_cache = {
            'cpu_percent': 0,  # NO FAKE CPU DATA!
            'memory_percent': mem.percent,
            'gpu_percent': 0,
            'neural_engine_percent': 0,
            'power_watts': 0,
            'thermal_pressure': 'unknown',
            'performance_cores_percent': 0,
            'efficiency_cores_percent': 0,
            'source': 'memory_only_fallback'
        }
        self._last_hardware_fetch = current_time
        return self._hardware_cache

    async def _enhanced_emergency_memory_monitor(self):
        """PHASE 1 ENHANCED: Memory monitor with cognitive bloat awareness."""
        logger.info("🔧 Enhanced memory monitor started with cognitive bloat analysis")
        
        while True:
            if self.context.stop_event and self.context.stop_event.is_set():
                logger.info("🔧 Enhanced memory monitor: Stop event set, halting.")
                break
                
            try:
                await asyncio.sleep(self._memory_check_interval)
                
                # Get real hardware metrics
                hw_metrics = self._get_m4_hardware_metrics()
                memory_percent = hw_metrics['memory_percent']
                current_time = time.time()
                
                # Run cognitive bloat analysis periodically
                if current_time - self.context.cognitive_bloat_analyzer.last_analysis_time > 60:
                    try:
                        await self.context.cognitive_bloat_analyzer.collect_bubble_metrics(self.context)
                    except Exception as e:
                        logger.warning(f"🔧 Cognitive bloat analysis failed: {e}")
                
                # Enhanced memory management decisions
                if memory_percent > MEMORY_THRESHOLD_CRITICAL:
                    logger.warning(f"🔧 CRITICAL MEMORY ({memory_percent:.1f}%) - Enhanced cognitive bloat pause!")
                    await self.context._enhanced_emergency_pause_bubbles(3)
                elif memory_percent > MEMORY_THRESHOLD_PAUSE:
                    logger.warning(f"🔧 HIGH MEMORY ({memory_percent:.1f}%) - Enhanced cognitive bloat pause!")
                    await self.context._enhanced_emergency_pause_bubbles(1)
                elif memory_percent < 70 and len(self.context._paused_bubbles) > 0:
                    # Try to resume paused bubbles when memory is low
                    paused_ids = list(self.context._paused_bubbles.keys())
                    if paused_ids and len(self.context.bubbles) < MAX_CONCURRENT_BUBBLES:
                        await self.context.resume_bubble(paused_ids[0])
                        logger.info(f"🔧 Enhanced resume: Resumed bubble {paused_ids[0]} (low memory)")
                
                # Log status periodically
                if current_time - self._last_memory_check > 30:
                    paused_count = len(self.context._paused_bubbles)
                    active_count = len(self.context.bubbles)
                    bloat_paused = self.context.cognitive_bloat_analyzer.paused_for_bloat
                    logger.info(f"🔧 Memory: {memory_percent:.1f}% (Active: {active_count}, "
                               f"Paused: {paused_count}, Bloat Paused: {bloat_paused})")
                    self._last_memory_check = current_time
                    
            except asyncio.CancelledError:
                logger.info("🔧 Enhanced memory monitor cancelled.")
                break
            except Exception as e:
                logger.error(f"🔧 Error in enhanced memory monitor: {e}", exc_info=True)
                await asyncio.sleep(self._memory_check_interval)

    async def _subscribe_to_tracked_events(self):
        """Subscribes to events for metric and frequency tracking."""
        await asyncio.sleep(0.1)
        try:
            for event_type in self.tracked_event_types:
                await EventService.subscribe(event_type, self._track_event)
            logger.debug(f"ResourceManager: Subscribed to {len(self.tracked_event_types)} Action events.")
        except Exception as e:
            logger.error(f"ResourceManager: Failed to subscribe to events: {e}", exc_info=True)

    async def _track_event(self, event: Event):
        """Updates metrics and frequency data when a relevant event occurs."""
        event_type = event.type
        current_time = time.time()

        async with self._metrics_lock:
            self.metrics["events_published_total"] += 1
            self.metrics["events_published_by_type"][event_type.name] += 1
            self.recent_event_timestamps[event_type].append(current_time)

            if event_type == Actions.LLM_RESPONSE:
                metadata = event.data.metadata or {}
                duration = metadata.get("duration_s")
                is_error = metadata.get("is_error", False)
                self.metrics["llm_call_count"] += 1
                if duration is not None:
                    self.metrics["llm_response_times"].append(duration)
                if is_error:
                    self.metrics["llm_error_count"] += 1
            elif event_type == Actions.ACTION_TAKEN:
                payload = event.data.value if event.data.tag == Tags.DICT else {}
                action_detail = payload.get("action_type")
                if action_detail == Actions.ACTION_TYPE_CODE_UPDATE.name:
                    self.metrics["code_update_count"] += 1
                elif action_detail == Actions.ACTION_TYPE_SPAWN_BUBBLE.name:
                    self.metrics["bubbles_spawned"] += 1
                elif action_detail == Actions.ACTION_TYPE_DESTROY_BUBBLE.name:
                    self.metrics["bubbles_destroyed"] += 1

    def get_event_frequencies(self) -> Dict[str, float]:
        """Calculates events per minute for tracked event types."""
        frequencies = {}
        current_time = time.time()
        window_start = current_time - self.event_frequency_window
        tracked_types_copy = list(self.recent_event_timestamps.keys())

        for event_type in tracked_types_copy:
            if event_type not in self.tracked_event_types:
                continue
            timestamps = self.recent_event_timestamps[event_type]
            count_in_window = sum(1 for ts in timestamps if ts >= window_start)
            freq = (count_in_window / self.event_frequency_window) * 60 if self.event_frequency_window > 0 else 0
            frequencies[f"{event_type.name}_freq_per_min"] = round(freq, 2)
        return frequencies

    async def add_resource(self, resource_type: str, amount: Union[int, float]):
        """Adds a specified amount to a resource."""
        if amount <= 0:
            return
        async with self._resource_lock:
            current_level = self.resources.get(resource_type, 0)
            self.resources[resource_type] = current_level + amount
            logger.debug(f"ResourceManager: Added {amount} {resource_type}. New: {self.resources[resource_type]:.2f}")

    async def consume_resource(self, resource_type: str, amount: Union[int, float]) -> bool:
        """Consumes a resource if available, returning True on success."""
        if amount <= 0:
            return True
        async with self._resource_lock:
            current_level = self.resources.get(resource_type, 0)
            if current_level >= amount:
                self.resources[resource_type] = current_level - amount
                logger.debug(f"ResourceManager: Consumed {amount:.2f} {resource_type}. Rem: {self.resources[resource_type]:.2f}")
                return True
            else:
                logger.warning(f"ResourceManager: Failed consume {amount:.2f} {resource_type}. Avail: {current_level:.2f}")
                return False

    def get_resource_level(self, resource_type: str) -> Union[int, float]:
        """Gets the current level of a resource (fetches dynamic resources from REAL hardware)."""
        # Get real hardware metrics
        hw_metrics = self._get_m4_hardware_metrics()
        
        if resource_type == 'cpu_percent':
            return hw_metrics['cpu_percent']
        elif resource_type == 'memory_percent':
            return hw_metrics['memory_percent']
        elif resource_type == 'gpu_percent':
            return hw_metrics.get('gpu_percent', 0)
        elif resource_type == 'neural_engine_percent':
            return hw_metrics.get('neural_engine_percent', 0)
        elif resource_type == 'power_watts':
            return hw_metrics.get('power_watts', 0)
        elif resource_type == 'thermal_pressure':
            return hw_metrics.get('thermal_pressure', 'unknown')
        elif resource_type == 'performance_cores_percent':
            return hw_metrics.get('performance_cores_percent', 0)
        elif resource_type == 'efficiency_cores_percent':
            return hw_metrics.get('efficiency_cores_percent', 0)
        else:
            return self.resources.get(resource_type, 0)

    def log_prediction_cache_hit(self):
        asyncio.create_task(self._async_log_metric("prediction_cache_hits"))

    def log_prediction_cache_miss(self):
        asyncio.create_task(self._async_log_metric("prediction_cache_misses"))

    async def _async_log_metric(self, metric_key: str, increment: int = 1):
        """Safely increments a metric counter asynchronously."""
        async with self._metrics_lock:
            self.metrics[metric_key] = self.metrics.get(metric_key, 0) + increment

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Returns a summary of key performance metrics."""
        response_times = list(self.metrics["llm_response_times"])
        avg_resp_time_s = statistics.mean(response_times) if response_times else 0
        median_resp_time_s = statistics.median(response_times) if response_times else 0
        total_preds = self.metrics["prediction_cache_hits"] + self.metrics["prediction_cache_misses"]
        cache_hit_rate = (self.metrics["prediction_cache_hits"] / total_preds) if total_preds > 0 else 0

        return {
            "avg_llm_response_time_ms": round(avg_resp_time_s * 1000, 1),
            "median_llm_response_time_ms": round(median_resp_time_s * 1000, 1),
            "llm_call_count": self.metrics["llm_call_count"],
            "llm_error_count": self.metrics["llm_error_count"],
            "code_update_count": self.metrics["code_update_count"],
            "events_published_total": self.metrics["events_published_total"],
            "prediction_cache_hits": self.metrics["prediction_cache_hits"],
            "prediction_cache_misses": self.metrics["prediction_cache_misses"],
            "prediction_cache_hit_rate": round(cache_hit_rate, 3),
            "bubbles_spawned": self.metrics["bubbles_spawned"],
            "bubbles_destroyed": self.metrics["bubbles_destroyed"],
        }

    def get_current_system_state(self) -> Dict[str, Any]:
        """Constructs a dictionary representing the current system state with REAL hardware data."""
        num_bubbles = 0
        bubble_types: Dict[str, int] = defaultdict(int)
        if self.context:
            all_bubbles = self.context.get_all_bubbles()
            num_bubbles = len(all_bubbles)
            for bubble in all_bubbles:
                bubble_types[type(bubble).__name__] += 1

        # Get real hardware metrics
        hw_metrics = self._get_m4_hardware_metrics()

        state = {
            "timestamp": time.time(),
            "energy": self.get_resource_level('energy'),
            "cpu_percent": hw_metrics['cpu_percent'],
            "memory_percent": hw_metrics['memory_percent'],
            "gpu_percent": hw_metrics.get('gpu_percent', 0),
            "neural_engine_percent": hw_metrics.get('neural_engine_percent', 0),
            "power_watts": hw_metrics.get('power_watts', 0),
            "thermal_pressure": hw_metrics.get('thermal_pressure', 'unknown'),
            "performance_cores_percent": hw_metrics.get('performance_cores_percent', 0),
            "efficiency_cores_percent": hw_metrics.get('efficiency_cores_percent', 0),
            "hardware_source": hw_metrics.get('source', 'unknown'),
            "metrics": self.get_metrics_summary(),
            "num_bubbles": num_bubbles,
            "bubble_type_counts": dict(bubble_types),
            "event_frequencies": self.get_event_frequencies(),
        }
        asyncio.create_task(self._async_set_metric("last_state_update_time", state["timestamp"]))
        return state

    async def _async_set_metric(self, metric_key: str, value: Any):
        """Safely sets a metric value asynchronously."""
        async with self._metrics_lock:
            self.metrics[metric_key] = value

    async def periodic_state_publisher(self, interval_seconds: int = 60):
        """Periodically publishes the system state via an event."""
        await asyncio.sleep(10)
        while True:
            if self.context.stop_event and self.context.stop_event.is_set():
                logger.info("ResourceManager: Stop event set, halting periodic state publisher.")
                break
            try:
                await asyncio.sleep(interval_seconds)
                if not self.context or not self.context.event_dispatcher:
                    logger.error("ResourceManager: Context/Dispatcher unavailable for state publish.")
                    continue
                current_state = self.get_current_system_state()
                state_uc = UniversalCode(Tags.DICT, current_state, description="Periodic system state update")
                state_event = Event(type=Actions.SYSTEM_STATE_UPDATE, data=state_uc, origin="ResourceManager", priority=10)
                await self.context.dispatch_event(state_event)
                queue_size = self.context.event_dispatcher.get_queue_size() if hasattr(self.context.event_dispatcher, 'get_queue_size') else -1
                if queue_size > 5000:
                    logger.warning(f"ResourceManager: High event queue size after periodic update: {queue_size}")
                logger.debug(f"ResourceManager: Event queue size after periodic update: {queue_size}")
            except asyncio.CancelledError:
                logger.info("ResourceManager: Periodic state publisher cancelled.")
                break
            except Exception as e:
                logger.error(f"ResourceManager: Error in periodic_state_publisher: {e}", exc_info=True)
                await asyncio.sleep(interval_seconds)

    async def trigger_state_update(self):
        """Manually triggers a SYSTEM_STATE_UPDATE event for testing."""
        try:
            if not self.context or not self.context.event_dispatcher:
                logger.error("ResourceManager: Context/Dispatcher unavailable for manual state update.")
                raise RuntimeError("Context or event dispatcher unavailable")
            state = self.get_current_system_state()
            state['manual_trigger'] = True
            state_uc = UniversalCode(Tags.DICT, state, description="Manually triggered system state")
            trigger_id = str(uuid.uuid4())
            state_event = Event(
                type=Actions.SYSTEM_STATE_UPDATE,
                data=state_uc,
                origin="ResourceManager",
                priority=15,
                metadata={"trigger_id": trigger_id}
            )
            await self.context.dispatch_event(state_event)
            logger.info(f"ResourceManager: Manually triggered SYSTEM_STATE_UPDATE (ts: {state.get('timestamp', 0):.2f}, trigger_id: {trigger_id[:8]})")
            queue_size = self.context.event_dispatcher.get_queue_size() if hasattr(self.context.event_dispatcher, 'get_queue_size') else -1
            if queue_size > 5000:
                logger.warning(f"ResourceManager: High event queue size after manual trigger: {queue_size}")
            logger.debug(f"ResourceManager: Event queue size after trigger: {queue_size}")
        except Exception as e:
            logger.error(f"ResourceManager: Failed to trigger SYSTEM_STATE_UPDATE: {e}", exc_info=True)
            raise

    async def monitor_state_updates(self):
        """Logs state updates to confirm they are being dispatched (disabled to avoid congestion)."""
        logger.warning("ResourceManager: monitor_state_updates is disabled to prevent event queue congestion.")
        return

# --- ChatBox Class ---
class ChatBox:
    """Manages messages displayed on the console interface."""
    def __init__(self, maxlen: int = 200):
        self.messages = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()
        self.recent_queries = deque(maxlen=10) # Track last 10 queries for deduplication
        self.query_cooldown = 10.0 # Seconds before allowing same query

    async def add_message(self, message: str):
        """Adds a message to the chat display and prints it."""
        async with self._lock:
            ts = time.strftime("%H:%M:%S")
            formatted_message = f"[{ts}] {message}"
            self.messages.append(formatted_message)
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, print, formatted_message)
            except RuntimeError:
                print(formatted_message, file=sys.stderr)
            except Exception as e:
                 print(f"Error printing message to console: {e}", file=sys.stderr)

    async def is_duplicate_query(self, query: str) -> bool:
        """Checks if a query is a duplicate within the cooldown period."""
        current_time = time.time()
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        for q_hash, q_time in self.recent_queries:
            if q_hash == query_hash and current_time - q_time < self.query_cooldown:
                return True
        self.recent_queries.append((query_hash, current_time))
        return False

# --- UniversalBubble Base Class ---
class UniversalBubble:
    """Abstract base class for all autonomous agents (Bubbles)."""
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("Bubble requires a non-empty string object_id.")
        if not isinstance(context, SystemContext):
             raise TypeError("Bubble requires a valid SystemContext instance.")

        self.object_id = object_id
        self.context = context
        self.dispatcher = context.event_dispatcher
        self.resource_manager = context.resource_manager
        self.chat_box = context.chat_box

        self.event_queue = asyncio.Queue(maxsize=8000)
        self.should_stop = False
        self.execution_count = 0
        self._process_task: Optional[asyncio.Task] = None

        # PHASE 1: Add tracking attributes for cognitive bloat analysis
        self.creation_time = time.time()
        self.last_activity_time = time.time()
        self.recent_event_count = 0
        self.error_count = 0

        self.context.register_bubble(self) # Register on successful init

    async def handle_event(self, event: Event):
        """Default event handler: Queues the event."""
        try:
            if self.event_queue.full():
                logger.warning(f"{self.object_id}: Event queue full. Discarding event {event.event_id[:8]} ({event.type.name}).")
                return
            await self.event_queue.put(event)
            
            # PHASE 1: Update activity tracking
            self.last_activity_time = time.time()
            self.recent_event_count += 1
            
            # logger.debug(f"{self.object_id}: Queued event {event.event_id[:8]} ({event.type.name})")
        except Exception as e:
            logger.error(f"{self.object_id}: Error queueing event {event}: {e}", exc_info=True)
            self.error_count += 1

    async def process_event_queue(self):
        """Processes one event from the internal queue."""
        try:
            event = await asyncio.wait_for(self.event_queue.get(), timeout=0.1)
            # logger.debug(f"{self.object_id}: Processing event from queue: {event}")
            await self.process_single_event(event)
            self.event_queue.task_done()
            
            # PHASE 1: Update activity tracking
            self.last_activity_time = time.time()
            
        except asyncio.TimeoutError: 
            pass
        except asyncio.CancelledError: 
            raise
        except Exception as e:
            logger.error(f"{self.object_id}: Error processing event from queue: {e}", exc_info=True)
            self.error_count += 1

    async def process_single_event(self, event: Event):
        """Placeholder for bubble-specific event processing logic."""
        logger.debug(f"{self.object_id}: Default handler ignoring event {event.type.name}")

    async def autonomous_step(self):
        """Placeholder for the bubble's main autonomous logic per cycle."""
        await asyncio.sleep(0.01) # Prevent busy-waiting in base class

    async def _internal_loop(self):
        """The main asynchronous loop driving the bubble's behavior."""
        logger.info(f"{self.object_id}: Internal loop starting.")
        while not self.should_stop:
            try:
                self.execution_count += 1
                await self.process_event_queue()
                await self.autonomous_step()
                
                # PHASE 1: Reset recent event count periodically
                if self.execution_count % 1000 == 0:
                    self.recent_event_count = max(0, self.recent_event_count - 1)
                    
            except asyncio.CancelledError:
                logger.info(f"{self.object_id}: Autonomous loop cancelled.")
                break
            except Exception as e:
                logger.error(f"{self.object_id}: Error in autonomous loop (iter {self.execution_count}): {e}", exc_info=True)
                self.error_count += 1
                await asyncio.sleep(5) # Avoid fast error loop
        logger.info(f"{self.object_id}: Autonomous loop stopped.")

    async def start_autonomous_loop(self):
        """Starts the bubble's internal processing loop."""
        if self._process_task and not self._process_task.done():
            logger.warning(f"{self.object_id}: Loop already running.")
            return
        logger.info(f"{self.object_id}: Starting autonomous loop.")
        self.should_stop = False
        self._process_task = asyncio.create_task(self._internal_loop(), name=f"{self.object_id}_loop")

    async def stop_autonomous_loop(self):
        """Signals the bubble's internal loop to stop and waits for it."""
        if not self._process_task or self._process_task.done():
            # logger.info(f"{self.object_id}: Loop not running or already stopped.")
            return

        logger.info(f"{self.object_id}: Stopping autonomous loop...")
        self.should_stop = True
        try:
            await asyncio.wait_for(self._process_task, timeout=5.0)
            logger.info(f"{self.object_id}: Loop stopped gracefully.")
        except asyncio.TimeoutError:
            logger.warning(f"{self.object_id}: Loop stop timed out. Cancelling task.")
            self._process_task.cancel()
            try: await self._process_task
            except asyncio.CancelledError: logger.info(f"{self.object_id}: Loop cancellation confirmed.")
        except Exception as e:
            logger.error(f"{self.object_id}: Error during loop stop/cleanup: {e}", exc_info=True)
        finally:
            self._process_task = None

    async def self_destruct(self):
        """Stops the loop and unregisters the bubble."""
        logger.info(f"{self.object_id}: Initiating self-destruct sequence...")
        await self.stop_autonomous_loop()
        self.context.unregister_bubble(self.object_id)
        logger.info(f"{self.object_id}: Self-destruct complete.")

    def get_resource_level(self, resource_type: str) -> Union[int, float]:
        """Convenience method to get resource levels."""
        if self.resource_manager: return self.resource_manager.get_resource_level(resource_type)
        logger.warning(f"{self.object_id}: RM unavailable for get_resource '{resource_type}'.")
        return 0

    async def add_chat_message(self, message: str):
        """Convenience method to add a message to the chat box."""
        if self.chat_box: await self.chat_box.add_message(f"[{self.object_id}] {message}")
        else: logger.warning(f"{self.object_id}: ChatBox unavailable for message: {message[:5000]}...")

    async def publish_action_taken(self, action_enum: Actions, payload: Dict[str, Any]):
        """Helper method to publish an ACTION_TAKEN event."""
        if not self.dispatcher:
             logger.error(f"{self.object_id}: Cannot publish ACTION_TAKEN, dispatcher unavailable.")
             return
        try:
            action_data = {"action_type": action_enum.name, "payload": payload}
            action_uc = UniversalCode(Tags.DICT, action_data,
                                      description=f"Action taken: {action_enum.name}",
                                      metadata={"timestamp": time.time()})
            action_event = Event(type=Actions.ACTION_TAKEN, data=action_uc, origin=self.object_id, priority=5)
            await self.context.dispatch_event(action_event)
            logger.debug(f"{self.object_id}: Published ACTION_TAKEN event: {action_enum.name}")
        except Exception as e:
             logger.error(f"{self.object_id}: Failed publish ACTION_TAKEN for {action_enum.name}: {e}", exc_info=True)

    # CRITICAL FIX: send_response_to_event method
    async def send_response_to_event(self, original_event: Event, response_data: Any):
        """
        Send a response back to an event that requested a response.
        """
        if hasattr(original_event, 'correlation_id') and original_event.correlation_id:
            await self.context.send_response(original_event.correlation_id, response_data)





# =============================================================================
# PHASE 1 HELPER FUNCTIONS
# =============================================================================

def initialize_specialized_analysis(context: SystemContext) -> EventAnalysisRouter:
    """Initialize the specialized analysis router system."""
    context.analysis_router = EventAnalysisRouter(context)
    return context.analysis_router

async def route_event_for_specialized_analysis(context: SystemContext, event: Event) -> Dict[str, Any]:
    """
    Route an event for specialized analysis instead of broadcasting to all bubbles.
    
    This is the main function you call to replace broadcasting events to all bubbles.
    """
    if context.analysis_router:
        return await context.analysis_router.route_for_analysis(event)
    else:
        logger.warning("Specialized analysis router not initialized - skipping routing")
        return {
            "analysis_performed": False,
            "reason": "Router not initialized",
            "category": "unknown",
            "urgency": "unknown"
        }

def get_analysis_routing_stats(context: SystemContext) -> Dict[str, Any]:
    """Get statistics about the specialized analysis routing."""
    if context.analysis_router:
        return context.analysis_router.get_routing_stats()
    else:
        return {
            "error": "Analysis router not initialized",
            "requests_routed": 0,
            "redundancy_eliminated": 0,
            "efficiency_gain": "0x",
            "registered_specialists": {},
            "total_specialists": 0
        }

def get_llm_flood_control_stats() -> Dict[str, Any]:
    """Get LLM flood control statistics."""
    global _llm_request_controller, _llm_circuit_breaker
    
    flood_stats = _llm_request_controller.get_stats()
    circuit_stats = _llm_circuit_breaker.get_status()
    
    return {
        "flood_control": flood_stats,
        "circuit_breaker": circuit_stats,
        "combined_stats": {
            "total_protection_events": flood_stats["requests_blocked"] + flood_stats["requests_deduplicated"],
            "efficiency_improvement": f"{flood_stats['dedup_rate'] + flood_stats['block_rate']:.1%}",
            "system_stability": "healthy" if circuit_stats["state"] == "closed" else circuit_stats["state"]
        }
    }

def get_cognitive_bloat_report(context: SystemContext) -> Dict[str, Any]:
    """Get cognitive bloat analysis report."""
    return context.cognitive_bloat_analyzer.get_bloat_report()

# --- Emergency Utility Functions ---
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        memory = psutil.virtual_memory()
        return {
            "percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "total_gb": memory.total / (1024**3)
        }
    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return {"percent": 0, "available_gb": 0, "used_gb": 0, "total_gb": 0}

def emergency_cleanup():
    """Emergency cleanup function for critical situations."""
    try:
        import gc
        collected = gc.collect()
        logger.warning(f"🔧 EMERGENCY: Garbage collected {collected} objects")
        
        # Force memory cleanup
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(700, 10, 10)  # More aggressive GC
            
        return True
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")
        return False

def get_phase1_status() -> Dict[str, Any]:
    """Get overall Phase 1 system status."""
    global _llm_circuit_breaker, _llm_request_controller
    
    return {
        "phase": "Phase 1 Complete",
        "features": [
            "Circuit Breaker for LLM Requests",
            "LLM Request Flood Control", 
            "Specialized Analysis Routing",
            "Cognitive Bloat Analysis",
            "Enhanced Memory Management",
            "Emergency Bubble Pausing",
            "Real M4 Hardware Integration - NO FAKE CPU DATA!"
        ],
        "circuit_breaker_state": _llm_circuit_breaker.state.value,
        "flood_control_active": True,
        "specialized_routing_available": True,
        "memory_monitoring": True,
        "real_hardware_metrics": True,
        "emergency_fixes": {
            "mps_memory_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "max_concurrent_bubbles": MAX_CONCURRENT_BUBBLES,
            "memory_thresholds": {
                "pause": MEMORY_THRESHOLD_PAUSE,
                "critical": MEMORY_THRESHOLD_CRITICAL
            }
        }
    }

# --- Module Initialization ---
logger.info("🔧 Phase 1 bubbles_core.py COMPLETE: All flood control and optimization systems loaded!")
logger.info(f"🔧 Key systems: Circuit Breaker, Flood Control, Specialized Analysis, Cognitive Bloat, Emergency Memory Management")
logger.info(f"🔧 Memory thresholds: Pause at {MEMORY_THRESHOLD_PAUSE}%, Critical at {MEMORY_THRESHOLD_CRITICAL}%")
logger.info(f"🔧 LLM Protection: Rate limit {_llm_request_controller.max_requests_per_minute}/min, Circuit breaker active")
logger.info(f"🔧 REAL M4 HARDWARE: ResourceManager now uses authentic M4 metrics - NO FAKE CPU DATA!")
logger.info(f"🔧 Phase 1 Status: {get_phase1_status()['phase']} - Ready for integration!")
