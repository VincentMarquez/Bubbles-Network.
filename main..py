# main.py: Entry point for the Bubbles Framework
# Initializes SystemContext, bubbles, and event-driven system
# Integrates Full Enhanced PPO with Meta-Learning, QMLBubble, QuantumOracleBubble, M4 Hardware Monitor, and APEP
# WITH SURGICAL M4 QML OPTIMIZATION FIXES (No Global dtype Changes)
# WITH TARGETED 3-BUBBLE FLOOD CONTROL (Oracle, Overseer, M4Hardware)
# WITH EVENT HANDLING ERROR FIXES
# WITH APEP BUBBLE INTEGRATION
# WITH SERIALIZATION BUBBLE INTEGRATION
# WITH LOG MONITOR BUBBLE INTEGRATION
# WITH QFD BUBBLE INTEGRATION - FIXED EVENT TYPE HANDLING

import sys
import asyncio
import logging
import time
import psutil
import os
import json
import uuid
import site
import numpy as np
from fixed_exploratory_llm2 import AtEngineV3, LLMDiscussionManager
import warnings
warnings.filterwarnings("ignore", message=".*correlation_id.*")


# In your main initialization code
async def setup_bubbles_with_serialization():
    """
    Set up the bubbles framework with serialization support.
    
    Creates and initializes the serialization bubble first, then the hardware bubble,
    and applies necessary patches for compatibility.
    
    Returns:
        tuple: (context, serialization_bubble, hardware_bubble)
    """
    # Create the context
    context = SystemContext()
    
    # IMPORTANT: Create serialization bubble FIRST
    from serialization_bubble import SerializationBubble, integrate_serialization_bubble
    serialization_bubble = integrate_serialization_bubble(context)
    
    # Now create the hardware bubble
    from m4_hardware_bubble import M4HardwareBubble
    hardware_bubble = M4HardwareBubble(
        object_id="m4_hardware_bubble",
        context=context
    )
    
    # Apply the serialization fix
    from hardware_bubble_serialization_integration import patch_m4_hardware_bubble
    patch_m4_hardware_bubble(hardware_bubble, context)
    
    # Start both bubbles
    await serialization_bubble.start_autonomous_loop()
    await hardware_bubble.start_autonomous_loop()
    
    return context, serialization_bubble, hardware_bubble


# ========== SURGICAL M4 QML OPTIMIZATION FIXES ==========
import torch
import warnings

def apply_surgical_qml_m4_fixes():
    """
    Apply surgical M4 fixes for QMLBubble ONLY - Don't affect other bubbles.
    
    This function applies optimizations specific to QML operations on M4 hardware
    without affecting the global PyTorch settings that would break MPS compatibility
    for other bubbles like DreamerV3 and Overseer.
    
    Returns:
        bool: True if fixes were applied successfully
    """
    print("🔧 Applying surgical QML M4 optimization fixes...")
    
    # 1. DO NOT set global dtype - keep float32 for MPS compatibility
    # torch.set_default_dtype(torch.float64)  # ❌ This breaks MPS!
    
    # 2. Set NumPy precision (safe)
    np.set_printoptions(precision=8)
    
    # 3. Enable PennyLane optimizations (safe)
    os.environ["PENNYLANE_ENABLE_JIT"] = "true"
    os.environ["PENNYLANE_CACHE_SIZE"] = "1000"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 4. Suppress precision warnings (safe)
    warnings.filterwarnings("ignore", message="Finite differences with float32 detected")
    warnings.filterwarnings("ignore", category=UserWarning, module="pennylane.gradients.finite_difference")
    warnings.filterwarnings("ignore", message=".*gradient of the QNode.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="pennylane")
    warnings.filterwarnings("ignore", message=".*ComplexWarning.*")
    
    # 5. MPS optimization without breaking other bubbles
    if torch.backends.mps.is_available():
        print("✅ M4 MPS detected - keeping float32 for classical ops")
        print("✅ QML will use CPU with float64 for precision")
        
        # Clear MPS cache for optimization
        try:
            torch.mps.empty_cache()
        except:
            pass
    else:
        print("✅ CPU optimization enabled for all operations")
    
    print("✅ Surgical QML M4 fixes applied successfully!")
    print("📊 Global precision: float32 (MPS compatible)")
    print("📊 QML precision: float64 (will be set in QMLBubble only)")
    print("🚀 PennyLane JIT and caching enabled")
    print("🔇 All precision warnings suppressed")
    print("⚡ M4 memory optimizations applied")
    print("✅ Other bubbles (Overseer, DreamerV3) remain MPS compatible!")
    
    return True

# Apply SURGICAL M4 QML fixes - won't break other bubbles
apply_surgical_qml_m4_fixes()

# Your existing environment setting (now redundant but keeping for compatibility)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up basic logging before any imports that might fail
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
print("Starting imports...")

try:
    print("Importing pennylane (with surgical M4 optimizations applied)...")
    import pennylane as qml
    print("✅ Pennylane imported successfully with surgical M4 fixes")
except Exception as e:
    print(f"❌ Failed to import pennylane: {e}")
    sys.exit(1)

print("Importing bubbles_core...")
from bubbles_core import (
    SystemContext, EventService, Actions, UniversalCode, Tags, logger, Event,
    EventDispatcher, ResourceManager, ChatBox
)
print("bubbles_core imported successfully")

# ========== EVENT HANDLING ERROR FIXES ==========
def ensure_required_actions():
    """
    Ensure all required action types exist to prevent AttributeError.
    
    This function dynamically adds missing action types to the Actions enum
    to prevent errors when events are created or subscribed to. Note that
    QFD-specific events are now registered by QFDBubble itself.
    """
    required_actions = [
        'CONSCIOUSNESS_EMERGENCE',
        'SPAWN_ALGORITHM', 
        'USER_HELP_REQUEST',
        'USER_HELP_RESPONSE',
        'ERROR_REPORT',
        'QML_PREDICTION',
        'QML_RESULT',
        'HARDWARE_ALERT',
        'HARDWARE_HEALTH_CHECK',
        'APEP_REFINEMENT_COMPLETE',
        'APEP_STATUS_REQUEST',
        # NEW: Log Monitor action types
        'WARNING_EVENT',
        'CORRELATION_WARNING',
        'PERFORMANCE_WARNING',
        'MEMORY_WARNING',
        'FIX_APPLIED',
        'RECOVERY_SUCCESS',
        # NOTE: QFD event types removed - QFDBubble registers its own
    ]
    
    for action_name in required_actions:
        if not hasattr(Actions, action_name):
            setattr(Actions, action_name, action_name)
            logger.info(f"Added missing action type: {action_name}")

# Apply the fixes immediately after importing bubbles_core
ensure_required_actions()

def safe_event_subscribe(event_type, handler, event_service=None):
    """
    Safely subscribe to events with proper error handling.
    
    Args:
        event_type: The event type to subscribe to (can be string or Actions member)
        handler: The async function to handle the event
        event_service: Optional EventService instance (uses global if not provided)
        
    Returns:
        The handler function or a dummy handler if subscription fails
    """
    try:
        # Normalize event_type to proper format
        if isinstance(event_type, str):
            # If it's a string, try to get it from Actions
            if hasattr(Actions, event_type):
                event_type = getattr(Actions, event_type)
            # If not found in Actions, create it
            else:
                setattr(Actions, event_type, event_type)
                event_type = getattr(Actions, event_type)
        
        # Use EventService.subscribe with proper error handling
        if event_service:
            return event_service.subscribe(event_type, handler)
        else:
            return EventService.subscribe(event_type, handler)
            
    except AttributeError as e:
        logger.error(f"Event subscription error for {event_type}: {e}")
        # Create a dummy handler that logs the issue
        async def dummy_handler(event):
            logger.warning(f"Dummy handler called for {event_type} - original handler failed to subscribe")
        return dummy_handler
    except Exception as e:
        logger.error(f"Unexpected error in event subscription: {e}")
        return None

def safe_json_serialize(obj):
    """
    Safely serialize objects to JSON, handling special types.
    
    This function handles serialization of complex objects including those with
    __dict__, named tuples, enums, and other special types that standard JSON
    can't handle directly.
    
    Args:
        obj: The object to serialize
        
    Returns:
        A JSON-serializable version of the object
    """
    if hasattr(obj, '__dict__'):
        # For objects with __dict__, convert to dictionary
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, '_asdict'):
        # For named tuples
        return obj._asdict()
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # For any other type, convert to string
        return str(obj)

print("Importing DreamerV3Bubble...")
from DreamerV3Bubble import DreamerV3Bubble
print("DreamerV3Bubble imported successfully")

print("Importing TuningBubble...")
from TuningBubble import TuningBubble
print("TuningBubble imported successfully")

print("Importing APIBubble...")
from APIBubble import APIBubble
print("APIBubble imported successfully")

# Comment out basic PPOBubble - we'll use Full Enhanced PPO instead
# print("Importing PPOBubble...")
# from PPOBubble import PPOBubble
# print("PPOBubble imported successfully")

print("Importing bubbles_Agent9...")
from bubbles_Agent9 import (
    CompositeBubble, SimpleLLMBubble, FeedbackBubble, CreativeSynthesisBubble,
    MetaReasoningBubble, DynamicManagerBubble
)
print("bubbles_Agent9 imported successfully")

print("Importing bubbles_home_assistant...")
from bubbles_home_assistant import HomeAssistantBubble
print("bubbles_home_assistant imported successfully")

print("Importing bubbles_overseer...")
from bubbles_overseer import OverseerBubble
print("✅ bubbles_overseer imported successfully (MPS compatible!)")

print("Importing RAGBubble...")
from RAGBubble import RAGBubble
print("RAGBubble imported successfully")

print("Importing pool_control...")
from pool_control import PoolControlBubble
print("pool_control imported successfully")

# NEW: Import LogMonitorBubble
print("Importing LogMonitorBubble...")
try:
    from log_monitor_bubble import LogMonitorBubble
    LOG_MONITOR_AVAILABLE = True
    print("✅ LogMonitorBubble imported successfully")
except ImportError as e:
    print(f"⚠️ LogMonitorBubble not available: {e}")
    LOG_MONITOR_AVAILABLE = False

# NEW: Import SerializationBubble
print("Importing SerializationBubble...")
try:
    from serialization_bubble import SerializationBubble, integrate_serialization_bubble
    SERIALIZATION_AVAILABLE = True
    print("✅ SerializationBubble imported successfully")
except ImportError as e:
    print(f"⚠️ SerializationBubble not available: {e}")
    SERIALIZATION_AVAILABLE = False

# NEW: Import APEP Bubble
print("Importing APEP Bubble...")
try:
    from apep_bubble_code import APEPBubble
    APEP_AVAILABLE = True
    print("✅ APEP Bubble imported successfully")
except ImportError as e:
    print(f"⚠️ APEP Bubble not available: {e}")
    APEP_AVAILABLE = False

# NEW: Import flood control system
print("Importing flood control system...")
try:
    from flood_control import (
        enable_flood_control_for_bubble,
        disable_flood_control_for_bubble, 
        get_flood_control_stats,
        get_recent_flood_control_requests,
        clear_flood_control_stats,
        is_flood_control_enabled_for_bubble
    )
    FLOOD_CONTROL_AVAILABLE = True
    print("✅ Flood control system imported successfully")
except ImportError as e:
    print(f"⚠️ Flood control not available: {e}")
    FLOOD_CONTROL_AVAILABLE = False

# Import Full Enhanced PPO System
print("Importing Full Enhanced PPO with Meta-Learning...")
try:
    from full_integrated_ppo_system import (
        setup_full_ppo_with_meta_learning,
        FullyIntegratedPPO,
        FullEnhancedPPOWithMetaLearning,
        MetaLearningOrchestrator
    )
    FULL_PPO_AVAILABLE = True
    print("Full Enhanced PPO imported successfully")
except ImportError as e:
    print(f"Full Enhanced PPO not available: {e}")
    print("Falling back to basic PPO...")
    from PPOBubble import PPOBubble
    FULL_PPO_AVAILABLE = False

print("Importing QMLBubble (with surgical M4 optimizations)...")
try:
    from QMLBubble import QMLBubble
    print("✅ QMLBubble imported successfully with surgical M4 fixes")
except Exception as e:
    print(f"❌ Failed to import QMLBubble: {e}")
    import traceback
    traceback.print_exc()
    QMLBubble = None

print("Importing QuantumOracleBubble...")
try:
    from QuantumOracleBubble import QuantumOracleBubble
    print("QuantumOracleBubble imported successfully")
except Exception as e:
    print(f"Failed to import QuantumOracleBubble: {e}")
    import traceback
    traceback.print_exc()
    QuantumOracleBubble = None

# NEW: Import M4 Hardware Monitor
print("Importing M4 Hardware Monitor...")
try:
    from m4_hardware_bubble import M4HardwareBubble, enhance_bubbles_with_hardware, HardwareActions
    M4_HARDWARE_AVAILABLE = True
    print("M4 Hardware Monitor imported successfully")
    
    # Ensure hardware action types exist
    hardware_actions = ['HARDWARE_ALERT', 'HARDWARE_HEALTH_CHECK', 'MEMORY_CLEANUP_REQUEST', 
                       'THERMAL_THROTTLE_REQUEST', 'PERFORMANCE_PROFILE_CHANGE',
                       'SYSTEM_DIAGNOSTICS_REQUEST', 'HARDWARE_CAPABILITY_QUERY']
    for action in hardware_actions:
        if not hasattr(Actions, action):
            setattr(Actions, action, action)
        if not hasattr(HardwareActions, action):
            setattr(HardwareActions, action, action)
            
except Exception as e:
    print(f"Failed to import M4 Hardware Monitor: {e}")
    import traceback
    traceback.print_exc()
    M4_HARDWARE_AVAILABLE = False

# NEW: Import QFDBubble
print("Importing QFDBubble...")
try:
    from QFDBubble import QFDBubble
    QFD_AVAILABLE = True
    print("✅ QFDBubble imported successfully")
except ImportError as e:
    print(f"⚠️ QFDBubble not available: {e}")
    QFD_AVAILABLE = False

print("All imports completed!")

# Configure logging levels for noisy modules (Options 1 & 4)
logging.getLogger("bubbles_core.ResourceManager").setLevel(logging.WARNING)
logging.getLogger("bubbles_core.EventDispatcher").setLevel(logging.INFO)
logging.getLogger("bubbles_core.EventService").setLevel(logging.INFO)

# Optional: Set INFO level for all of bubbles_core to reduce overall verbosity
# logging.getLogger("bubbles_core").setLevel(logging.INFO)

# Custom filter to suppress specific debug messages
class DreamerDebugFilter(logging.Filter):
    """
    Filter to suppress tensor shape and other verbose debug messages from DreamerV3 and ResourceManager.
    
    This filter helps reduce log noise by suppressing repetitive debug messages that aren't
    useful for normal operation, while still allowing important warnings and errors through.
    """
    def filter(self, record):
        # Filter ResourceManager messages
        if record.name == "bubbles_core.ResourceManager":
            # Only allow WARNING and above
            return record.levelno >= logging.WARNING
            
        # Only filter DEBUG level messages
        if record.levelno == logging.DEBUG:
            msg = record.getMessage()
            # Suppress messages containing these patterns - UPDATED with QML and APEP patterns
            suppress_patterns = [
                '.shape=',
                'forward:',
                'input:',
                '_vectorize_state:',
                'train_actor_critic:',
                'state_logits',
                'z_indices',
                'z_t.shape',
                'action_mixer',
                'Actor forward',
                'Imagination step',
                'collect_transitions:',
                'train_world_model:',
                'SYSTEM_STATE_UPDATE',
                'Dispatching event: Actions.SYSTEM_STATE_UPDATE',
                'ResourceManager: Broadcasting system state',
                # NEW: Suppress QML debug noise
                'quantum_circuit_base:',
                'QNode execution:',
                'pennylane device:',
                'gradient computation:',
                'Q-learning circuit:',
                'memory circuit:',
                'QMLBubble:.*forward:',
                'QMLBubble:.*train:',
                'tensor.dtype',
                'parameter update:',
                # NEW: Suppress APEP debug noise
                'APEP:.*technique applied:',
                'APEP:.*cache hit:',
                'APEP:.*refinement iteration:',
                'APEPBubble:.*processing prompt:',
                'APEPBubble:.*cache lookup:'
            ]
            if any(pattern in msg for pattern in suppress_patterns):
                return False
        return True

# Configure logging to avoid duplicates
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"))

# Add the debug filter to the handler
handler.addFilter(DreamerDebugFilter())

logger.addHandler(handler)
file_handler = logging.FileHandler(f"bubbles_log_{int(time.time())}.json")
file_handler.setFormatter(logging.Formatter('{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "filename": "%(filename)s", "lineno": "%(lineno)d", "message": "%(message)s"}'))

# Also add filter to file handler
file_handler.addFilter(DreamerDebugFilter())

logger.addHandler(file_handler)

# Optional: Set specific module logging levels
# logging.getLogger("bubbles_core").setLevel(logging.INFO)  # Uncomment to suppress ALL debug from bubbles_core

async def handle_simple_chat(context: SystemContext):
    """
    Enhanced chat handler with Full Enhanced PPO, M4 Hardware commands, QML status, APEP commands, QFD commands, and Flood Control monitoring.
    
    This function provides the main user interface for interacting with the bubbles system
    through text commands. It handles various system commands, bubble-specific commands,
    and routes user queries to appropriate bubbles.
    
    Args:
        context: The SystemContext instance containing all bubbles and system state
    """
    logger.debug("Starting chat handler")
    if not context.chat_box or not context.stop_event:
        logger.critical("Chat handler cannot start: ChatBox or StopEvent missing")
        return
    chat_box = context.chat_box
    stop_event = context.stop_event
    loop = asyncio.get_running_loop()
    await chat_box.add_message("System: Chat handler started. Type 'quit' to exit")
    
    # Build command list based on available features
    command_list = [
        "quit, status, trigger_state, llm <query>, turn_on_pool"
    ]
    
    if FULL_PPO_AVAILABLE:
        command_list.append("ppo_status, explore_consciousness, ppo_algorithms, ppo_patterns, ppo_help <msg>")
    
    if M4_HARDWARE_AVAILABLE:
        command_list.append("hw_status, hw_temp, hw_cleanup, hw_throttle, hw_profile <mode>")
    
    # NEW: QML-specific commands
    if QMLBubble is not None:
        command_list.append("qml_status, qml_train, qml_predict <metrics>, qml_optimize")
    
    # NEW: APEP-specific commands
    if APEP_AVAILABLE:
        command_list.append("apep_status, apep_cache_stats, apep_techniques, apep_config <mode>")
    
    # NEW: Flood control commands
    if FLOOD_CONTROL_AVAILABLE:
        command_list.append("flood status, show flood stats, show recent requests, enable flood control, disable flood control, reset flood control")
    
    # NEW: Serialization test command
    if SERIALIZATION_AVAILABLE:
        command_list.append("test_serialization")
    
    # NEW: Log monitor commands
    if LOG_MONITOR_AVAILABLE:
        command_list.append("log_status, log_stats")
    
    # NEW: QFD commands
    if QFD_AVAILABLE:
        command_list.append("qfd_status, qfd_start, qfd_pause, qfd_metrics")
    
    await chat_box.add_message(f"Commands: {', '.join(command_list)}")
    
    bubble_id_lock = asyncio.Lock()

    async def handle_llm_response(event: Event):
        """Handle LLM response events and display them in the chat."""
        if event.type == Actions.LLM_RESPONSE and isinstance(event.data, UniversalCode):
            try:
                response = event.data.value
                correlation_id = event.data.metadata.get("correlation_id", "unknown")
                origin = event.data.metadata.get("response_to", "unknown")
                
                # NEW: Check if this is a flood-controlled response
                flood_controlled = event.data.metadata.get("flood_control_managed", False)
                flood_indicator = " 🌊" if flood_controlled else ""
                
                # NEW: Check if this is an APEP-refined response
                apep_refined = event.data.metadata.get("apep_processed", False) or event.data.metadata.get("apep_code_refined", False)
                apep_indicator = " 🧠" if apep_refined else ""
                
                display_response = response[:5000] + ("..." if len(response) > 5000 else "")
                await chat_box.add_message(f"LLM Response{flood_indicator}{apep_indicator} (cid: {correlation_id[:6]}, origin: {origin}): {display_response}")
                with open("llm_responses.log", "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] cid: {correlation_id}, origin: {origin}, flood_controlled: {flood_controlled}, apep_refined: {apep_refined}\n{response}\n{'='*80}\n")
                logger.debug(f"Displayed LLM response (cid: {correlation_id[:6]})")
            except Exception as e:
                logger.error(f"Error displaying LLM response: {e}", exc_info=True)

    async def handle_overseer_control(event: Event):
        """Handle overseer control events for spawning new bubbles."""
        if event.type == Actions.OVERSEER_CONTROL and isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
            action_type = event.data.value.get("action_type")
            payload = event.data.value.get("payload", {})
            if action_type == "SPAWN_BUBBLE":
                try:
                    bubble_type = payload.get("bubble_type", "SimpleLLMBubble")
                    async with bubble_id_lock:
                        max_attempts = 20
                        for _ in range(max_attempts):
                            bubble_id = f"{bubble_type.lower()}_{uuid.uuid4().hex[:16]}"
                            if bubble_id not in context.get_all_bubbles():
                                break
                        else:
                            raise ValueError(f"Failed to generate unique bubble ID after {max_attempts} attempts")
                    bubble_class = globals().get(bubble_type, SimpleLLMBubble)
                    kwargs = payload.get("kwargs", {"use_mock": False})
                    bubble = bubble_class(object_id=bubble_id, context=context, **kwargs)
                    context.register_bubble(bubble)
                    await bubble.start_autonomous_loop()
                    context.resource_manager.metrics["bubbles_spawned"] = context.resource_manager.metrics.get("bubbles_spawned", 0) + 1
                    logger.info(f"Spawned new bubble: {bubble_id} (type: {bubble_type}, kwargs={kwargs})")
                except Exception as e:
                    logger.error(f"Failed to spawn bubble: {e}", exc_info=True)

    # Enhanced PPO event handlers
    async def handle_consciousness_emergence(event: Event):
        """Handle consciousness emergence events from the PPO system."""
        if event.type == Actions.CONSCIOUSNESS_EMERGENCE:
            data = event.data.value if hasattr(event.data, 'value') else {}
            entropy = data.get('current_entropy', 0)
            confidence = data.get('confidence', 0)
            
            if confidence > 0.7:
                await chat_box.add_message(
                    f"🌟 HIGH CONSCIOUSNESS EMERGENCE! "
                    f"Entropy: {entropy:.1f}, Confidence: {confidence:.1%}"
                )
                
                # Log visualization if available
                if data.get('visualization_available'):
                    await chat_box.add_message(
                        f"📊 Visualization saved: ppo_consciousness_{data.get('exploration_count', 0)}.png"
                    )
    
    async def handle_algorithm_spawn(event: Event):
        """Handle algorithm spawning events from the PPO system."""
        if event.type == Actions.SPAWN_ALGORITHM:
            data = event.data.value if hasattr(event.data, 'value') else {}
            algo_type = data.get('algorithm_type', 'unknown')
            algo_id = data.get('algorithm_id', 'unknown')
            purpose = data.get('purpose', 'optimization')
            
            await chat_box.add_message(
                f"🔬 Spawned {algo_type} algorithm: {algo_id} for {purpose}"
            )
    
    async def handle_user_help_request(event: Event):
        """Handle when PPO asks for help from the user."""
        if event.type == Actions.USER_HELP_REQUEST:
            data = event.data.value if hasattr(event.data, 'value') else {}
            problem = data.get('problem', 'Unknown')
            
            await chat_box.add_message(
                f"\n{'='*60}\n"
                f"🆘 PPO REQUESTS HELP!\n"
                f"Problem: {problem}\n"
                f"Type 'ppo_help <suggestion>' to provide guidance\n"
                f"{'='*60}"
            )
    
    async def handle_user_help_response(event: Event):
        """Handle user responses to PPO help requests."""
        if hasattr(event.data, 'value') and isinstance(event.data.value, dict):
            if event.data.value.get('user_response'):
                await chat_box.add_message("✅ PPO received your help suggestion")

    # NEW: APEP event handlers
    async def handle_apep_refinement(event: Event):
        """Handle APEP refinement completion events."""
        if event.type == Actions.APEP_REFINEMENT_COMPLETE:
            data = event.data.value if hasattr(event.data, 'value') else {}
            refinement_type = data.get('type', 'unknown')
            techniques_used = data.get('techniques_applied', [])
            improvement_score = data.get('improvement_score', 0)
            
            if improvement_score > 0.2:  # Only show significant improvements
                await chat_box.add_message(
                    f"🧠 APEP {refinement_type} refinement: {improvement_score:.1%} improvement using {len(techniques_used)} techniques"
                )

    # NEW: M4 Hardware event handlers with safe serialization
    async def handle_hardware_alert(event: Event):
        """Handle hardware alerts from the M4 monitoring system."""
        try:
            if event.type == getattr(HardwareActions, 'HARDWARE_ALERT', 'HARDWARE_ALERT'):
                data = event.data.value if hasattr(event.data, 'value') else {}
                # Use safe serialization for hardware data
                data = safe_json_serialize(data)
                severity = data.get('severity', 'info')
                message = data.get('message', 'Hardware alert')
                
                # Color code by severity
                if severity == 'critical':
                    prefix = "🚨 CRITICAL HARDWARE ALERT"
                elif severity == 'warning':
                    prefix = "⚠️ Hardware Warning"
                else:
                    prefix = "ℹ️ Hardware Info"
                    
                await chat_box.add_message(f"{prefix}: {message}")
        except Exception as e:
            logger.error(f"Error handling hardware alert: {e}")
    
    async def handle_hardware_health(event: Event):
        """Handle hardware health updates from the M4 monitoring system."""
        try:
            if event.type == getattr(HardwareActions, 'HARDWARE_HEALTH_CHECK', 'HARDWARE_HEALTH_CHECK'):
                data = event.data.value if hasattr(event.data, 'value') else {}
                # Use safe serialization for hardware data
                data = safe_json_serialize(data)
                status = data.get('overall_status', 'unknown')
                
                if status == 'critical':
                    await chat_box.add_message("🩺 Hardware Health: CRITICAL - Check system immediately!")
                elif status == 'unhealthy':
                    await chat_box.add_message("🩺 Hardware Health: UNHEALTHY - System degraded")
                elif status == 'degraded':
                    await chat_box.add_message("🩺 Hardware Health: DEGRADED - Performance affected")
                else:
                    await chat_box.add_message(f"🩺 Hardware Health: {status.upper()}")
        except Exception as e:
            logger.error(f"Error handling hardware health check: {e}")

    # NEW: QFD event handlers
    async def handle_qfd_metrics(event: Event):
        """Handle QFD performance metrics from the quantum fractal dynamics simulation."""
        if event.type == Actions.PERFORMANCE_METRIC and event.origin == "qfd_bubble":
            data = event.data.value if hasattr(event.data, 'value') else {}
            phi = data.get('phi', 0)
            entropy = data.get('entropy', 0)
            fd = data.get('fd', 0)
            
            # Only show significant events
            if phi > 2.0:
                await chat_box.add_message(
                    f"🌌 QFD: High consciousness detected! φ={phi:.3f}"
                )
            
            if abs(entropy - getattr(handle_qfd_metrics, 'last_entropy', entropy)) > 0.5:
                await chat_box.add_message(
                    f"🌌 QFD: Phase transition! Entropy: {getattr(handle_qfd_metrics, 'last_entropy', 0):.3f} → {entropy:.3f}"
                )
            
            handle_qfd_metrics.last_entropy = entropy

    async def handle_qfd_complete(event: Event):
        """Handle QFD simulation completion events."""
        if event.type == Actions.QFD_SIMULATION_COMPLETE:  # FIXED: Use Actions enum
            data = event.data.value if hasattr(event.data, 'value') else {}
            await chat_box.add_message(
                f"🌌 QFD Simulation Complete!\n"
                f"   Final entropy: {data.get('final_entropy', 0):.3f}\n"
                f"   Final FD: {data.get('final_fd', 0):.3f}\n"
                f"   Final φ: {data.get('final_phi', 0):.3f}"
            )

    # FIXED: Safe event subscription with proper error handling
    try:
        await safe_event_subscribe(Actions.LLM_RESPONSE, handle_llm_response)
        await safe_event_subscribe(Actions.OVERSEER_CONTROL, handle_overseer_control)
        
        # Subscribe to Full Enhanced PPO events with safe handling
        if FULL_PPO_AVAILABLE:
            await safe_event_subscribe(Actions.CONSCIOUSNESS_EMERGENCE, handle_consciousness_emergence)
            await safe_event_subscribe(Actions.SPAWN_ALGORITHM, handle_algorithm_spawn)
            await safe_event_subscribe(Actions.USER_HELP_REQUEST, handle_user_help_request)
            await safe_event_subscribe(Actions.USER_HELP_RESPONSE, handle_user_help_response)
        
        # NEW: APEP subscriptions with safe handling
        if APEP_AVAILABLE:
            await safe_event_subscribe(Actions.APEP_REFINEMENT_COMPLETE, handle_apep_refinement)
        
        # NEW: M4 Hardware subscriptions with safe handling
        if M4_HARDWARE_AVAILABLE:
            await safe_event_subscribe('HARDWARE_ALERT', handle_hardware_alert)
            await safe_event_subscribe('HARDWARE_HEALTH_CHECK', handle_hardware_health)
        
        # Subscribe to QFD events if available
        if QFD_AVAILABLE:
            await safe_event_subscribe(Actions.PERFORMANCE_METRIC, handle_qfd_metrics)
            await safe_event_subscribe(Actions.QFD_SIMULATION_COMPLETE, handle_qfd_complete)
        
        logger.debug("Chat handler subscribed to events successfully")
    except Exception as e:
        logger.error(f"Failed to subscribe to events: {e}", exc_info=True)
        await chat_box.add_message(f"System Error: Failed to initialize chat handler: {e}")
        return

    while not stop_event.is_set():
        try:
            user_input = await loop.run_in_executor(None, input, f"You (time: {time.strftime('%H:%M:%S')}): ")
            logger.debug(f"Chat: Received input: {user_input}")
            original_input = user_input.strip()
            user_input = user_input.strip().lower()
            
            if user_input == 'quit':
                await chat_box.add_message("System: Shutting down...")
                stop_event.set()
                break
            elif user_input == 'trigger_state':
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
            elif user_input == 'status':
                if context.resource_manager:
                    state = context.resource_manager.get_current_system_state()
                    await chat_box.add_message(f"System State:\n{json.dumps(state, indent=2)}")
                    logger.info("Chat: Generated status report")
                else:
                    await chat_box.add_message("System Error: ResourceManager unavailable")
            elif user_input.startswith('llm '):
                query_text = user_input[4:].strip()
                if not query_text:
                    await chat_box.add_message("System: LLM query cannot be empty.")
                    continue
                await chat_box.add_message(f"System: Sending query to SimpleLLMBubble: {query_text[:50]}...")
                query_uc = UniversalCode(
                    Tags.STRING,
                    f"User query: {query_text}\nRespond directly.",
                    metadata={"correlation_id": str(uuid.uuid4()), "target_bubble": "simplellm_bubble"}
                )
                query_event = Event(type=Actions.LLM_QUERY, data=query_uc, origin="user_chat", priority=3)
                await context.dispatch_event(query_event)
                logger.debug(f"Chat: Dispatched LLM_QUERY for: {query_text[:50]}...")
            elif user_input == 'turn_on_pool':
                entity_id = "switch.my_pool_chlorinator"
                await chat_box.add_message(f"System: Triggering pool control...")
                correlation_id = str(uuid.uuid4())
                ha_control_data = {
                    "entity_id": entity_id,
                    "action": "control_pool",
                    "correlation_id": correlation_id
                }
                ha_event = Event(
                    type=Actions.HA_CONTROL,
                    data=UniversalCode(Tags.DICT, ha_control_data),
                    origin="user_chat",
                    metadata={"correlation_id": correlation_id}
                )
                await context.dispatch_event(ha_event)
                logger.debug(f"Chat: Dispatched HA_CONTROL for pool control (cid: {correlation_id[:6]})")
                await chat_box.add_message(f"System: Pool control command sent to {entity_id}")
            
            # NEW: Serialization test command
            elif user_input == 'test_serialization' and SERIALIZATION_AVAILABLE:
                serializer = context.serialization_bubble
                if serializer:
                    try:
                        # Test data with enums
                        from enum import Enum
                        class TestStatus(Enum):
                            GOOD = "good"
                            BAD = "bad"
                        
                        from datetime import datetime
                        
                        test_data = {
                            "status": TestStatus.GOOD,
                            "tag": Tags.DICT,
                            "action": Actions.SYSTEM_STATE_UPDATE,
                            "timestamp": datetime.now()
                        }
                        
                        # Test serialization
                        uc = serializer.create_universal_code(
                            Tags.DICT,
                            test_data,
                            description="Serialization test"
                        )
                        
                        await chat_box.add_message("✅ Serialization test passed! Enums work correctly.")
                        await chat_box.add_message(f"Metrics: {serializer.get_metrics_report()}")
                        
                    except Exception as e:
                        await chat_box.add_message(f"❌ Serialization test failed: {e}")
                else:
                    await chat_box.add_message("Serialization Bubble not found")
            
            # Full Enhanced PPO commands
            elif user_input == 'ppo_status' and FULL_PPO_AVAILABLE:
                ppo = context.get_bubble("enhanced_ppo") or context.get_bubble("ppo_bubble")
                if ppo:
                    try:
                        status = await ppo.get_status()
                        if hasattr(ppo, 'decisions_made'):  # Full PPO
                            await chat_box.add_message(f"""
Enhanced PPO Status:
==================
Decisions Made: {status.get('decisions_made', 0):,}
Patterns Discovered: {status.get('patterns_discovered', 0)}
Algorithms Spawned: {status.get('algorithms_spawned', 0)}
Error Handlers Created: {status.get('handlers_created', 0)}
Consciousness Patterns: {status.get('consciousness_patterns', 0)}
Current Strategy: {status.get('current_strategy', 'unknown')}
Average Reward: {status.get('recent_reward_avg', 0):.3f}

Active Algorithms: {', '.join(status.get('active_algorithms', []))}
Error Patterns: {status.get('error_patterns_detected', 0)} ({status.get('active_error_patterns', 0)} active)

Objective Weights:
{json.dumps(status.get('objective_weights', {}), indent=2)}
""")
                        else:  # Basic PPO
                            await chat_box.add_message(f"Basic PPO Status:\n{json.dumps(status, indent=2)}")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting PPO status: {e}")
                else:
                    await chat_box.add_message("PPO not found")
            
            elif user_input == 'explore_consciousness' and FULL_PPO_AVAILABLE:
                ppo = context.get_bubble("enhanced_ppo")
                if ppo and hasattr(ppo, 'consciousness') and ppo.consciousness.enabled:
                    await chat_box.add_message("🌟 Starting consciousness exploration with AtEngineV3...")
                    await ppo.consciousness.explore_consciousness_patterns()
                    
                    # Get latest pattern
                    if ppo.consciousness.consciousness_patterns:
                        latest = ppo.consciousness.consciousness_patterns[-1]
                        await chat_box.add_message(f"""
Consciousness Exploration Complete!
==================================
Entropy: {latest['entropy']:.1f}
Fractal Dimension: {latest['fractal_dimension']:.3f}
Nodes: {latest['metrics']['node_count']}
Edges: {latest['metrics']['edge_count']}

Indicators:
{json.dumps(latest.get('indicators', {}), indent=2)}

Insights:
{chr(10).join('- ' + i for i in latest.get('insights', []))}
""")
                else:
                    await chat_box.add_message("Consciousness exploration not available (Enhanced PPO required)")
            
            elif user_input == 'ppo_algorithms' and FULL_PPO_AVAILABLE:
                ppo = context.get_bubble("enhanced_ppo")
                if ppo and hasattr(ppo, 'spawned_algorithms'):
                    if ppo.spawned_algorithms:
                        await chat_box.add_message("Active Algorithms:")
                        for algo_id, algo_info in ppo.spawned_algorithms.items():
                            perf = algo_info.get('performance_history', [])
                            avg_perf = np.mean(perf[-10:]) if perf else 0
                            await chat_box.add_message(f"""
- {algo_id}:
  Type: {algo_info['type']}
  Spawned: {time.strftime('%H:%M:%S', time.localtime(algo_info['spawned_at']))}
  Performance: {avg_perf:.3f if perf else 'N/A'}
""")
                    else:
                        await chat_box.add_message("No algorithms spawned yet")
                else:
                    await chat_box.add_message("Algorithm spawning not available (Enhanced PPO required)")
            
            elif user_input == 'ppo_patterns' and FULL_PPO_AVAILABLE:
                ppo = context.get_bubble("enhanced_ppo")
                if ppo and hasattr(ppo, 'meta_knowledge_base'):
                    if ppo.meta_knowledge_base:
                        await chat_box.add_message(f"Discovered Patterns ({len(ppo.meta_knowledge_base)}):")
                        
                        # Group by category
                        from collections import defaultdict
                        by_category = defaultdict(list)
                        for pid, pattern in ppo.meta_knowledge_base.items():
                            category = pattern.context.get('category', 'general')
                            by_category[category].append(pattern)
                        
                        for category, patterns in by_category.items():
                            await chat_box.add_message(f"\n{category.upper()}:")
                            for p in patterns[:3]:  # Show top 3 per category
                                await chat_box.add_message(f"- {p.context.get('name', p.pattern_id)} "
                                                         f"(confidence: {p.confidence:.1%}, "
                                                         f"applications: {p.applications})")
                    else:
                        await chat_box.add_message("No patterns discovered yet")
                else:
                    await chat_box.add_message("Pattern discovery not available (Enhanced PPO required)")
            
            elif user_input.startswith('ppo_help ') and FULL_PPO_AVAILABLE:
                suggestion = original_input[9:].strip()  # Use original case for suggestion
                ppo = context.get_bubble("enhanced_ppo")
                if ppo and hasattr(ppo, 'user_interaction'):
                    # Create help response
                    response_event = Event(
                        type=Actions.USER_HELP_RESPONSE,
                        data=UniversalCode(Tags.DICT, {
                            "user_response": {
                                "suggestion": suggestion,
                                "confidence": 0.8,
                                "reasoning": "User provided guidance"
                            }
                        }),
                        origin="user_chat"
                    )
                    await context.dispatch_event(response_event)
                    await chat_box.add_message(f"✅ Sent help to PPO: {suggestion}")
                else:
                    await chat_box.add_message("PPO help system not available (Enhanced PPO required)")

            # NEW: APEP Commands
            elif user_input == 'apep_status' and APEP_AVAILABLE:
                apep_bubble = context.get_bubble("apep_bubble")
                if apep_bubble:
                    try:
                        status = await apep_bubble.get_apep_status()
                        await chat_box.add_message(f"""
🧠 APEP Status:
===============
Mode: {status.get('mode', 'unknown')}
Safety Level: {status.get('safety_level', 'unknown')}
Total Prompt Refinements: {status.get('total_prompt_refinements', 0)}
Total Code Refinements: {status.get('total_code_refinements', 0)}
Prompt Cache Size: {status.get('prompt_cache_size', 0)}
Code Cache Size: {status.get('code_cache_size', 0)}

Active Prompt Techniques: {', '.join(status.get('active_prompt_techniques', []))}
Active Code Techniques: {', '.join(status.get('active_code_techniques', []))}

High-Value Bubbles: {', '.join(status.get('high_value_bubbles', []))}
Cache Enabled: {status.get('cache_enabled', False)}

Performance by Bubble:
{json.dumps(status.get('performance_by_bubble', {}), indent=2)}
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting APEP status: {e}")
                        logger.error(f"APEP status error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("APEP Bubble not found")
            
            elif user_input == 'apep_cache_stats' and APEP_AVAILABLE:
                apep_bubble = context.get_bubble("apep_bubble")
                if apep_bubble:
                    try:
                        status = await apep_bubble.get_apep_status()
                        prompt_cache_size = status.get('prompt_cache_size', 0)
                        code_cache_size = status.get('code_cache_size', 0)
                        total_prompt_refinements = status.get('total_prompt_refinements', 0)
                        total_code_refinements = status.get('total_code_refinements', 0)
                        
                        # Calculate hit rates
                        prompt_hit_rate = (prompt_cache_size / max(1, total_prompt_refinements)) * 100
                        code_hit_rate = (code_cache_size / max(1, total_code_refinements)) * 100
                        
                        await chat_box.add_message(f"""
🧠 APEP Cache Statistics:
========================
Prompt Cache:
- Size: {prompt_cache_size} entries
- Total Refinements: {total_prompt_refinements}
- Estimated Hit Rate: {prompt_hit_rate:.1f}%

Code Cache:
- Size: {code_cache_size} entries  
- Total Refinements: {total_code_refinements}
- Estimated Hit Rate: {code_hit_rate:.1f}%

Cache Status: {'✅ ENABLED' if status.get('cache_enabled', False) else '❌ DISABLED'}
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting APEP cache stats: {e}")
                        logger.error(f"APEP cache stats error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("APEP Bubble not found")
            
            elif user_input == 'apep_techniques' and APEP_AVAILABLE:
                apep_bubble = context.get_bubble("apep_bubble")
                if apep_bubble:
                    try:
                        # Get technique effectiveness from the bubble
                        prompt_techniques = apep_bubble.get_applied_techniques()
                        
                        await chat_box.add_message(f"""
🧠 APEP Technique Effectiveness:
===============================
Recently Applied Prompt Techniques:
{chr(10).join('- ' + t for t in prompt_techniques) if prompt_techniques else '- None applied yet'}

Available Prompt Techniques:
- Clarity & Specificity: Replace vague terms
- Constraint Addition: Add relevant constraints
- Few-Shot Prompting: Add examples for complex tasks  
- Output Structuring: Add format requirements
- Persona Assignment: Add appropriate expert role
- Chain-of-Thought: Add step-by-step reasoning
- Negative Prompting: Add "avoid" constraints
- Contextual Enrichment: Add system state info

Available Code Techniques:
- Safety Checks: Add null checks and validation
- Error Handling: Wrap in try-catch blocks
- Documentation: Add comments and docstrings
- Bubble Integration: Ensure framework compatibility
- Resource Awareness: Add consumption tracking
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting APEP techniques: {e}")
                        logger.error(f"APEP techniques error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("APEP Bubble not found")
            
            elif user_input.startswith('apep_config ') and APEP_AVAILABLE:
                mode = original_input[12:].strip().lower()
                valid_modes = ["manual", "semi_automated", "fully_automated"]
                
                if mode in valid_modes:
                    apep_bubble = context.get_bubble("apep_bubble")
                    if apep_bubble:
                        try:
                            # Update APEP mode
                            from apep_bubble_code import APEPMode
                            apep_bubble.mode = APEPMode(mode)
                            await chat_box.add_message(f"✅ APEP mode changed to: {mode}")
                            
                            # Show mode description
                            mode_descriptions = {
                                "manual": "APEP will only refine when explicitly requested",
                                "semi_automated": "APEP will refine high-value prompts automatically",
                                "fully_automated": "APEP will refine all applicable prompts"
                            }
                            await chat_box.add_message(f"📝 {mode_descriptions[mode]}")
                            
                        except Exception as e:
                            await chat_box.add_message(f"Error changing APEP mode: {e}")
                            logger.error(f"APEP config error: {e}", exc_info=True)
                    else:
                        await chat_box.add_message("APEP Bubble not found")
                else:
                    await chat_box.add_message(f"Invalid mode. Valid options: {', '.join(valid_modes)}")

            # NEW: M4 Hardware commands
            elif user_input == 'hw_status' and M4_HARDWARE_AVAILABLE:
                m4_bubble = context.get_bubble("m4_hardware_bubble")
                if m4_bubble:
                    try:
                        status = m4_bubble.get_hardware_status()
                        # Use safe serialization for display
                        status = safe_json_serialize(status)
                        metrics = status.get('current_metrics', {})
                        health = status.get('health', {})
                        
                        await chat_box.add_message(f"""
M4 Hardware Status:
==================
CPU: {metrics.get('cpu', {}).get('total_usage_percent', 0):.1f}% (P-cores: {metrics.get('cpu', {}).get('p_core_usage_percent', 0):.1f}%, E-cores: {metrics.get('cpu', {}).get('e_core_usage_percent', 0):.1f}%)
Memory: {metrics.get('memory', {}).get('usage_percent', 0):.1f}% ({metrics.get('memory', {}).get('available_gb', 0):.1f}GB available)
Temperature: {metrics.get('thermal', {}).get('cpu_temp_celsius') or 'N/A'}°C ({metrics.get('thermal', {}).get('thermal_state', 'unknown')})
Power: {metrics.get('power', {}).get('estimated_total_watts', 0):.1f}W (efficiency: {metrics.get('power', {}).get('power_efficiency', 0):.2f})
GPU: {metrics.get('gpu', {}).get('estimated_utilization', 0):.1f}% utilization
Neural Engine: {metrics.get('neural_engine', {}).get('estimated_utilization_percent', 0):.1f}%

Health: {health.get('overall_status', 'unknown').upper()}
Performance Profile: {status.get('performance_profile', 'unknown')}
Constraints: {status.get('constraints', {}).get('active_constraint_count', 0)} active
                        """)
                    except Exception as e:
                        await chat_box.add_message(f"Error getting hardware status: {e}")
                        logger.error(f"Hardware status error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("M4 Hardware Monitor not found")
            
            elif user_input == 'hw_temp' and M4_HARDWARE_AVAILABLE:
                m4_bubble = context.get_bubble("m4_hardware_bubble")
                if m4_bubble and hasattr(m4_bubble, 'metrics_history') and m4_bubble.metrics_history:
                    try:
                        latest = safe_json_serialize(m4_bubble.metrics_history[-1])
                        temp = latest.get('hardware', {}).get('thermal', {}).get('cpu_temp_celsius')
                        thermal_state = latest.get('hardware', {}).get('thermal', {}).get('thermal_state', 'unknown')
                        
                        if temp:
                            status_emoji = "🟢" if temp < 70 else "🟡" if temp < 80 else "🔴"
                            await chat_box.add_message(f"{status_emoji} CPU Temperature: {temp:.1f}°C ({thermal_state})")
                            
                            if temp > 85:
                                await chat_box.add_message("⚠️ High temperature detected! Consider thermal throttling.")
                        else:
                            await chat_box.add_message("Temperature monitoring not available")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting temperature: {e}")
                        logger.error(f"Temperature check error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("M4 Hardware Monitor not found or no data")
            
            elif user_input == 'hw_cleanup' and M4_HARDWARE_AVAILABLE:
                await chat_box.add_message("🧹 Requesting memory cleanup...")
                cleanup_event = Event(
                    type=getattr(HardwareActions, 'MEMORY_CLEANUP_REQUEST', 'MEMORY_CLEANUP_REQUEST'),
                    data=UniversalCode(Tags.STRING, "user_requested"),
                    origin="user_chat"
                )
                await context.dispatch_event(cleanup_event)
            
            elif user_input == 'hw_throttle' and M4_HARDWARE_AVAILABLE:
                await chat_box.add_message("❄️ Requesting thermal throttling...")
                throttle_event = Event(
                    type=getattr(HardwareActions, 'THERMAL_THROTTLE_REQUEST', 'THERMAL_THROTTLE_REQUEST'),
                    data=UniversalCode(Tags.STRING, "user_requested"),
                    origin="user_chat"
                )
                await context.dispatch_event(throttle_event)
            
            elif user_input.startswith('hw_profile ') and M4_HARDWARE_AVAILABLE:
                profile = original_input[11:].strip().lower()
                valid_profiles = ["power_save", "balanced", "performance"]
                
                if profile in valid_profiles:
                    await chat_box.add_message(f"⚙️ Changing performance profile to: {profile}")
                    profile_event = Event(
                        type=getattr(HardwareActions, 'PERFORMANCE_PROFILE_CHANGE', 'PERFORMANCE_PROFILE_CHANGE'),
                        data=UniversalCode(Tags.STRING, profile),
                        origin="user_chat"
                    )
                    await context.dispatch_event(profile_event)
                else:
                    await chat_box.add_message(f"Invalid profile. Valid options: {', '.join(valid_profiles)}")

            # NEW: QML Commands with safe error handling
            elif user_input == 'qml_status' and QMLBubble is not None:
                qml_bubble = context.get_bubble("qml_bubble")
                if qml_bubble:
                    try:
                        # Try to get optimization status
                        if hasattr(qml_bubble, 'get_optimization_status'):
                            opt_status = qml_bubble.get_optimization_status()
                            # Use safe serialization
                            opt_status = safe_json_serialize(opt_status)
                            await chat_box.add_message(f"""
QML Bubble Status (M4 Optimized):
=================================
{json.dumps(opt_status, indent=2)}
""")
                        else:
                            # Fallback to basic status
                            training_status = qml_bubble.get_training_status() if hasattr(qml_bubble, 'get_training_status') else {}
                            quantum_metrics = qml_bubble.get_quantum_state_metrics() if hasattr(qml_bubble, 'get_quantum_state_metrics') else {}
                            
                            # Use safe serialization
                            training_status = safe_json_serialize(training_status)
                            quantum_metrics = safe_json_serialize(quantum_metrics)
                            
                            await chat_box.add_message(f"""
QML Bubble Status:
==================
Training Status:
{json.dumps(training_status, indent=2)}

Quantum Metrics:
{json.dumps(quantum_metrics, indent=2)}
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting QML status: {e}")
                        logger.error(f"QML status error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("QML Bubble not found")
            
            elif user_input == 'qml_train' and QMLBubble is not None:
                qml_bubble = context.get_bubble("qml_bubble")
                if qml_bubble:
                    await chat_box.add_message("🧠 Starting QML training session...")
                    
                    # Trigger training re-initialization
                    if hasattr(qml_bubble, '_initialize_training'):
                        try:
                            await qml_bubble._initialize_training()
                            await chat_box.add_message("✅ QML training completed successfully!")
                        except Exception as e:
                            await chat_box.add_message(f"❌ QML training failed: {e}")
                            logger.error(f"QML training error: {e}", exc_info=True)
                    else:
                        await chat_box.add_message("❌ QML training method not available")
                else:
                    await chat_box.add_message("QML Bubble not found")
            
            elif user_input.startswith('qml_predict ') and QMLBubble is not None:
                qml_bubble = context.get_bubble("qml_bubble")
                if qml_bubble:
                    try:
                        # Parse metrics from input
                        metrics_str = original_input[12:].strip()
                        if metrics_str:
                            # Try to parse as JSON
                            try:
                                metrics = json.loads(metrics_str)
                            except:
                                # Fallback to simple format
                                metrics = {"energy_usage": float(metrics_str)}
                        else:
                            # Use default test metrics
                            metrics = {
                                "device_state": 0.7,
                                "user_presence": 1.0,
                                "temperature": 22.0,
                                "time_of_day": 14.0,
                                "humidity": 45.0
                            }
                        
                        # Process quantum prediction
                        result = await qml_bubble.process_quantum_task("prediction", {"metrics": metrics})
                        # Use safe serialization
                        result = safe_json_serialize(result)
                        await chat_box.add_message(f"""
🔮 QML Prediction Result:
========================
{json.dumps(result, indent=2)}
""")
                    except Exception as e:
                        await chat_box.add_message(f"❌ QML prediction failed: {e}")
                        logger.error(f"QML prediction error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("QML Bubble not found")
            
            elif user_input == 'qml_optimize' and QMLBubble is not None:
                qml_bubble = context.get_bubble("qml_bubble")
                if qml_bubble:
                    await chat_box.add_message("⚡ Running quantum optimization...")
                    
                    try:
                        # Test both energy and comfort optimization
                        test_metrics = {
                            "device_state": 0.8,
                            "user_presence": 0.9,
                            "temperature": 25.0,
                            "energy_tariff": 0.25,
                            "time_of_day": 16.0
                        }
                        
                        energy_result = await qml_bubble._perform_quantum_optimization("energy", {"metrics": test_metrics})
                        comfort_result = await qml_bubble._perform_quantum_optimization("comfort", {"metrics": test_metrics})
                        
                        # Use safe serialization
                        energy_result = safe_json_serialize(energy_result)
                        comfort_result = safe_json_serialize(comfort_result)
                        
                        await chat_box.add_message(f"""
⚡ Quantum Optimization Results:
===============================
Energy Optimization:
{json.dumps(energy_result, indent=2)}

Comfort Optimization:
{json.dumps(comfort_result, indent=2)}
""")
                    except Exception as e:
                        await chat_box.add_message(f"❌ Quantum optimization failed: {e}")
                        logger.error(f"QML optimization error: {e}", exc_info=True)
                else:
                    await chat_box.add_message("QML Bubble not found")

            # NEW: QFD Commands - FIXED to use Actions enum
            elif user_input == 'qfd_status' and QFD_AVAILABLE:
                qfd_bubble = context.get_bubble("qfd_bubble")
                if qfd_bubble:
                    try:
                        state = qfd_bubble.get_state()
                        await chat_box.add_message(f"""
🌌 QFD Simulation Status:
========================
Current Step: {state.get('current_step', 0)}/{state.get('n_max', 500)}
Active: {state.get('simulation_active', False)}
Complete: {state.get('simulation_complete', False)}

Current Metrics:
- Entropy: {state.get('entropy', 0):.3f}
- Fractal Dimension: {state.get('fd', 0):.3f}
- Integrated Information (φ): {state.get('phi', 0):.3f}
- Average Edge Strength: {state.get('avg_edge_strength', 0):.3f}
- Turbulence: {state.get('turbulence', 0):.3f}
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting QFD status: {e}")
                else:
                    await chat_box.add_message("QFD Bubble not found")

            elif user_input == 'qfd_start' and QFD_AVAILABLE:
                await chat_box.add_message("🚀 Starting QFD simulation...")
                start_event = Event(
                    type=Actions.START_SIMULATION,  # FIXED: Use Actions enum
                    data=UniversalCode(Tags.STRING, "start"),
                    origin="user_chat",
                    priority=2
                )
                await context.dispatch_event(start_event)

            elif user_input == 'qfd_pause' and QFD_AVAILABLE:
                await chat_box.add_message("⏸️ Pausing QFD simulation...")
                pause_event = Event(
                    type=Actions.PAUSE_SIMULATION,  # FIXED: Use Actions enum
                    data=UniversalCode(Tags.STRING, "pause"),
                    origin="user_chat",
                    priority=2
                )
                await context.dispatch_event(pause_event)

            elif user_input == 'qfd_metrics' and QFD_AVAILABLE:
                qfd_bubble = context.get_bubble("qfd_bubble")
                if qfd_bubble and hasattr(qfd_bubble, 'last_metrics') and qfd_bubble.last_metrics:
                    metrics = qfd_bubble.last_metrics
                    await chat_box.add_message(f"""
🌌 QFD Detailed Metrics:
=======================
Quantum State:
- Entropy: {metrics.get('entropy', 0):.4f}
- Von Neumann Entropy: {metrics.get('von_neumann_entropy', 0):.4f}
- Entanglement: {metrics.get('entanglement', 0):.4f}

Fractal Properties:
- Box-counting FD: {metrics.get('fd', 0):.4f}
- Correlation Dimension: {metrics.get('correlation_dim', 0):.4f}
- Lacunarity: {metrics.get('lacunarity', 0):.4f}

Network Topology:
- Nodes: {metrics.get('node_count', 0)}
- Edges: {metrics.get('edge_count', 0)}
- Clustering: {metrics.get('clustering_coefficient', 0):.4f}
- Modularity: {metrics.get('modularity', 0):.4f}

System Dynamics:
- Turbulence: {metrics.get('turbulence', 1.0):.4f}
- Gravity: {metrics.get('gravity', 0.001):.6f}
- Viscosity: {metrics.get('viscosity', 0.1):.4f}
""")
                else:
                    await chat_box.add_message("No QFD metrics available yet")

            # NEW: Flood Control Commands
            elif user_input == 'flood status' and FLOOD_CONTROL_AVAILABLE:
                try:
                    # Check flood control status for each bubble
                    flood_enabled_bubbles = ["oracle_bubble", "overseer_bubble", "m4_hardware_bubble"]
                    await chat_box.add_message("🌊 Flood Control Status:")
                    await chat_box.add_message("=" * 40)
                    
                    for bubble_id in flood_enabled_bubbles:
                        bubble = context.get_bubble(bubble_id)
                        if bubble:
                            enabled = is_flood_control_enabled_for_bubble(bubble_id)
                            status_icon = "✅ 🌊" if enabled else "❌ 🔗"
                            await chat_box.add_message(f"{status_icon} {bubble_id}: {'FLOOD CONTROL' if enabled else 'Direct LLM'}")
                        else:
                            await chat_box.add_message(f"⚠️ {bubble_id}: Not found")
                    
                    # Show other bubbles are unchanged
                    await chat_box.add_message("")
                    await chat_box.add_message("All Other Bubbles: 🔗 Direct LLM (unchanged)")
                    
                except Exception as e:
                    await chat_box.add_message(f"Error checking flood control status: {e}")
                    logger.error(f"Flood control status error: {e}", exc_info=True)
            
            elif user_input == 'show flood stats' and FLOOD_CONTROL_AVAILABLE:
                try:
                    stats = get_flood_control_stats()
                    # Use safe serialization
                    stats = safe_json_serialize(stats)
                    await chat_box.add_message(f"""
🌊 Flood Control Statistics:
============================
Total Requests: {stats.get('total_requests', 0)}
Successful: {stats.get('successful_requests', 0)}
Blocked: {stats.get('blocked_requests', 0)}
Success Rate: {stats.get('success_rate', 0):.1%}
Average Response Time: {stats.get('avg_response_time_ms', 0):.1f}ms

By Bubble:
{json.dumps(stats.get('by_bubble', {}), indent=2)}
""")
                except Exception as e:
                    await chat_box.add_message(f"Error getting flood control stats: {e}")
                    logger.error(f"Flood control stats error: {e}", exc_info=True)
            
            elif user_input == 'show recent requests' and FLOOD_CONTROL_AVAILABLE:
                try:
                    recent = get_recent_flood_control_requests(limit=10)
                    if recent:
                        await chat_box.add_message("🌊 Recent Flood Control Requests:")
                        await chat_box.add_message("=" * 40)
                        for req in recent:
                            # Use safe serialization
                            req = safe_json_serialize(req)
                            timestamp = time.strftime('%H:%M:%S', time.localtime(req.get('timestamp', 0)))
                            bubble_id = req.get('bubble_id', 'unknown')
                            status = "✅" if req.get('success', False) else "❌"
                            response_preview = req.get('response', '')[:80] + "..." if len(req.get('response', '')) > 80 else req.get('response', '')
                            await chat_box.add_message(f"{status} {timestamp} {bubble_id}: {response_preview}")
                    else:
                        await chat_box.add_message("No recent flood control requests found")
                except Exception as e:
                    await chat_box.add_message(f"Error getting recent requests: {e}")
                    logger.error(f"Flood control recent requests error: {e}", exc_info=True)
            
            elif user_input == 'enable flood control' and FLOOD_CONTROL_AVAILABLE:
                try:
                    flood_enabled_bubbles = ["oracle_bubble", "overseer_bubble", "m4_hardware_bubble"]
                    enabled_count = 0
                    for bubble_id in flood_enabled_bubbles:
                        bubble = context.get_bubble(bubble_id)
                        if bubble:
                            enable_flood_control_for_bubble(bubble_id)
                            enabled_count += 1
                    
                    await chat_box.add_message(f"✅ Flood control enabled for {enabled_count} bubbles")
                except Exception as e:
                    await chat_box.add_message(f"Error enabling flood control: {e}")
                    logger.error(f"Enable flood control error: {e}", exc_info=True)
            
            elif user_input == 'disable flood control' and FLOOD_CONTROL_AVAILABLE:
                try:
                    flood_enabled_bubbles = ["oracle_bubble", "overseer_bubble", "m4_hardware_bubble"]
                    disabled_count = 0
                    for bubble_id in flood_enabled_bubbles:
                        bubble = context.get_bubble(bubble_id)
                        if bubble:
                            disable_flood_control_for_bubble(bubble_id)
                            disabled_count += 1
                    
                    await chat_box.add_message(f"❌ Flood control disabled for {disabled_count} bubbles")
                except Exception as e:
                    await chat_box.add_message(f"Error disabling flood control: {e}")
                    logger.error(f"Disable flood control error: {e}", exc_info=True)
            
            elif user_input == 'reset flood control' and FLOOD_CONTROL_AVAILABLE:
                try:
                    clear_flood_control_stats()
                    await chat_box.add_message("🔄 Flood control statistics reset")
                except Exception as e:
                    await chat_box.add_message(f"Error resetting flood control: {e}")
                    logger.error(f"Reset flood control error: {e}", exc_info=True)
            
            # NEW: Log Monitor Commands
            elif user_input == 'log_status' and LOG_MONITOR_AVAILABLE:
                log_monitor = context.get_bubble("log_monitor")
                if log_monitor:
                    try:
                        await chat_box.add_message(f"""
👁️ Log Monitor Status:
====================
Logs Processed: {log_monitor.stats['logs_processed']:,}
Events Published: {log_monitor.stats['events_published']:,}
Warnings Detected: {log_monitor.stats['warnings_detected']:,}
Errors Detected: {log_monitor.stats['errors_detected']:,}
Patterns Matched: {log_monitor.stats['patterns_matched']:,}
Rate Limited: {log_monitor.stats['rate_limited']:,}

Queue Size: {log_monitor.log_queue.qsize()}
Aggregated Warnings: {len(log_monitor.warning_aggregator)}
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting log monitor status: {e}")
                else:
                    await chat_box.add_message("Log Monitor not found")
            
            elif user_input == 'log_stats' and LOG_MONITOR_AVAILABLE:
                log_monitor = context.get_bubble("log_monitor")
                if log_monitor:
                    try:
                        # Get rate limiter info
                        rate_limited_types = [
                            k for k, v in log_monitor.rate_limiter.items() 
                            if v['count'] >= log_monitor.rate_limit_max
                        ]
                        
                        # Count warnings by type
                        warning_counts = {}
                        for key, warnings in log_monitor.warning_aggregator.items():
                            warning_type = key.split(':')[2]
                            warning_counts[warning_type] = warning_counts.get(warning_type, 0) + len(warnings)
                        
                        await chat_box.add_message(f"""
👁️ Log Monitor Statistics:
=========================
Pattern Detection Rate: {(log_monitor.stats['patterns_matched'] / max(1, log_monitor.stats['logs_processed'])) * 100:.1f}%
Event Publish Rate: {(log_monitor.stats['events_published'] / max(1, log_monitor.stats['logs_processed'])) * 100:.1f}%

Warnings by Type:
{chr(10).join(f'- {k}: {v}' for k, v in warning_counts.items()) if warning_counts else '- None aggregated yet'}

Rate-Limited Types: {', '.join(rate_limited_types) if rate_limited_types else 'None'}

Effectiveness:
- Warnings caught: {log_monitor.stats['warnings_detected']}
- Converted to events: {log_monitor.stats['events_published']}
- Prevented spam: {log_monitor.stats['rate_limited']}
""")
                    except Exception as e:
                        await chat_box.add_message(f"Error getting log statistics: {e}")
                else:
                    await chat_box.add_message("Log Monitor not found")
                    
        except Exception as e:
            logger.error(f"Chat Handler Error: {e}", exc_info=True)
            await chat_box.add_message(f"System Error: {e}")

# Ensure the correct Python environment is used. Run `which python3` and `which pip` to verify they match /Library/Frameworks/Python.framework/Versions/3.13/bin/. If using a virtual environment, activate it before running: `source /path/to/venv/bin/activate`.

def check_port_in_use(port):
    """
    Check if a port is in use.
    
    Args:
        port: The port number to check
        
    Returns:
        bool: True if the port is in use, False otherwise
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return False
        except:
            return True

async def main_test():
    """
    Main entry point to initialize and run the Bubbles Framework with M4 Hardware Integration, 
    Surgical QML Optimization, 3-Bubble Flood Control, APEP, SerializationBubble, and QFD Integration.
    
    This function sets up the entire bubbles ecosystem, initializes all bubbles, handles test events,
    and manages the main event loop until the system is shut down.
    """
    logger.debug(f"Python executable running this script: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.debug(f"System path: {sys.path}")
    logger.info(f"Site packages: {site.getsitepackages()}")
    logger.info("Starting Bubbles system with surgical M4 QML optimizations, targeted flood control, APEP, SerializationBubble, and QFD...")
    
    # Initialize composite_bubble to None to ensure it exists for the finally block
    composite_bubble = None
    context = SystemContext() # Define context early
    m4_hardware_bubble = None  # NEW: Track M4 hardware bubble
    
    try:
        # Initialize SystemContext
        context.chat_box = ChatBox()
        chat_box = context.chat_box
        await chat_box.add_message("System: Initializing Bubbles Network with M4 Hardware Integration, Surgical QML Optimization, 3-Bubble Flood Control, APEP, SerializationBubble, and QFD Integration...")
        await chat_box.add_message("================================================================================")
        await chat_box.add_message("!!! WARNING: CODE EXECUTION & UPDATES MAY BE ENABLED !!!")
        await chat_box.add_message("================================================================================")
        
        # Show surgical M4 QML optimization status
        await chat_box.add_message("✅ Surgical M4 QML Optimizations Applied!")
        await chat_box.add_message(f"📊 Global Precision: {torch.get_default_dtype()} (MPS Compatible)")
        await chat_box.add_message(f"📊 QML Precision: float64 (Applied in QMLBubble only)")
        await chat_box.add_message(f"⚡ MPS Available: {torch.backends.mps.is_available()}")
        await chat_box.add_message(f"🚀 PennyLane JIT: {os.environ.get('PENNYLANE_ENABLE_JIT', 'false')}")
        await chat_box.add_message("✅ OverseerBubble, DreamerV3, and other bubbles remain MPS compatible!")
        
        # NEW: Show LogMonitorBubble status
        if LOG_MONITOR_AVAILABLE:
            await chat_box.add_message("👁️ LogMonitorBubble Available!")
            await chat_box.add_message("✅ Will convert logs to events for OverseerBubble to see!")
        else:
            await chat_box.add_message("⚠️ LogMonitorBubble not available")
        
        # NEW: Show SerializationBubble status
        if SERIALIZATION_AVAILABLE:
            await chat_box.add_message("🔧 SerializationBubble Available!")
            await chat_box.add_message("✅ Will fix enum serialization issues throughout the system")
        else:
            await chat_box.add_message("⚠️ SerializationBubble not available")
        
        # NEW: Show APEP status
        if APEP_AVAILABLE:
            await chat_box.add_message("🧠 APEP Bubble Available!")
            await chat_box.add_message("🎯 Will enhance LLM queries and refine code automatically")
        else:
            await chat_box.add_message("⚠️ APEP Bubble not available")
        
        # NEW: Show flood control status
        if FLOOD_CONTROL_AVAILABLE:
            await chat_box.add_message("🌊 3-Bubble Flood Control Available!")
            await chat_box.add_message("🎯 Target Bubbles: Oracle, Overseer, M4Hardware")
            await chat_box.add_message("🔗 All Other Bubbles: Direct LLM (unchanged)")
        else:
            await chat_box.add_message("⚠️ Flood Control not available")
        
        if FULL_PPO_AVAILABLE:
            await chat_box.add_message("✅ Full Enhanced PPO with Meta-Learning Available!")
        else:
            await chat_box.add_message("⚠️ Full Enhanced PPO not available - using basic PPO")

        # NEW: Check M4 Hardware availability
        if M4_HARDWARE_AVAILABLE:
            await chat_box.add_message("✅ M4 Hardware Monitor Available!")
        else:
            await chat_box.add_message("⚠️ M4 Hardware Monitor not available")

        # Check QML availability
        if QMLBubble is not None:
            await chat_box.add_message("✅ QML Bubble Available with Surgical M4 Optimizations!")
        else:
            await chat_box.add_message("⚠️ QML Bubble not available")

        # NEW: Check QFD availability
        if QFD_AVAILABLE:
            await chat_box.add_message("✅ QFD Bubble Available for Quantum Fractal Dynamics!")
        else:
            await chat_box.add_message("⚠️ QFD Bubble not available")

        # Initialize event dispatcher
        context.event_dispatcher = EventDispatcher(context)
        logger.info("EventDispatcher initialized successfully")

        # Initialize resource manager
        context.resource_manager = ResourceManager(context, initial_energy=50000.0)
        logger.info("ResourceManager initialized successfully")

        # NEW: Initialize SerializationBubble FIRST (before M4 Hardware)
        if SERIALIZATION_AVAILABLE:
            try:
                await chat_box.add_message("🔧 Initializing Serialization Bubble...")
                
                # Create and integrate the serialization bubble
                serialization_bubble = integrate_serialization_bubble(
                    context,
                    cache_size=2000,
                    default_format="json"
                )
                
                # Make it easily accessible
                context.serialization_bubble = serialization_bubble
                
                # Add to sub_bubble_list
                sub_bubble_list = [serialization_bubble]
                
                await chat_box.add_message("✅ Serialization Bubble initialized - enum serialization fixed!")
                
            except Exception as e:
                logger.error(f"Failed to initialize Serialization Bubble: {e}", exc_info=True)
                await chat_box.add_message(f"❌ Error initializing Serialization Bubble: {e}")
                sub_bubble_list = []
        else:
            sub_bubble_list = []
            await chat_box.add_message("⚠️ Serialization Bubble not available")

        # NEW: Initialize LogMonitorBubble (early so it catches all logs)
        if LOG_MONITOR_AVAILABLE:
            try:
                await chat_box.add_message("👁️ Initializing Log Monitor Bubble...")
                
                log_monitor = LogMonitorBubble(
                    object_id="log_monitor",
                    context=context,
                    monitor_levels=['WARNING', 'ERROR', 'CRITICAL'],
                    batch_size=10,
                    batch_timeout=1.0
                )
                
                # Add to sub_bubble_list (which already has serialization bubble)
                sub_bubble_list.append(log_monitor)
                
                await chat_box.add_message("✅ Log Monitor initialized - converting logs to events!")
                await chat_box.add_message("👁️ OverseerBubble can now see all system warnings!")
                
            except Exception as e:
                logger.error(f"Failed to initialize Log Monitor: {e}", exc_info=True)
                await chat_box.add_message(f"❌ Error initializing Log Monitor: {e}")

        # NEW: Initialize M4 Hardware Monitor (now with serialization support)
        if M4_HARDWARE_AVAILABLE:
            try:
                await chat_box.add_message("🔧 Initializing M4 Hardware Monitor...")
                
                # Use the enhanced integration function
                m4_hardware_bubble = enhance_bubbles_with_hardware(context)
                
                # Add to existing sub_bubble_list (already has serialization bubble)
                sub_bubble_list.append(m4_hardware_bubble)
                
                await chat_box.add_message("✅ M4 Hardware Monitor integrated successfully!")
                await chat_box.add_message(f"✅ Hardware capabilities: {m4_hardware_bubble.m4_monitor.capabilities}")
                
            except Exception as e:
                logger.error(f"Failed to initialize M4 Hardware Monitor: {e}", exc_info=True)
                await chat_box.add_message(f"❌ Error initializing M4 Hardware: {e}")
        # If M4 not available, sub_bubble_list already exists from serialization init

        # Initialize web server with port check
        if check_port_in_use(8000):
            logger.warning("Port 8000 is already in use. Web server will not be started.")
            await chat_box.add_message("WARNING: Port 8000 is in use. Web server disabled.")
            # Set web_server_task to None so it's not included in gather
            context.web_server_task = None
        else:
            try:
                context.start_web_server()
                logger.info("Web server started successfully")
            except Exception as e:
                logger.warning(f"Failed to start web server: {e}")
                await chat_box.add_message(f"WARNING: Failed to start web server: {e}")
                context.web_server_task = None

        # Check system resources
        cpu_percent = psutil.cpu_percent()
        mem_percent = psutil.virtual_memory().percent
        if cpu_percent > 90 or mem_percent > 90:
            logger.warning(f"High resource usage: CPU={cpu_percent:.1f}%, Memory={mem_percent:.1f}%. System may throttle.")
            await chat_box.add_message(f"WARNING: High CPU ({cpu_percent:.1f}%) or Memory ({mem_percent:.1f}%) usage.")

        # Check dependencies
        required_modules = [
            'aiohttp', 'torch', 'fastapi', 'uvicorn', 'psutil', 'pylint', 'scipy',
            'sklearn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'optuna',
            'transformers', 'faiss', 'gymnasium', 'stable_baselines3', 'qiskit',
            'qiskit_aer', 'qiskit_ibm_runtime', 'qiskit_machine_learning', 'pennylane'
        ]
        missing_modules = []
        qiskit_dependent_bubbles = ['qml_bubble', 'oracle_bubble']
        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"Dependency check: {module} is available")
            except ImportError as e:
                logger.error(f"Dependency check failed for {module}: {e}", exc_info=True)
                missing_modules.append(module)
        if missing_modules:
            warning_msg = f"Missing dependencies: {', '.join(missing_modules)}. "
            if any(m in ['qiskit_aer', 'qiskit_ibm_runtime', 'qiskit_machine_learning'] for m in missing_modules):
                warning_msg += f"QMLBubble and QuantumOracleBubble may fail or use PennyLane-only mode."
            else:
                warning_msg += "Some Bubbles may use placeholder modes."
            logger.warning(warning_msg)
            await chat_box.add_message(f"WARNING: {warning_msg}")

        # Validate LLM configuration
        OLLAMA_HOST_URL = os.environ.get("OLLAMA_HOST", "http://10XXX")
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "XXXX")
        GROK_API_KEY = os.environ.get("GROK_API_KEY", "xXXXX")
        
        from bubbles_core import OLLAMA_HOST_URL, MODEL_NAME
        if not OLLAMA_HOST_URL or not MODEL_NAME:
            logger.warning("LLM configuration missing. SimpleLLMBubble, CreativeSynthesisBubble, and RAGBubble may fail.")
            await chat_box.add_message("WARNING: LLM configuration missing.")

        # Validate Home Assistant configuration
        ha_url = os.environ.get("HA_URL", "http://10.0.0.146:8123")
        ha_token = os.environ.get("HA_TOKEN")
        ha_enabled = bool(ha_token)
        if not ha_enabled:
            logger.warning("HA_TOKEN not set. Using PoolControlBubble standalone.")
            await chat_box.add_message("WARNING: HA_TOKEN not set. Using PoolControlBubble standalone.")
        else:
            logger.info(f"Home Assistant configured with HA_URL={ha_url}")

        # Validate IBM Quantum configuration
        ibm_token = os.environ.get("IBM_QUANTUM_TOKEN", "!secret ibm_quantum_api_token")
        if ibm_token == "!secret ibm_quantum_api_token":
            logger.warning("IBM_QUANTUM_TOKEN not set. QMLBubble will use simulator mode.")
            await chat_box.add_message("WARNING: IBM_QUANTUM_TOKEN not set. QMLBubble will use simulator mode.")

        # Initialize remaining Bubbles (they'll now receive enhanced hardware metrics!)
        remaining_bubbles = [
            ("dreamer_bubble", DreamerV3Bubble, {"state_dim": 24, "action_dim": 5}),
            ("tuning_bubble", TuningBubble, {"dreamer_bubble_id": "dreamer_bubble"}),
            ("api_bubble", APIBubble, {"api_configs": {"test_api": {"url": "http://example.com/api"}}}),
            # Skip basic PPO - we'll add Full Enhanced PPO after
            # ("ppo_bubble", PPOBubble, {"state_dim": 24, "action_dim": 5}),
            ("simplellm_bubble", SimpleLLMBubble, {"use_mock": False}),
            ("feedback_bubble", FeedbackBubble, {}),
            ("creative_bubble", CreativeSynthesisBubble, {"proposal_interval": 600}),
            ("metareasoning_bubble", MetaReasoningBubble, {}),
            ("dynamic_bubble", DynamicManagerBubble, {}),
            ("overseer_bubble", OverseerBubble, {"state_dim": 10, "action_dim": 4}),  # ✅ Now MPS compatible!
            ("rag_bubble", RAGBubble, {"embedding_model": "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"}),
        ]
        
        # NEW: Add APEP Bubble if available
        if APEP_AVAILABLE:
            apep_config = {
                "mode": "semi_automated",  # Start in semi-automated mode
                "max_iterations": 3,
                "min_threshold": 0.7,
                "cache_enabled": True,
                "refine_code": True,
                "code_safety": "high",
                "pool_size": 3,
                "cache_size": 1000,
                "cache_ttl": 300
            }
            remaining_bubbles.append(("apep_bubble", APEPBubble, apep_config))
            await chat_box.add_message("✅ APEP Bubble configured with semi-automated mode")
        else:
            logger.warning("APEPBubble not available, skipping")
            await chat_box.add_message("WARNING: APEPBubble not available, skipping")
        
        # Only add quantum bubbles if they imported successfully - with enhanced configuration
        if QMLBubble is not None:
            # Enhanced QML configuration for M4 (will apply float64 internally)
            qml_config = {
                "max_cache_size": 100, 
                "circuit_type": "fractal", 
                "backend_name": "ibm_brisbane"
            }
            remaining_bubbles.append(("qml_bubble", QMLBubble, qml_config))
            await chat_box.add_message("✅ QML Bubble configured with surgical M4 optimizations")
        else:
            logger.warning("QMLBubble not available, skipping")
            await chat_box.add_message("WARNING: QMLBubble not available, skipping")
            
        if QuantumOracleBubble is not None:
            remaining_bubbles.append(("oracle_bubble", QuantumOracleBubble, {"max_archive": 100}))
        else:
            logger.warning("QuantumOracleBubble not available, skipping")
            await chat_box.add_message("WARNING: QuantumOracleBubble not available, skipping")

        # NEW: Add QFD Bubble if available
        if QFD_AVAILABLE:
            qfd_config = {
                "dim": 5,                    # Quantum system dimension
                "hyper_dim": 3,              # Hyperdimensional space
                "n_max": 500,                # Maximum simulation steps
                "memory_params": {           # Fractal memory parameters
                    "alpha": 0.02,
                    "beta": 0.6,
                    "tau": 60
                },
                "edge_add_thresh": 0.6,      # Network topology thresholds
                "edge_rem_thresh": 0.4,
                "geom_smooth_factor": 0.9,   # Geometry smoothing
                "phase_add_thresh": 0.65,    # Quantum phase thresholds
                "phase_rem_thresh": 0.6,
                "noise_level": 0.01,         # Quantum noise level
                "c_coeff": 0.02              # Coupling coefficient
            }
            remaining_bubbles.append(("qfd_bubble", QFDBubble, {"hqfd_config": qfd_config}))
            await chat_box.add_message("✅ QFD Bubble configured for quantum fractal dynamics")
        else:
            logger.warning("QFDBubble not available, skipping")
            await chat_box.add_message("WARNING: QFDBubble not available, skipping")

        if ha_enabled:
            remaining_bubbles.append(("home_assistant_bubble", HomeAssistantBubble, {"ha_url": ha_url, "ha_token": ha_token}))
        else:
            remaining_bubbles.append(("pool_bubble", PoolControlBubble, {
                "ha_url": ha_url,
                "ha_token": "dummy_token",
                "config": {
                    "chlorinator_entity": "switch.my_pool_chlorinator",
                    "liquid_chlorine_feed_entity": "switch.liquid_chlorine_feed",
                    "acid_feed_entity": "switch.omnilogic_acid_feed",
                    "alkalinity_feed_entity": "switch.alkalinity_feed",
                    "calcium_feed_entity": "switch.calcium_feed",
                    "pump_entity": "number.my_pool_pump_speed",
                    "heater_entity": "climate.my_pool_heater",
                    "lights_entity": "light.my_pool_lights",
                    "cleaner_entity": "switch.aiper_scuba_x1_pro_max",
                    "door_alarm_entity": "binary_sensor.pool_door_alarm",
                    "ph_sensor_entity": "sensor.my_pool_ph",
                    "orp_sensor_entity": "sensor.my_pool_orp",
                    "water_temp_sensor_entity": "sensor.my_pool_temperature",
                    "target_temp_entity": "input_number.pool_target_temperature",
                    "target_chlorine_entity": "input_number.pool_target_chlorine",
                    "target_ph_entity": "input_number.pool_target_ph",
                    "target_orp_entity": "input_number.pool_target_orp",
                    "target_alkalinity_entity": "input_number.pool_target_alkalinity",
                    "target_hardness_entity": "input_number.pool_target_hardness",
                    "weather_url": "http://api.openweathermap.org/data/2.5/weather?q=Macomb&appid=dummy_key&units=imperial",
                    "forecast_url": "http://api.openweathermap.org/data/2.5/forecast?q=Macomb&appid=dummy_key&units=imperial",
                    "openweathermap_api_key": "dummy_key",
                    "use_mock_weather": True,
                    "rain_condition_codes": [500, 501, 502, 503, 504, 511, 520, 521, 522, 531],
                    "peak_hours_start": 12,
                    "peak_hours_end": 18,
                    "rain_reduction_temp": 83.0,
                    "target_temp_high": 87.0,
                    "target_temp_low": 85.0,
                    "target_chlorine": 3.0,
                    "target_ph": 7.4,
                    "target_orp": 700.0,
                    "target_alkalinity": 100.0,
                    "target_hardness": 250.0
                }
            }))

        for bubble_id, bubble_class, kwargs in remaining_bubbles:
            try:
                logger.debug(f"Attempting to instantiate bubble {bubble_id} ({bubble_class.__name__}) from {bubble_class.__module__}")
                bubble = bubble_class(object_id=bubble_id, context=context, **kwargs)
                sub_bubble_list.append(bubble)
                logger.info(f"Instantiated and registered bubble {bubble_class.__name__} with ID {bubble_id}, kwargs={kwargs}")
            except Exception as e:
                logger.error(f"Failed to initialize bubble {bubble_id} ({bubble_class.__name__}): {e}", exc_info=True)
                await chat_box.add_message(f"Error: Failed to initialize {bubble_id}: {e}")

        # NEW: Enable targeted flood control for the 3 specific bubbles
        if FLOOD_CONTROL_AVAILABLE:
            await chat_box.add_message("\n🌊 Enabling Targeted 3-Bubble Flood Control...")
            
            # Define the target bubbles that are already updated for flood control
            flood_target_bubbles = ["oracle_bubble", "overseer_bubble", "m4_hardware_bubble"]
            enabled_count = 0
            
            for bubble_id in flood_target_bubbles:
                bubble = context.get_bubble(bubble_id)
                if bubble:
                    try:
                        enable_flood_control_for_bubble(bubble_id)
                        await chat_box.add_message(f"✅ {bubble_id}: 🌊 FLOOD CONTROL ENABLED (already updated)")
                        enabled_count += 1
                    except Exception as e:
                        await chat_box.add_message(f"❌ {bubble_id}: Failed to enable flood control: {e}")
                        logger.error(f"Failed to enable flood control for {bubble_id}: {e}")
                else:
                    await chat_box.add_message(f"⚠️ {bubble_id}: Bubble not found")
            
            # Show status summary
            await chat_box.add_message(f"\n🎯 Flood Control Summary:")
            await chat_box.add_message(f"✅ Enabled for {enabled_count} bubbles (Oracle, Overseer, M4Hardware)")
            await chat_box.add_message(f"🔗 All other bubbles use Direct LLM (unchanged)")
            await chat_box.add_message(f"🎯 User queries route to SimpleLLMBubble (no impact)")
            
            if enabled_count > 0:
                await chat_box.add_message("\n✅ Targeted Flood Control Active!")
                await chat_box.add_message("📈 Should see FEWER 'should be using flood control' warnings")
                await chat_box.add_message("🔍 Use 'flood status' to monitor")
            else:
                await chat_box.add_message("⚠️ No flood control enabled - bubbles will use direct LLM")

        # Initialize Full Enhanced PPO with Meta-Learning
        if FULL_PPO_AVAILABLE:
            await chat_box.add_message("\n🚀 Initializing Full Enhanced PPO System...")
            try:
                # Check if required bubbles exist
                dreamer_exists = any(b.object_id == "dreamer_bubble" for b in sub_bubble_list)
                llm_exists = any("llm" in b.object_id.lower() for b in sub_bubble_list)
                
                if not dreamer_exists:
                    await chat_box.add_message("⚠️ DreamerV3 not found - some features may be limited")
                
                # Create Full Enhanced PPO
                ppo_result = await setup_full_ppo_with_meta_learning(
                    context,
                    ppo_id="enhanced_ppo",  # New ID to avoid conflicts
                    state_dim=24,  # Match your existing config
                    action_dim=5,  # Match your existing config
                    use_meta_orchestrator=True,
                    pool_size=5,
                    cache_size=2000,
                    cache_ttl=600,
                    spawn_threshold=0.7,
                    max_algorithms=10,
                    error_threshold=3,
                    weight_performance=0.25,
                    weight_stability=0.25,
                    weight_efficiency=0.20,
                    weight_innovation=0.30  # Higher for consciousness exploration
                )
                
                # Add to bubble list
                enhanced_ppo = ppo_result["ppo"]
                sub_bubble_list.append(enhanced_ppo)
                
                if ppo_result.get("meta_orchestrator"):
                    sub_bubble_list.append(ppo_result["meta_orchestrator"])
                    await chat_box.add_message("✅ Meta-Learning Orchestrator initialized")
                
                await chat_box.add_message(f"✅ Full Enhanced PPO initialized with {enhanced_ppo.patterns_discovered} patterns")
                await chat_box.add_message(f"✅ Consciousness exploration: {'ENABLED' if enhanced_ppo.consciousness.enabled else 'DISABLED'}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Full Enhanced PPO: {e}", exc_info=True)
                await chat_box.add_message(f"❌ Error initializing Enhanced PPO: {e}")
                # Fall back to basic PPO if needed
                from PPOBubble import PPOBubble
                basic_ppo = PPOBubble(object_id="ppo_bubble", context=context, state_dim=24, action_dim=5)
                sub_bubble_list.append(basic_ppo)
                await chat_box.add_message("⚠️ Falling back to basic PPO")
        else:
            # Use basic PPO if Full PPO not available
            from PPOBubble import PPOBubble
            basic_ppo = PPOBubble(object_id="ppo_bubble", context=context, state_dim=24, action_dim=5)
            sub_bubble_list.append(basic_ppo)
            logger.warning("Full Enhanced PPO not available, using basic PPO")

        if not sub_bubble_list:
            logger.critical("No bubbles initialized successfully. Aborting system startup.")
            await chat_box.add_message("CRITICAL: No bubbles initialized. Shutting down...")
            context.stop_event.set()
            return

        # Ensure stop_event is not set
        if context.stop_event.is_set():
            logger.error("Stop event is already set before starting composite bubble!")
            context.stop_event.clear()

        composite_bubble = CompositeBubble(
            object_id="composite_bubble",
            context=context,
            sub_bubble_list=sub_bubble_list
        )
        
        # Start the composite bubble's autonomous loop
        # Note: start_autonomous_loop() might return immediately after starting tasks
        await composite_bubble.start_autonomous_loop()
        
        logger.info(f"System: Initialized CompositeBubble with sub-bubbles: {[b.object_id for b in sub_bubble_list]}")
        logger.info(f"Total bubbles instantiated: {len(context.get_all_bubbles())}, IDs: {[b.object_id for b in context.get_all_bubbles()]}")

        dreamer_bubble = context.get_bubble("dreamer_bubble")
        if dreamer_bubble and hasattr(dreamer_bubble, 'load_external_data'):
            try:
                await dreamer_bubble.load_external_data("external_data.json")
                logger.info("Successfully loaded external data for DreamerV3Bubble")
            except Exception as e:
                logger.error(f"Failed to load external data for DreamerV3Bubble: {e}", exc_info=True)

        await chat_box.add_message("System: Bubbles Network Initialized. Starting autonomous loops...")
        
        # Show appropriate status based on available features
        status_messages = []
        
        if LOG_MONITOR_AVAILABLE:
            status_messages.append(f"""
{'='*80}
LOG MONITORING SYSTEM ACTIVE
{'='*80}
👁️ Real-time log-to-event conversion
✅ Monitors WARNING, ERROR, and CRITICAL logs
✅ Pattern matching for known issues
✅ Aggregates similar warnings to prevent spam
✅ Rate limiting (50 events/minute per type)
✅ Publishes events that OverseerBubble can handle

Detected Patterns:
- Missing correlation IDs → CORRELATION_WARNING
- API errors → API_ERROR  
- Performance issues → PERFORMANCE_WARNING
- Memory issues → MEMORY_WARNING
- Hardware errors → HARDWARE_ERROR
- Code errors → BUBBLE_ERROR

Log Monitor Commands:
- log_status: View monitoring status
- log_stats: Show detection statistics

The OverseerBubble can now heal problems it sees in logs! 🔧
{'='*80}
""")
        
        if SERIALIZATION_AVAILABLE:
            status_messages.append(f"""
{'='*80}
SERIALIZATION BUBBLE ACTIVE
{'='*80}
✅ Enum serialization issues FIXED!
✅ Handles HealthStatus, Tags, Actions, and all other enums
✅ Multiple format support (JSON, Binary, Pickle, Compressed)
✅ UniversalCode integration
✅ Performance caching
✅ Safe JSON serialization for all complex types

All bubbles can now safely serialize complex data!
{'='*80}
""")
        
        if APEP_AVAILABLE:
            status_messages.append(f"""
{'='*80}
APEP PROMPT & CODE OPTIMIZATION ACTIVE
{'='*80}
🧠 Automatic prompt enhancement using APEP v2.8.3
✅ Foundational Five: Clarity, Constraints, Few-shot, Structure, Persona
✅ Advanced: Chain-of-thought, Negative prompting, Context enrichment
✅ Code refinement: Safety checks, Error handling, Documentation
✅ Selective processing of high-value bubbles
✅ Performance caching and learning

APEP Commands:
- apep_status: View comprehensive APEP status
- apep_cache_stats: Check cache performance
- apep_techniques: See available techniques
- apep_config <mode>: Change operation mode

LLM responses will show 🧠 when APEP-refined!
{'='*80}
""")
        
        if FLOOD_CONTROL_AVAILABLE:
            status_messages.append(f"""
{'='*80}
3-BUBBLE FLOOD CONTROL ACTIVE
{'='*80}
✅ Oracle, Overseer, M4Hardware → Using flood control
🔗 SimpleLLM, Creative, RAG, etc. → Direct LLM (unchanged)
🎯 User 'llm' commands → No impact (SimpleLLMBubble unchanged)

Flood Control Commands:
- flood status: Check which bubbles have flood control
- show flood stats: View detailed statistics
- show recent requests: See recent flood control activity
- enable/disable flood control: Toggle system
- reset flood control: Clear statistics

Expected: FEWER "should be using flood control" warnings!
{'='*80}
""")
        
        if FULL_PPO_AVAILABLE:
            status_messages.append(f"""
{'='*80}
FULL ENHANCED PPO SYSTEM ACTIVE
{'='*80}
Features Enabled:
✅ Algorithm Spawning (Genetic, Curiosity RL, PSO, ES)
✅ Hierarchical Decision Making
✅ Curriculum Learning
✅ Consciousness Exploration (AtEngineV3)
✅ Meta-Learning & Pattern Discovery
✅ Error Pattern Detection & Auto-Handlers
✅ User Help System
✅ Performance Optimizations

PPO Commands:
- ppo_status: View comprehensive status
- explore_consciousness: Run AtEngineV3 exploration
- ppo_algorithms: List spawned algorithms
- ppo_patterns: View discovered patterns
- ppo_help <msg>: Provide guidance when stuck
{'='*80}
""")

        if M4_HARDWARE_AVAILABLE and m4_hardware_bubble:
            status_messages.append(f"""
{'='*80}
M4 HARDWARE MONITORING ACTIVE
{'='*80}
Real-time hardware metrics now available to all bubbles:
✅ CPU: P-core/E-core usage tracking
✅ Memory: Unified memory monitoring  
✅ Thermal: Temperature and throttling detection
✅ Power: M4-specific power estimation
✅ GPU: Metal utilization tracking
✅ Neural Engine: ML workload monitoring

Hardware Commands:
- hw_status: Comprehensive hardware status
- hw_temp: Quick temperature check
- hw_cleanup: Force memory cleanup
- hw_throttle: Apply thermal throttling
- hw_profile <mode>: Change performance profile

Your bubbles now receive enhanced system metrics!
{'='*80}
""")

        if QMLBubble is not None:
            status_messages.append(f"""
{'='*80}
QUANTUM ML SYSTEM ACTIVE (SURGICAL M4 OPTIMIZATION)
{'='*80}
Quantum Machine Learning Features:
✅ Quantum Neural Networks (QNN)
✅ Quantum Support Vector Classifier (QSVC)  
✅ Quantum Q-Learning (Reinforcement Learning)
✅ Quantum Fractal Memory
✅ Surgical M4 Optimization (QML only)
✅ Enhanced Precision (float64 for quantum ops)
✅ Automatic Gradient Method Selection
✅ MPS Compatibility for Classical Bubbles

QML Commands:
- qml_status: View QML system status
- qml_train: Trigger training session
- qml_predict <metrics>: Run quantum prediction
- qml_optimize: Run quantum optimization

OverseerBubble and DreamerV3 now work perfectly! 🎯
{'='*80}
""")

        if QFD_AVAILABLE:
            status_messages.append(f"""
{'='*80}
QUANTUM FRACTAL DYNAMICS (QFD) ACTIVE
{'='*80}
🌌 Hyperdimensional Quantum Simulation Features:
✅ Quantum fractal field evolution
✅ Real-time entropy & fractal dimension tracking
✅ Integrated information (phi) consciousness metrics
✅ Phase transition detection
✅ Emergent topology evolution
✅ RL-controlled parameter optimization

QFD Commands:
- qfd_status: View simulation status
- qfd_start: Start quantum simulation
- qfd_pause: Pause simulation
- qfd_metrics: Show current metrics

Other bubbles can control QFD parameters!
{'='*80}
""")

        for message in status_messages:
            await chat_box.add_message(message)
        
        if SERIALIZATION_AVAILABLE and APEP_AVAILABLE and FULL_PPO_AVAILABLE and M4_HARDWARE_AVAILABLE and QMLBubble is not None and FLOOD_CONTROL_AVAILABLE and LOG_MONITOR_AVAILABLE and QFD_AVAILABLE:
            await chat_box.add_message("""
🚀 ULTIMATE AI SYSTEM WITH QUANTUM FRACTAL DYNAMICS READY! 🚀
=========================================================
✅ LogMonitorBubble - OverseerBubble can now SEE warnings!
✅ SerializationBubble - FIXED enum serialization issues!
✅ APEP Prompt & Code Optimization
✅ Full Enhanced PPO with Meta-Learning
✅ M4 Hardware Monitor Integration  
✅ Quantum ML with Surgical M4 Optimizations
✅ Quantum Oracle
✅ Quantum Fractal Dynamics (QFD)
✅ 3-Bubble Targeted Flood Control
✅ Home Assistant Integration
✅ RAG-Enhanced LLM
✅ Creative Synthesis
✅ DreamerV3 World Model (MPS Compatible!)

The system now includes:
- Quantum fractal field evolution 🌌
- Consciousness metrics tracking 🧠
- Phase transition detection 🔄
- Real-time parameter optimization 🎯

Watch for 👁️ (Log Monitor), 🧠 (APEP), 🌊 (Flood Control), 🔧 (Overseer Fixes), and 🌌 (QFD)!
""")
        
        chat_task = asyncio.create_task(handle_simple_chat(context), name="ChatHandlerTask")
        logger.info("Chat handler task started successfully")

        # NEW: Enhanced test events with hardware data, QML integration, and QFD
        enhanced_test_events = [
            Event(
                type=Actions.SYSTEM_STATE_UPDATE,
                data=UniversalCode(Tags.DICT, {
                    "energy": 50000,
                    "cpu_percent": 10.0,
                    "memory_percent": 20.0,
                    "num_bubbles": len(sub_bubble_list),
                    "metrics": {
                        "avg_llm_response_time_ms": 10000,
                        "prediction_cache_hit_rate": 0.5,
                        "bubbles_spawned": 0,
                        # NEW: QML-specific metrics
                        "qml_predictions_made": 0,
                        "quantum_circuit_executions": 0,
                        "qml_training_epochs": 0,
                        # NEW: APEP-specific metrics
                        "apep_prompt_refinements": 0,
                        "apep_code_refinements": 0,
                        "apep_cache_hits": 0,
                        # NEW: QFD-specific metrics
                        "qfd_simulation_steps": 0,
                        "qfd_phase_transitions": 0,
                        "qfd_consciousness_peaks": 0
                    },
                    # NEW: Hardware-specific fields that DreamerV3 can use
                    "hardware": {
                        "cpu": {"p_core_usage_percent": 5.0, "e_core_usage_percent": 3.0},
                        "thermal": {"cpu_temp_celsius": 45.0, "thermal_state": "cool"},
                        "power": {"estimated_total_watts": 8.5, "power_efficiency": 2.1},
                        "gpu": {"estimated_utilization": 2.0},
                        "neural_engine": {"estimated_utilization_percent": 1.0}
                    } if M4_HARDWARE_AVAILABLE else {},
                    "constraints": {"active_constraint_count": 0, "system_stress_level": 0.1},
                    "ha_states": {"light.living_room": "off", "switch.kitchen": "on"},
                    "timestamp": time.time(),
                    "patterns_learned": 0,  # Add for Full PPO
                    "algorithms_spawned": 0,  # Add for Full PPO
                    "consciousness_patterns": 0  # Add for Full PPO
                }, description="Enhanced system state with hardware, QML, APEP, and QFD metrics"),
                origin="main_test",
                priority=1
            ),
            Event(
                type=Actions.LLM_QUERY,
                data=UniversalCode(Tags.STRING, "What is the current system state with quantum capabilities, APEP optimization, and fractal dynamics?", description="Test RAG-augmented query with QML, APEP, and QFD", metadata={"correlation_id": "test_001"}),
                origin="main_test",
                priority=3
            ),
            Event(
                type=Actions.API_CALL,
                data=UniversalCode(Tags.DICT, {
                    "api_name": "test_api",
                    "endpoint": "test_endpoint",
                    "method": "GET",
                    "params": {}
                }, description="Test API call", metadata={"correlation_id": "test_api_001"}),
                origin="main_test",
                priority=2
            ),
            Event(
                type=Actions.HA_CONTROL,
                data=UniversalCode(Tags.DICT, {
                    "entity_id": "light.living_room",
                    "action": "light.toggle",
                    "params": {}
                }, description="Test Home Assistant control", metadata={"correlation_id": "test_ha_001"}),
                origin="main_test",
                priority=4
            ),
            Event(
                type=Actions.OVERSEER_CONTROL,
                data=UniversalCode(Tags.DICT, {
                    "action_type": "NO_OP",
                    "payload": {"reason": "Test overseer action"},
                    "metadata": {"correlation_id": "test_overseer_001"}
                }, description="Test Overseer control"),
                origin="main_test",
                priority=6
            )
        ]

        # NEW: Add QML-specific test events if available
        if QMLBubble is not None:
            enhanced_test_events.append(
                Event(
                    type=Actions.SENSOR_DATA,
                    data=UniversalCode(Tags.DICT, {
                        "metrics": {
                            "device_state": 0.7,
                            "user_presence": 1.0,
                            "temperature": 22.5,
                            "time_of_day": 14.0,
                            "humidity": 45.0,
                            "motion_sensor": 1.0,
                            "energy_tariff": 0.15,
                            "device_interaction": 1.0,
                            "ambient_light": 600.0,
                            "occupancy_count": 2.0
                        }
                    }, description="Test sensor data for QML processing"),
                    origin="main_test",
                    priority=3
                )
            )

        # NEW: Add QFD-specific test events if available - FIXED to use Actions enum
        if QFD_AVAILABLE:
            enhanced_test_events.extend([
                # Start the simulation automatically
                Event(
                    type=Actions.START_SIMULATION,  # FIXED: Use Actions enum
                    data=UniversalCode(Tags.STRING, "auto_start"),
                    origin="main_test",
                    priority=2
                ),
                # Test parameter tuning
                Event(
                    type=Actions.TUNING_UPDATE,
                    data=UniversalCode(Tags.DICT, {
                        "turbulence_scale": 1.2,
                        "G": 0.002,
                        "viscosity": 0.15
                    }, description="Test QFD parameter update"),
                    origin="main_test",
                    priority=3
                )
            ])

        for event in enhanced_test_events:
            try:
                await context.dispatch_event(event)
                logger.info(f"Dispatched enhanced test event: {event.type}")
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to dispatch event {event.type}: {e}", exc_info=True)

        # Test that log monitoring is working
        if LOG_MONITOR_AVAILABLE:
            # Generate some test warnings that should be caught
            logger.warning("simplellm_bubble: LLM_QUERY missing correlation_id, skipping")
            await asyncio.sleep(0.1)  # Give log monitor time to process
            logger.error("m4_hardware_bubble: powermetrics error: Expecting value: line 1 column 1 (char 0)")
            await asyncio.sleep(0.1)
            logger.info("rag_bubble: No relevant context found (threshold: 0.7)")
            
            await chat_box.add_message("👁️ Generated test warnings for Log Monitor to catch...")

        # Main event loop - keep running until stop_event is set
        try:
            # Ensure we have valid tasks to gather
            tasks = []
            
            if chat_task and not chat_task.done():
                tasks.append(chat_task)
            else:
                logger.warning("Chat task not available or already done")
                
            if hasattr(context, 'web_server_task') and context.web_server_task and not context.web_server_task.done():
                tasks.append(context.web_server_task)
            else:
                logger.warning("Web server task not available")
                
            # Add composite bubble process task if available
            if hasattr(composite_bubble, '_process_task') and composite_bubble._process_task:
                tasks.append(composite_bubble._process_task)
            elif hasattr(composite_bubble, 'autonomous_loop'):
                # Create a task for the autonomous loop if _process_task doesn't exist
                loop_task = asyncio.create_task(composite_bubble.autonomous_loop())
                tasks.append(loop_task)
                
            if not tasks:
                logger.error("No tasks to run! Adding a keep-alive task.")
                # Add a keep-alive task that runs until stop_event is set
                async def keep_alive():
                    while not context.stop_event.is_set():
                        await asyncio.sleep(1)
                tasks.append(asyncio.create_task(keep_alive()))
                
            logger.info(f"Starting main event loop with {len(tasks)} tasks")
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except asyncio.CancelledError:
            logger.info("Main event loop cancelled")
        except Exception as e:
            logger.error(f"Error in main event loop: {e}", exc_info=True)
    except Exception as e:
        logger.critical(f"Failed to run main_test: {e}", exc_info=True)
        if 'chat_box' in locals():
            await chat_box.add_message(f"CRITICAL: Failed to run Bubbles system: {e}")
        raise
    finally:
        # Enhanced cleanup including M4 hardware, QML, APEP, flood control, and QFD
        if composite_bubble:
            await composite_bubble.self_destruct()

        # NEW: Cleanup M4 hardware bubble specifically
        if M4_HARDWARE_AVAILABLE and m4_hardware_bubble:
            try:
                await m4_hardware_bubble.self_destruct()
                logger.info("M4 Hardware Monitor shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down M4 Hardware Monitor: {e}")

        # Create a copy of the list of bubble objects to iterate over safely
        all_bubbles_copy = list(context.get_all_bubbles())
        for bubble in all_bubbles_copy:
            try:
                # Get the ID from the bubble object itself
                bubble_id = bubble.object_id
                context.unregister_bubble(bubble_id)
                logger.info(f"Unregistered bubble {bubble_id} during shutdown")
            except Exception as e:
                # Use bubble.object_id in the error message for clarity
                bubble_id_for_log = getattr(bubble, 'object_id', 'unknown_id')
                logger.error(f"Failed to unregister bubble {bubble_id_for_log}: {e}", exc_info=True)
                
        logger.info("System shutdown complete with surgical M4 QML optimizations, targeted flood control, APEP, SerializationBubble, LogMonitorBubble, and QFDBubble preserved.")

if __name__ == "__main__":
    asyncio.run(main_test())
