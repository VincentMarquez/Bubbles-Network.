Bubbles Network: Architecting Emergence
Cite this work as:
Vincent Marquez, “Architecting Emergence: The Bubbles Network,” 2025.
GitHub: https://github.com/VincentMarquez/Bubbles-Network

What is This Project?
The Bubbles Network is a modular, event-driven research toolkit for connecting and orchestrating diverse technologies—AI, quantum simulation, smart devices—using a flexible "bubble" architecture. Think of it as a digital Lego set where each “bubble” is a standalone module: AI, quantum simulation, hardware monitoring, or home automation. The goal: let these parts interact freely to spark new behaviors and insights.

This is an experimental system. Some parts are prototypes, some are robust. Use with care, especially if you’re integrating with real hardware.

How It Works
Bubbles: Each technology module is wrapped as a “bubble.” For example, there’s a bubble for hardware monitoring, a bubble for AI, one for quantum simulation, and so on.
Messaging: Bubbles communicate only by sending events (“messages”) across the system, never by direct calls. Any bubble can listen, ignore, or react to any event.
Plug-and-play: You can add, remove, or swap out bubbles without breaking the rest of the system.

AI and RL: Supports deep reinforcement learning (DreamerV3, PPO, etc.), LLM-based assistants, meta-learning, and self-improving agents.
Quantum Simulation & ML: Simulate quantum systems, run QML (via Qiskit, PennyLane), experiment with new quantum-inspired RL algorithms.
Real-World Control: Integrates with smart home platforms (Home Assistant, pool controllers) to connect AI/quantum logic to physical devices and sensors.
Self-Monitoring: Continuously tracks its own health and performance, logs issues, and has an “Overseer” that can attempt self-repair or request help.
Automated Discovery: Includes tools for automated pattern/law discovery in both simulated and real-world environments.

Core System Architecture
EventBus/Dispatcher: Routes all messages/events by type and priority.
UniversalBubble: Base class; all modules inherit core event logic.
SerializationBubble: Handles safe serialization of all events/objects.
LogMonitorBubble: Aggregates logs, detects patterns, escalates warnings.
M4HardwareBubble: Tracks system resources; can trigger cleanup or diagnostics.
QFDBubble: Runs high-dimensional quantum fractal simulations and metrics.
QMLBubble: Quantum machine learning with real hardware or simulation.
QuantumOracleBubble: Quantum info queries and LLM integration.
DreamerV3Bubble: Model-based RL (DreamerV3).
FullEnhancedPPOWithMetaLearning: Advanced PPO/Meta-RL agent.
TuningBubble: Automatic hyperparameter sweeps and tuning.
HomeAssistantBubble/PoolControlBubble: Smart home and IoT event integration.
APIBubble: REST/web API interface.
CompositeBubble: Container for bubble groups (start/stop lifecycle).
MetaReasoning/Overseer/DynamicManager: High-level orchestration, spawning, auto-repair, resource management.
RAGBubble, APEPBubble: LLM prompt optimization, retrieval-augmented search, and grounding.

See code for more bubbles and optional modules (RL, pattern discovery, UI, feedback, etc.)
Typical Startup Sequence
Hardware optimization (Apple Silicon, QML, etc.)
Start SerializationBubble (safe data transfer)
Start LogMonitorBubble and hardware/resource monitoring
Import/init all core & optional bubbles (RL, QML, automation)
Group bubbles under CompositeBubble for unified management
Start async event loop(s)
Launch CLI or web/chat interface; begin dispatching events

Example Workflow
Say you type qfd_start in the chat interface:
The command is broadcast by the EventBus.
QFDBubble receives it and starts a quantum fractal simulation.
Metrics/events produced by QFDBubble are routed to relevant RL agents, tuning bubbles, log monitors, and discovery tools.
The system can autonomously adjust parameters, spawn new agents, or alert the user, all in real time.
PPO Agent Usage Example
PPO agent with hierarchical decision layers, event-driven training, and meta-learning. Usage (Python):

python
Copy
Edit
ppo = FullEnhancedPPOWithMetaLearning(
    object_id="ppo_01",
    context=your_system_context,
    state_dim=32,
    action_dim=16,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    ppo_epochs=10,
    batch_size=64
)
await ppo._initialize_pools()
await ppo._subscribe_to_all_events()
await ppo.train()
Agent will start training and react to live events.

