If you use this project, please give appropriate credit by citing:

Vincent Marquez, "Architecting Emergence: The Bubbles Network", 2025.
GitHub: https://github.com/VincentMarquez/Bubbles-Network.

Disclaimer:
This project is experimental and under active development. Some components are fully functional, while others are prototypes or works in progress and may not work as intended. Please use caution when integrating into production environments. Feedback and contributions are welcome!


Bubbles Network: Modular AI–Quantum Event-Driven Research Framework: Architecting Emergence: The Bubbles Network
A fully modular, event-driven AI, quantum, and real-world automation research platform.
Built for explainable, extensible, real-time AI systems in both simulation and the physical world.*





Overview
The Bubbles Network is a next-gen AI research and automation orchestration framework that combines:
- Event-driven agent architecture ("bubbles")
- Quantum fractal simulation (QFD, QML, Qiskit, PennyLane)
- Meta-learning reinforcement learning (DreamerV3, Enhanced PPO)
- LLM-powered synthesis, prompt/code auto-enhancement (APEP, Oracle, RAG)
- Live hardware/resource monitoring (M4, Mac, multi-platform)
- Home automation & IoT control (Home Assistant, pool, HVAC, lighting, sensors)
- Automated discovery of mathematical and physical laws
- Real-time web/CLI interaction, robust logging, and repair

You can run massive, composable experiments with AI, RL, and quantum algorithms, observe emergent phenomena, discover new patterns and laws in complex systems, and control real hardware in your home automated or via chat/voice/web interface.

---

Real-World Automation & IoT Integration

Beyond simulation, the Bubbles Network directly manages and adapts your real-world environment:
- Home Assistant integration: Control lights, pumps, sensors, alarms, heating/cooling, and more, via the HomeAssistantBubble or PoolControlBubble.
- Adaptive RL and Quantum control: Use AI or quantum RL to dynamically optimize pool chemistry, energy use, comfort, and device scheduling.
- Event-driven automation: Set up custom automations, triggers, and feedback loops, or let the system learn emergent patterns for you.
- Live feedback: All real-world state, metrics, and events are available to your RL agents, LLMs, and quantum models in real time.

**Example:**  
- Type `turn_on_pool` or `hw_profile performance` in chat—the system dispatches the right command, controls Home Assistant, and updates the state everywhere (including quantum/AI agents).

---

BUBBLES SYSTEM (main.py, SystemContext)
│
├── EventBus / Dispatcher
│    ├─ Event subscriptions by type, bubble, priority
│    └─ Async event routing (system-wide)
│
├── UniversalBubble (Base class for all Bubbles)
│    └─ All "Bubble" subclasses inherit core event logic
│
├── SerializationBubble (must start first)
│    ├─ UniversalCode/Tags/Actions enum serialization
│    └─ Handles special object serialization for all bubbles
│
├── LogMonitorBubble
│    ├─ Listens to logs (WARNING/ERROR)
│    ├─ Pattern detection → events for Overseer
│    └─ Aggregates & rate-limits warnings
│
├── M4HardwareBubble
│    ├─ Monitors CPU, GPU, memory, thermal, power, neural engine
│    ├─ Publishes hardware metrics/events
│    └─ Handles cleanup, throttling, diagnostics
│
├── QFDBubble
│    ├─ Quantum Fractal Dynamics simulation (HyperdimensionalQFD)
│    ├─ Emits PERFORMANCE_METRIC, PHASE_TRANSITION_DETECTED, etc.
│    ├─ Accepts TUNING_UPDATE, RL_ACTION, START_SIMULATION, PAUSE_SIMULATION
│    └─ Outputs advanced consciousness, entropy, fd, phi, topology metrics
│
├── QMLBubble
│    ├─ Quantum ML (QNN, QSVC, QRL, etc.), PennyLane+Qiskit
│    ├─ M4 optimization, precision fixes
│    ├─ Receives SENSOR_DATA
│    ├─ Accepts qml_status, qml_train, qml_predict, qml_optimize
│    └─ Publishes quantum predictions/optimizations
│
├── QuantumOracleBubble
│    ├─ Quantum information queries, code assistance
│    ├─ Uses QMLBubble for backend when needed
│    └─ Integrates with LLM and pattern recognition
│
├── DreamerV3Bubble
│    ├─ Model-based RL world model (DreamerV3)
│    ├─ RL agent (observation/policy/model loss)
│    ├─ Used by PPO/meta-RL orchestrators
│
├── FullEnhancedPPOWithMetaLearning
│    ├─ Top-level RL agent with meta-learning
│    ├─ Hierarchical, algorithm spawning, consciousness explorer
│    ├─ Can propose/tune/monitor all parameterized bubbles
│    └─ MetaLearningOrchestrator (controls, evaluates, explores)
│
├── TuningBubble
│    ├─ Auto-tuning and parameter sweep orchestrator
│    ├─ Interfaces with DreamerV3, QFD, QML for parameter updates
│
├── APIBubble
│    ├─ Handles REST API calls, connects to external services
│
├── HomeAssistantBubble / PoolControlBubble
│    ├─ Home automation control, pool management
│    └─ Receives/dispatches HA_CONTROL events
│
├── CompositeBubble
│    ├─ Container for all other bubbles (sub_bubble_list)
│    ├─ Unified lifecycle management (start/stop/self_destruct)
│
├── SimpleLLMBubble / CreativeSynthesisBubble / FeedbackBubble
│    ├─ Various LLM-based assistant, creativity, feedback, and knowledge bubbles
│    ├─ Receive LLM_QUERY, output LLM_RESPONSE
│
├── MetaReasoningBubble / DynamicManagerBubble / OverseerBubble
│    ├─ High-level reasoning, task spawning, resource allocation, self-healing
│    └─ Can spawn new Bubbles, repair, manage failures
│
├── RAGBubble
│    ├─ Retrieval-Augmented Generation (vector DB, embeddings)
│    └─ Used for better LLM responses
│
├── APEPBubble (Automatic Prompt & Code Enhancement Pipeline)
│    ├─ Refines prompts/code for LLMs
│    ├─ Applies “foundational five” and advanced techniques
│    ├─ Maintains performance cache, triggers improvements
│
├── Flood Control System
│    ├─ Limits LLM/LLM-assisted requests from Oracle/Overseer/M4Hardware
│    └─ Logs, stats, enable/disable at runtime
│
├── (If needed) PPOBubble (fallback basic RL agent)
│
├── [Possible User extensions]
│    ├─ MathematicalDiscoveryBubble (pattern/law detection, see above)
│    ├─ PhysicsDiscoveryBubble    (physics/phase transitions/etc.)
│    ├─ HypothesisTestingBubble   (validate/experiment on discoveries)
│
├── [User-Facing]
│    ├─ ChatBox      (user command input/output)
│    ├─ Web Server   (optional, exposes REST/Web interface)
│    └─ All event-driven, live status/metrics/visuals
│
└── (SystemContext ties it all together, maintains registry)

====================================================================
[ Event/Data Flow & Feedback ]
====================================================================
User / Autonomous events
   ↓
[ EventBus / Dispatcher ]
   ↓
All Bubbles (in parallel, by subscription)
   │
   ├─ Hardware events/metrics → DreamerV3, QFD, RL agents, QML, API, LogMonitor
   ├─ LLM queries/responses → LLM, Oracle, Creative, RAG, APEP, etc.
   ├─ Performance/metrics (QFD, QML, PPO, DreamerV3) → PPO, Overseer, Tuning, Discovery, LogMonitor
   ├─ Error/warning logs   → LogMonitor → Overseer
   ├─ Flood Control events → Oracle, Overseer, M4Hardware
   ├─ Serialization events → SerializationBubble
   ├─ Home Automation      → HomeAssistant/PoolControl
   └─ Parameter tuning     → QFDBubble, QMLBubble, DreamerV3Bubble, TuningBubble

Feedback:
   - Discovery/Pattern Bubbles publish events, request experiments, or escalate to user.
   - RL/MetaRL agents receive metrics, propose new actions, or spawn new algorithms.
   - OverseerBubble and MetaLearningOrchestrator coordinate repairs and system-level adaptation.
   - APEP continually refines prompts and code for all LLM flows.
   - LogMonitor triggers auto-repair and alerts via event feedback.

====================================================================
[ Composition / Inheritance ]
====================================================================
UniversalBubble
   │
   ├─ CompositeBubble
   │     └─ sub_bubble_list: [DreamerV3Bubble, QFDBubble, QMLBubble, ...]
   ├─ DreamerV3Bubble
   ├─ QFDBubble
   ├─ QMLBubble
   ├─ QuantumOracleBubble
   ├─ TuningBubble
   ├─ APIBubble
   ├─ PoolControlBubble
   ├─ HomeAssistantBubble
   ├─ FeedbackBubble
   ├─ SimpleLLMBubble
   ├─ CreativeSynthesisBubble
   ├─ MetaReasoningBubble
   ├─ DynamicManagerBubble
   ├─ OverseerBubble
   ├─ RAGBubble
   ├─ LogMonitorBubble
   ├─ SerializationBubble
   ├─ APEPBubble
   └─ (Others)

====================================================================
[ Critical Startup Sequence (Simple) ]
====================================================================
main.py
   ├─ Apply QML/M4 surgical fixes
   ├─ Import/initialize SerializationBubble
   ├─ Import/initialize LogMonitorBubble
   ├─ Import/initialize M4HardwareBubble
   ├─ Import/initialize all core/optional Bubbles
   ├─ Initialize CompositeBubble(sub_bubble_list)
   ├─ Start event loop(s)
   └─ Start chat/webserver, dispatch enhanced test events

====================================================================
[ Real-World Data/Command Example ]
====================================================================
User types "qfd_start" → ChatBox → EventDispatcher
    → QFDBubble (starts simulation)
        → emits PERFORMANCE_METRIC
            → DreamerV3, PPO, MetaLearningOrchestrator, QMLBubble, TuningBubble, Mathematical/PhysicsDiscoveryBubble, LogMonitorBubble, etc.
                → LLM/Oracle/Creative Bubbles handle queries, APEP refines prompts
                → OverseerBubble watches for errors, fixes, or spawns new bubbles
                → All bubbles can publish events or escalate to user

====================================================================









This module implements a PPO agent with some extra features like:
Breaking decisions into three parts (strategy, tactics, and actions).
A standard PPO training loop using clipped loss and advantage estimation.
A system that gradually increases task difficulty during training.
explanations for why actions were taken.
Detcting when the agent struggles and asking for help.
Running multiple algorithms and picking the best.
Managing resources efficiently with object pooling and caching.
It’s built to run inside an event-driven system that handles training and reacting to external events asynchronously.
How to use it
Create an instance by giving it a unique ID and system context. Configure basic PPO parameters if you want:


python
Copy
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

The agent will start training and responding to system events.
Things to keep in mind
This code is complex and best suited for experimentation and research, not production out of the box.
The hierarchical decision-making adds layers that make debugging harder.
Make sure to monitor memory and CPU usage and tune batch sizes accordingly.
Logging is detailed; adjust verbosity for your needs.
The system will ask for user help if it gets stuck.
Before production, test carefully and consider disabling some  features.
