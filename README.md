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

Example: 
- Type `turn_on_pool` or `hw_profile performance` in chat—the system dispatches the right command, controls Home Assistant, and updates the state everywhere (including quantum/AI agents).


---

Bubbles System Architecture

The Bubbles Network is structured around a modular, event-driven framework orchestrated by `main.py` and the `SystemContext`. The architecture is composed of multiple specialized components, each called a “bubble,” which are independently developed, event-subscribing agents. Below is a breakdown of the major components and their roles:



Core Infrastructure

EventBus / Dispatcher:
  Central message router. All events (commands, metrics, status, errors) are routed by type and priority to all subscribing bubbles asynchronously.

UniversalBubble (Base class):
  Every functional “bubble” inherits this base. It defines core event-handling, registration, and lifecycle management.

---

Core Bubbles

SerializationBubble:
  This must be initialized first. It ensures safe and robust serialization for all events and custom types (enums, UniversalCode, Tags, Actions).

LogMonitorBubble:
  Listens for WARNING and ERROR logs from all components. Detects patterns, escalates events to the Overseer, and rate-limits spammy warnings.

M4HardwareBubble:
  Monitors hardware resources (CPU, GPU, memory, thermal, power, neural engine). Publishes real-time hardware metrics and can trigger throttling, diagnostics, and cleanups.

QFDBubble:
  Runs the Quantum Fractal Dynamics simulation (HyperdimensionalQFD). Emits performance and phase transition metrics, accepts runtime tuning and RL-driven control, and exposes advanced quantum/consciousness metrics (entropy, fractal dimension, phi, topology, etc.).

QMLBubble:
  Handles quantum machine learning (QNN, QSVC, QRL) using PennyLane and Qiskit. Optimized for Apple Silicon (M4), accepts sensor data and training commands, and can publish quantum predictions or optimizations.

QuantumOracleBubble:
  Provides quantum information queries, code assistance, and integrates with both QMLBubble and LLM-based pattern recognition.

DreamerV3Bubble:
  Implements model-based RL using the DreamerV3 architecture. Works as a standalone world model or as a module in PPO/meta-RL orchestrators.

FullEnhancedPPOWithMetaLearning:
  Advanced, hierarchical RL agent with meta-learning and algorithm spawning. Coordinates, monitors, and tunes all parameterized bubbles. Includes a meta-learning orchestrator for system-wide evaluation and exploration.

TuningBubble:
  Handles automatic tuning, parameter sweeps, and interfaces with DreamerV3, QFD, and QML for parameter updates.

APIBubble:
  Provides REST API integration, allowing the system to connect and interact with external web services.

HomeAssistantBubble / PoolControlBubble:
  Bridges the event-driven AI with real-world IoT. Controls home automation, pools, and appliances, subscribing to and dispatching HA\_CONTROL events.

CompositeBubble:
  Container/manager for all other bubbles. Provides unified start/stop and destruction for grouped bubble lifecycles.

* **SimpleLLMBubble, CreativeSynthesisBubble, FeedbackBubble:**
  LLM-based modules for assistant/chat, creativity, feedback, and knowledge retrieval. They process LLM\_QUERY events and return LLM\_RESPONSE.

* **MetaReasoningBubble, DynamicManagerBubble, OverseerBubble:**
  High-level system reasoning, spawning of new agents, dynamic resource allocation, and automatic repair and recovery.

* **RAGBubble:**
  Retrieval-Augmented Generation module for semantic search and LLM grounding.

* **APEPBubble:**
  Automatic Prompt & Code Enhancement Pipeline. Refines prompts/code for LLMs using foundational and advanced prompt engineering, maintains a performance cache, and applies improvements system-wide.

* **Flood Control System:**
  Targeted rate limiter for LLM or LLM-assisted requests (for example, from Oracle, Overseer, M4Hardware). Tracks logs and can be toggled at runtime.

---

Optional or User-Defined Bubbles

* **PPOBubble:**
  Basic RL agent, used as a fallback if FullEnhancedPPO is unavailable.

* **Discovery/Pattern Bubbles:**
  MathematicalDiscoveryBubble, PhysicsDiscoveryBubble, and HypothesisTestingBubble for advanced research, pattern/law detection, phase transitions, and automated experimentation.

---

User Interface and System Control

* **ChatBox:**
  Command-line or web-based user command input/output.

* **Web Server:**
  Optional REST/web interface for monitoring and control.

* **SystemContext:**
  The central registry, maintains all bubbles, manages registration, event dispatching, and high-level configuration.

---

Event Flow and Feedback

* **User and autonomous events** are dispatched by the EventBus to all relevant bubbles.
* Hardware events, metrics, and warnings reach RL, DreamerV3, QFD, QML, API, and LogMonitor.
* LLM queries and responses are handled by all LLM/Oracle/Creative modules, with APEP continuously refining prompts.
* Performance/metrics from all major simulations and agents are routed to tuning, optimization, and meta-reasoning bubbles.
* Errors and warnings are elevated to LogMonitor and Overseer for diagnosis, repair, and even auto-spawn of new bubbles.
* Flood Control is enforced for high-risk bubbles.
* Home automation and IoT events are controlled and monitored through HomeAssistantBubble and PoolControlBubble.
* All events, discoveries, and escalations can ultimately reach the user through chat, logs, or web.

---

## **Startup Sequence**

1. Apply QML/M4-specific hardware optimizations.
2. Initialize the SerializationBubble (enabling safe serialization for the whole network).
3. Start the LogMonitorBubble and hardware monitoring.
4. Import and initialize all core and optional Bubbles, including advanced RL, quantum, and automation modules.
5. Group all initialized bubbles within the CompositeBubble for unified management.
6. Start the async event loop(s).
7. Launch the chat/web interface, and begin dispatching events.

---

## **Example: Real-World Event Handling**

When a user types `qfd_start` into the chat:

* The command is dispatched to the EventDispatcher.
* The QFDBubble receives it and starts the quantum fractal simulation.
* As metrics are produced (PERFORMANCE\_METRIC), they are automatically routed to RL agents, tuning modules, LLM assistants, mathematical/physics discovery bubbles, and the log monitor.
* These agents can respond, escalate, or take autonomous action (tuning parameters, spawning new agents, or alerting the user) based on those metrics.
* All logs, errors, and significant discoveries are handled in real time.






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
