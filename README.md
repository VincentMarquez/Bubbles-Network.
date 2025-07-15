If you use this project, please give appropriate credit by citing:

Vincent Marquez, "Architecting Emergence: The Bubbles Network", 2025.
GitHub: https://github.com/VincentMarquez/Bubbles-Network.

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
