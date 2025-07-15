If you use this project, please give appropriate credit by citing:

Vincent Marquez, "Architecting Emergence: The Bubbles Network", 2025.
GitHub: https://github.com/VincentMarquez/Bubbles-Network.


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
Before production, test carefully and consider disabling some advanced features.
