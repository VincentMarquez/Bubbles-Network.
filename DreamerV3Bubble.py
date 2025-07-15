"""
DreamerV3Bubble: A Hybrid STORM-DreamerV3 Agent for Predictive System Management
This module implements DreamerV3Bubble, a novel model-based reinforcement learning agent
that synthesizes the best features from STORM and DreamerV3 for real-world system
management tasks.

Key Innovations:
1. Transformer-based world model (STORM) with stochastic latents for robustness
2. Imagination-based policy learning (DreamerV3) for safe exploration
3. Distributional value learning for handling diverse metric scales
4. Real telemetry integration with safety constraints
5. Epistemic uncertainty quantification for risk-aware decisions

The agent is specifically designed for production system management where:
- Actions must be validated before execution
- Exploration must happen in imagination, not on live systems
- Metrics vary wildly in scale (CPU: 0-100%, Memory: GBs, Energy: thousands)
- Long-term dependencies matter (transformer advantage)

References:
- STORM: "Efficient Stochastic Transformer based World Models for RL" (arXiv:2310.09615)
- DreamerV3: "Mastering Diverse Domains through World Models" (arXiv:2301.04104)

Version: 1.0
Author: Enhanced from community implementation
License: Same as original bubbles_core
"""

import asyncio
import json
import time
import logging
import random
import os
import pickle
from typing import Dict, Optional, Tuple, Any, List, Union
from collections import deque
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass, field
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Placeholder classes for when PyTorch is not available
    class nn: Module = object; Sequential = object; Linear = object; ReLU = object; MSELoss = object; Dropout = object; LayerNorm = object; TransformerEncoderLayer = object; TransformerEncoder = object
    class torch: Tensor = object; cat = staticmethod(lambda *a, **kw: None); stack = staticmethod(lambda *a, **kw: None); tensor = staticmethod(lambda *a, **kw: None); float32 = None; no_grad = staticmethod(lambda: nullcontext()); zeros = staticmethod(lambda *a, **kw: None); zeros_like = staticmethod(lambda *a, **kw: None); unsqueeze = staticmethod(lambda *a, **kw: None); squeeze = staticmethod(lambda *a, **kw: None); detach = staticmethod(lambda *a, **kw: None); cpu = staticmethod(lambda *a, **kw: a[0]); numpy = staticmethod(lambda *a, **kw: None); arange = staticmethod(lambda *a, **kw: None); triu = staticmethod(lambda *a, **kw: None); __version__ = "N/A"
    class optim: Adam = object
    class F: cross_entropy = staticmethod(lambda *a, **kw: None); softmax = staticmethod(lambda *a, **kw: None)
    np = None
    print("WARNING: PyTorch not found. DreamerV3Bubble will be disabled.", file=sys.stderr)

from bubbles_core import UniversalBubble, Actions, Event, UniversalCode, Tags, SystemContext, logger, EventService


@dataclass
class DreamerV3Config:
    """Configuration for SystemDreamer (STORM-DreamerV3 Hybrid) with sensible defaults."""
    # Model architecture
    state_dim: int = 24
    action_dim: int = 5
    hidden_dim: int = 512
    num_categories: int = 32
    num_classes: int = 32
    horizon: int = 16  # Aligned with DreamerV3 paper
    num_transformer_layers: int = 2
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    sequence_length: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    entropy_coeff: float = 3e-4
    ema_alpha: float = 0.98
    gamma: float = 0.997  # DreamerV3 paper value (updated from 0.985)
    lambda_: float = 0.95  # TD-lambda parameter
    beta1: float = 0.5  # STORM paper
    beta2: float = 0.1  # STORM paper
    kl_free_bits: float = 1.0  # DreamerV3 uses 1 nat
    critic_real_data_scale: float = 0.3  # DreamerV3 trains critic on both imagined and real
    
    # Buffer sizes
    replay_buffer_size: int = 10000
    validation_buffer_size: int = 1000
    sequence_buffer_size: int = 1000
    
    # Distributional RL parameters
    num_bins: int = 41
    v_min: float = -20.0
    v_max: float = 20.0
    
    # Imagination parameters
    imagination_horizon: int = 16  # Aligned with DreamerV3
    num_imagination_rollouts: int = 64
    
    # STORM stochasticity parameters
    use_stochastic_latents: bool = True
    latent_noise_std: float = 0.1
    vae_beta: float = 0.001  # VAE KL weight
    
    # Real data integration
    use_real_data: bool = False
    telemetry_source: str = "prometheus"  # prometheus, cloudwatch, custom
    telemetry_endpoint: str = ""
    telemetry_auth: Dict[str, str] = field(default_factory=dict)
    
    # Enhanced safety and security
    enable_action_validation: bool = True
    min_bubbles_threshold: int = 2
    max_cpu_threshold: float = 90.0
    max_memory_threshold: float = 85.0
    max_action_frequency: Dict[str, float] = field(default_factory=lambda: {
        "SPAWN_BUBBLE": 1.0,  # Max 1 spawn per minute
        "DESTROY_BUBBLE": 0.5,  # Max 0.5 destroys per minute
        "CODE_UPDATE": 2.0,  # Max 2 updates per minute
    })
    action_cooldowns: Dict[str, float] = field(default_factory=dict)
    telemetry_encryption_key: Optional[str] = None
    require_action_confirmation: bool = False
    safety_mode: str = "normal"  # normal, conservative, aggressive
    
    # Multi-agent support
    enable_multi_agent: bool = False
    agent_id: str = "agent_0"
    num_agents: int = 1
    agent_communication_freq: int = 100  # Steps between communication
    
    # Uncertainty quantification  
    use_epistemic_uncertainty: bool = True
    risk_aversion_factor: float = 0.1
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Ablation study support
    ablation_mode: str = "none"  # none, no_transformer, no_distributional, no_imagination
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_dir: str = "./checkpoints"
    auto_load_checkpoint: bool = True
    checkpoint_interval: int = 1000
    keep_last_n_checkpoints: int = 5
    
    # Logging and debugging
    debug_mode: bool = False
    log_interval: int = 10
    summary_interval: int = 100

class StateField:
    """Descriptor for state vector fields with validation."""
    def __init__(self, name: str, scale: float, field_type: str = "float"):
        self.name = name
        self.scale = scale
        self.field_type = field_type
    
    def normalize(self, value: float) -> float:
        """Normalize value by scale."""
        return value / self.scale
    
    def denormalize(self, value: float) -> float:
        """Denormalize value by scale."""
        result = value * self.scale
        
        if self.field_type == "percent":
            result = max(0.0, min(100.0, result))
        elif self.field_type == "int":
            result = max(0, int(round(result)))
        elif self.field_type == "probability":
            result = max(0.0, min(1.0, result))
            
        return result


class ResidualBlock(nn.Module):
    """
    Residual block with dropout for stable deep learning.
    
    Args:
        dim (int): Dimension of input/output features
        dropout_rate (float): Dropout probability for regularization
    """
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection with two FC layers."""
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out


class CategoricalDistribution:
    """Helper class for categorical distributional RL (C51-style)."""
    
    @staticmethod
    def compute_projection(next_distr: torch.Tensor, rewards: torch.Tensor, 
                          gammas: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """
        Project the next state distribution back through the Bellman equation.
        
        Args:
            next_distr: Next state value distribution (batch_size, num_bins)
            rewards: Rewards (batch_size,)
            gammas: Discount factors (batch_size,)
            bins: Bin centers (num_bins,)
            
        Returns:
            Projected distribution (batch_size, num_bins)
        """
        batch_size = next_distr.shape[0]
        num_bins = bins.shape[0]
        v_min, v_max = bins[0], bins[-1]
        delta_z = (v_max - v_min) / (num_bins - 1)
        
        # Compute projected bin values
        rewards = rewards.unsqueeze(1)
        gammas = gammas.unsqueeze(1)
        projected_bins = rewards + gammas * bins.unsqueeze(0)
        projected_bins = torch.clamp(projected_bins, v_min, v_max)
        
        # Compute projection indices
        b = (projected_bins - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Handle edge cases
        l = torch.clamp(l, 0, num_bins - 1)
        u = torch.clamp(u, 0, num_bins - 1)
        
        # Distribute probability mass
        proj_dist = torch.zeros_like(next_distr)
        
        for i in range(batch_size):
            for j in range(num_bins):
                # Distribute the probability mass
                proj_dist[i, l[i, j]] += next_distr[i, j] * (u[i, j].float() - b[i, j])
                proj_dist[i, u[i, j]] += next_distr[i, j] * (b[i, j] - l[i, j].float())
                
        return proj_dist


class STORMWorldModel(nn.Module):
    """
    STORM (Stochastic Transformer-based Object-centric Recurrent Model) World Model.
    
    This model learns to predict future states, rewards, and continuation probabilities
    using categorical representations and transformer architectures, aligned with arXiv:2310.09615.
    
    Enhanced with DreamerV3's robustness features while maintaining STORM's transformer backbone
    and stochastic latent variables for handling model-reality gaps.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                 num_categories: int = 32, num_classes: int = 32, 
                 num_layers: int = 2, num_heads: int = 8, dropout_rate: float = 0.1,
                 horizon: int = 16, num_bins: int = 41, num_objects: int = 8,
                 use_stochastic_latents: bool = True, latent_noise_std: float = 0.1):
        super().__init__()
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.input_dim = num_categories * num_classes + action_dim
        self.horizon = horizon
        self.num_bins = num_bins
        self.num_objects = num_objects  # For future object-centric extension
        self.use_stochastic_latents = use_stochastic_latents
        self.latent_noise_std = latent_noise_std
        
        # Stochastic encoder (STORM-style VAE component)
        if self.use_stochastic_latents:
            self.stochastic_encoder = nn.Sequential(
                nn.Linear(state_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * 2)  # Mean and log_var
            )
        
        # Object-centric state encoder (for future extension if needed)
        # Note: STORM paper doesn't actually use object-centric representations
        # This is kept for potential future extensions
        # self.object_encoder = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     ResidualBlock(hidden_dim, dropout_rate),
        #     nn.Linear(hidden_dim, num_objects * hidden_dim // 8)  # Object slots
        # )
        
        # State encoder to categorical representation
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, num_categories * num_classes)
        )
        
        # Reconstruction decoder (added for STORM alignment)
        self.recon_decoder = nn.Sequential(
            nn.Linear(num_categories * num_classes, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Action mixer combines categorical states with actions
        self.action_mixer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learnable positional encoding (expanded for horizon)
        self.positional_encoding = nn.Parameter(torch.randn(1, horizon + 10, hidden_dim))
        
        # Transformer for temporal modeling (GPT-like with causal masking)
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout_rate,
                activation='relu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.use_transformer = True
        else:
            # GRU fallback for ablation studies
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0
            )
            self.use_transformer = False
        
        # Decoders for various predictions
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Distributional reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_bins)
        )
        
        self.continuation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dynamics_predictor = nn.Linear(hidden_dim, num_categories * num_classes)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, 
                device: torch.device = torch.device("cpu"), hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the world model with optional stochastic latents.
        
        Args:
            states: Tensor of shape (batch_size, seq_len, state_dim)
            actions: Tensor of shape (batch_size, seq_len, action_dim)
            device: Device to run computations on
            hidden_state: Optional hidden state for stochastic encoding
            
        Returns:
            Tuple of (next_state, reward_logits, continuation, hidden_state, kl_loss, z_t, recon_state, vae_kl_loss)
        """
        batch_size, seq_len, state_dim = states.shape
        
        # Stochastic latent variables (STORM enhancement)
        vae_kl_loss = torch.tensor(0.0, device=device)
        if self.use_stochastic_latents and hidden_state is not None:
            # Encode to mean and log variance
            stoch_input = torch.cat([states[:, -1, :], hidden_state], dim=-1)
            stoch_params = self.stochastic_encoder(stoch_input)
            mean, log_var = torch.chunk(stoch_params, 2, dim=-1)
            
            # Sample using reparameterization trick
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z_stoch = mean + eps * std * self.latent_noise_std
            
            # KL divergence loss
            vae_kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            
            # Add stochastic component to states
            states = states + z_stoch.unsqueeze(1).expand(-1, seq_len, -1)[:, :, :state_dim]
        
        # Encode states to categorical representation
        state_logits = self.state_encoder(states.to(device)).view(
            batch_size, seq_len, self.num_categories, self.num_classes
        )
        
        # Add small uniform noise for numerical stability
        state_logits = 0.99 * state_logits + 0.01 * torch.ones_like(state_logits) / self.num_classes
        
        # Sample categorical latents
        dist = torch.distributions.Categorical(logits=state_logits)
        z_indices = dist.sample()
        
        # Convert to one-hot representation
        z_one_hot = torch.zeros(batch_size, seq_len, self.num_categories, self.num_classes, device=device)
        z_one_hot.scatter_(3, z_indices.unsqueeze(-1), 1.0)
        z_t = z_one_hot.view(batch_size, seq_len, -1)
        
        # Reconstruction (added for STORM)
        recon_state = self.recon_decoder(z_t.view(batch_size * seq_len, -1)).view(batch_size, seq_len, state_dim)
        
        # Combine with actions
        combined_input = torch.cat([z_t, actions.to(device)], dim=-1)
        
        if combined_input.shape[-1] != self.input_dim:
            logger.error(f"Input dimension mismatch: got {combined_input.shape[-1]}, expected {self.input_dim}")
            raise ValueError(f"Input dimension mismatch: got {combined_input.shape[-1]}, expected {self.input_dim}")
        
        # Mix state and action information
        e_t = self.action_mixer(combined_input)
        e_t = e_t + self.positional_encoding[:, :seq_len, :].to(device)
        
        # Apply transformer or GRU based on configuration
        if self.use_transformer:
            # Create causal mask for autoregressive attention
            src_mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            # Apply transformer with mask
            h = self.transformer(e_t, mask=src_mask)
        else:
            # Use GRU for ablation
            h, _ = self.gru(e_t)
            
        # Decode predictions
        next_state = self.decoder(h[:, -1, :])
        reward_logits = self.reward_predictor(h[:, -1, :])
        continuation = self.continuation_predictor(h[:, -1, :])
        
        # Dynamics prediction for KL regularization
        dynamics_logits = self.dynamics_predictor(h[:, -1, :]).view(
            batch_size, self.num_categories, self.num_classes
        )
        dynamics_dist = torch.distributions.Categorical(logits=dynamics_logits)
        
        # KL loss between current and predicted categorical distributions
        last_state_logits = state_logits[:, -1, :, :]
        kl_loss = torch.mean(torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=last_state_logits),
            dynamics_dist
        ))
        
        return next_state, reward_logits, continuation, h[:, -1, :], kl_loss, z_t[:, -1, :], recon_state[:, -1, :], vae_kl_loss

    def imagine_trajectory(self, initial_state: torch.Tensor, actor, horizon: int, 
                          device: torch.device) -> Dict[str, List[torch.Tensor]]:
        """
        Imagine a trajectory using the world model and actor.
        
        Args:
            initial_state: Initial state tensor (batch_size, state_dim)
            actor: Actor network for action selection
            horizon: Number of steps to imagine
            device: Device to run computations on
            
        Returns:
            Dictionary containing imagined states, actions, rewards, continuations, and hidden states
        """
        batch_size = initial_state.shape[0]
        trajectory = {
            'states': [initial_state],
            'actions': [],
            'rewards': [],
            'continuations': [],
            'hidden_states': [],
            'reward_logits': []
        }
        
        current_state = initial_state.unsqueeze(1)  # Add sequence dimension
        hidden = None
        
        for _ in range(horizon):
            # Get action from actor
            with torch.no_grad():
                action_dist, _ = actor(trajectory['states'][-1], hidden)
                action_idx = action_dist.sample()
                
                # Convert to one-hot
                action_one_hot = torch.zeros(batch_size, self.input_dim - self.num_categories * self.num_classes, device=device)
                action_one_hot.scatter_(1, action_idx.unsqueeze(1), 1.0)
            
            # Predict next state
            next_state, reward_logits, continuation, hidden, _, z_t, _, vae_kl = self.forward(
                current_state, action_one_hot.unsqueeze(1), device, hidden_state=hidden
            )
            
            # Store trajectory
            trajectory['states'].append(next_state)
            trajectory['actions'].append(action_idx)
            trajectory['reward_logits'].append(reward_logits)
            trajectory['continuations'].append(continuation)
            trajectory['hidden_states'].append(hidden)
            
            # Decode reward from distribution
            reward_probs = F.softmax(reward_logits, dim=-1)
            bins = self._symexp(torch.linspace(-20, 20, self.num_bins).to(device))
            reward = torch.sum(reward_probs * bins, dim=-1)
            trajectory['rewards'].append(reward)
            
            # Update state
            current_state = next_state.unsqueeze(1)
        
        return trajectory

    def _symlog(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric log transformation for numerical stability."""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    def _symexp(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric exponential (inverse of symlog)."""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class DreamerV3Bubble(UniversalBubble):
    """
    DreamerV3Bubble: A hybrid STORM-DreamerV3 agent for predictive system management.
    
    This agent combines the best of both worlds:
    - STORM's efficient transformer-based world modeling with stochastic latents
    - DreamerV3's robust training techniques (distributional RL, imagination, symlog)
    
    Designed specifically for real-world system telemetry tasks including:
    - Predictive resource allocation
    - Anomaly detection and prevention
    - Safe exploration through imagination
    
    Key Features:
    - Transformer-based world model for long-range dependencies
    - Purely imagination-based policy learning (safe for production)
    - Distributional value learning for varying metric scales
    - Epistemic uncertainty quantification for risk-aware decisions
    - Real telemetry integration (Prometheus, CloudWatch, custom)
    - Action validation and safety constraints
    - Comprehensive ablation study support
    
    Paper References:
    - STORM: arXiv:2310.09615 (Efficient Stochastic Transformer based World Models)
    - DreamerV3: arXiv:2301.04104 (Mastering Diverse Domains through World Models)
    """
    
    # State field definitions with types and scales
    STATE_FIELDS = [
        StateField("energy", 10000.0, "float"),
        StateField("cpu_percent", 100.0, "percent"),
        StateField("memory_percent", 100.0, "percent"),
        StateField("num_bubbles", 20.0, "int"),
        StateField("avg_llm_response_time_ms", 60000.0, "float"),
        StateField("code_update_count", 100.0, "int"),
        StateField("prediction_cache_hit_rate", 1.0, "probability"),
        StateField("LLM_QUERY_freq_per_min", 60.0, "float"),
        StateField("CODE_UPDATE_freq_per_min", 10.0, "float"),
        StateField("ACTION_TAKEN_freq_per_min", 60.0, "float"),
        StateField("gravity_force", 10.0, "float"),
        StateField("gravity_direction", 360.0, "float"),
        StateField("bubble_pos_x", 100.0, "float"),
        StateField("bubble_pos_y", 100.0, "float"),
        StateField("cluster_id", 10.0, "int"),
        StateField("cluster_strength", 1.0, "probability"),
        StateField("energy_avg", 10000.0, "float"),
        StateField("cpu_avg", 100.0, "float"),
        StateField("memory_avg", 100.0, "float"),
        StateField("energy_var", 100000.0, "float"),
        StateField("cpu_var", 100.0, "float"),
        StateField("memory_var", 100.0, "float"),
        StateField("energy_trend", 1000.0, "float"),
        StateField("cpu_trend", 10.0, "float"),
    ]
    
    # Action type ordering with parameterized actions support
    ACTION_TYPES = [
        Actions.ACTION_TYPE_CODE_UPDATE.name,
        Actions.ACTION_TYPE_SELF_QUESTION.name,
        Actions.ACTION_TYPE_SPAWN_BUBBLE.name,
        Actions.ACTION_TYPE_DESTROY_BUBBLE.name,
        Actions.ACTION_TYPE_NO_OP.name
    ]
    
    # Parameterized action definitions (NEW)
    # Note: Current implementation only supports discrete actions
    # To add continuous parameters, extend action vector with parameter values
    ACTION_PARAMS = {
        Actions.ACTION_TYPE_CODE_UPDATE.name: {
            "intensity": {"type": "continuous", "range": [0.0, 1.0], "default": 1.0}
        },
        Actions.ACTION_TYPE_SELF_QUESTION.name: {
            "complexity": {"type": "continuous", "range": [0.0, 1.0], "default": 1.0}
        },
        Actions.ACTION_TYPE_SPAWN_BUBBLE.name: {
            "size": {"type": "continuous", "range": [0.0, 1.0], "default": 0.5},
            "priority": {"type": "continuous", "range": [0.0, 1.0], "default": 0.5}
        },
        Actions.ACTION_TYPE_DESTROY_BUBBLE.name: {
            "target_id": {"type": "discrete", "range": [0, 20], "default": 0.0}
        },
        Actions.ACTION_TYPE_NO_OP.name: {}
    }
    
    def __init__(self, object_id: str, context: SystemContext, 
                 config: Optional[DreamerV3Config] = None, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Use provided config or defaults
        self.config = config or DreamerV3Config()
        
        # Verify dimensions match
        if len(self.STATE_FIELDS) != self.config.state_dim:
            raise ValueError(f"State fields mismatch: {len(self.STATE_FIELDS)} fields but config.state_dim={self.config.state_dim}")
        if len(self.ACTION_TYPES) != self.config.action_dim:
            raise ValueError(f"Action types mismatch: {len(self.ACTION_TYPES)} types but config.action_dim={self.config.action_dim}")
        
        # Experience replay buffers with configured sizes
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)
        self.validation_buffer = deque(maxlen=self.config.validation_buffer_size)
        self.sequence_buffer = deque(maxlen=self.config.sequence_buffer_size)
        self.current_sequence = []
        
        # Monitoring
        self.execution_count = 0
        self.nan_inf_count = 0
        self.training_metrics = {
            "state_loss": 0.0, "reward_loss": 0.0, "kl_loss": 0.0,
            "continuation_loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0,
            "entropy": 0.0, "disagreement_loss": 0.0, "recon_loss": 0.0,
            "rep_loss": 0.0, "avg_return": 0.0, "validation_loss": None
        }
        self.return_range = None
        
        # State tracking
        self.current_known_state: Optional[Dict[str, Any]] = None
        self.state_action_history: Dict[str, Tuple[Dict, Dict]] = {}
        self.state_history = deque(maxlen=20)
        
        # Async initialization
        self._initialized = False
        self._init_task = None

        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: PyTorch unavailable, switching to placeholder mode.")
            self.world_model = None
            self.world_model_ensemble = None
            self.actor = None
            self.critic = None
            self.critic_ema = None
            self.device = None
            self.bins = None
        else:
            # Initialize device with memory awareness
            self.device = self._get_best_device()
            
            # Initialize distributional bins
            self.bins = self._symexp(torch.linspace(
                self.config.v_min, self.config.v_max, self.config.num_bins
            ).to(self.device))
            
            # Initialize models
            self._initialize_models()
            
        # Setup checkpointing
        if self.config.enable_checkpointing:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{self.object_id}_checkpoint.pt")
            self.best_checkpoint_path = os.path.join(self.config.checkpoint_dir, f"{self.object_id}_best.pt")
            self.best_validation_loss = float('inf')
            
        # Multi-agent support
        if self.config.enable_multi_agent:
            self._initialize_multi_agent_communication()
            
            logger.info(f"{self.object_id}: Initialized DreamerV3Bubble (Mode: {'NN' if TORCH_AVAILABLE else 'Placeholder'}, "
                   f"Device: {self.device if TORCH_AVAILABLE else 'None'}, "
                   f"Config: hidden_dim={self.config.hidden_dim}, "
                   f"transformer_layers={self.config.num_transformer_layers}, "
                   f"checkpoint_dir={self.config.checkpoint_dir if self.config.enable_checkpointing else 'disabled'})")

    def _get_best_device(self) -> torch.device:
        """Get the best available device with memory considerations."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage if available."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / 1024**3  # GB
        elif self.device.type == "mps":
            # MPS doesn't support memory queries yet
            return None
        return None

    def _initialize_models(self):
        """Initialize all neural network models with ablation support."""
        # Apply ablation configurations
        if self.config.ablation_mode == "no_transformer":
            # Replace transformer with GRU for ablation
            self.config.num_transformer_layers = 0  # Will use GRU fallback
            logger.info(f"{self.object_id}: Ablation mode - using GRU instead of transformer")
        elif self.config.ablation_mode == "no_distributional":
            # Disable distributional RL
            self.config.num_bins = 1  # Single value output
            logger.info(f"{self.object_id}: Ablation mode - disabled distributional RL")
        elif self.config.ablation_mode == "no_imagination":
            # Will be handled in train_actor_critic
            logger.info(f"{self.object_id}: Ablation mode - training on real data only")
        
        # Main world model
        self.world_model = STORMWorldModel(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim, 
            self.config.num_categories, self.config.num_classes,
            num_layers=self.config.num_transformer_layers, 
            num_heads=self.config.num_heads, 
            dropout_rate=self.config.dropout_rate, 
            horizon=self.config.horizon,
            num_bins=self.config.num_bins,
            use_stochastic_latents=self.config.use_stochastic_latents,
            latent_noise_std=self.config.latent_noise_std
        )
        
        # Ensemble for uncertainty estimation
        self.world_model_ensemble = [
            STORMWorldModel(
                self.config.state_dim, self.config.action_dim, self.config.hidden_dim,
                self.config.num_categories, self.config.num_classes,
                num_layers=self.config.num_transformer_layers, 
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout_rate, 
                horizon=self.config.horizon,
                num_bins=self.config.num_bins,
                use_stochastic_latents=self.config.use_stochastic_latents,
                latent_noise_std=self.config.latent_noise_std
            ) for _ in range(3)
        ]
        
        # Actor-critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.critic_ema = self._build_critic()
        
        # Optimizers
        self.world_optimizer = optim.Adam(
            self.world_model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        self.ensemble_optimizers = [
            optim.Adam(m.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            for m in self.world_model_ensemble
        ]
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.learning_rate * 0.1,  # Lower LR for actor
            weight_decay=self.config.weight_decay
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.learning_rate * 0.1,  # Lower LR for critic
            weight_decay=self.config.weight_decay
        )
        
        # Move models to device
        self.world_model.to(self.device)
        for m in self.world_model_ensemble:
            m.to(self.device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.critic_ema.to(self.device)
        
        # Initialize weights
        nn.init.zeros_(self.world_model.reward_predictor[-1].weight)
        nn.init.zeros_(self.critic.net[-1].weight)
        self._update_critic_ema(1.0)
        
        # Clear buffers
        self.replay_buffer.clear()
        self.validation_buffer.clear()
        self.sequence_buffer.clear()

    async def ensure_initialized(self):
        """Ensure async initialization is complete."""
        if not self._initialized:
            if self._init_task is None:
                self._init_task = asyncio.create_task(self._async_init())
            await self._init_task

    async def _async_init(self):
        """Perform async initialization tasks."""
        try:
            await self._subscribe_to_events()
            
            # Auto-load checkpoint if available
            if self.config.enable_checkpointing and self.config.auto_load_checkpoint:
                self.load_checkpoint()
            
            self._initialized = True
            logger.info(f"{self.object_id}: Async initialization complete")
        except Exception as e:
            logger.error(f"{self.object_id}: Async initialization failed: {e}", exc_info=True)
            raise

    def debug_log(self, message: str):
        """Log debug messages only when debug mode is enabled."""
        if self.config.debug_mode:
            logger.debug(message)

    def _symlog(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric log transformation for numerical stability."""
        if not TORCH_AVAILABLE:
            return x
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def _symexp(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric exponential (inverse of symlog)."""
        if not TORCH_AVAILABLE:
            return x
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def _twohot_encode(self, y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """
        Two-hot encoding for continuous values.
        
        Args:
            y: Values to encode (supports batched input)
            bins: Bin edges
            
        Returns:
            Two-hot encoded tensor
        """
        if not TORCH_AVAILABLE:
            return torch.zeros(y.size(0), len(bins))
        
        # Preserve original shape
        original_shape = y.shape
        y_flat = y.view(-1)
        
        y_flat = y_flat.clamp(bins[0], bins[-1])
        idx = torch.searchsorted(bins, y_flat)
        idx = idx.clamp(0, len(bins) - 2)
        
        lower = bins[idx]
        upper = bins[idx + 1]
        weight = (y_flat - lower) / (upper - lower + 1e-8)
        
        twohot = torch.zeros(y_flat.size(0), len(bins), device=self.device)
        twohot.scatter_(1, idx.unsqueeze(1), 1.0 - weight.unsqueeze(1))
        twohot.scatter_(1, (idx + 1).unsqueeze(1), weight.unsqueeze(1))
        
        # Restore original shape
        if len(original_shape) > 1:
            twohot = twohot.view(*original_shape[:-1], len(bins))
        
        return twohot

    def _build_actor(self) -> Optional[nn.Module]:
        """Build the actor network for action selection with parameterized action support."""
        if not TORCH_AVAILABLE:
            return None
        
        config = self.config  # Capture for closure
        
        class ParameterizedActor(nn.Module):
            def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                        device: torch.device, dropout_rate: float, action_params: Dict):
                super().__init__()
                self.device = device
                self.hidden_dim = hidden_dim
                self.action_params = action_params
                
                # Discrete action head
                self.action_net = nn.Sequential(
                    nn.Linear(state_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    ResidualBlock(hidden_dim, dropout_rate),
                    nn.Linear(hidden_dim, action_dim)
                )
                
                # Continuous parameter heads for each action type
                self.param_nets = nn.ModuleDict()
                for action_type, params in action_params.items():
                    if params:  # If action has parameters
                        param_dim = sum(1 for p in params.values() if p["type"] == "continuous")
                        if param_dim > 0:
                            self.param_nets[action_type] = nn.Sequential(
                                nn.Linear(state_dim + hidden_dim, hidden_dim // 2),
                                nn.ReLU(),
                                nn.Linear(hidden_dim // 2, param_dim),
                                nn.Tanh()  # Output in [-1, 1], will be scaled
                            )

            def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.distributions.Categorical, Dict[str, torch.Tensor]]:
                if hidden is not None:
                    state_hidden = torch.cat([self._symlog(state), hidden], dim=-1)
                else:
                    padding = torch.zeros(state.shape[0], self.hidden_dim, device=self.device)
                    state_hidden = torch.cat([self._symlog(state), padding], dim=-1)
                
                # Get discrete action distribution
                action_logits = self.action_net(state_hidden.to(self.device))
                action_dist = torch.distributions.Categorical(logits=action_logits)
                
                # Get continuous parameters for each action type
                param_outputs = {}
                for action_type, param_net in self.param_nets.items():
                    param_outputs[action_type] = param_net(state_hidden.to(self.device))
                
                return action_dist, param_outputs
            
            def sample_action(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict[str, Any]:
                """Sample a complete action with parameters."""
                action_dist, param_outputs = self.forward(state, hidden)
                action_idx = action_dist.sample()
                
                # Get action type
                action_types = list(self.action_params.keys())
                action_type = action_types[action_idx.item()]
                
                # Get parameters for selected action
                action_dict = {"action_type": action_type, "action_idx": action_idx}
                
                if action_type in param_outputs:
                    params = param_outputs[action_type]
                    param_configs = self.action_params[action_type]
                    
                    # Scale parameters to their ranges
                    param_idx = 0
                    for param_name, param_config in param_configs.items():
                        if param_config["type"] == "continuous":
                            # Scale from [-1, 1] to [min, max]
                            param_value = params[0, param_idx]  # Assuming batch size 1
                            min_val, max_val = param_config["range"]
                            scaled_value = (param_value + 1) / 2 * (max_val - min_val) + min_val
                            action_dict[param_name] = scaled_value.item()
                            param_idx += 1
                        elif param_config["type"] == "discrete":
                            # For discrete params, use default for now
                            action_dict[param_name] = param_config["default"]
                
                return action_dict

            def _symlog(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return ParameterizedActor(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim, 
            self.device, self.config.dropout_rate, self.ACTION_PARAMS
        )

    def _build_critic(self) -> Optional[nn.Module]:
        """Build the critic network for distributional value estimation."""
        if not TORCH_AVAILABLE:
            return None
        
        config = self.config  # Capture for closure
        
        class Critic(nn.Module):
            def __init__(self, state_dim: int, hidden_dim: int, device: torch.device, 
                        dropout_rate: float, num_bins: int):
                super().__init__()
                self.device = device
                self.hidden_dim = hidden_dim
                self.num_bins = num_bins
                self.net = nn.Sequential(
                    nn.Linear(state_dim + hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    ResidualBlock(hidden_dim, dropout_rate),
                    nn.Linear(hidden_dim, num_bins)  # Distributional output
                )

            def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
                if hidden is not None:
                    state = torch.cat([self._symlog(state), hidden], dim=-1)
                else:
                    padding = torch.zeros(state.shape[0], self.hidden_dim, device=self.device)
                    state = torch.cat([self._symlog(state), padding], dim=-1)
                
                return self.net(state.to(self.device))
            
            def value(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None, 
                     bins: torch.Tensor = None) -> torch.Tensor:
                """Get expected value from distribution."""
                logits = self.forward(state, hidden)
                probs = F.softmax(logits, dim=-1)
                if bins is not None:
                    return torch.sum(probs * bins.unsqueeze(0), dim=-1)
                else:
                    # If no bins provided, just return mean of distribution
                    return torch.sum(probs * torch.arange(self.num_bins, device=self.device).float().unsqueeze(0), dim=-1)

            def _symlog(self, x: torch.Tensor) -> torch.Tensor:
                return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

        return Critic(
            self.config.state_dim, self.config.hidden_dim, 
            self.device, self.config.dropout_rate, self.config.num_bins
        )

    def _update_critic_ema(self, alpha: float = 0.02):
        """Update exponential moving average of critic parameters."""
        if not TORCH_AVAILABLE:
            return
        
        for param, ema_param in zip(self.critic.parameters(), self.critic_ema.parameters()):
            ema_param.data.mul_(1.0 - alpha).add_(param.data, alpha=alpha)

    async def _subscribe_to_events(self):
        """Subscribe to relevant system events."""
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            await EventService.subscribe(Actions.PREDICT_STATE_QUERY, self.handle_event)
            self.debug_log(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, ACTION_TAKEN, PREDICT_STATE_QUERY")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)
            raise

    def _vectorize_state(self, state: Dict) -> Optional[torch.Tensor]:
        """Convert state dictionary to tensor representation."""
        if not TORCH_AVAILABLE:
            logger.warning(f"{self.object_id}: Cannot vectorize state, PyTorch unavailable.")
            return None
        
        try:
            self.state_history.append(state)
            metrics = state.get("metrics", {})
            event_frequencies = state.get("event_frequencies", {})
            perturbation = state.get("response_time_perturbation", 0.0)

            # Chain of Thought: Compute historical context
            # Step 1: Calculate averages over history
            energy_avg = sum(s.get("energy", 0) for s in self.state_history) / len(self.state_history)
            cpu_avg = sum(s.get("cpu_percent", 0) for s in self.state_history) / len(self.state_history)
            memory_avg = sum(s.get("memory_percent", 0) for s in self.state_history) / len(self.state_history)
            
            # Step 2: Calculate variances for stability assessment
            energy_var = sum((s.get("energy", 0) - energy_avg) ** 2 for s in self.state_history) / len(self.state_history)
            cpu_var = sum((s.get("cpu_percent", 0) - cpu_avg) ** 2 for s in self.state_history) / len(self.state_history)
            memory_var = sum((s.get("memory_percent", 0) - memory_avg) ** 2 for s in self.state_history) / len(self.state_history)
            
            # Step 3: Calculate recent trends
            trend_length = min(5, len(self.state_history))
            recent_states = list(self.state_history)[-trend_length:]
            energy_trend = (recent_states[-1].get("energy", 0) - recent_states[0].get("energy", 0)) / trend_length if trend_length > 1 else 0
            cpu_trend = (recent_states[-1].get("cpu_percent", 0) - recent_states[0].get("cpu_percent", 0)) / trend_length if trend_length > 1 else 0

            # Build state vector using field definitions
            vector = []
            
            # Map field names to values
            field_values = {
                "energy": state.get("energy", 0),
                "cpu_percent": state.get("cpu_percent", 0),
                "memory_percent": state.get("memory_percent", 0),
                "num_bubbles": state.get("num_bubbles", 0),
                "avg_llm_response_time_ms": metrics.get("avg_llm_response_time_ms", 0) * (1 + perturbation),
                "code_update_count": metrics.get("code_update_count", 0),
                "prediction_cache_hit_rate": metrics.get("prediction_cache_hit_rate", 0),
                "LLM_QUERY_freq_per_min": event_frequencies.get("LLM_QUERY_freq_per_min", 0),
                "CODE_UPDATE_freq_per_min": event_frequencies.get("CODE_UPDATE_freq_per_min", 0),
                "ACTION_TAKEN_freq_per_min": event_frequencies.get("ACTION_TAKEN_freq_per_min", 0),
                "gravity_force": state.get("gravity_force", 0.0),
                "gravity_direction": state.get("gravity_direction", 0.0),
                "bubble_pos_x": state.get("bubble_pos_x", 0.0),
                "bubble_pos_y": state.get("bubble_pos_y", 0.0),
                "cluster_id": state.get("cluster_id", 0),
                "cluster_strength": state.get("cluster_strength", 0.0),
                "energy_avg": energy_avg,
                "cpu_avg": cpu_avg,
                "memory_avg": memory_avg,
                "energy_var": energy_var,
                "cpu_var": cpu_var,
                "memory_var": memory_var,
                "energy_trend": energy_trend,
                "cpu_trend": cpu_trend,
            }
            
            # Normalize using field definitions
            for field in self.STATE_FIELDS:
                value = field_values.get(field.name, 0.0)
                normalized = field.normalize(value)
                vector.append(normalized)

            tensor = torch.tensor(vector, dtype=torch.float32).to(self.device)
            
            # Add small noise for exploration
            noise = torch.normal(mean=0.0, std=0.01, size=tensor.shape, device=self.device)
            tensor = tensor + noise
            
            self.debug_log(f"_vectorize_state: tensor.shape={tensor.shape}")
            return self._symlog(tensor)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing state: {e}", exc_info=True)
            return None

    def _rebuild_models(self):
        """Rebuild all models after instability detection."""
        if not TORCH_AVAILABLE:
            return
        
        logger.info(f"{self.object_id}: Rebuilding models due to training instability")
        
        # Clear buffers
        self.replay_buffer.clear()
        self.validation_buffer.clear()
        self.sequence_buffer.clear()
        self.current_sequence.clear()
        
        # Reinitialize models
        self._initialize_models()
        
        logger.info(f"{self.object_id}: Models rebuilt successfully")

    def _vectorize_action(self, action: Dict) -> Optional[torch.Tensor]:
        """Convert action dictionary to tensor representation."""
        if not TORCH_AVAILABLE:
            logger.warning(f"{self.object_id}: Cannot vectorize action, PyTorch unavailable.")
            return None
        
        try:
            action_type_str = action.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
            
            # One-hot encoding using class constant
            vector = [1.0 if action_type_str == at else 0.0 for at in self.ACTION_TYPES]
            return torch.tensor(vector, dtype=torch.float32).to(self.device)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error vectorizing action: {e}", exc_info=True)
            return None

    def _devectorize_state(self, state_vector: torch.Tensor) -> Dict:
        """Convert state tensor back to dictionary representation."""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        try:
            vec = self._symexp(state_vector).detach().cpu().numpy()
            
            if len(vec) != self.config.state_dim:
                return {"error": f"State vector dim mismatch: got {len(vec)}, expected {self.config.state_dim}"}
            
            # Denormalize using field definitions
            state = {"metrics": {}, "event_frequencies": {}}
            
            for i, field in enumerate(self.STATE_FIELDS):
                value = field.denormalize(vec[i])
                
                # Map to appropriate dictionary location
                if field.name in ["energy", "cpu_percent", "memory_percent", "num_bubbles",
                                  "gravity_force", "gravity_direction", "bubble_pos_x", 
                                  "bubble_pos_y", "cluster_id", "cluster_strength"]:
                    state[field.name] = value
                elif field.name in ["avg_llm_response_time_ms", "code_update_count", 
                                    "prediction_cache_hit_rate"]:
                    state["metrics"][field.name] = value
                elif field.name in ["LLM_QUERY_freq_per_min", "CODE_UPDATE_freq_per_min", 
                                    "ACTION_TAKEN_freq_per_min"]:
                    state["event_frequencies"][field.name] = value
                else:
                    # Historical stats
                    state[field.name] = value
            
            # Add metadata
            state.update({
                "timestamp": time.time(),
                "categorical_confidence": 0.7,
                "continuation_probability": 0.9,
                "response_time_perturbation": 0.1
            })
            
            return state
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error devectorizing state: {e}", exc_info=True)
            return {"error": f"State devectorization failed: {e}"}

    def _devectorize_action(self, action_vector: torch.Tensor) -> str:
        """Convert action tensor back to action type string."""
        if not TORCH_AVAILABLE:
            return Actions.ACTION_TYPE_NO_OP.name
        
        try:
            action_idx = torch.argmax(action_vector).item()
            return self.ACTION_TYPES[action_idx]
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error devectorizing action: {e}", exc_info=True)
            return Actions.ACTION_TYPE_NO_OP.name

    async def fetch_real_telemetry(self) -> Optional[Dict[str, Any]]:
        """
        Fetch real telemetry data from configured source with encryption support.
        
        Returns:
            Dictionary of system metrics or None if unavailable
        """
        if not self.config.use_real_data:
            return None
            
        try:
            # Decrypt telemetry endpoint if encrypted
            endpoint = self.config.telemetry_endpoint
            if self.config.telemetry_encryption_key:
                # In production, use proper encryption library
                # This is a placeholder for demonstration
                endpoint = self._decrypt_endpoint(endpoint)
            
            if self.config.telemetry_source == "prometheus":
                # Example Prometheus query
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    params = {
                        'query': 'up',  # Replace with actual queries
                        'time': time.time()
                    }
                    headers = self.config.telemetry_auth.copy()
                    
                    # Add encryption headers if configured
                    if self.config.telemetry_encryption_key:
                        headers['X-Encryption'] = 'AES256'
                        
                    async with session.get(
                        f"{endpoint}/api/v1/query",
                        params=params,
                        headers=headers,
                        ssl=True  # Ensure SSL/TLS
                    ) as response:
                        data = await response.json()
                        
                        # Decrypt response if needed
                        if self.config.telemetry_encryption_key:
                            data = self._decrypt_telemetry_data(data)
                            
                        # Parse Prometheus response format
                        return self._parse_prometheus_metrics(data)
                        
            elif self.config.telemetry_source == "cloudwatch":
                # CloudWatch integration would go here
                pass
                
            elif self.config.telemetry_source == "custom":
                # Custom telemetry source
                return await self._fetch_custom_telemetry()
                
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to fetch telemetry: {e}")
            return None
    
    def _decrypt_endpoint(self, endpoint: str) -> str:
        """Decrypt telemetry endpoint (placeholder - use real encryption in production)."""
        # In production, use cryptography library with proper key management
        return endpoint
    
    def _decrypt_telemetry_data(self, data: Dict) -> Dict:
        """Decrypt telemetry data (placeholder - use real encryption in production)."""
        # In production, implement proper decryption
        return data
    
    def _parse_prometheus_metrics(self, data: Dict) -> Dict[str, Any]:
        """Parse Prometheus response into state format."""
        # This would parse actual Prometheus metrics
        # For now, return a template
        return {
            "energy": 10000.0,
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "num_bubbles": len(self.state_history),
            # ... other fields
        }
    
    async def _fetch_custom_telemetry(self) -> Dict[str, Any]:
        """Override this method for custom telemetry sources."""
        raise NotImplementedError("Custom telemetry source not implemented")
    
    def validate_action(self, action: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced action validation with safety constraints, rate limiting, and security checks.
        
        Args:
            action: Proposed action with parameters
            current_state: Current system state
            
        Returns:
            Validated (possibly modified) action
        """
        if not self.config.enable_action_validation:
            return action
            
        action_type = action.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
        current_time = time.time()
        
        # Enhanced safety checks based on mode
        safety_multiplier = {
            "normal": 1.0,
            "conservative": 0.5,  # More restrictive
            "aggressive": 1.5     # Less restrictive
        }.get(self.config.safety_mode, 1.0)
        
        # Rate limiting check
        if action_type in self.config.max_action_frequency:
            last_time = self.config.action_cooldowns.get(action_type, 0)
            min_interval = 60.0 / (self.config.max_action_frequency[action_type] * safety_multiplier)
            
            if current_time - last_time < min_interval:
                logger.warning(f"{self.object_id}: Action {action_type} rate limited "
                             f"(cooldown: {min_interval - (current_time - last_time):.1f}s)")
                return {"action_type": Actions.ACTION_TYPE_NO_OP.name}
        
        # Resource-based safety checks
        if action_type == Actions.ACTION_TYPE_DESTROY_BUBBLE.name:
            if current_state.get("num_bubbles", 0) <= self.config.min_bubbles_threshold:
                logger.warning(f"{self.object_id}: Blocked DESTROY_BUBBLE - too few bubbles")
                return {"action_type": Actions.ACTION_TYPE_NO_OP.name}
                
        elif action_type == Actions.ACTION_TYPE_SPAWN_BUBBLE.name:
            cpu = current_state.get("cpu_percent", 0)
            memory = current_state.get("memory_percent", 0)
            
            # Adjust thresholds based on safety mode
            cpu_threshold = self.config.max_cpu_threshold * safety_multiplier
            memory_threshold = self.config.max_memory_threshold * safety_multiplier
            
            if cpu > cpu_threshold or memory > memory_threshold:
                logger.warning(f"{self.object_id}: Blocked SPAWN_BUBBLE - resources too high "
                             f"(CPU: {cpu:.1f}%, Memory: {memory:.1f}%)")
                return {"action_type": Actions.ACTION_TYPE_NO_OP.name}
            
            # Check action parameters
            if "size" in action:
                max_size = 0.5 if self.config.safety_mode == "conservative" else 1.0
                action["size"] = min(action["size"], max_size)
                
        elif action_type == Actions.ACTION_TYPE_CODE_UPDATE.name:
            # Limit intensity based on current system load
            if "intensity" in action:
                cpu_load_factor = current_state.get("cpu_percent", 50) / 100.0
                max_intensity = (1.0 - cpu_load_factor) * safety_multiplier
                action["intensity"] = min(action.get("intensity", 0.5), max_intensity)
        
        # Update cooldown
        self.config.action_cooldowns[action_type] = current_time
        
        # Require confirmation for critical actions
        if self.config.require_action_confirmation and action_type != Actions.ACTION_TYPE_NO_OP.name:
            action["requires_confirmation"] = True
            action["safety_score"] = self._calculate_safety_score(action, current_state)
        
        return action
    
    def _calculate_safety_score(self, action: Dict[str, Any], state: Dict[str, Any]) -> float:
        """Calculate safety score for an action (0-1, higher is safer)."""
        score = 1.0
        
        # Penalize based on resource usage
        cpu_factor = state.get("cpu_percent", 0) / 100.0
        memory_factor = state.get("memory_percent", 0) / 100.0
        score *= (1.0 - 0.5 * max(cpu_factor, memory_factor))
        
        # Penalize risky actions
        risk_factors = {
            Actions.ACTION_TYPE_DESTROY_BUBBLE.name: 0.8,
            Actions.ACTION_TYPE_SPAWN_BUBBLE.name: 0.9,
            Actions.ACTION_TYPE_CODE_UPDATE.name: 0.85,
            Actions.ACTION_TYPE_SELF_QUESTION.name: 0.95,
            Actions.ACTION_TYPE_NO_OP.name: 1.0
        }
        score *= risk_factors.get(action.get("action_type"), 0.5)
        
        return max(0.0, min(1.0, score))




    def _simulate_state(self) -> Dict[str, Any]:
        """Generate a simulated state for training (fallback when no real data)."""
        # First try to get real telemetry
        if self.config.use_real_data:
            real_data = asyncio.run(self.fetch_real_telemetry())
            if real_data:
                return real_data
        state = {
            "energy": random.uniform(5000, 15000),
            "cpu_percent": random.uniform(5, 80),
            "memory_percent": random.uniform(20, 90),
            "num_bubbles": random.randint(1, 20),
            "metrics": {
                "avg_llm_response_time_ms": random.uniform(100, 5000),
                "code_update_count": random.randint(0, 50),
                "prediction_cache_hit_rate": random.uniform(0, 1),
            },
            "event_frequencies": {
                "LLM_QUERY_freq_per_min": random.uniform(0, 60),
                "CODE_UPDATE_freq_per_min": random.uniform(0, 10),
                "ACTION_TAKEN_freq_per_min": random.uniform(0, 60),
            },
            "gravity_force": random.uniform(0, 10),
            "gravity_direction": random.uniform(0, 360),
            "bubble_pos_x": random.uniform(-100, 100),
            "bubble_pos_y": random.uniform(-100, 100),
            "cluster_id": random.randint(0, 10),
            "cluster_strength": random.uniform(0, 1),
            "response_time_perturbation": random.uniform(-0.1, 0.1),
            "timestamp": time.time()
        }
        
        # Add momentum from previous states
        if self.state_history:
            prev_state = self.state_history[-1]
            energy_trend = prev_state.get("energy", 10000) * random.uniform(-0.05, 0.05)
            state["energy"] = max(0, min(20000, state["energy"] + energy_trend))
            
            cpu_load = prev_state.get("cpu_percent", 50) * random.uniform(0.9, 1.1)
            state["cpu_percent"] = max(0, min(100, cpu_load))
            
            memory_load = prev_state.get("memory_percent", 50) * random.uniform(0.95, 1.05)
            state["memory_percent"] = max(0, min(100, memory_load))
            
            state["num_bubbles"] = max(1, min(20, prev_state.get("num_bubbles", 5) + random.randint(-1, 1)))
        
        return state

    def _simulate_next_state(self, current_state: Dict[str, Any], action: Dict) -> Dict[str, Any]:
        """Simulate next state based on current state and action."""
        next_state = current_state.copy()
        action_type = action.get("action_type", Actions.ACTION_TYPE_NO_OP.name)
        metrics = next_state.get("metrics", {}).copy()
        event_frequencies = next_state.get("event_frequencies", {}).copy()
        
        system_load = random.uniform(0.8, 1.2)
        momentum = 0.9 if self.state_history else 1.0

        # Apply action effects
        if action_type == Actions.ACTION_TYPE_CODE_UPDATE.name:
            next_state["energy"] = max(0, next_state["energy"] - 200 * system_load)
            next_state["cpu_percent"] = min(100, next_state["cpu_percent"] + 10 * system_load)
            metrics["code_update_count"] = metrics.get("code_update_count", 0) + 1
            event_frequencies["CODE_UPDATE_freq_per_min"] = event_frequencies.get("CODE_UPDATE_freq_per_min", 0) + 1
            
        elif action_type == Actions.ACTION_TYPE_SELF_QUESTION.name:
            next_state["cpu_percent"] = min(100, next_state["cpu_percent"] + 5 * system_load)
            metrics["avg_llm_response_time_ms"] = metrics.get("avg_llm_response_time_ms", 1000) * random.uniform(0.9, 1.1)
            event_frequencies["LLM_QUERY_freq_per_min"] = event_frequencies.get("LLM_QUERY_freq_per_min", 0) + 2
            
        elif action_type == Actions.ACTION_TYPE_SPAWN_BUBBLE.name:
            next_state["cpu_percent"] = min(100, next_state["cpu_percent"] + 15 * system_load)
            next_state["memory_percent"] = min(100, next_state["memory_percent"] + 10 * system_load)
            next_state["num_bubbles"] = min(20, next_state["num_bubbles"] + 1)
            next_state["energy"] = max(0, next_state["energy"] - 100 * system_load)
            
        elif action_type == Actions.ACTION_TYPE_DESTROY_BUBBLE.name:
            next_state["cpu_percent"] = max(0, next_state["cpu_percent"] - 10 * system_load)
            next_state["memory_percent"] = max(0, next_state["memory_percent"] - 8 * system_load)
            next_state["num_bubbles"] = max(1, next_state["num_bubbles"] - 1)
            
        elif action_type == Actions.ACTION_TYPE_NO_OP.name:
            pass  # No operation

        # Apply environmental dynamics
        next_state["energy"] = max(0, min(20000, next_state["energy"] + random.uniform(-50, 50) * momentum))
        next_state["cpu_percent"] = max(0, min(100, next_state["cpu_percent"] * momentum + random.uniform(-5, 5) * system_load))
        next_state["memory_percent"] = max(0, min(100, next_state["memory_percent"] * momentum + random.uniform(-3, 3) * system_load))
        
        # Update physics
        next_state["gravity_force"] = max(0, min(10, next_state["gravity_force"] + random.uniform(-1, 1)))
        next_state["gravity_direction"] = (next_state["gravity_direction"] + random.uniform(-10, 10)) % 360
        next_state["bubble_pos_x"] += random.uniform(-5, 5)
        next_state["bubble_pos_y"] += random.uniform(-5, 5)
        next_state["cluster_strength"] = max(0, min(1, next_state["cluster_strength"] + random.uniform(-0.1, 0.1)))
        
        # Update metadata
        next_state["response_time_perturbation"] = random.uniform(-0.1, 0.1)
        next_state["timestamp"] = time.time()
        next_state["metrics"] = metrics
        next_state["event_frequencies"] = event_frequencies
        
        # Update derived metrics
        metrics["avg_llm_response_time_ms"] = max(100, metrics.get("avg_llm_response_time_ms", 1000) * random.uniform(0.95, 1.05))
        metrics["prediction_cache_hit_rate"] = max(0, min(1, metrics.get("prediction_cache_hit_rate", 0.5) + random.uniform(-0.05, 0.05)))
        
        return next_state  # Return the dictionary, not a tensor!



    def _compute_reward(self, prev_state: Dict[str, Any], next_state: Dict[str, Any]) -> torch.Tensor:
        """
        Compute reward signal for state transition based on system performance metrics.
        
        Args:
            prev_state: Previous system state dictionary
            next_state: Current system state dictionary
            
        Returns:
            Reward tensor scaled and bounded between -10 and 10
        """
        if not TORCH_AVAILABLE:
            return torch.tensor([0.0])
        
        # Energy efficiency component - positive reward for energy gains
        energy_delta = next_state.get("energy", 0) - prev_state.get("energy", 0)
        energy_reward = energy_delta / 1000.0
        
        # Resource utilization penalties - encourage efficient resource usage
        cpu_utilization = next_state.get("cpu_percent", 0) / 100.0
        memory_utilization = next_state.get("memory_percent", 0) / 100.0
        resource_penalty = -0.05 * cpu_utilization - 0.05 * memory_utilization
        
        # Performance incentive - reward high cache hit rates
        cache_metrics = next_state.get("metrics", {})
        cache_hit_rate = cache_metrics.get("prediction_cache_hit_rate", 0)
        performance_bonus = 0.1 * cache_hit_rate
        
        # System stability component - penalize high variance
        energy_variance = next_state.get("energy_var", 0) / 100000.0
        stability_penalty = -0.01 * energy_variance
        
        # Aggregate reward with bounds
        total_reward = (energy_reward + resource_penalty + 
                       performance_bonus + stability_penalty)
        total_reward = max(-10.0, min(10.0, total_reward))
        
        self.debug_log(
            f"{self.object_id}: Reward calculation - Total: {total_reward:.4f} "
            f"(energy: {energy_reward:.4f}, resources: {resource_penalty:.4f}, "
            f"performance: {performance_bonus:.4f}, stability: {stability_penalty:.4f})"
        )
        
        return torch.tensor([total_reward], dtype=torch.float32).to(self.device)

    def compute_epistemic_uncertainty(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute epistemic uncertainty using ensemble disagreement.
        
        Args:
            states: State tensor
            actions: Action tensor
            
        Returns:
            Uncertainty estimates for each state-action pair
        """
        if not TORCH_AVAILABLE or not self.config.use_epistemic_uncertainty:
            return torch.zeros(states.shape[0], device=self.device)
            
        with torch.no_grad():
            # Get predictions from each ensemble member
            predictions = []
            
            # Ensure correct dimensions
            if states.dim() == 2:  # (batch_size, state_dim)
                states_input = states.unsqueeze(1)  # (batch_size, 1, state_dim)
            else:  # Already has sequence dimension
                states_input = states
                
            if actions.dim() == 2:  # (batch_size, action_dim)
                actions_input = actions.unsqueeze(1)  # (batch_size, 1, action_dim)
            else:  # Already has sequence dimension
                actions_input = actions
            
            for model in self.world_model_ensemble:
                pred_state, _, _, _, _, _, _, _ = model(
                    states_input,
                    actions_input,
                    device=self.device
                )
                predictions.append(pred_state)
            
            # Compute variance across ensemble
            predictions_stack = torch.stack(predictions, dim=0)
            epistemic_uncertainty = torch.var(predictions_stack, dim=0).mean(dim=-1)
            
            return epistemic_uncertainty
    
    def get_risk_adjusted_action(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Get action with risk adjustment based on epistemic uncertainty.
        
        Args:
            state: Current state
            hidden: Hidden state
            
        Returns:
            Risk-adjusted action dictionary with parameters
        """
        if not self.actor or not self.config.use_epistemic_uncertainty:
            if self.actor:
                return self.actor.sample_action(state, hidden)
            return {"action_type": Actions.ACTION_TYPE_NO_OP.name}
            
        # Get action distribution and parameters
        action_dist, param_outputs = self.actor(state, hidden)
        
        # Sample multiple actions and evaluate uncertainty
        num_samples = 10
        action_candidates = []
        uncertainties = []
        
        for _ in range(num_samples):
            action_idx = action_dist.sample()
            action_one_hot = F.one_hot(action_idx, num_classes=self.config.action_dim).float()
            uncertainty = self.compute_epistemic_uncertainty(state, action_one_hot)
            
            # Create full action dict
            action_types = list(self.ACTION_PARAMS.keys())
            action_type = action_types[action_idx.item()]
            action_dict = {"action_type": action_type, "action_idx": action_idx}
            
            # Add parameters if available
            if action_type in param_outputs:
                params = param_outputs[action_type]
                param_configs = self.ACTION_PARAMS[action_type]
                
                param_idx = 0
                for param_name, param_config in param_configs.items():
                    if param_config["type"] == "continuous":
                        param_value = params[0, param_idx]
                        min_val, max_val = param_config["range"]
                        scaled_value = (param_value + 1) / 2 * (max_val - min_val) + min_val
                        action_dict[param_name] = scaled_value.item()
                        param_idx += 1
            
            action_candidates.append(action_dict)
            uncertainties.append(uncertainty)
        
        # Select action with lowest uncertainty (risk-averse)
        uncertainties_tensor = torch.stack(uncertainties)
        risk_adjusted_scores = -uncertainties_tensor * self.config.risk_aversion_factor
        best_idx = torch.argmax(risk_adjusted_scores).item()
        
        return action_candidates[best_idx]

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Validate tensor for NaN/Inf with detailed logging."""
        if torch.any(torch.isnan(tensor)):
            logger.error(f"{self.object_id}: NaN detected in {name}, shape={tensor.shape}")
            return False
        if torch.any(torch.isinf(tensor)):
            logger.error(f"{self.object_id}: Inf detected in {name}, shape={tensor.shape}")
            return False
        return True

    def _validate_transition(self, state: torch.Tensor, action: torch.Tensor, 
                           reward: torch.Tensor, next_state: torch.Tensor) -> bool:
        """Validate complete transition."""
        tensors = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        }
        
        for name, tensor in tensors.items():
            if not self._validate_tensor(tensor, name):
                return False
        
        return True

    def check_training_stability(self, loss: torch.Tensor) -> bool:
        """Check if training is stable and handle instability."""
        if not TORCH_AVAILABLE:
            return False
        
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_inf_count += 1
            logger.error(f"{self.object_id}: Detected unstable training (NaN/Inf loss). "
                        f"Consecutive count: {self.nan_inf_count}")
            
            if self.nan_inf_count > 2:
                # Rebuild models after repeated instability
                self._rebuild_models()
                self.nan_inf_count = 0
                logger.info(f"{self.object_id}: Rebuilt models due to repeated NaN/Inf losses.")
            else:
                # Reduce learning rate in-place
                self.config.learning_rate *= 0.5
                for opt in [self.world_optimizer, self.actor_optimizer, self.critic_optimizer] + self.ensemble_optimizers:
                    for param_group in opt.param_groups:
                        param_group['lr'] = self.config.learning_rate
                logger.info(f"{self.object_id}: Reduced learning rate to {self.config.learning_rate}")
            
            return False
        
        self.nan_inf_count = 0
        return True

    def add_transition(self, transition: Tuple[torch.Tensor, ...]):
        """Add transition to buffers and update sequence buffer."""
        # Add to regular buffers
        if random.random() < 0.8:
            self.replay_buffer.append(transition)
        else:
            self.validation_buffer.append(transition)
        
        # Add to sequence building
        self.current_sequence.append(transition)
        if len(self.current_sequence) >= self.config.sequence_length:
            self.sequence_buffer.append(list(self.current_sequence))
            self.current_sequence = self.current_sequence[1:]  # Sliding window

    def collect_transitions(self, num_transitions: int):
        """Collect simulated transitions for training."""
        skipped_count = 0
        
        for _ in range(num_transitions):
            # Generate transition
            current_state = self._simulate_state()
            action_types = [{"action_type": at} for at in self.ACTION_TYPES]
            action = random.choice(action_types)
            next_state = self._simulate_next_state(current_state, action)
            reward = self._compute_reward(current_state, next_state)
            
            # Vectorize
            state_vec = self._vectorize_state(current_state)
            action_vec = self._vectorize_action(action)
            next_state_vec = self._vectorize_state(next_state)
            
            if state_vec is None or action_vec is None or next_state_vec is None:
                skipped_count += 1
                continue
            
            # Validate transition
            if not self._validate_transition(state_vec, action_vec, reward, next_state_vec):
                skipped_count += 1
                if skipped_count == 1:  # Only log first occurrence
                    logger.warning(f"{self.object_id}: Skipping invalid transition")
                continue
            
            # Add valid transition
            transition = (state_vec, action_vec, reward, next_state_vec)
            self.add_transition(transition)
        
        if skipped_count > 0:
            logger.warning(f"{self.object_id}: Skipped {skipped_count} invalid transitions")
        
        logger.info(f"{self.object_id}: Collected {num_transitions} transitions. "
                   f"Replay: {len(self.replay_buffer)}, "
                   f"Validation: {len(self.validation_buffer)}, "
                   f"Sequences: {len(self.sequence_buffer)}")

    async def compute_validation_loss(self) -> Optional[float]:
        """Compute validation loss on held-out data."""
        if not TORCH_AVAILABLE or not self.world_model or len(self.validation_buffer) == 0:
            return None
        
        try:
            batch_size = min(self.config.batch_size, len(self.validation_buffer))
            batch = random.sample(self.validation_buffer, batch_size)
            
            states, actions, rewards, next_states = zip(*batch)
            states_tensor = torch.stack(states).to(self.device)
            actions_tensor = torch.stack(actions).to(self.device)
            rewards_tensor = torch.stack(rewards).to(self.device)
            next_states_tensor = torch.stack(next_states).to(self.device)

            self.world_model.eval()
            with torch.no_grad():
                # Handle dimensions properly - add sequence dimension for single transitions
                states_input = states_tensor.unsqueeze(1)  # (batch_size, 1, state_dim)
                actions_input = actions_tensor.unsqueeze(1)  # (batch_size, 1, action_dim)
                
                predicted_next_states, reward_logits, predicted_continuations, _, kl_loss, _, recon_state, vae_kl_loss = self.world_model(
                    states_input, actions_input, device=self.device
                )
                
                # State prediction loss
                state_loss = torch.mean((predicted_next_states - next_states_tensor) ** 2)
                
                # Distributional reward loss
                rewards_tensor = torch.clamp(rewards_tensor, min=self.config.v_min, max=self.config.v_max)
                reward_targets = self._twohot_encode(rewards_tensor, self.bins)
                reward_loss = F.cross_entropy(reward_logits, reward_targets, reduction='mean')
                
                # Continuation loss
                continuation_tensor = torch.ones(batch_size, 1, dtype=torch.float32).to(self.device)
                continuation_loss = nn.BCELoss()(predicted_continuations, continuation_tensor)
                
                # Reconstruction loss
                recon_loss = torch.mean((recon_state - states_tensor) ** 2)
                
                # Total loss (STORM-aligned)
                total_loss = state_loss + reward_loss + continuation_loss + self.config.beta1 * kl_loss + self.config.beta2 * recon_loss
                
                self.debug_log(f"{self.object_id}: Validation loss components: "
                              f"state={state_loss.item():.4f}, reward={reward_loss.item():.4f}, "
                              f"continuation={continuation_loss.item():.4f}, kl={kl_loss.item():.4f}, recon={recon_loss.item():.4f}")
                
                return total_loss.item()
                
        except Exception as e:
            logger.error(f"{self.object_id}: Validation loss computation error: {e}", exc_info=True)
            return None

    @contextmanager
    def memory_efficient_compute(self):
        """Context manager for memory-efficient computation."""
        try:
            # Clear cache before
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            yield
            
        finally:
            # Clear cache after
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    @contextmanager
    def training_error_handler(self, operation: str):
        """Context manager for handling training errors."""
        try:
            yield
        except torch.cuda.OutOfMemoryError:
            logger.error(f"{self.object_id}: OOM during {operation}")
            self._handle_oom()
        except Exception as e:
            logger.error(f"{self.object_id}: Error in {operation}: {e}", exc_info=True)
            raise

    def _handle_oom(self):
        """Handle out-of-memory errors."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Reduce batch size
        self.config.batch_size = max(1, self.config.batch_size // 2)
        logger.info(f"{self.object_id}: Reduced batch size to {self.config.batch_size} due to OOM")

    async def train_actor_critic(self):
        """
        Train actor and critic networks using PURELY IMAGINED rollouts.
        This is the key DreamerV3 innovation - we train on imagined data, not real transitions.
        
        Supports ablation mode where we can train on real data for comparison.
        """
        if not TORCH_AVAILABLE or not self.actor or not self.critic or len(self.replay_buffer) < 10:
            self.debug_log(f"{self.object_id}: Skipping actor-critic training "
                          f"(Torch: {TORCH_AVAILABLE}, Buffer: {len(self.replay_buffer)})")
            return
        
        # Ablation: Train on real data instead of imagination
        if self.config.ablation_mode == "no_imagination":
            await self._train_actor_critic_on_real_data()
            return
        
        with self.training_error_handler("actor-critic training"):
            with self.memory_efficient_compute():
                # Step 1: Sample starting states from replay buffer
                num_starts = min(self.config.num_imagination_rollouts, len(self.replay_buffer))
                start_transitions = random.sample(self.replay_buffer, num_starts)
                start_states = torch.stack([t[0] for t in start_transitions]).to(self.device)
                
                self.debug_log(f"train_actor_critic: Imagining from {num_starts} start states")
                
                self.world_model.eval()  # World model in eval mode
                self.actor.train()
                self.critic.train()

                # Step 2: Imagine trajectories from these start states
                all_trajectories = []
                
                for i in range(num_starts):
                    trajectory = self.world_model.imagine_trajectory(
                        start_states[i:i+1], 
                        self.actor, 
                        self.config.imagination_horizon,
                        self.device
                    )
                    all_trajectories.append(trajectory)
                
                # Step 3: Process imagined trajectories for training
                # Collect all imagined transitions
                all_states = []
                all_actions = []
                all_rewards = []
                all_continuations = []
                all_hidden_states = []
                all_values = []
                
                for traj in all_trajectories:
                    # Skip the initial state (index 0) as we need state-action-next_state tuples
                    for t in range(len(traj['states']) - 1):
                        all_states.append(traj['states'][t])
                        all_actions.append(traj['actions'][t])
                        all_rewards.append(traj['rewards'][t])
                        all_continuations.append(traj['continuations'][t])
                        all_hidden_states.append(traj['hidden_states'][t])
                        
                        # Get value estimates
                        with torch.no_grad():
                            value_logits = self.critic(traj['states'][t], traj['hidden_states'][t])
                            value_probs = F.softmax(value_logits, dim=-1)
                            value = torch.sum(value_probs * self.bins.unsqueeze(0), dim=-1)
                            all_values.append(value)

                # Convert lists to tensors using cat to avoid extra dim
                states_tensor = torch.cat(all_states, dim=0)  # (num_transitions, state_dim)
                actions_tensor = torch.cat(all_actions, dim=0)  # (num_transitions,)
                rewards_tensor = torch.cat(all_rewards, dim=0)  # (num_transitions,)
                continuations_tensor = torch.cat(all_continuations, dim=0).squeeze(-1)  # (num_transitions,)
                hidden_states_tensor = torch.cat(all_hidden_states, dim=0)  # (num_transitions, hidden_dim)
                values_tensor = torch.cat(all_values, dim=0).squeeze(-1)  # (num_transitions,)
                
                # Step 4: Compute lambda returns for all imagined transitions
                num_transitions = states_tensor.shape[0]
                returns = torch.zeros(num_transitions, device=self.device)
                
                # Process each trajectory separately for return calculation
                idx = 0
                for traj_idx, traj in enumerate(all_trajectories):
                    traj_len = len(traj['states']) - 1
                    
                    # Get value of final state in trajectory
                    with torch.no_grad():
                        final_value_logits = self.critic_ema(
                            traj['states'][-1], 
                            traj['hidden_states'][-1] if traj['hidden_states'] else None
                        )
                        final_value_probs = F.softmax(final_value_logits, dim=-1)
                        final_value = torch.sum(final_value_probs * self.bins.unsqueeze(0), dim=-1)
                    
                    # Backward pass to compute lambda returns
                    lambda_return = final_value.squeeze()
                    for t in reversed(range(traj_len)):
                        r = rewards_tensor[idx + t].squeeze()
                        c = continuations_tensor[idx + t]
                        lambda_return = r + self.config.gamma * c * lambda_return
                        returns[idx + t] = lambda_return
                    
                    idx += traj_len
                
                # Normalize returns
                if self.return_range is None:
                    self.return_range = torch.tensor(1.0, device=self.device)
                else:
                    percentiles = torch.quantile(returns, torch.tensor([0.05, 0.95], device=self.device))
                    range_estimate = percentiles[1] - percentiles[0]
                    self.return_range = self.config.ema_alpha * self.return_range + (1 - self.config.ema_alpha) * range_estimate
                
                norm_factor = torch.max(torch.tensor(1.0, device=self.device), self.return_range)
                
                # Step 5: Compute advantages
                advantages = (returns - values_tensor) / norm_factor
                advantages = torch.clamp(advantages, min=-5.0, max=5.0)
                
                # IMPORTANT: Detach hidden states to prevent gradient issues
                hidden_states_detached = hidden_states_tensor.detach()
                
                # Step 6: Actor loss (policy gradient with entropy bonus)
                self.actor_optimizer.zero_grad()
                
                # Recompute action distributions for gradient flow
                action_dists = []
                for i in range(num_transitions):
                    # Use detached hidden states for actor
                    dist, _ = self.actor(states_tensor[i:i+1], hidden_states_detached[i:i+1])
                    action_dists.append(dist)
                
                # Compute log probabilities
                log_probs = torch.stack([
                    dist.log_prob(actions_tensor[i]) 
                    for i, dist in enumerate(action_dists)
                ])
                
                # Policy gradient loss
                actor_loss = -(log_probs * advantages.detach()).mean()
                
                # Entropy bonus for exploration
                entropy = torch.stack([dist.entropy() for dist in action_dists]).mean()
                
                # Exploration bonus from ensemble disagreement
                with torch.no_grad():
                    ensemble_next_states = []
                    for m in self.world_model_ensemble:
                        # Use only a subset for efficiency
                        sample_indices = torch.randperm(num_transitions)[:min(32, num_transitions)]
                        sampled_states = states_tensor[sample_indices]
                        sampled_actions_one_hot = F.one_hot(actions_tensor[sample_indices], num_classes=self.config.action_dim).float()
                        
                        # Add sequence dimension since these are single transitions
                        sampled_states_input = sampled_states.unsqueeze(1)
                        sampled_actions_input = sampled_actions_one_hot.unsqueeze(1)
                        
                        next_state, _, _, _, _, _, _, _ = m(sampled_states_input, sampled_actions_input, device=self.device)
                        ensemble_next_states.append(next_state)
                    
                    disagreement = torch.var(torch.stack(ensemble_next_states, dim=0), dim=0).mean()
                    disagreement_bonus = 0.01 * disagreement
                
                total_actor_loss = actor_loss - self.config.entropy_coeff * entropy + disagreement_bonus
                
                if self.check_training_stability(total_actor_loss):
                    # Use retain_graph=True since we'll need the graph for critic training
                    total_actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.config.gradient_clip_norm)
                    self.actor_optimizer.step()
                
                # Step 7: Critic loss (distributional TD learning)
                self.critic_optimizer.zero_grad()
                
                # Recompute critic predictions for gradient flow with fresh hidden states
                critic_logits = self.critic(states_tensor, hidden_states_tensor)
                
                # Target distribution (two-hot encoding of returns)
                returns_clamped = torch.clamp(returns, min=self.config.v_min, max=self.config.v_max)
                returns_twohot = self._twohot_encode(returns_clamped, self.bins)
                
                # Categorical cross-entropy loss
                critic_loss = F.cross_entropy(critic_logits, returns_twohot, reduction='mean')
                
                if self.check_training_stability(critic_loss):
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.config.gradient_clip_norm)
                    self.critic_optimizer.step()
                
                # Update EMA critic
                self._update_critic_ema()
                
                # Optional: Also train critic on real data (DreamerV3 paper)
                if self.config.critic_real_data_scale > 0 and len(self.replay_buffer) > 32:
                    real_batch = random.sample(self.replay_buffer, min(32, len(self.replay_buffer)))
                    real_states, real_actions, real_rewards, real_next_states = zip(*real_batch)
                    real_states_tensor = torch.stack(real_states).to(self.device)
                    real_next_states_tensor = torch.stack(real_next_states).to(self.device)
                    real_rewards_tensor = torch.stack(real_rewards).to(self.device).squeeze()
                    
                    # Get hidden states from world model
                    with torch.no_grad():
                        # Add sequence dimension for single transitions
                        real_states_input = real_states_tensor.unsqueeze(1)
                        real_actions_input = torch.stack(real_actions).to(self.device).unsqueeze(1)
                        
                        _, _, _, real_hidden, _, _, _, _ = self.world_model(
                            real_states_input, 
                            real_actions_input, 
                            device=self.device
                        )
                    
                    # Compute TD target
                    next_values = self.critic_ema.value(real_next_states_tensor, None, self.bins)
                    real_returns = real_rewards_tensor + self.config.gamma * next_values
                    real_returns_clamped = torch.clamp(real_returns, min=self.config.v_min, max=self.config.v_max)
                    real_returns_twohot = self._twohot_encode(real_returns_clamped, self.bins)
                    
                    # Critic loss on real data
                    real_critic_logits = self.critic(real_states_tensor, real_hidden)
                    real_critic_loss = F.cross_entropy(real_critic_logits, real_returns_twohot, reduction='mean')
                    real_critic_loss = real_critic_loss * self.config.critic_real_data_scale
                    
                    if self.check_training_stability(real_critic_loss):
                        real_critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.config.gradient_clip_norm)
                        self.critic_optimizer.step()
                
                # Update metrics
                self.training_metrics.update({
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy": entropy.item(),
                    "disagreement_loss": disagreement.item() if isinstance(disagreement, torch.Tensor) else 0.0,
                    "avg_return": returns.mean().item()
                })
                
                logger.info(f"{self.object_id}: Trained actor-critic on {num_transitions} imagined transitions, "
                           f"actor loss: {actor_loss.item():.6f}, critic loss: {critic_loss.item():.6f}, "
                           f"avg_return: {returns.mean().item():.4f}")




    async def _train_actor_critic_on_real_data(self):
        """Train actor-critic on real transitions for ablation study."""
        with self.training_error_handler("real data actor-critic training"):
            with self.memory_efficient_compute():
                # Sample real transitions
                batch_size = min(self.config.batch_size, len(self.replay_buffer))
                batch = random.sample(self.replay_buffer, batch_size)
                
                states, actions, rewards, next_states = zip(*batch)
                states_tensor = torch.stack(states).to(self.device)
                actions_tensor = torch.stack(actions).to(self.device)
                rewards_tensor = torch.stack(rewards).to(self.device)
                next_states_tensor = torch.stack(next_states).to(self.device)
                
                # Get hidden states from world model
                with torch.no_grad():
                    # Add sequence dimension for single transitions
                    states_input = states_tensor.unsqueeze(1)  # (batch_size, 1, state_dim)
                    actions_input = actions_tensor.unsqueeze(1)  # (batch_size, 1, action_dim)
                    
                    _, _, _, hidden_states, _, _, _, _ = self.world_model(
                        states_input, actions_input, device=self.device
                    )
                
                # Compute values and returns
                values = self.critic.value(states_tensor, hidden_states, self.bins)
                next_values = self.critic_ema.value(next_states_tensor, None, self.bins)
                
                # Simple TD returns
                returns = rewards_tensor.squeeze() + self.config.gamma * next_values
                advantages = returns - values
                
                # Update actor
                self.actor_optimizer.zero_grad()
                action_dist, _ = self.actor(states_tensor, hidden_states)
                action_indices = torch.argmax(actions_tensor, dim=-1)
                log_probs = action_dist.log_prob(action_indices)
                actor_loss = -(log_probs * advantages.detach()).mean()
                entropy = action_dist.entropy().mean()
                total_actor_loss = actor_loss - self.config.entropy_coeff * entropy
                
                if self.check_training_stability(total_actor_loss):
                    total_actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
                    self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_logits = self.critic(states_tensor, hidden_states)
                returns_clamped = torch.clamp(returns, min=self.config.v_min, max=self.config.v_max)
                returns_twohot = self._twohot_encode(returns_clamped, self.bins)
                critic_loss = F.cross_entropy(critic_logits, returns_twohot, reduction='mean')
                
                if self.check_training_stability(critic_loss):
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
                    self.critic_optimizer.step()
                
                self._update_critic_ema()
                
                logger.info(f"{self.object_id}: [ABLATION] Trained actor-critic on {batch_size} REAL transitions, "
                           f"actor loss: {actor_loss.item():.6f}, critic loss: {critic_loss.item():.6f}")

    async def handle_state_update(self, event: Event):
        """Handle system state update events."""
        if event.origin != "ResourceManager" or not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            self.debug_log(f"{self.object_id}: Invalid SYSTEM_STATE_UPDATE event: "
                          f"origin={event.origin}, data_type={type(event.data)}")
            return
        
        new_state = event.data.value
        new_ts = new_state.get("timestamp")
        
        if not new_ts:
            logger.warning(f"{self.object_id}: SYSTEM_STATE_UPDATE missing timestamp")
            return
        
        self.debug_log(f"{self.object_id}: Received SYSTEM_STATE_UPDATE (ts: {new_ts:.2f})")

        # Link state updates with actions
        if self.current_known_state is not None and TORCH_AVAILABLE:
            prev_ts = self.current_known_state.get("timestamp", 0)
            
            # Simplified action matching
            matched_action = None
            for act_ts_str, (state_before, action_data) in list(self.state_action_history.items()):
                act_ts = float(act_ts_str)
                
                # Match actions within reasonable time window
                if prev_ts <= act_ts <= new_ts:
                    matched_action = (state_before, action_data, act_ts_str)
                    break
            
            # Clean up old history entries
            current_time = time.time()
            for act_ts_str in list(self.state_action_history.keys()):
                if float(act_ts_str) < current_time - 300:  # 5 minutes
                    self.state_action_history.pop(act_ts_str, None)
            
            if matched_action:
                state_before, action_data, act_ts_str = matched_action
                
                # Create transition
                state_vec = self._vectorize_state(state_before)
                action_vec = self._vectorize_action(action_data)
                next_state_vec = self._vectorize_state(new_state)
                
                if state_vec is not None and action_vec is not None and next_state_vec is not None:
                    reward = self._compute_reward(state_before, new_state)
                    
                    # Validate transition
                    if self._validate_transition(state_vec, action_vec, reward, next_state_vec):
                        transition = (state_vec, action_vec, reward, next_state_vec)
                        self.add_transition(transition)
                        
                        act_type = action_data.get("action_type", "UNKNOWN")
                        logger.info(f"{self.object_id}: Stored transition (Action '{act_type}') "
                                   f"in buffers. Replay: {len(self.replay_buffer)}, "
                                   f"Sequences: {len(self.sequence_buffer)}")
                else:
                    logger.warning(f"{self.object_id}: Failed to vectorize state/action for transition at ts {new_ts}")
                
                # Remove matched action from history
                self.state_action_history.pop(act_ts_str, None)
            else:
                self.debug_log(f"{self.object_id}: No matching action found for state update at ts {new_ts}")

        self.current_known_state = new_state

    async def handle_action_taken(self, event: Event):
        """Handle action taken events."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            self.debug_log(f"{self.object_id}: Invalid ACTION_TAKEN event: data_type={type(event.data)}")
            return
        
        action_data = event.data.value
        timestamp = event.data.metadata.get("timestamp", time.time())
        
        if self.current_known_state is not None and TORCH_AVAILABLE:
            self.state_action_history[str(timestamp)] = (self.current_known_state.copy(), action_data)
            self.debug_log(f"{self.object_id}: Stored action '{action_data.get('action_type', 'UNKNOWN')}' "
                          f"at ts {timestamp:.2f}. History size: {len(self.state_action_history)}")
        else:
            logger.warning(f"{self.object_id}: ACTION_TAKEN at {timestamp} but no current state or PyTorch unavailable")

    async def handle_predict_query(self, event: Event):
        """Handle prediction query events."""
        if not TORCH_AVAILABLE:
            await self._send_prediction_response(
                event.origin, event.data.metadata.get("correlation_id"), 
                error="PyTorch unavailable"
            )
            return
        
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            await self._send_prediction_response(
                event.origin, event.data.metadata.get("correlation_id"), 
                error="Invalid query format"
            )
            return
        
        query_data = event.data.value
        origin_bubble_id = event.origin
        correlation_id = event.data.metadata.get("correlation_id")
        
        if not correlation_id:
            return
        
        current_state = query_data.get("current_state")
        action_to_simulate = query_data.get("action")
        
        if not current_state or not action_to_simulate or not isinstance(action_to_simulate, dict):
            await self._send_prediction_response(
                origin_bubble_id, correlation_id, 
                error="Missing state or valid action"
            )
            return
        
        act_type = action_to_simulate.get('action_type', 'UNKNOWN')
        logger.info(f"{self.object_id}: Received PREDICT_STATE_QUERY {correlation_id[:8]} "
                   f"from {origin_bubble_id} for action: {act_type}")

        if not self.world_model:
            await self._send_prediction_response(
                origin_bubble_id, correlation_id, 
                error="DreamerV3 not available"
            )
            return

        try:
            # Vectorize inputs
            state_vector = self._vectorize_state(current_state)
            action_vector = self._vectorize_action(action_to_simulate)
            
            if state_vector is None or action_vector is None:
                raise ValueError("Vectorization failed")

            # Run prediction
            self.world_model.eval()
            with torch.no_grad():
                # Ensure correct dimensions - add batch and sequence dimensions
                state_seq = state_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)
                action_seq = action_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, action_dim)
                
                predicted_states, predicted_continuations = [], []
                
                # Rollout future states
                for _ in range(self.config.horizon):
                    next_state, _, continuation, _, _, _, _, _ = self.world_model(
                        state_seq, action_seq, device=self.device
                    )
                    predicted_states.append(next_state)
                    predicted_continuations.append(continuation)
                    # Update state sequence for next step
                    state_seq = torch.cat([state_seq, next_state.unsqueeze(1)], dim=1)

            # Return final predicted state
            predicted_state = self._devectorize_state(predicted_states[-1].squeeze(0))
            predicted_state["continuation_probability"] = predicted_continuations[-1].item()
            
            if "error" in predicted_state:
                raise ValueError(predicted_state["error"])

            await self._send_prediction_response(
                origin_bubble_id, correlation_id, 
                prediction=predicted_state
            )
            
        except Exception as e:
            logger.error(f"{self.object_id}: Prediction error for {correlation_id[:8]}: {e}", exc_info=True)
            await self._send_prediction_response(
                origin_bubble_id, correlation_id, 
                error=f"Prediction failed: {e}"
            )

    async def _send_prediction_response(self, requester_id: Optional[str], 
                                      correlation_id: Optional[str], 
                                      prediction: Optional[Dict] = None, 
                                      error: Optional[str] = None):
        """Send prediction response to requester."""
        if not requester_id or not correlation_id:
            logger.error(f"{self.object_id}: Cannot send prediction response - "
                        f"missing requester_id or correlation_id")
            return
        
        if not self.dispatcher:
            logger.error(f"{self.object_id}: Cannot send prediction response, dispatcher unavailable")
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

        response_uc = UniversalCode(
            Tags.DICT, response_payload, 
            description=f"Predicted state response ({status})"
        )
        response_event = Event(
            type=Actions.PREDICT_STATE_RESPONSE, 
            data=response_uc, 
            origin=self.object_id, 
            priority=2
        )
        
        await self.context.dispatch_event(response_event)
        logger.info(f"{self.object_id}: Sent PREDICT_STATE_RESPONSE ({status}) "
                   f"for {correlation_id[:8]} to {requester_id}")

    async def train_world_model_sequences(self):
        """Train world model using proper sequences for transformer."""
        if len(self.sequence_buffer) < self.config.batch_size // 4:
            return
        
        with self.training_error_handler("sequence world model training"):
            with self.memory_efficient_compute():
                # Sample sequences
                batch_size = min(self.config.batch_size // 4, len(self.sequence_buffer))
                sequence_batch = random.sample(self.sequence_buffer, batch_size)
                
                # Prepare batch tensors
                all_states, all_actions, all_rewards, all_next_states = [], [], [], []
                
                for sequence in sequence_batch:
                    seq_states, seq_actions, seq_rewards, seq_next_states = zip(*sequence)
                    all_states.append(torch.stack(seq_states))
                    all_actions.append(torch.stack(seq_actions))
                    all_rewards.append(torch.stack(seq_rewards))
                    all_next_states.append(torch.stack(seq_next_states))
                
                states_tensor = torch.stack(all_states).to(self.device)
                actions_tensor = torch.stack(all_actions).to(self.device)
                rewards_tensor = torch.stack(all_rewards).to(self.device)
                next_states_tensor = torch.stack(all_next_states).to(self.device)
                
                self.debug_log(f"Training with sequences: states_tensor.shape={states_tensor.shape}")
                
                # Train models
                await self._train_world_model_step(
                    states_tensor, actions_tensor, rewards_tensor[:, :, 0], next_states_tensor, 
                    is_sequence=True
                )

    async def train_world_model_singles(self):
        """Train world model using single transitions."""
        with self.training_error_handler("single world model training"):
            with self.memory_efficient_compute():
                batch_size = min(self.config.batch_size, len(self.replay_buffer))
                batch = random.sample(self.replay_buffer, batch_size)
                
                states, actions, rewards, next_states = zip(*batch)
                # Don't unsqueeze here - let _train_world_model_step handle it
                states_tensor = torch.stack(states).to(self.device)
                actions_tensor = torch.stack(actions).to(self.device)
                rewards_tensor = torch.stack(rewards).to(self.device)
                next_states_tensor = torch.stack(next_states).to(self.device)
                
                await self._train_world_model_step(
                    states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                    is_sequence=False
                )

    async def _train_world_model_step(self, states_tensor, actions_tensor, rewards_tensor, 
                                    next_states_tensor, is_sequence=False):
        """Core world model training step with distributional reward prediction."""
        batch_size = states_tensor.shape[0]
        seq_len = states_tensor.shape[1] if is_sequence else 1
        
        # Log memory usage
        memory_usage = self._get_memory_usage()
        if memory_usage is not None:
            self.debug_log(f"Memory usage before training: {memory_usage:.2f} GB")
        
        self.world_model.train()
        for m in self.world_model_ensemble:
            m.train()
        
        self.world_optimizer.zero_grad()
        for opt in self.ensemble_optimizers:
            opt.zero_grad()

        # Forward pass - FIX: Don't unsqueeze if already has correct dimensions
        if states_tensor.dim() == 2:  # Single transitions: (batch_size, state_dim)
            states_input = states_tensor.unsqueeze(1)  # Make it (batch_size, 1, state_dim)
            actions_input = actions_tensor.unsqueeze(1)  # Make it (batch_size, 1, action_dim)
        else:  # Sequences: already (batch_size, seq_len, state_dim)
            states_input = states_tensor
            actions_input = actions_tensor
            
        predicted_next_states, reward_logits, predicted_continuations, _, kl_loss, _, recon_state, vae_kl_loss = self.world_model(
            states_input, actions_input, device=self.device)

        # Handle sequence vs single-step predictions correctly
        if is_sequence:
            target_next_state = next_states_tensor[:, -1, :]
            target_reward = rewards_tensor[:, -1]
        else:
            target_next_state = next_states_tensor
            target_reward = rewards_tensor

        # Compute losses (STORM-aligned)
        state_loss = torch.mean((predicted_next_states - target_next_state) ** 2)
        
        # Distributional reward loss with two-hot encoding
        target_reward = torch.clamp(target_reward, min=self.config.v_min, max=self.config.v_max)
        reward_targets = self._twohot_encode(target_reward, self.bins)
        reward_loss = F.cross_entropy(reward_logits, reward_targets, reduction='mean')
        
        # Continuation loss (assume always continuing)
        continuation_tensor = torch.ones(batch_size, 1, dtype=torch.float32).to(self.device)
        continuation_loss = nn.BCELoss()(predicted_continuations, continuation_tensor)
        
        # Reconstruction loss
        if is_sequence:
            recon_loss = torch.mean((recon_state - states_tensor[:, -1, :]) ** 2)
        else:
            recon_loss = torch.mean((recon_state - states_tensor) ** 2)
        
        # Representation KL loss with free bits
        prior_logits = torch.ones(batch_size, self.config.num_categories, self.config.num_classes, device=self.device) / self.config.num_classes
        if is_sequence:
            state_for_kl = states_tensor[:, -1, :]
        else:
            state_for_kl = states_tensor
            
        state_logits = self.world_model.state_encoder(state_for_kl).view(
            batch_size, self.config.num_categories, self.config.num_classes
        )
        rep_kl_loss = torch.mean(torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=state_logits),
            torch.distributions.Categorical(logits=prior_logits)
        ))
        # Apply free bits to both KL losses as per DreamerV3
        rep_kl_loss = torch.max(rep_kl_loss - self.config.kl_free_bits, torch.tensor(0.0, device=self.device))
        kl_loss = torch.max(kl_loss - self.config.kl_free_bits, torch.tensor(0.0, device=self.device))
        
        # Total loss (weighted per STORM with VAE component)
        total_loss = (recon_loss + reward_loss + continuation_loss + 
                     self.config.beta1 * kl_loss + 
                     self.config.beta2 * rep_kl_loss +
                     self.config.vae_beta * vae_kl_loss)

        if not self.check_training_stability(total_loss):
            return

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=self.config.gradient_clip_norm)
        self.world_optimizer.step()

        # Train ensemble with same dimension handling
        for m, opt in zip(self.world_model_ensemble, self.ensemble_optimizers):
            m.train()
            opt.zero_grad()
            
            pred_next_states, _, _, _, _, _, _, _ = m(states_input, actions_input, device=self.device)
            ensemble_loss = torch.mean((pred_next_states - target_next_state) ** 2)
            
            ensemble_loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=self.config.gradient_clip_norm)
            opt.step()

        # Compute disagreement for monitoring
        with torch.no_grad():
            ensemble_preds = []
            for m in self.world_model_ensemble:
                pred, _, _, _, _, _, _, _ = m(states_input, actions_input, device=self.device)
                ensemble_preds.append(pred)
            disagreement_loss = torch.mean(torch.var(torch.stack(ensemble_preds, dim=0), dim=0))
            del ensemble_preds

        # Compute validation loss
        validation_loss = await self.compute_validation_loss()

        # Update metrics
        self.training_metrics.update({
            "state_loss": state_loss.item(),
            "reward_loss": reward_loss.item(),
            "continuation_loss": continuation_loss.item(),
            "kl_loss": kl_loss.item(),
            "disagreement_loss": disagreement_loss.item(),
            "recon_loss": recon_loss.item(),
            "rep_loss": rep_kl_loss.item(),
            "validation_loss": validation_loss
        })

        # Format validation loss string properly
        validation_str = f"{validation_loss:.6f}" if validation_loss is not None else "N/A"
        
        logger.info(f"{self.object_id}: Trained world model with {batch_size} "
                   f"{'sequences' if is_sequence else 'samples'}, "
                   f"loss: {total_loss.item():.6f}, state: {state_loss.item():.6f}, "
                   f"reward: {reward_loss.item():.6f}, recon: {recon_loss.item():.6f}, "
                   f"validation: {validation_str}")
        
        # Memory cleanup
        del states_tensor, actions_tensor, rewards_tensor, next_states_tensor
        del predicted_next_states, reward_logits, predicted_continuations, recon_state

    async def train_world_model(self):
        """Train the world model on replay buffer data."""
        if not TORCH_AVAILABLE or not self.world_model:
            self.debug_log(f"{self.object_id}: Skipping world model training (Torch: {TORCH_AVAILABLE})")
            return
            
        # Use sequence buffer if available for proper transformer training
        if len(self.sequence_buffer) >= self.config.batch_size // 4:
            await self.train_world_model_sequences()
        elif len(self.replay_buffer) >= self.config.batch_size:
            await self.train_world_model_singles()

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint with distributed training support."""
        if not TORCH_AVAILABLE or not self.config.enable_checkpointing:
            return
            
        # Only save on rank 0 in distributed training
        if self.config.distributed and self.config.rank != 0:
            return
        
        try:
            checkpoint = {
                # Model states
                'world_model_state_dict': self.world_model.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_ema_state_dict': self.critic_ema.state_dict(),
                
                # Ensemble states
                'ensemble_state_dicts': [m.state_dict() for m in self.world_model_ensemble],
                
                # Optimizer states
                'world_optimizer_state_dict': self.world_optimizer.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'ensemble_optimizer_state_dicts': [opt.state_dict() for opt in self.ensemble_optimizers],
                
                # Training state
                'execution_count': self.execution_count,
                'training_metrics': self.training_metrics,
                'nan_inf_count': self.nan_inf_count,
                'return_range': self.return_range,
                'best_validation_loss': self.best_validation_loss,
                
                # Buffers (optional - can be large)
                'replay_buffer_sample': list(self.replay_buffer)[-1000:] if len(self.replay_buffer) > 0 else [],
                'validation_buffer_sample': list(self.validation_buffer)[-200:] if len(self.validation_buffer) > 0 else [],
                
                # Configuration for verification
                'config': self.config,
                'state_dim': self.config.state_dim,
                'action_dim': self.config.action_dim,
                
                # Metadata
                'timestamp': time.time(),
                'device': str(self.device),
                'torch_version': torch.__version__ if TORCH_AVAILABLE else None,
            }
            
            # Save main checkpoint
            checkpoint_path = self._get_checkpoint_path()
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"{self.object_id}: Saved checkpoint to {checkpoint_path}")
            
            # Save best checkpoint if applicable
            if is_best:
                torch.save(checkpoint, self.best_checkpoint_path)
                logger.info(f"{self.object_id}: Saved best checkpoint (validation loss: {self.best_validation_loss:.6f})")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to save checkpoint: {e}", exc_info=True)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Load model checkpoint."""
        if not TORCH_AVAILABLE or not self.config.enable_checkpointing:
            return False
        
        try:
            # Find checkpoint to load
            if checkpoint_path is None:
                checkpoint_path = self._find_latest_checkpoint()
                if checkpoint_path is None:
                    logger.info(f"{self.object_id}: No checkpoint found to load")
                    return False
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Verify compatibility
            if checkpoint.get('state_dim') != self.config.state_dim:
                logger.error(f"{self.object_id}: State dimension mismatch in checkpoint")
                return False
            if checkpoint.get('action_dim') != self.config.action_dim:
                logger.error(f"{self.object_id}: Action dimension mismatch in checkpoint")
                return False
            
            # Load model states
            self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_ema.load_state_dict(checkpoint['critic_ema_state_dict'])
            
            # Load ensemble states
            for i, m in enumerate(self.world_model_ensemble):
                if i < len(checkpoint['ensemble_state_dicts']):
                    m.load_state_dict(checkpoint['ensemble_state_dicts'][i])
            
            # Load optimizer states
            self.world_optimizer.load_state_dict(checkpoint['world_optimizer_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            for i, opt in enumerate(self.ensemble_optimizers):
                if i < len(checkpoint['ensemble_optimizer_state_dicts']):
                    opt.load_state_dict(checkpoint['ensemble_optimizer_state_dicts'][i])
            
            # Load training state
            self.execution_count = checkpoint.get('execution_count', 0)
            self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
            self.nan_inf_count = checkpoint.get('nan_inf_count', 0)
            self.return_range = checkpoint.get('return_range', None)
            self.best_validation_loss = checkpoint.get('best_validation_loss', float('inf'))
            
            # Optionally load buffer samples
            if 'replay_buffer_sample' in checkpoint and len(checkpoint['replay_buffer_sample']) > 0:
                # Add some of the saved transitions back to buffer
                for transition in checkpoint['replay_buffer_sample'][-100:]:  # Only load recent ones
                    self.replay_buffer.append(transition)
            
            if 'validation_buffer_sample' in checkpoint and len(checkpoint['validation_buffer_sample']) > 0:
                for transition in checkpoint['validation_buffer_sample'][-20:]:
                    self.validation_buffer.append(transition)
            
            logger.info(f"{self.object_id}: Loaded checkpoint from {checkpoint_path} "
                       f"(execution_count: {self.execution_count}, "
                       f"best_validation_loss: {self.best_validation_loss:.6f})")
            
            return True
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to load checkpoint from {checkpoint_path}: {e}", exc_info=True)
            return False

    def _get_checkpoint_path(self) -> str:
        """Generate checkpoint path with timestamp."""
        timestamp = int(time.time())
        return os.path.join(
            self.config.checkpoint_dir, 
            f"{self.object_id}_checkpoint_{self.execution_count}_{timestamp}.pt"
        )

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        if not os.path.exists(self.config.checkpoint_dir):
            return None
        
        checkpoints = []
        for filename in os.listdir(self.config.checkpoint_dir):
            if filename.startswith(f"{self.object_id}_checkpoint_") and filename.endswith(".pt"):
                filepath = os.path.join(self.config.checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        if not checkpoints:
            return None
        
        # Sort by modification time and return the latest
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        if not os.path.exists(self.config.checkpoint_dir):
            return
        
        checkpoints = []
        for filename in os.listdir(self.config.checkpoint_dir):
            if filename.startswith(f"{self.object_id}_checkpoint_") and filename.endswith(".pt"):
                filepath = os.path.join(self.config.checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old checkpoints
        for filepath, _ in checkpoints[self.config.keep_last_n_checkpoints:]:
            try:
                os.remove(filepath)
                logger.debug(f"{self.object_id}: Removed old checkpoint: {filepath}")
            except Exception as e:
                logger.warning(f"{self.object_id}: Failed to remove old checkpoint {filepath}: {e}")

    def save_training_history(self):
        """Save training history and metrics to a separate file for analysis."""
        if not self.config.enable_checkpointing:
            return
        
        history_path = os.path.join(self.config.checkpoint_dir, f"{self.object_id}_training_history.json")
        
        try:
            history = {
                'execution_count': self.execution_count,
                'training_metrics': self.training_metrics,
                'config': {
                    'state_dim': self.config.state_dim,
                    'action_dim': self.config.action_dim,
                    'hidden_dim': self.config.hidden_dim,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                },
                'timestamp': time.time(),
            }
            
            # Load existing history if available
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    existing_history = json.load(f)
                    if 'history' not in existing_history:
                        existing_history = {'history': [existing_history]}
            else:
                existing_history = {'history': []}
            
            # Append new entry
            existing_history['history'].append(history)
            
            # Save updated history
            with open(history_path, 'w') as f:
                json.dump(existing_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to save training history: {e}")

    def export_model(self, export_path: str):
        """Export model for deployment (without training-specific data)."""
        if not TORCH_AVAILABLE:
            logger.error(f"{self.object_id}: Cannot export model without PyTorch")
            return
        
        try:
            export_data = {
                'world_model_state_dict': self.world_model.state_dict(),
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic_ema.state_dict(),  # Use EMA for deployment
                'config': self.config,
                'state_fields': [(f.name, f.scale, f.field_type) for f in self.STATE_FIELDS],
                'action_types': self.ACTION_TYPES,
                'device': 'cpu',  # Export for CPU by default
                'version': '1.0',
                'export_timestamp': time.time(),
            }
            
            torch.save(export_data, export_path)
            logger.info(f"{self.object_id}: Exported model to {export_path}")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to export model: {e}", exc_info=True)

    async def run_ablation_study(self, num_steps: int = 1000):
        """
        Run ablation study comparing different configurations.
        
        Args:
            num_steps: Number of training steps per configuration
        """
        ablation_configs = ["none", "no_transformer", "no_distributional", "no_imagination"]
        results = {}
        
        for ablation in ablation_configs:
            logger.info(f"{self.object_id}: Starting ablation study - {ablation}")
            
            # Reset and reconfigure
            self.config.ablation_mode = ablation
            self._initialize_models()
            self.replay_buffer.clear()
            self.validation_buffer.clear()
            
            # Collect initial data
            self.collect_transitions(100)
            
            # Train for specified steps
            start_time = time.time()
            initial_loss = await self.compute_validation_loss()
            
            for step in range(num_steps):
                if step % 10 == 0:
                    self.collect_transitions(10)
                    await self.train_world_model()
                    await self.train_actor_critic()
            
            # Evaluate
            final_loss = await self.compute_validation_loss()
            train_time = time.time() - start_time
            
            results[ablation] = {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "improvement": (initial_loss - final_loss) / initial_loss if initial_loss else 0,
                "train_time": train_time,
                "final_metrics": dict(self.training_metrics)
            }
            
            logger.info(f"{self.object_id}: Ablation {ablation} complete - "
                       f"Loss: {initial_loss:.4f} -> {final_loss:.4f}, Time: {train_time:.1f}s")
        
        # Summary
        logger.info(f"{self.object_id}: Ablation study complete. Results:")
        for config, metrics in results.items():
            logger.info(f"  {config}: {metrics['improvement']:.1%} improvement, "
                       f"final loss: {metrics['final_loss']:.4f}")
        
        return results

    def _initialize_multi_agent_communication(self):
        """Initialize multi-agent communication channels."""
        # Placeholder for multi-agent support
        logger.info(f"{self.object_id}: Multi-agent communication initialized")

    async def coordinate_multi_agent_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate action with other agents."""
        # Placeholder for multi-agent coordination
        return action

    async def share_agent_state(self):
        """Share current state with other agents."""
        # Placeholder for state sharing
        pass

    def _sync_gradients(self):
        """Synchronize gradients across distributed training."""
        # Placeholder for distributed training
        pass

    async def autonomous_step(self):
        """Periodic training and maintenance step."""
        await super().autonomous_step()
        
        # Ensure initialization is complete
        await self.ensure_initialized()
        
        self.execution_count += 1
        
        # Train periodically
        if self.execution_count % self.config.log_interval == 0:
            # Collect simulated data
            self.collect_transitions(10)
            
            logger.info(f"{self.object_id}: Training step {self.execution_count}, "
                       f"buffers - replay: {len(self.replay_buffer)}, "
                       f"sequence: {len(self.sequence_buffer)}")
            
            # Train models
            await self.train_world_model()
            await self.train_actor_critic()
        
        # Save checkpoint periodically
        if self.config.enable_checkpointing and self.execution_count % self.config.checkpoint_interval == 0:
            # Check if this is the best model based on validation loss
            validation_loss = self.training_metrics.get('validation_loss')
            is_best = False
            if validation_loss is not None and validation_loss < self.best_validation_loss:
                self.best_validation_loss = validation_loss
                is_best = True
            
            self.save_checkpoint(is_best=is_best)
            self.save_training_history()
        
        # Log aggregate performance periodically
        if self.execution_count % self.config.summary_interval == 0:
            avg_metrics = {k: v for k, v in self.training_metrics.items() 
                          if v not in (0, None)}
            if avg_metrics:
                logger.info(f"{self.object_id}: Training summary - Iterations: {self.execution_count}, "
                           f"Metrics: {avg_metrics}")
        
        await asyncio.sleep(0.5)

    async def handle_event(self, event: Event):
        """Main event handler routing to specific handlers."""
        try:
            if event.type == Actions.SYSTEM_STATE_UPDATE:
                await self.handle_state_update(event)
            elif event.type == Actions.ACTION_TAKEN:
                await self.handle_action_taken(event)
            elif event.type == Actions.PREDICT_STATE_QUERY:
                await self.handle_predict_query(event)
            else:
                self.debug_log(f"{self.object_id}: Ignoring event type: {event.type}")
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event {event.type}: {e}", exc_info=True)
