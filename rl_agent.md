# Reinforcement Learning Agent

The Reinforcement Learning (RL) agent is a key component of the cognitive system, enabling adaptive decision-making through continuous learning from feedback. This document explains the agent's architecture, key features, and integration with the larger system.

## Overview

The RL agent uses Proximal Policy Optimization (PPO), a state-of-the-art policy gradient method, to learn optimal behaviors over time. It incorporates several advanced features:

1. **Prioritized Experience Replay**: Focuses learning on surprising or high-error experiences
2. **Knowledge Distillation**: Extracts pattern knowledge from successful experiences
3. **Meta-Learning**: Adaptively optimizes hyperparameters based on performance
4. **Transfer Learning**: Enables cross-domain knowledge sharing

```
┌───────────────────────────────────────────────────────┐
│                     RL Agent                          │
│                                                       │
│  ┌─────────────┐      ┌───────────────┐              │
│  │   PPO       │◄────►│  Experience   │              │
│  │  Policy     │      │    Buffer     │              │
│  └─────────────┘      └───────────────┘              │
│         ▲                     ▲                       │
│         │                     │                       │
│         ▼                     │                       │
│  ┌─────────────┐      ┌───────────────┐              │
│  │    Meta     │      │   Transfer    │              │
│  │  Controller │      │   Learning    │              │
│  └─────────────┘      └───────────────┘              │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Key Components

### 1. PPO Policy

The core learning algorithm uses Proximal Policy Optimization (PPO) with the following key features:

- **MLP Policy Network**: Multi-layer perceptron for action selection
- **Value Function**: Learns to estimate state values
- **Clip Objective**: Prevents excessive policy updates
- **Multiple Epochs**: Performs multiple optimization passes over each batch

### 2. Experience Buffer

Stores and manages experiences for batched learning:

- **Experience Storage**: Collects state, action, reward, next_state tuples
- **Prioritization**: Calculates TD errors to identify important experiences
- **Batch Formation**: Creates optimized batches for efficient learning

### 3. Meta-Controller

Adaptively tunes hyperparameters to optimize learning:

- **Learning Rate Adjustment**: Dynamically changes learning rate based on performance
- **Exploration Control**: Adjusts exploration vs. exploitation balance
- **Cross-Validation**: Evaluates parameter changes through validation sets

### 4. Transfer Learning Bridge

Enables knowledge sharing between different domains/tasks:

- **Domain Embeddings**: Creates embeddings to represent different tasks
- **Similarity Calculation**: Identifies related domains for knowledge transfer
- **Policy Adaptation**: Modifies policies based on domain similarities

## Code Implementation

### Agent Initialization

```python
def __init__(self, 
             learning_rate: float = 1e-4, 
             gamma: float = 0.99, 
             n_steps: int = 128,
             batch_size: int = 64,
             n_epochs: int = 10,
             action_set: Optional[List[str]] = None,
             state_features: Optional[List[str]] = None,
             model_path: Optional[str] = "ppo_agent",
             use_gpu: bool = True,
             enable_meta_learning: bool = True,
             enable_transfer_learning: bool = True,
             verbose: int = 1):
    """Initialize the RL agent with PPO, meta-learning and transfer learning"""
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.n_steps = n_steps
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.model_path = model_path
    self.verbose = verbose
    self.actions_taken = 0
    self.lock = threading.Lock()  # Thread safety
    
    # Configure device
    self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"PPO agent using device: {self.device}")
    
    # Configure action set
    self.action_set = action_set or [
        "increase_context", "reduce_context", 
        "focus_search", "broaden_search",
        "default"
    ]
    
    # State features (what the agent observes)
    self.state_features = state_features or [
        "success_rate", "memory_size", "correction_needed", 
        "query_complexity", "response_length"
    ]
    
    # Create the environment and PPO model
    # ... (implementation details)
    
    # Initialize meta-learning controller and transfer learning bridge
    self.meta_controller = MetaLearningController() if enable_meta_learning else None
    self.transfer_bridge = TransferLearningBridge(use_gpu=use_gpu) if enable_transfer_learning else None
```

### Action Selection

The RL agent selects actions based on the current state using its learned policy:

```python
def get_action(self, state: Dict[str, float]) -> str:
    """
    Select an action for the given state using the PPO policy
    with meta-learning hyperparameter optimization
    
    Args:
        state: The current state as a dictionary of features
        
    Returns:
        Selected action as string
    """
    with self.lock:
        try:
            # Apply meta-learning hyperparameter adjustments
            if self.enable_meta_learning and self.meta_controller:
                # Extract current hyperparameters from model
                current_params = {
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                }
                
                # Get adjusted parameters from meta-controller
                adjusted_params = self.meta_controller.adjust_hyperparameters(current_params)
                
                # Apply adjustments if different from current
                # ... (implementation details)
            
            # Convert state to vector
            state_vector = self._state_to_vector(state)
            
            # Update environment state
            self.env.set_state(state_vector)
            
            # Get action from policy
            action, _ = self.model.predict(state_vector, deterministic=False)
            
            # Convert action index to string
            action_str = self.action_set[int(action)]
            
            # Track action
            self.actions_taken += 1
            self.action_history[time.time()] = {
                'state': state,
                'action': action_str
            }
            
            return action_str
            
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            # Fallback to default action
            return self.action_set[-1]  # Default action
```

### Prioritized Experience Replay

The agent implements prioritized experience replay to focus learning on surprising or high-error experiences:

```python
async def learn_from_experiences_with_priority(self, batch_size=None) -> bool:
    """
    Learn from experiences with prioritized replay, focusing on surprising
    or high-error experiences first.
    
    Args:
        batch_size: Optional batch size, defaults to self.batch_size
        
    Returns:
        True if learning occurred, False otherwise
    """
    if not self.experiences:
        return False
        
    if batch_size is None:
        batch_size = self.batch_size
        
    try:
        # Calculate TD errors for prioritization
        experiences_with_errors = self._calculate_experience_priorities()
        
        # Sort by error (highest first)
        experiences_with_errors.sort(key=lambda x: x[1], reverse=True)
        
        # Take top batch_size experiences
        selected_experiences = [exp for exp, _ in experiences_with_errors[:batch_size]]
        
        # Learn from selected experiences
        self._learn_from_batch(selected_experiences)
        
        # Log learning and update stats
        logger.info(f"Trained on {len(selected_experiences)} prioritized experiences")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in prioritized learning: {e}")
        return False
```

### Knowledge Distillation

The agent can distill knowledge from successful experiences to improve policy efficiency:

```python
async def distill_knowledge(self):
    """
    Distill knowledge from successful experiences to improve
    the agent's policy. This creates a more compact representation
    of successful behaviors.
    
    Returns:
        True if distillation occurred, False otherwise
    """
    # Need a minimum number of successful experiences
    successful_experiences = [exp for exp in self.experiences 
                            if exp[2] > 0.7]  # Reward threshold
    
    if len(successful_experiences) < 10:
        return False
        
    try:
        logger.info(f"Distilling knowledge from {len(successful_experiences)} successful experiences")
        
        # Create pseudo-labels from successful experiences
        distillation_states = []
        distillation_actions = []
        
        for state, action, reward, _, _ in successful_experiences:
            distillation_states.append(state)
            distillation_actions.append(self.action_set.index(action))
            
        # Convert to tensors
        states_tensor = torch.FloatTensor(distillation_states)
        actions_tensor = torch.LongTensor(distillation_actions)
        
        # Train on this high-quality dataset multiple times
        distillation_epochs = 5
        for epoch in range(distillation_epochs):
            # Get current action probabilities
            action_probs = self.model.policy.get_action_probs(states_tensor)
            
            # Calculate log probabilities of correct actions
            action_logs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) + 1e-8)
            
            # Loss is negative log likelihood
            loss = -action_logs.mean()
            
            # Optimize
            self.model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
            self.model.optimizer.step()
            
        return True
        
    except Exception as e:
        logger.error(f"Error in knowledge distillation: {e}")
        return False
```

### Transfer Learning

The agent can transfer knowledge between different domains:

```python
def set_domain(self, domain_name: str, domain_descriptor: str) -> bool:
    """
    Set the current domain/task for transfer learning
    
    Args:
        domain_name: Name identifier for the domain
        domain_descriptor: Descriptive text for the domain
        
    Returns:
        True if domain was set, False otherwise
    """
    if not self.enable_transfer_learning or not self.transfer_bridge:
        return False
        
    try:
        # Register domain if not already known
        if domain_name not in self.transfer_bridge.domain_embeddings:
            self.transfer_bridge.register_domain(domain_name, domain_descriptor)
        
        # Set as current domain
        self.current_domain = domain_name
        
        # Check if we can apply transfer learning from similar domains
        similar_domains = self.transfer_bridge.find_similar_domains(domain_name)
        
        if similar_domains and hasattr(self, 'model') and hasattr(self.model, 'policy'):
            # Get the most similar domain
            source_domain, similarity = similar_domains[0]
            
            # Apply transfer learning
            adapted_policy = self.transfer_bridge.transfer_knowledge(
                source_domain,
                domain_name,
                current_policy_np
            )
            
            # Apply the adapted policy
            self.model.policy.load_state_dict(adapted_policy_torch)
            logger.info(f"Applied transfer learning from domain '{source_domain}' to '{domain_name}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting domain '{domain_name}': {e}")
        return False
```

## Integration with Recursive Learning Loop

The RL agent is a key component of the recursive learning loop, continuously improving its policy based on system feedback:

```python
# In the Orchestrator class
async def _recursive_learning_loop(self) -> None:
    """Main recursive learning loop that processes experiences and improves the system"""
    
    while self.learning_active:
        try:
            # ... other learning steps ...
            
            # RL agent learning from accumulated experiences
            if self.rl_agent and len(self.rl_agent.experiences) >= self.rl_agent.batch_size:
                # Prioritized learning from surprising experiences
                await self.rl_agent.learn_from_experiences_with_priority()
                
                # Periodically distill knowledge from successful experiences
                if random.random() < 0.2:  # 20% chance each loop
                    await self.rl_agent.distill_knowledge()
            
            # ... other learning steps ...
            
        except Exception as e:
            logger.error(f"Error in recursive learning loop: {e}")
        
        # Wait until next learning iteration
        await asyncio.sleep(self.recursive_learning_interval)
```

## Configuration Options

The RL agent can be configured with various parameters:

### Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `learning_rate` | Rate of policy updates | 1e-4 |
| `gamma` | Future reward discount factor | 0.99 |
| `n_steps` | Steps per update | 128 |
| `batch_size` | Batch size for learning | 64 |
| `n_epochs` | Training epochs per batch | 10 |
| `action_set` | Available actions | ["increase_context", "reduce_context", "focus_search", "broaden_search", "default"] |
| `state_features` | Observable state features | ["success_rate", "memory_size", "correction_needed", "query_complexity", "response_length"] |

### Advanced Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_meta_learning` | Enable adaptive hyperparameters | True |
| `enable_transfer_learning` | Enable cross-domain knowledge sharing | True |
| `use_gpu` | Use GPU acceleration if available | True |
| `clip_range` | PPO clipping parameter | 0.2 |
| `entropy_coeff` | Entropy bonus coefficient | 0.01 |
| `value_coeff` | Value loss coefficient | 0.5 |

## Best Practices

### 1. Action Design

Design your action space carefully:
- Use discrete actions with clear semantics
- Limit the number of actions (usually 3-7 is optimal)
- Ensure actions have measurable impact

Example:
```python
action_set = [
    "increase_depth",      # Increase search depth
    "decrease_depth",      # Decrease search depth
    "increase_breadth",    # Increase search breadth
    "decrease_breadth",    # Decrease search breadth
    "prioritize_recency",  # Focus on recent information
    "prioritize_relevance" # Focus on semantic relevance
]
```

### 2. State Representation

Choose state features that:
- Capture relevant information
- Are normalized (0-1 range)
- Include both internal and external factors

Example:
```python
state = {
    "query_complexity": 0.85,      # Complexity measure (0-1)
    "memory_utilization": 0.32,    # Memory usage (0-1)
    "user_satisfaction": 0.91,     # Estimated satisfaction (0-1)
    "context_size": 0.45,          # Normalized context size
    "execution_time": 0.28,        # Normalized execution time
    "correction_frequency": 0.15   # How often corrections needed
}
```

### 3. Reward Design

Create a balanced reward function:
- Include user feedback when available
- Consider response quality
- Include efficiency metrics
- Balance immediate vs. long-term feedback

Example:
```python
def calculate_reward(
    user_feedback: Optional[float] = None,
    task_success: Optional[bool] = None,
    response_time: Optional[float] = None,
    accuracy: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None
) -> float:
    """Calculate composite reward signal"""
    
    reward = 0.0
    metrics = metrics or {}
    
    # User feedback component (weight: 0.5)
    if user_feedback is not None:
        reward += 0.5 * user_feedback
    
    # Task success component (weight: 0.3)
    if task_success is not None:
        reward += 0.3 if task_success else 0.0
    
    # Response time component (weight: 0.1)
    if response_time is not None:
        # Convert response time to a 0-1 value (lower is better)
        time_score = max(0.0, 1.0 - (response_time / 10.0))
        reward += 0.1 * time_score
    
    # Accuracy component (weight: 0.2)
    if accuracy is not None:
        reward += 0.2 * accuracy
    
    # Additional metrics (weight: 0.1)
    if metrics:
        metrics_avg = sum(metrics.values()) / len(metrics)
        reward += 0.1 * metrics_avg
    
    return reward
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| Agent chooses same action repeatedly | Reward function bias, Insufficient exploration | Adjust reward function, Increase exploration rate, Check for state feature normalization |
| Unstable learning | Learning rate too high, Poor state representation | Reduce learning rate, Improve state features, Add experience replay |
| Poor transfer learning | Insufficient domain descriptions, Too few samples | Improve domain descriptions, Collect more experiences, Adjust similarity threshold |
| Slow learning | Batch size too small, Complex network | Increase batch size, Simplify policy network, Enable GPU |
| High resource usage | Excessive experiences stored, Large network | Limit experience buffer size, Simplify policy network, Add regular cleanup |

### Monitoring PPO Training

Monitor these metrics during training:
- **Policy Loss**: Should decrease and stabilize over time
- **Value Loss**: Should decrease and stabilize over time
- **Explained Variance**: Should increase toward 1.0
- **Average Reward**: Should increase over time
- **Entropy**: Should gradually decrease as policy becomes more certain

## Advanced Topics

### 1. Custom Policy Networks

You can customize the policy network architecture:

```python
from stable_baselines3.common.torch_layers import MlpExtractor

class CustomPolicy(MlpExtractor):
    def __init__(self, feature_dim, net_arch, activation_fn):
        super().__init__(feature_dim, net_arch, activation_fn)
        
        # Add custom layers
        self.attention = SelfAttention(feature_dim)
        
    def forward(self, features):
        # Apply attention to features
        attended_features = self.attention(features)
        
        # Process through standard MLP
        policy_latent, value_latent = super().forward(attended_features)
        
        return policy_latent, value_latent

# Use custom policy
custom_policy = {
    'features_extractor_class': CustomPolicy,
    'features_extractor_kwargs': {
        'feature_dim': 64,
        'net_arch': [64, 64],
        'activation_fn': nn.Tanh
    }
}

agent = RLAgent(policy_kwargs=custom_policy)
```

### 2. Multi-Objective Reinforcement Learning

For complex systems, you can implement multi-objective RL:

```python
class MultiObjectiveRLAgent(RLAgent):
    def __init__(self, objectives=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define multiple objectives
        self.objectives = objectives or {
            'efficiency': 0.3,   # Weight for efficiency objective
            'quality': 0.4,      # Weight for output quality
            'user_exp': 0.3      # Weight for user experience
        }
        
    def calculate_multi_objective_reward(self, rewards_dict):
        """Calculate weighted reward across multiple objectives"""
        total_reward = 0.0
        
        for objective, weight in self.objectives.items():
            if objective in rewards_dict:
                total_reward += weight * rewards_dict[objective]
                
        return total_reward
        
    # Override add_experience to support multi-objective rewards
    def add_experience(self, state, action, rewards_dict, next_state, done=False):
        # Calculate composite reward
        composite_reward = self.calculate_multi_objective_reward(rewards_dict)
        
        # Store experience with composite reward
        super().add_experience(state, action, composite_reward, next_state, done)
        
        # Also store per-objective rewards for analysis
        if not hasattr(self, 'objective_rewards'):
            self.objective_rewards = {obj: [] for obj in self.objectives}
            
        for obj in self.objectives:
            if obj in rewards_dict:
                self.objective_rewards[obj].append(rewards_dict[obj])
```

### 3. Curriculum Learning

Implement curriculum learning to gradually increase task difficulty:

```python
class CurriculumManager:
    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0, success_threshold=0.7):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.success_threshold = success_threshold
        self.success_history = []
        
    def update_difficulty(self, success_rate):
        """Update difficulty based on recent success rate"""
        # Add to history
        self.success_history.append(success_rate)
        
        # Keep only recent history
        if len(self.success_history) > 10:
            self.success_history = self.success_history[-10:]
            
        # Calculate average success rate
        avg_success = sum(self.success_history) / len(self.success_history)
        
        # Adjust difficulty
        if avg_success > self.success_threshold:
            # Increase difficulty
            self.current_difficulty = min(
                self.max_difficulty, 
                self.current_difficulty + 0.05
            )
        elif avg_success < self.success_threshold - 0.2:
            # Decrease difficulty
            self.current_difficulty = max(
                0.1, 
                self.current_difficulty - 0.05
            )
            
        return self.current_difficulty
        
    def get_environment_params(self):
        """Get environment parameters based on current difficulty"""
        return {
            'task_complexity': self.current_difficulty,
            'time_constraint': int(30 * self.current_difficulty),
            'noise_level': 0.1 * self.current_difficulty
        }
```

## Example Usage

Here's a complete example of using the RL agent with the orchestrator:

```python
# Initialize the RL agent
rl_agent = RLAgent(
    learning_rate=1e-4,
    gamma=0.99,
    batch_size=64,
    action_set=[
        "increase_context", "reduce_context", 
        "focus_search", "broaden_search",
        "specialize_query", "generalize_query", 
        "default"
    ],
    state_features=[
        "success_rate", "memory_size", "correction_needed", 
        "query_complexity", "response_length", "user_satisfaction"
    ],
    enable_meta_learning=True,
    enable_transfer_learning=True
)

# Initialize the orchestrator with the agent
orchestrator = QuantumSynapseOrchestrator(
    rl_agent=rl_agent,
    enable_recursive_learning=True
)

# Start the learning loop
orchestrator.start_learning_loop()

# Process user interaction
state = {
    "success_rate": 0.75,
    "memory_size": 0.32,
    "correction_needed": 0,
    "query_complexity": 0.68,
    "response_length": 0.45,
    "user_satisfaction": 0.81
}

# Get action from agent
action = rl_agent.get_action(state)

# Apply the action
success = await orchestrator._apply_rl_action(action)

# Add experience
reward = rl_agent.calculate_reward(
    user_feedback=0.9,
    task_success=True,
    response_time=2.3
)

rl_agent.add_experience(
    state=state,
    action=action,
    reward=reward,
    next_state=new_state,
    done=False
)
```

## API Reference

For complete API documentation, see [API Reference](api_reference.md).

## Related Components

- [Memory Core](memory_core.md)
- [Recursive Learning System](recursive_learning.md)
- [HyperPlanner](hyperplanner.md)
- [Meta-Learning Controller](meta_learning.md) 
