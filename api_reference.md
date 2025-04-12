# API Reference

This document provides a comprehensive reference for the key classes and methods in the Recursive Cognitive AGI system.

## Core Components

### QuantumSynapseOrchestrator

The central coordinator that manages all system components and the recursive learning loop.

#### Initialization

```python
orchestrator = QuantumSynapseOrchestrator(
    use_gpu: bool = True, 
    embedding_model: str = 'all-MiniLM-L6-v2', 
    max_planning_depth: int = 3, 
    batch_size: int = 8, 
    encryption_key: Optional[str] = None,
    enable_caching: bool = True,
    beam_width: int = 3,
    max_chat_history: int = 10,
    max_token_limit: int = 4000,
    enable_recursive_learning: bool = True,
    recursive_learning_interval: int = 60,
    memory_reflection_depth: int = 3,
    learning_checkpoint_interval: int = 600,
    enable_transfer_learning: bool = True,
    memory_params: Optional[Dict[str, Any]] = None,
    rl_params: Optional[Dict[str, Any]] = None,
    security_params: Optional[Dict[str, Any]] = None
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `chat` | Process a user message and generate a response | `user_input: str` | `str` |
| `process_user_feedback` | Process explicit user feedback | `feedback_score: float`, `response_time: Optional[float] = None`, `task_success: Optional[bool] = None`, `metrics: Optional[Dict[str, float]] = None` | `None` |
| `start_learning_loop` | Start the recursive learning loop | None | `None` |
| `stop_learning_loop` | Stop the recursive learning loop | None | `None` |
| `cleanup` | Release resources | None | `None` |
| `detect_domain` | Detect the domain of a text | `text: str` | `Optional[str]` |
| `set_task_domain` | Set the current task domain | `domain_name: str`, `domain_description: str` | `bool` |

### QuantumMemory

The memory system that manages storage, retrieval, and optimization of knowledge.

#### Initialization

```python
memory = QuantumMemory(
    model_name: str = 'all-MiniLM-L6-v2', 
    use_gpu: bool = True, 
    batch_size: int = 32, 
    encryption_key: Optional[str] = None,
    max_cache_size: int = 5000,
    index_type: str = 'flat',
    max_memory_items: int = 100000,
    retention_days: Optional[int] = None,
    min_memory_score: float = 0.0,
    production_mode: bool = False,
    max_episodic_buffer_size: int = 10, 
    max_buffer_tokens: int = 128,
    episodic_transfer_threshold: Optional[float] = 0.65,
    gate_scaling_factor: float = 10.0,
    max_priority_queue_size: int = 25,
    min_transfer_threshold: float = 0.6,
    enable_zettelkasten: bool = True,
    link_threshold: float = 0.85,
    hebbian_learning_rate: float = 0.01,
    hebbian_decay: float = 0.85
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `add_memory` | Add a memory item to long-term storage | `text: str`, `task: str`, `score: float` | `int` |
| `semantic_search` | Search memory using semantic similarity | `query: str`, `top_k: int = 5`, `threshold: Optional[float] = None` | `List[Dict[str, Any]]` |
| `add_to_episodic_buffer` | Add interaction to episodic buffer | `user_input: str`, `response: str` | `None` |
| `add_chat_interaction` | Add chat interaction and process to memory | `user_input: str`, `response: str` | `bool` |
| `_process_priority_queue` | Process priority queue items | `max_transfers: int = 3` | `int` |
| `_replay_random_memories` | Replay random memories | `count: int = 2` | `int` |
| `_reinforce_memory_schemas` | Reinforce knowledge schemas | None | `int` |
| `_merge_similar_memories` | Merge similar memories | `similarity_threshold: float = 0.92` | `int` |
| `create_second_order_connections` | Create transitive connections | `memory_indices: List[int]`, `strength_multiplier: float = 0.7` | `int` |
| `cleanup` | Release resources | None | `None` |

### RLAgent

The reinforcement learning agent that adapts system behavior based on feedback.

#### Initialization

```python
rl_agent = RLAgent(
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
    verbose: int = 1
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `get_action` | Select an action for a given state | `state: Dict[str, float]` | `str` |
| `add_experience` | Add an experience for learning | `state: Dict[str, float]`, `action: str`, `reward: float`, `next_state: Dict[str, float]`, `done: bool = False`, `success: Optional[bool] = None` | `None` |
| `learn_from_experiences_with_priority` | Learn from experiences with prioritization | `batch_size: Optional[int] = None` | `bool` |
| `learn_from_experiences` | Learn from experiences with standard sampling | `batch_size: Optional[int] = None` | `bool` |
| `distill_knowledge` | Distill knowledge from successful experiences | None | `bool` |
| `set_domain` | Set current domain for transfer learning | `domain_name: str`, `domain_descriptor: str` | `bool` |
| `calculate_reward` | Calculate reward from feedback | `user_feedback: Optional[float] = None`, `task_success: Optional[bool] = None`, `response_time: Optional[float] = None`, `accuracy: Optional[float] = None`, `metrics: Optional[Dict[str, float]] = None` | `float` |
| `save_model` | Save the model | `path: Optional[str] = None` | `bool` |
| `load_model` | Load the model | `path: Optional[str] = None` | `bool` |
| `cleanup` | Release resources | None | `None` |

### HyperPlanner

The planning system that decomposes complex goals into manageable tasks.

#### Initialization

```python
planner = HyperPlanner(
    memory: Optional[QuantumMemory] = None,
    rl_agent: Optional[RLAgent] = None,
    max_depth: int = 3,
    beam_width: int = 3,
    enable_caching: bool = True,
    temporal_decay_factor: float = 0.85,
    cross_validation_k: int = 5,
    concept_drift_threshold: float = 2.5
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `plan` | Generate a hierarchical plan | `goal: str`, `context: Optional[Dict] = None`, `confidence_threshold: float = 0.01` | `Optional[TaskNode]` |
| `record_plan_outcome` | Record outcome for learning | `plan_id: int`, `success: bool`, `execution_time: Optional[float] = None`, `complexity_score: Optional[float] = None`, `metrics: Optional[Dict[str, float]] = None` | `bool` |
| `_beam_search_decomposition` | Perform beam search decomposition | `goal: str`, `context: Dict` | `List[TaskNode]` |
| `_decompose_subtask` | Decompose a subtask | `task: str`, `context: Dict` | `List[str]` |
| `_prioritize_tasks_with_rl` | Prioritize tasks using RL | `tasks: List[TaskNode]`, `context: Dict` | `List[TaskNode]` |
| `_detect_concept_drift` | Detect concept drift | `errors: List[float]`, `threshold: float = 2.5` | `bool` |
| `_validate_plan_prediction` | Validate planning predictions | None | `Dict[str, float]` |
| `_adapt_planning_parameters` | Adapt planning parameters | None | `bool` |

## Utility Classes

### TaskNode

Represents a task in a hierarchical plan.

#### Initialization

```python
task = TaskNode(
    id: int,
    description: str,
    parent: Optional[TaskNode] = None
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `add_child` | Add a child task | `child: TaskNode` | `None` |
| `get_all_tasks` | Get all tasks in the tree | None | `List[TaskNode]` |
| `get_leaf_tasks` | Get all leaf tasks | None | `List[TaskNode]` |
| `print_tree` | Print the task tree | `indent: int = 0` | `None` |
| `to_dict` | Convert to dictionary | None | `Dict[str, Any]` |
| `from_dict` | Create from dictionary | `data: Dict[str, Any]` | `TaskNode` |

### MetaLearningController

Controls adaptive hyperparameter optimization.

#### Initialization

```python
meta_controller = MetaLearningController(
    learning_rate_bounds: Tuple[float, float] = (1e-5, 1e-2),
    gamma_bounds: Tuple[float, float] = (0.9, 0.99),
    persistence_path: str = "meta_learning_state.json"
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `adjust_hyperparameters` | Adjust hyperparameters based on performance | `current_params: Dict[str, float]`, `domain: Optional[str] = None` | `Dict[str, float]` |
| `update_performance_metrics` | Update performance metrics | `reward: float`, `action: Optional[str] = None`, `success: Optional[bool] = None` | `None` |
| `_persist_hyperparameters` | Save hyperparameters | None | `bool` |
| `load_persisted_hyperparameters` | Load saved hyperparameters | None | `bool` |

### TransferLearningBridge

Enables cross-domain knowledge transfer.

#### Initialization

```python
transfer_bridge = TransferLearningBridge(
    embedding_model: str = 'all-MiniLM-L6-v2',
    similarity_threshold: float = 0.7,
    transfer_strength: float = 0.5,
    use_gpu: bool = True,
    persistence_path: str = "transfer_learning_state.json"
)
```

#### Key Methods

| Method | Description | Parameters | Return Type |
|--------|-------------|------------|-------------|
| `register_domain` | Register a new domain | `domain_name: str`, `domain_descriptor: str` | `bool` |
| `find_similar_domains` | Find similar domains | `target_domain: str`, `min_similarity: float = 0.7` | `List[Tuple[str, float]]` |
| `transfer_knowledge` | Transfer knowledge between domains | `source_domain: str`, `target_domain: str`, `current_policy: Dict[str, Any]` | `Dict[str, Any]` |
| `store_policy` | Store policy for a domain | `domain_name: str`, `policy: Dict[str, Any]` | `bool` |
| `save_bridge_state` | Save bridge state | None | `bool` |
| `load_bridge_state` | Load bridge state | None | `bool` |

## Events and Callbacks

### Event Types

| Event | Description | Data Format |
|-------|-------------|------------|
| `memory_added` | Memory item added | `{'id': int, 'text': str, 'task': str, 'score': float}` |
| `plan_created` | Plan created | `{'plan_id': int, 'goal': str, 'task_count': int}` |
| `learning_checkpoint` | Learning checkpoint created | `{'timestamp': float, 'metrics': Dict[str, float]}` |
| `concept_drift_detected` | Concept drift detected | `{'domain': str, 'error_std_dev': float, 'threshold': float}` |
| `transfer_learning_applied` | Transfer learning applied | `{'source_domain': str, 'target_domain': str, 'similarity': float}` |

### Callback Registration

```python
# Register callback for event
orchestrator.register_callback('memory_added', callback_function)

# Callback function signature
def callback_function(event_type: str, event_data: Dict[str, Any]) -> None:
    # Process event
    pass
```

## Error Handling

### Custom Exceptions

| Exception | Description |
|-----------|-------------|
| `SecureMemoryError` | Base exception for secure memory operations |
| `EncryptionError` | Exception raised when encryption fails |
| `DecryptionError` | Exception raised when decryption fails |
| `KeyRotationError` | Exception raised when key rotation fails |
| `PlanningError` | Exception raised when planning fails |
| `RecursiveLearningError` | Exception raised when recursive learning fails |
| `MemoryCorruptionError` | Exception raised when memory corruption detected |

### Error Handling Example

```python
try:
    response = await orchestrator.chat(user_input)
except SecureMemoryError as e:
    print(f"Memory error: {e}")
    # Fallback to stateless mode
    response = await orchestrator.alignment.generate_response(user_input)
except PlanningError as e:
    print(f"Planning error: {e}")
    # Fallback to simpler planning
    response = "I'm having trouble planning this. Could you break it down further?"
except Exception as e:
    print(f"Unexpected error: {e}")
    # General fallback
    response = "I'm sorry, I encountered an unexpected error. Please try again."
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QUANTUM_MEMORY_MODEL` | Embedding model for memory | 'all-MiniLM-L6-v2' |
| `QUANTUM_MEMORY_USE_GPU` | Use GPU for memory operations | 'True' |
| `QUANTUM_MAX_MEMORY` | Maximum memory items | '100000' |
| `QUANTUM_ENCRYPTION_KEY` | Memory encryption key | None |
| `RECURSIVE_LEARNING_INTERVAL` | Recursive learning interval (seconds) | '60' |
| `RL_AGENT_MODEL_PATH` | Path to RL agent model | 'ppo_agent' |
| `ENABLE_TRANSFER_LEARNING` | Enable transfer learning | 'True' |

### Configuration File

Example `config.json`:

```json
{
  "memory": {
    "model_name": "all-MiniLM-L6-v2",
    "use_gpu": true,
    "max_memory_items": 100000,
    "enable_zettelkasten": true,
    "hebbian_learning_rate": 0.05
  },
  "rl_agent": {
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 64,
    "enable_meta_learning": true
  },
  "planner": {
    "max_depth": 3,
    "beam_width": 5,
    "enable_caching": true
  },
  "orchestrator": {
    "enable_recursive_learning": true,
    "recursive_learning_interval": 60,
    "learning_checkpoint_interval": 600
  }
}
```

Loading from configuration:

```python
import json

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize orchestrator with configuration
orchestrator = QuantumSynapseOrchestrator(
    memory_params=config['memory'],
    rl_params=config['rl_agent'],
    **config['orchestrator']
) 
