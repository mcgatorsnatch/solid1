# HyperPlanner

The HyperPlanner is an advanced planning system that implements hierarchical task decomposition with multiple cognitive mechanisms including beam search, temporal attention, concept drift detection, and adaptive validation.

## Overview

The HyperPlanner breaks down complex goals into manageable subtasks using:

1. **Beam Search Decomposition**: Explores multiple decomposition options simultaneously
2. **Temporal Attention**: Weights memory items based on time sensitivity
3. **RL-Driven Prioritization**: Uses reinforcement learning to prioritize tasks
4. **Continuous Validation**: Adapts planning based on outcomes and predictions
5. **Concept Drift Detection**: Identifies when environment assumptions have changed

```
┌───────────────────────────────────────────────────┐
│                   HyperPlanner                     │
│                                                   │
│  ┌─────────────┐      ┌───────────────────┐       │
│  │ Beam Search │◄────►│  RL Prioritization│       │
│  │ Decomposer  │      │                   │       │
│  └─────────────┘      └───────────────────┘       │
│         ▲                      ▲                   │
│         │                      │                   │
│         ▼                      ▼                   │
│  ┌─────────────┐      ┌───────────────────┐       │
│  │  Temporal   │◄────►│   Drift Detection │       │
│  │  Attention  │      │                   │       │
│  └─────────────┘      └───────────────────┘       │
│         ▲                      ▲                   │
│         │                      │                   │
│         ▼                      ▼                   │
│  ┌─────────────┐      ┌───────────────────┐       │
│  │ Cross-      │◄────►│     Dynamic       │       │
│  │ Validation  │      │    Thresholds     │       │
│  └─────────────┘      └───────────────────┘       │
└───────────────────────────────────────────────────┘
```

## Key Components

### 1. TaskNode Structure

The `TaskNode` class represents the hierarchical structure of tasks:

```python
class TaskNode:
    def __init__(self, id: int, description: str, parent=None):
        self.id = id
        self.description = description
        self.parent = parent
        self.children = []
        self.is_completed = False
        self.duration_estimate = None
        self.difficulty_estimate = None
        self.is_time_sensitive = False
        self.priority = 0
        self.knowledge_requirements = []
        self.completion_criteria = ""
        self.dependencies = []
        self.estimated_success_probability = 0.5
        self.metadata = {}
```

### 2. Beam Search Decomposition

Explores multiple task decomposition options in parallel:

```python
async def _beam_search_decomposition(self, goal: str, context: Dict) -> List[TaskNode]:
    """
    Perform beam search to find multiple promising task decomposition paths.
    
    Args:
        goal: The goal to decompose
        context: Additional context for planning
        
    Returns:
        List of task decomposition options
    """
    # Start with the goal as a root node
    root = TaskNode(self._generate_id(), goal)
    
    # Initial beam contains just the root node
    beam = [root]
    
    # Perform beam search for max_depth iterations
    for depth in range(self.max_depth):
        # Candidate lists for the next beam
        candidates = []
        
        # Expand each node in the current beam
        for node in beam:
            # Generate subtasks for this node
            subtasks = await self._decompose_subtask(node.description, context)
            
            # Create child nodes
            child_nodes = []
            for subtask in subtasks:
                child = TaskNode(self._generate_id(), subtask, parent=node)
                node.children.append(child)
                child_nodes.append(child)
            
            # Add this decomposition as a candidate
            candidates.append(node)
        
        # Select top-k candidates for the next beam based on heuristic score
        beam = self._select_top_k_candidates(candidates, self.beam_width)
    
    return beam
```

### 3. Temporal Attention

Applies time-sensitive weighting to memory search and task prioritization:

```python
async def _apply_temporal_attention(self, query: str, memory_items: List[Dict], context: Dict) -> List[Dict]:
    """
    Apply temporal attention to weight memory items based on time sensitivity.
    
    Args:
        query: The search query
        memory_items: List of memory items
        context: Context with temporal information
        
    Returns:
        Re-weighted memory items
    """
    # Default time decay factor (lower = faster decay)
    decay_factor = self.temporal_decay_factor
    
    # Check if the query is time-sensitive
    is_time_sensitive = self._detect_time_sensitivity(query)
    
    # Adjust decay factor for time-sensitive queries
    if is_time_sensitive:
        decay_factor *= 0.5  # More aggressive decay for time-sensitive tasks
    
    # Current time reference
    current_time = time.time()
    
    # Apply temporal weighting
    for item in memory_items:
        if 'timestamp' in item:
            # Calculate time-based decay
            time_diff = (current_time - item['timestamp']) / (3600 * 24)  # Days
            temporal_weight = math.exp(-decay_factor * time_diff)
            
            # Adjust the item's score
            if 'score' in item:
                item['original_score'] = item['score']  # Preserve original
                item['score'] *= temporal_weight
    
    # Re-sort items by adjusted score
    memory_items.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return memory_items
```

### 4. RL-Driven Task Prioritization

Uses reinforcement learning to prioritize tasks based on context:

```python
async def _prioritize_tasks_with_rl(self, tasks: List[TaskNode], context: Dict) -> List[TaskNode]:
    """
    Prioritize tasks using reinforcement learning guidance.
    
    Args:
        tasks: List of tasks to prioritize
        context: Context information
        
    Returns:
        Prioritized list of tasks
    """
    if not self.rl_agent:
        return tasks  # No prioritization without RL agent
    
    try:
        prioritized_tasks = tasks.copy()
        
        # For each task, extract features and get RL-based priority
        for task in prioritized_tasks:
            # Extract task features
            features = self._extract_task_features(task, context)
            
            # Get priority weight from RL agent
            priority = await self.rl_agent.evaluate_priority(features)
            
            # Store priority score
            task.priority = priority
        
        # Sort by priority (descending)
        prioritized_tasks.sort(key=lambda x: x.priority, reverse=True)
        
        return prioritized_tasks
        
    except Exception as e:
        logger.error(f"Error in RL prioritization: {e}")
        return tasks  # Return original ordering on error
```

### 5. Concept Drift Detection

Detects when planning assumptions have changed:

```python
async def _detect_concept_drift(self, errors: List[float], threshold: float = 2.5) -> bool:
    """
    Detect concept drift based on prediction error statistics.
    
    Args:
        errors: List of recent prediction errors
        threshold: Standard deviation threshold for drift detection
        
    Returns:
        True if concept drift is detected, False otherwise
    """
    if len(errors) < 5:
        return False  # Need enough samples
        
    # Calculate rolling statistics
    mean_error = sum(errors) / len(errors)
    variance = sum((x - mean_error) ** 2 for x in errors) / len(errors)
    std_dev = math.sqrt(variance)
    
    # Check for drift condition
    drift_detected = std_dev > threshold
    
    if drift_detected:
        logger.warning(f"Concept drift detected: error std_dev={std_dev:.2f} exceeds threshold={threshold}")
        
        # Trigger adaptation
        await self._adapt_to_concept_drift()
        
    return drift_detected
```

### 6. Dynamic Planning Thresholds

Implements confidence-based plan rejection:

```python
async def plan(self, goal: str, context: Optional[Dict] = None,
              confidence_threshold: float = 0.01) -> Optional[TaskNode]:
    """
    Generate a hierarchical plan with confidence-based rejection.
    
    Args:
        goal: The high-level goal to plan for
        context: Additional context for planning
        confidence_threshold: Minimum confidence threshold for accepting plans
        
    Returns:
        A TaskNode representing the hierarchical plan, or None if confidence
        below threshold
    """
    # Plan generation logic...
    
    # Calculate plan confidence score
    confidence_score = self._calculate_plan_confidence(plan, context)
    
    # Reject low-confidence plans
    if confidence_score < confidence_threshold:
        logger.warning(f"Plan rejected: confidence {confidence_score:.4f} below threshold {confidence_threshold}")
        return None
        
    return plan
```

## Integration with the Learning Loop

The planner continuously improves through the recursive learning loop:

```python
# In Orchestrator._recursive_learning_loop
# Planning improvement phase
if self.planner:
    # Update planning parameters based on success metrics
    if hasattr(self.planner, '_adapt_planning_parameters'):
        await self.planner._adapt_planning_parameters()
    
    # Evaluate success rates of different planning strategies
    if hasattr(self.planner, '_validate_plan_prediction'):
        await self.planner._validate_plan_prediction()
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_depth` | Maximum depth of task decomposition | 3 |
| `beam_width` | Number of parallel decomposition paths | 3 |
| `temporal_decay_factor` | Rate of time-based memory decay | 0.85 |
| `cross_validation_k` | K-fold cross-validation parameter | 5 |
| `enable_caching` | Enable plan caching for similar goals | True |
| `concept_drift_threshold` | Threshold for drift detection | 2.5 |
| `confidence_threshold` | Minimum confidence for plan acceptance | 0.01 |

## Example Usage

```python
# Initialize the planner
planner = HyperPlanner(
    memory=memory,
    rl_agent=rl_agent,
    beam_width=5,
    max_depth=3,
    temporal_decay_factor=0.85,
    enable_caching=True,
    cross_validation_k=5
)

# Generate a plan
goal = "Implement a REST API for user authentication"
context = {
    "domain": "software_development",
    "constraints": ["must use OAuth", "complete within 2 weeks"],
    "available_resources": ["Node.js", "MongoDB", "AWS"]
}

plan = await planner.plan(goal, context)

# Print the hierarchical plan
if plan:
    plan.print_tree()
else:
    print("Could not generate a confident plan")

# Record plan outcome for learning
await planner.record_plan_outcome(
    plan_id=plan.id,
    success=True,
    execution_time=156,  # minutes
    complexity_score=0.75
)
```

## Best Practices

### 1. Goal Formulation

- Be specific and clear in goal descriptions
- Include constraints and requirements in the context
- Break very complex goals into multiple planning sessions

### 2. Performance Optimization

- Use appropriate beam width (3-5 for most tasks)
- Enable caching for similar, repeated goals
- Adjust temporal decay factor based on domain time sensitivity

### 3. Learning Integration

- Always record plan outcomes for continuous improvement
- Periodically validate planning predictions
- Monitor concept drift in dynamic environments

## Advanced Topics

### 1. Multi-Agent Planning

Distribute planning across specialized agents:

```python
class MultiAgentPlanner(HyperPlanner):
    def __init__(self, domain_planners=None, **kwargs):
        super().__init__(**kwargs)
        self.domain_planners = domain_planners or {}
        
    async def plan(self, goal, context=None):
        # Detect domain
        domain = self._detect_domain(goal, context)
        
        # If specialized planner exists, use it
        if domain in self.domain_planners:
            return await self.domain_planners[domain].plan(goal, context)
            
        # Otherwise use general planner
        return await super().plan(goal, context)
```

### 2. Uncertainty-Aware Planning

Incorporate uncertainty estimation in planning:

```python
async def _calculate_plan_uncertainty(self, plan, context):
    """Calculate uncertainty estimates for each task in the plan"""
    
    for node in plan.get_all_tasks():
        # Monte Carlo dropout for uncertainty estimation
        predictions = []
        for _ in range(10):  # Run multiple forward passes
            pred = await self._predict_task_success(node.description, dropout_enabled=True)
            predictions.append(pred)
            
        # Calculate standard deviation as uncertainty measure
        mean = sum(predictions) / len(predictions)
        variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        uncertainty = math.sqrt(variance)
        
        # Store in node metadata
        node.metadata['uncertainty'] = uncertainty
        
        # Higher uncertainty = lower success probability
        uncertainty_factor = max(0, 1 - uncertainty)
        node.estimated_success_probability *= uncertainty_factor
```

### 3. Explainable Planning

Add explainability to planning decisions:

```python
class ExplainablePlanner(HyperPlanner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.explanations = {}
        
    async def _decompose_subtask(self, task, context):
        # Original decomposition
        subtasks = await super()._decompose_subtask(task, context)
        
        # Generate explanation
        explanation = await self._generate_decomposition_explanation(
            task, subtasks, context
        )
        
        # Store explanation
        self.explanations[task] = explanation
        
        return subtasks
        
    def get_plan_explanation(self, plan):
        """Generate a complete explanation of the planning process"""
        explanation = []
        
        for task in plan.get_all_tasks():
            if task.description in self.explanations:
                explanation.append({
                    'task': task.description,
                    'explanation': self.explanations[task.description],
                    'priority': task.priority,
                    'success_probability': task.estimated_success_probability
                })
                
        return explanation
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Possible Causes | Solutions |
|-------|----------------|-----------|
| Shallow task decomposition | Insufficient context, Low max_depth | Increase max_depth, Provide more context, Check memory retrieval quality |
| Over-complicated plans | High beam width, Complex goal | Reduce beam width, Break goal into subgoals, Adjust RL prioritization |
| Frequent plan rejections | Confidence threshold too high, Poor context match | Lower confidence threshold, Improve context specification, Check for concept drift |
| Poor temporal attention | Incorrect decay factor, Missing timestamps | Adjust temporal_decay_factor, Ensure timestamps in memory items, Set is_time_sensitive appropriately |
| Slow planning | Large beam width, Deep decomposition | Reduce beam width or max_depth, Enable caching, Optimize memory retrieval |

## API Reference

For complete API documentation, see [API Reference](api_reference.md).

## Related Components

- [RL Agent](rl_agent.md)
- [Memory Core](memory_core.md)
- [Recursive Learning System](recursive_learning.md) 
