# Memory Core

The Memory Core is the central knowledge repository of the cognitive system, implementing a brain-inspired memory architecture for storage, retrieval, and continuous optimization of experiences and knowledge.

## Overview

The QuantumMemory class provides a sophisticated neural-inspired memory system with multiple specialized components:

1. **Episodic Buffer**: Short-term storage for recent interactions
2. **Semantic Network**: Long-term knowledge with Zettelkasten-inspired organization
3. **Hebbian Learning**: Connection strengthening between related memories
4. **Memory Consolidation**: Sleep-like processing for organization and pruning

## Key Features

### Semantic Search and Storage

```python
async def add_memory(self, text: str, task: str, score: float) -> int:
    """
    Add a new memory item to the semantic store with embedding-based indexing.
    
    Args:
        text: The text content to store
        task: The associated task/context
        score: Importance score (0.0-1.0)
        
    Returns:
        ID of the newly created memory
    """
    # Implementation details...
```

```python
async def semantic_search(self, query: str, top_k: int = 5, 
                     threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Search memory using semantic similarity with the query.
    
    Args:
        query: The search query text
        top_k: Maximum number of results to return
        threshold: Optional similarity threshold (0.0-1.0)
        
    Returns:
        List of matching memory items with similarity scores
    """
    # Implementation details...
```

### Episodic Buffer Management

The episodic buffer provides a short-term cache with adaptive management:

```python
async def add_to_episodic_buffer(self, user_input: str, response: str) -> None:
    """
    Add a conversation interaction to the episodic buffer.
    If buffer exceeds token limit, items are selectively transferred 
    to long-term memory based on relevance.
    
    Args:
        user_input: The user's input text
        response: The system's response text
    """
    # Implementation details...
```

```python
async def _process_priority_queue(self, max_transfers: int = 3) -> int:
    """
    Process the highest priority items in the buffer,
    transferring them to long-term memory when appropriate.
    
    Args:
        max_transfers: Maximum number of items to transfer
        
    Returns:
        Number of items transferred
    """
    # Implementation details...
```

### Memory Consolidation

Brain-inspired sleep-like processing to strengthen important memories:

```python
async def _replay_random_memories(self, count: int = 2) -> int:
    """
    Randomly select and reprocess older memories to strengthen retention.
    This mimics hippocampal replay during sleep in biological brains.
    
    Returns:
        Number of memories successfully replayed
    """
    # Implementation details...
```

```python
async def _merge_similar_memories(self, similarity_threshold: float = 0.92) -> int:
    """
    Identify and merge highly similar memories to reduce redundancy.
    
    Args:
        similarity_threshold: Minimum similarity for merging
        
    Returns:
        Number of memory groups merged
    """
    # Implementation details...
```

### Hebbian Learning

Neural-inspired connection strengthening between related memories:

```python
async def _reinforce_memory_schemas(self) -> int:
    """
    Identify and reinforce knowledge schemas (patterns of connected memories).
    
    Returns:
        Number of schemas reinforced
    """
    # Implementation details...
```

```python
async def create_second_order_connections(self, memory_indices, strength_multiplier=0.7):
    """
    Create transitive connections between memories that share connections
    to the same memory items (A→B, B→C creates A→C).
    
    Args:
        memory_indices: List of memory indices to process
        strength_multiplier: Connection strength multiplier
        
    Returns:
        Number of new connections created
    """
    # Implementation details...
```

## Memory Architecture

The memory system uses a multi-layer architecture:

1. **Episodic Buffer** (short-term): Holds recent interactions in working memory
2. **Priority Queue**: Sorts memories for importance-based transfer
3. **Semantic Index** (long-term): Vector database with FAISS for fast similarity search
4. **Connection Graph**: Tracks relationships between memories (Hebbian connections)
5. **Schema Clusters**: Groups of related memories forming knowledge structures

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Embedding model for semantic encoding | 'all-MiniLM-L6-v2' |
| `max_episodic_buffer_size` | Maximum interactions in buffer | 10 |
| `max_buffer_tokens` | Maximum tokens in episodic buffer | 128 |
| `episodic_transfer_threshold` | Minimum relevance for transfer | 0.65 |
| `hebbian_learning_rate` | Rate of connection strengthening | 0.01 |
| `hebbian_decay` | Temporal decay of connections | 0.85 |
| `link_threshold` | Minimum similarity for connections | 0.85 |

## Integration with the Learning Loop

The memory system participates in the recursive learning loop through several methods:

```python
# In Orchestrator._recursive_learning_loop
# Memory consolidation phase
if self.memory:
    # 1. Process priority queue (fast consolidation)
    await self.memory._process_priority_queue(max_transfers=3)
    
    # 2. Random memory replay (hippocampal-inspired)
    await self.memory._replay_random_memories(count=2)
    
    # 3. Merge similar memories (reduce redundancy)
    if random.random() < 0.1:  # 10% chance each cycle
        await self.memory._merge_similar_memories()
    
    # 4. Schema reinforcement (strengthen patterns)
    if random.random() < 0.15:  # 15% chance each cycle
        await self.memory._reinforce_memory_schemas()
    
    # 5. Second-order connections (transitive relationships)
    if random.random() < 0.05:  # 5% chance each cycle
        # Select random subset of memories
        if self.memory.metadata:
            memory_subset = random.sample(
                range(len(self.memory.metadata)),
                min(5, len(self.memory.metadata))
            )
            await self.memory.create_second_order_connections(memory_subset)
```

## Example Usage

```python
# Initialize memory system
memory = QuantumMemory(
    model_name='all-MiniLM-L6-v2',
    use_gpu=True,
    max_episodic_buffer_size=15,
    hebbian_learning_rate=0.05,
    enable_zettelkasten=True
)

# Add to episodic buffer (short-term)
await memory.add_to_episodic_buffer(
    user_input="What's the capital of France?",
    response="The capital of France is Paris."
)

# Add directly to long-term memory
memory_id = await memory.add_memory(
    text="Paris is the capital and most populous city of France.",
    task="geography_facts",
    score=0.85
)

# Semantic search
results = await memory.semantic_search(
    query="Tell me about the largest cities in France",
    top_k=3,
    threshold=0.65
)

# Process memory consolidation
await memory._process_priority_queue()
await memory._replay_random_memories(count=3)
await memory._merge_similar_memories()
await memory._reinforce_memory_schemas()
```

## Best Practices

### 1. Memory Optimization

- Set appropriate buffer size based on expected interaction frequency
- Configure pruning thresholds based on expected memory growth
- Balance hebbian_learning_rate for plasticity vs. stability

### 2. Embedding Models

- For general knowledge: 'all-MiniLM-L6-v2' (balanced performance/size)
- For specialized domains: Consider fine-tuned domain-specific models
- For maximum accuracy: 'all-mpnet-base-v2' (higher resource usage)

### 3. Performance Tuning

- Use GPU acceleration for embedding generation when available
- Configure appropriate batch_size (16-64) for parallel processing
- Enable caching for frequently accessed items
- Implement regular pruning for very large memory stores

## Advanced Topics

### 1. Custom Memory Adapters

Connect to external knowledge sources:

```python
class PostgresMemoryAdapter:
    def __init__(self, connection_string):
        self.conn = psycopg2.connect(connection_string)
        
    async def store(self, memory_data):
        # Implementation for storing in PostgreSQL
        
    async def retrieve(self, query, top_k=5):
        # Vector similarity search in PostgreSQL
```

### 2. Memory-Guided Planning

Use memory statistics to guide the planner:

```python
# Get memory access patterns to inform planning
memory_clusters = await memory._identify_memory_clusters()
access_frequency = await memory._calculate_access_frequency()

# Use in planning
plan = await planner.plan(
    goal="Create marketing strategy",
    memory_guidance={
        'clusters': memory_clusters,
        'access_patterns': access_frequency
    }
)
```

### 3. Multi-Modal Memory

Extend the system to store and retrieve multi-modal data:

```python
class MultiModalMemory(QuantumMemory):
    async def add_image_memory(self, image_tensor, caption, score=0.7):
        # Extract image embeddings
        image_embedding = self.image_encoder(image_tensor)
        
        # Create multi-modal memory entry
        # Implementation details...
        
    async def multi_modal_search(self, text_query=None, image_query=None):
        # Search across text and image modalities
        # Implementation details...
```

## API Reference

For complete API documentation, see [API Reference](api_reference.md).

## Related Components

- [Recursive Learning System](recursive_learning.md)
- [HyperPlanner](hyperplanner.md)
- [RL Agent](rl_agent.md) 
