# Performance Benchmarks

This document provides reference benchmarks for the Recursive Cognitive Framework, as well as guidelines for maintaining optimal performance.

## Reference Benchmarks

The following benchmarks were collected on a reference system (16-core CPU, 64GB RAM):

| Component | Operation | Average Duration | Memory Usage |
|-----------|-----------|------------------|-------------|
| RL Agent | Learning Iteration | 45ms | 0.8MB |
| Orchestrator | Fast Cycle | 75ms | 1.2MB |
| Orchestrator | Medium Cycle | 420ms | 5.6MB |
| Orchestrator | Slow Cycle | 2.8s | 28MB |
| Memory | Retrieval (1000 items) | 12ms | 0.4MB |
| Memory | Consolidation | 350ms | 15MB |
| Planning | Task Generation | 80ms | 2.2MB |

## Optimization Guidelines

### Memory Usage

- Keep experience buffer size below 10,000 items
- Set `max_experiences` appropriate to your system memory
- Run `_clean_experience_buffer()` periodically
- Monitor memory growth with the health reporting system

### Cycle Timing

Optimal cycle intervals depend on your workload:

- **Fast Cycle**: 5-10 seconds (default: 5s)
  - Reduce to 2-3s for reactive systems
  - Increase to 15-20s for batch processing systems
  
