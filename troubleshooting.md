# Troubleshooting Guide

## Common Issues and Solutions

### System Performance Degradation

**Symptoms:**
- Increasing cycle durations
- Rising memory usage
- Frequent timeouts

**Potential Causes:**
1. Memory leaks
2. Resource contention
3. Excessive data accumulation

**Solutions:**
1. Check the health report logs for metrics trends
2. Verify that `_trim_metrics_history()` is running correctly
3. Confirm that experience buffers are being cleaned with `_clean_experience_buffer()`
4. Run garbage collection manually: `await orchestrator._clean_shutdown(force_gc=True)`

### Recursive Loop Instability

**Symptoms:**
- Frequent cycle failures
- System entering DEGRADED or RECOVERY state
- Increasing consecutive_failures count

**Potential Causes:**
1. Component deadlocks
2. Resource exhaustion
3. Logic errors in cycle processing

**Solutions:**
1. Check logs for timeout patterns and specific errors
2. Verify that timeouts are set appropriately for each cycle
3. Test individual cycle methods directly: `await orchestrator._fast_cycle()`
4. Reset the orchestrator state: `await orchestrator._emergency_recovery_procedure()`

### Memory Consolidation Issues

**Symptoms:**
- Growing memory usage
- Slow retrieval times
- Duplicate or conflicting knowledge

**Potential Causes:**
1. Failed consolidation due to timeouts
2. Errors in deduplication logic
3. Insufficient cleanup

**Solutions:**
1. Force memory optimization: `await memory.optimize_memory_usage(force=True)`
2. Check consolidation logs for specific errors
3. Verify vector store integrity: `await memory.verify_integrity()`
