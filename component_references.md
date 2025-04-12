# Component Reference Guide

## RL Agent

### Key Methods

#### `_recursive_learning_loop()`
The core learning loop that continuously improves the agent.

**Implementation Details:**
- Operates on multiple biological-inspired phases
- Handles prioritized experience replay
- Manages knowledge distillation
- Implements error recovery with biological-inspired mechanisms
- Uses oscillation timing for memory processing

#### `_select_recovery_strategy(error_streak)`
Selects appropriate recovery strategy based on error persistence.

**Parameters:**
- `error_streak`: Number of consecutive errors

**Returns:**
- Strategy name as string: "MILD_RESET", "BUFFER_FLUSH", or "FULL_ROLLBACK"

#### `_execute_recovery_protocol(strategy)`
Executes the selected recovery protocol.

**Parameters:**
- `strategy`: The recovery strategy to execute

**Recovery Levels:**
- **MILD_RESET**: Resets transient states while preserving core knowledge
- **BUFFER_FLUSH**: Cleans buffers and refreshes temporary storage
- **FULL_ROLLBACK**: Restores from last known good state

## Orchestrator

### Key Methods

#### `run_recursive_learning_loop()`
Main orchestration loop that coordinates all learning cycles.

**Implementation Details:**
- Manages fast (5s), medium (60s), and slow (300s) cycles
- Monitors system health and adjusts parameters accordingly
- Implements recovery mechanisms for different failure modes
- Tracks performance metrics for self-optimization

#### `_reset_cycle_components(cycle_type)`
Resets components after a cycle timeout.

**Parameters:**
- `cycle_type`: Type of cycle ("fast", "medium", or "slow")

#### `_report_system_health(metrics)`
Generates a health report with performance metrics.

**Parameters:**
- `metrics`: Dictionary containing cycle performance metrics
