"""
Orchestrator module for coordinating all components and managing the main interaction flow.
"""
import time
import json
import logging
import asyncio
import threading
import math
import random
import concurrent.futures
import uuid
import gc
import pickle
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import psutil

from .memory_core import QuantumMemory
from .hyperplanner import HyperPlanner
from .meta_learning import MetaLearningController
from .rl_agent import RLAgent
from .alignment import CulturalAlignmentModule

logger = logging.getLogger(__name__)

class QuantumSynapseOrchestrator:
    """
    Orchestrates planning, memory, learning, and execution for the Recursive AGI framework.
    Handles multi-turn chat interactions using an episodic buffer and long-term memory.
    Incorporates Hebbian learning for adaptive memory connections.
    """
    def __init__(self, 
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
                 security_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator with memory, learning, and planning components.
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            embedding_model: Name of the embedding model to use
            max_planning_depth: Maximum depth for task planning
            batch_size: Batch size for embedding operations
            encryption_key: Key for memory encryption
            enable_caching: Whether to enable plan and embedding caching
            beam_width: Beam width for task planning
            max_chat_history: Maximum number of chat turns to keep in context
            max_token_limit: Maximum number of tokens to use for context
            enable_recursive_learning: Whether to enable the recursive learning loop
            recursive_learning_interval: Seconds between learning cycles
            memory_reflection_depth: Depth of memory reflection
            learning_checkpoint_interval: Seconds between saving learning checkpoints
            memory_params: Additional parameters for memory module
            rl_params: Parameters for reinforcement learning agent
            security_params: Parameters for security features
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.embedding_model = embedding_model
        self.max_planning_depth = max_planning_depth
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.beam_width = beam_width
        self.max_chat_history = max_chat_history
        self.max_token_limit = max_token_limit
        self.learning_cycles_completed = 0
        
        # Configure recursive learning
        self.enable_recursive_learning = enable_recursive_learning
        self.recursive_learning_interval = recursive_learning_interval
        self.memory_reflection_depth = memory_reflection_depth
        self.learning_checkpoint_interval = learning_checkpoint_interval
        self.last_learning_save = time.time()
        
        # Set up memory system with default or custom parameters
        memory_config = memory_params or {}
        memory_config.update({
            'model_name': embedding_model,
            'use_gpu': use_gpu,
            'batch_size': batch_size,
            'encryption_key': encryption_key,
            'enable_zettelkasten': True,  # Enable Zettelkasten features by default
        })
        
        # Initialize memory system
        self.memory = QuantumMemory(**memory_config)
        
        # Initialize search parameters (used by RL agent actions)
        self.search_parameters = {
            'top_k': 5,
            'threshold': 0.7
        }
        
        # Initialize RL agent with PPO if parameters provided
        self.rl_agent = None
        if rl_params:
            # Ensure default PPO parameters if not specified
            default_rl_params = {
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'n_steps': 128,
                'batch_size': 64,
                'n_epochs': 10,
                'use_gpu': use_gpu
            }
            # Update with user-provided parameters
            default_rl_params.update(rl_params)
            self.rl_agent = RLAgent(**default_rl_params)
            logger.info(f"Initialized PPO-based RL agent with {len(self.rl_agent.action_set)} actions")
            
        # Set up planning system
        self.planner = HyperPlanner(
            max_depth=max_planning_depth,
            beam_width=beam_width,
            memory=self.memory,
            enable_caching=enable_caching,
            rl_agent=self.rl_agent,  # Connect RL agent for task prioritization
            cross_validation_k=5,    # Enable cross-validation for continuous refinement
            enable_continuous_validation=True,
            drift_detection_threshold=2.5  # Standard threshold for concept drift detection
        )
        
        # Chat and interaction history
        self.chat_history = []
        self.chat_context_turns = min(5, max_chat_history)  # Start with reasonable default
        
        # Initialize learning background task
        self.learning_task = None
        if self.enable_recursive_learning:
            # Start recursive learning loop
            self.learning_task = asyncio.create_task(self._recursive_learning_loop())
            
        logger.info(f"QuantumSynapseOrchestrator initialized with {embedding_model} on {self.device}")
        
        # Add transfer learning flag
        if rl_params is None:
            rl_params = {}
        rl_params['enable_transfer_learning'] = enable_transfer_learning
        
        # Track current domain/task
        self.current_domain = "general"
        self.domain_history = {}
        
        # Tracking timestamps for dual-time scale adaptation
        self.last_fast_adapt_time = time.time()
        self.last_slow_adapt_time = time.time()
        self.fast_adaptation_interval = 30  # 30 seconds
        self.slow_adaptation_interval = 86400  # 24 hours
        
    async def _recursive_learning_loop(self) -> None:
        """
        Advanced cognitive recursive learning loop that mimics brain-like memory processes.
        Implements memory consolidation, active replay, hierarchical attention mechanisms,
        predictive schema formation, and goal-driven self-improvement for continuous adaptation.
        
        This method integrates multiple cognitive mechanisms inspired by neuroscience:
        1. Hippocampal memory indexing and consolidation
        2. Neocortical knowledge extraction and schema formation
        3. Hebbian learning ("neurons that fire together, wire together")
        4. Synaptic pruning for network optimization
        5. Attention-weighted memory activation
        6. Dual-process learning (fast/implicit and slow/explicit)
        7. Predictive coding and error-driven learning
        8. Goal-directed adaptation with self-assessment
        """
        # --- Initialization of Learning Management Systems ---
        logger.info("Starting advanced cognitive learning loop")
        
        # Track learning cycles and timing for different adaptation processes
        # This implements time-separated learning processes similar to brain's 
        # different timescales for plasticity (minutes to months)
        learning_cycles = 0
        last_prune_time = time.time()         # For structural maintenance
        last_rl_training_time = time.time()   # For reinforcement learning updates
        last_memory_consolidation = time.time()  # For memory consolidation (like sleep)
        last_schema_update = time.time()      # For long-term knowledge structure updates
        
        # Goal-directed adaptation tracking
        last_fast_adaptation = time.time()   # For fast (30s) adaptation cycle
        last_slow_adaptation = time.time()   # For slow (24h) adaptation cycle
        
        # --- Define Biologically-Inspired Timing Intervals ---
        # These intervals mimic different biological learning timeframes
        prune_interval = 3600 * 6  # Synaptic pruning every 6 hours (like sleep phases)
        rl_training_interval = 3600  # Train RL model hourly (like working memory updates)
        consolidation_interval = 3600 * 2  # Memory consolidation every 2 hours (like REM cycles)
        schema_update_interval = 3600 * 12  # Update knowledge schemas every 12 hours (like overnight consolidation)
        
        # Goal-directed adaptation intervals
        fast_adaptation_interval = 30  # Fast adaptation every 30 seconds
        slow_adaptation_interval = 86400  # Slow adaptation every 24 hours
        
        # --- Error Resilience Mechanisms ---
        # Implement homeostatic regulation through adaptive error handling
        error_backoff = 30  # Initial backoff time in seconds
        max_backoff = 3600  # Maximum backoff time (1 hour)
        
        # --- Memory Optimization Structures ---
        # These structures accelerate memory access similar to attention mechanisms
        memory_access_patterns = {}  # Track pattern-based access (like prefrontal cortex)
        active_memory_cache = []  # Recently active memories (like working memory buffer)
        
        # --- Main Cognitive Learning Loop ---
        while self.enable_recursive_learning:
            try:
                # Sleep at the beginning to prevent resource overuse
                # This creates a pulsed learning pattern similar to brain oscillations
                await asyncio.sleep(self.recursive_learning_interval)
                
                # --- Prerequisite Check ---
                # Skip processing if memory system is not ready
                if not hasattr(self, 'memory') or not self.memory or len(self.memory.metadata) == 0:
                    logger.debug("Skipping learning cycle: memory not ready or empty")
                    continue
                
                # Increment learning cycles counter
                learning_cycles += 1
                current_time = time.time()
                
                # === GOAL-DIRECTED ADAPTATION CYCLES ===
                # Apply fast adaptation cycle (30s) for continuous improvement
                if current_time - last_fast_adaptation >= fast_adaptation_interval:
                    await self._apply_fast_adaptation()
                    last_fast_adaptation = current_time
                    
                # Apply slow adaptation cycle (24h) for major architectural changes
                if current_time - last_slow_adaptation >= slow_adaptation_interval:
                    await self._apply_slow_adaptation()
                    last_slow_adaptation = current_time
                    
                # ======================================================================
                # === PHASE 1: MEMORY ACTIVATION ANALYSIS =============================
                # ======================================================================
                # This phase implements attention-weighted memory activation similar to
                # how the hippocampus and prefrontal cortex select relevant memories
                
                # --- Temporal Memory Activation with Decay Gradient ---
                # Extract active memories using recency and importance weighting
                # This mimics the brain's attentional mechanisms that prioritize
                # recent and important information
                active_memories = []  # Indices of active memories
                active_weights = []   # Attention weights for active memories
                
                # Define recency window (24 hours) - memories accessed within this
                # window are considered "active" with an exponential decay gradient
                # This resembles the hippocampal indexing of recent episodic memories
                recent_cutoff = current_time - (24 * 3600)
                
                # --- Implement Attention-Weighted Memory Activation ---
                for i, item in enumerate(self.memory.metadata):
                    access_time = item.get('last_accessed', 0)
                    if access_time >= recent_cutoff:
                        # Calculate exponential decay based on recency (6-hour half-life)
                        # This implements temporal decay similar to biological memory
                        time_diff = current_time - access_time
                        recency_weight = math.exp(-time_diff / (6 * 3600))
                        
                        # Combine recency with importance for multi-dimensional weighting
                        # This mimics how the brain weights memories by both recency and salience
                        importance = item.get('importance', 0.5)
                        combined_weight = 0.7 * recency_weight + 0.3 * importance
                        
                        active_memories.append(i)
                        active_weights.append(combined_weight)
                        
                        # --- Track Access Patterns for Optimization ---
                        # This builds a model of memory usage for accelerating future access
                        # Similar to how the brain forms shortcuts for frequently used pathways
                        memory_id = item.get('id')
                        if memory_id not in memory_access_patterns:
                            memory_access_patterns[memory_id] = {'count': 0, 'last_access': 0}
                        memory_access_patterns[memory_id]['count'] += 1
                        memory_access_patterns[memory_id]['last_access'] = current_time
                
                # Skip processing if no active memories are available
                # This prevents wasting resources on empty learning cycles
                if not active_memories:
                    logger.debug("Skipping learning cycle: no active memories found")
                    continue
                
                # --- Update Working Memory Cache ---
                # Keep the most active memories in a quick-access cache
                # This mimics the function of working memory in the brain
                active_memory_cache = active_memories[:20]  # Keep top 20 most active
                
                # ======================================================================
                # === PHASE 2: HEBBIAN LEARNING & SYNAPTIC PLASTICITY =================
                # ======================================================================
                # This phase implements Hebbian learning ("neurons that fire together, wire together")
                # to strengthen connections between co-activated memories
                try:
                    if hasattr(self.memory, 'apply_hebbian_learning'):
                        # Apply Hebbian learning between active memories
                        # This strengthens connections between memories that are active together
                        if len(active_memories) >= 2:
                            # --- Attention-Guided Memory Selection ---
                            # Use weighted probability sampling based on activation strength
                            # This mimics the brain's attentional selection of memories
                            selected_indices = self._weighted_sample(active_memories, active_weights, 
                                                                   max_samples=min(10, len(active_memories)))
                            
                            # --- Apply Hebbian Learning ---
                            # Strengthen connections between selected memories
                            # This implements the core Hebbian learning principle
                            await self.memory.apply_hebbian_learning(selected_indices)
                            
                            # --- Create Higher-Order Associations ---
                            # Periodically create second-order connections 
                            # This implements transitive associations (A→B, B→C ⟹ A→C)
                            # similar to how the brain forms indirect associations
                            if learning_cycles % 5 == 0 and hasattr(self.memory, 'create_second_order_connections'):
                                # Select top memories by activation weight
                                top_memories = [active_memories[i] for i in 
                                               sorted(range(len(active_weights)), 
                                                     key=lambda i: active_weights[i], 
                                                     reverse=True)[:3]]
                                # Create secondary connections - "friends of friends become friends"
                                await self.memory.create_second_order_connections(top_memories)
                except Exception as e:
                    logger.error(f"Error during Hebbian learning: {e}", exc_info=True)
                
                # ======================================================================
                # === PHASE 3: MEMORY CONSOLIDATION (mimics sleep phases) =============
                # ======================================================================
                # This phase implements memory consolidation processes similar to
                # how the brain consolidates memories during sleep, particularly REM sleep
                if current_time - last_memory_consolidation > consolidation_interval:
                    try:
                        # --- 3.1 Process Episodic Buffer ---
                        # Handle recent episodic memories with dual focus strategy
                        # Similar to how the hippocampus transfers memories to neocortex
                        if hasattr(self.memory, 'episodic_buffer') and hasattr(self.memory, '_process_priority_queue'):
                            # Transfer important recent memories first (prioritized consolidation)
                            # This mimics the brain's prioritization of emotionally salient memories
                            recent_processed = await self.memory._process_priority_queue(max_transfers=3)
                            
                            # --- Memory Replay for Older Memories ---
                            # Replay older memories to prevent catastrophic forgetting
                            # This mimics hippocampal replay during slow-wave sleep
                            if hasattr(self.memory, '_replay_random_memories'):
                                old_processed = await self.memory._replay_random_memories(count=2)
                                logger.info(f"Memory consolidation: processed {recent_processed} recent and {old_processed} older memories")
                                
                        # --- 3.2 Schema Reinforcement ---
                        # Strengthen frequently accessed memory clusters (knowledge schemas)
                        # This mimics how the brain forms and reinforces semantic knowledge structures
                        if hasattr(self.memory, '_reinforce_memory_schemas'):
                            schemas_updated = await self.memory._reinforce_memory_schemas()
                            if schemas_updated:
                                logger.info(f"Reinforced {schemas_updated} memory schemas")
                                
                        # Update consolidation timestamp
                        last_memory_consolidation = current_time
                    except Exception as e:
                        logger.error(f"Error during memory consolidation: {e}", exc_info=True)
                        # Still update timestamp to prevent repeated failures
                        last_memory_consolidation = current_time
                
                # ======================================================================
                # === PHASE 4: ADAPTIVE PRUNING & STRUCTURAL MAINTENANCE ==============
                # ======================================================================
                # This phase implements synaptic pruning and structural optimization
                # similar to how the brain removes unused connections during development and sleep
                if current_time - last_prune_time > prune_interval:
                    try:
                        # --- 4.1 Synaptic Pruning ---
                        # Remove weak connections with adaptive threshold
                        # This mimics activity-dependent synaptic pruning in the brain
                        if hasattr(self.memory, '_prune_synaptic_connections'):
                            # Calculate adaptive threshold based on memory size
                            # Larger memory requires more aggressive pruning (resource optimization)
                            memory_size = len(self.memory.metadata)
                            adaptive_ratio = min(0.15, max(0.02, 0.05 + (memory_size / 10000) * 0.1))
                            
                            pruned = await self.memory._prune_synaptic_connections(adaptive_ratio)
                            logger.info(f"Pruned {pruned} weak synaptic connections (ratio: {adaptive_ratio:.2f})")
                            
                        # --- 4.2 Memory Deduplication ---
                        # Identify and merge redundant memories
                        # This implements efficient storage similar to how the brain avoids redundancy
                        if learning_cycles % 10 == 0 and hasattr(self.memory, '_merge_similar_memories'):
                            # Use high similarity threshold to only merge very similar memories
                            merged = await self.memory._merge_similar_memories(similarity_threshold=0.92)
                            if merged > 0:
                                logger.info(f"Merged {merged} redundant memories")
                                
                        # Update pruning timestamp
                        last_prune_time = current_time
                    except Exception as e:
                        logger.error(f"Error during memory pruning: {e}", exc_info=True)
                        last_prune_time = current_time
                
                # ======================================================================
                # === PHASE 5: LONG-TERM SCHEMA & KNOWLEDGE STRUCTURE UPDATES =========
                # ======================================================================
                # This phase implements schema formation and structural knowledge organization
                # similar to how the neocortex forms semantic knowledge networks
                if current_time - last_schema_update > schema_update_interval:
                    try:
                        # --- 5.1 Knowledge Pattern Extraction ---
                        # Identify recurring patterns across memories
                        # This mimics how the brain extracts general rules from specific examples
                        if hasattr(self.memory, '_extract_knowledge_patterns'):
                            patterns = await self.memory._extract_knowledge_patterns()
                            if patterns:
                                logger.info(f"Extracted {len(patterns)} knowledge patterns")
                                
                                # --- 5.2 Knowledge Organization ---
                                # Apply extracted patterns to reorganize memory
                                # This implements semantic organization similar to neocortical processing
                                if hasattr(self.memory, '_apply_knowledge_patterns'):
                                    reorganized = await self.memory._apply_knowledge_patterns(patterns)
                                    logger.info(f"Reorganized {reorganized} memories based on knowledge patterns")
                        
                        # Update schema timestamp
                        last_schema_update = current_time
                    except Exception as e:
                        logger.error(f"Error during schema updates: {e}", exc_info=True)
                        last_schema_update = current_time
                
                # ======================================================================
                # === PHASE 6: SAVE STATE & PRESERVATION ==============================
                # ======================================================================
                # This phase implements persistence of learned knowledge
                # Similar to how biological learning results in physical changes in the brain
                if current_time - self.last_learning_save > self.learning_checkpoint_interval:
                    try:
                        await self._save_learning_state()
                        self.last_learning_save = current_time
                    except Exception as e:
                        logger.error(f"Error saving learning state: {e}", exc_info=True)
                        self.last_learning_save = current_time
                
                # ======================================================================
                # === PHASE 7: MEMORY GRAPH ANALYTICS ================================= 
                # ======================================================================
                # This phase implements meta-cognitive analysis of the memory structure
                # Similar to how the brain monitors its own connectivity patterns
                if learning_cycles % 10 == 0:
                    try:
                        # --- Memory Network Analysis ---
                        # Calculate connectivity statistics for the memory graph
                        # This provides insight into the structure of knowledge representation
                        connection_count = len(self.memory.connection_weights) if hasattr(self.memory, 'connection_weights') else 0
                        memory_count = len(self.memory.metadata) if hasattr(self.memory, 'metadata') else 0
                        
                        if connection_count > 0 and memory_count > 0:
                            # Calculate basic connectivity density
                            avg_connections = connection_count / memory_count
                            
                            # --- Advanced Graph Analytics ---
                            # Calculate graph-theoretic metrics for the memory network
                            # This analyzes knowledge organization quality and efficiency
                            if hasattr(self.memory, '_calculate_graph_metrics'):
                                metrics = await self.memory._calculate_graph_metrics()
                                clustering = metrics.get('clustering_coefficient', 0)  # Local connectivity 
                                centrality = metrics.get('avg_centrality', 0)  # Importance distribution
                                
                                logger.info(f"Memory graph: {memory_count} nodes, {connection_count} connections, "
                                          f"{avg_connections:.2f} avg connections, clustering: {clustering:.2f}, "
                                          f"centrality: {centrality:.2f}")
                            else:
                                logger.info(f"Memory graph: {memory_count} nodes, {connection_count} connections, "
                                          f"{avg_connections:.2f} avg connections per memory")
                    except Exception as e:
                        logger.error(f"Error analyzing memory graph: {e}", exc_info=True)
                
                # ======================================================================
                # === PHASE 8: REINFORCEMENT LEARNING OPTIMIZATION ====================
                # ======================================================================
                # This phase implements reward-based learning optimization
                # Similar to how the brain's dopaminergic systems drive learning from rewards
                if hasattr(self, 'rl_agent') and self.rl_agent:
                    if current_time - last_rl_training_time > rl_training_interval:
                        try:
                            # --- 8.1 Prioritized Experience Replay ---
                            # Focus learning on surprising or high-error experiences
                            # This mimics how the brain prioritizes learning from unexpected outcomes
                            if hasattr(self.rl_agent, 'learn_from_experiences_with_priority'):
                                success = await self.rl_agent.learn_from_experiences_with_priority()
                                if success:
                                    logger.info("Updated RL agent with prioritized experience replay")
                            # --- Fallback Learning Mechanism ---
                            # Use standard learning if prioritized learning is unavailable
                            elif hasattr(self.rl_agent, 'learn_from_experiences'):
                                if self.rl_agent.learn_from_experiences():
                                    logger.info("Updated RL agent from recent experiences")
                                    
                            # --- 8.2 Knowledge Distillation ---
                            # Extract general principles from successful experiences
                            # This mimics how the brain consolidates successful strategies
                            if learning_cycles % 5 == 0 and hasattr(self.rl_agent, 'distill_knowledge'):
                                await self.rl_agent.distill_knowledge()
                                
                            last_rl_training_time = current_time
                        except Exception as e:
                            logger.error(f"Error in RL training: {e}", exc_info=True)
                            last_rl_training_time = current_time
                
                # ======================================================================
                # === PHASE 9: TIME-SCALED ADAPTATION =================================
                # ======================================================================
                # This phase implements dual-timescale learning adaptation
                # Similar to how the brain has both fast and slow learning processes
                
                # Fast adaptation cycle (every 30s)
                if current_time - self.last_fast_adapt_time >= self.fast_adaptation_interval:
                    await self._apply_fast_adaptation()
                    self.last_fast_adapt_time = current_time
                
                # Slow adaptation cycle (every 24h)
                if current_time - self.last_slow_adapt_time >= self.slow_adaptation_interval:
                    await self._apply_slow_adaptation()
                    self.last_slow_adapt_time = current_time
                
                # ======================================================================
                # === PHASE 10: COGNITIVE MONITORING ==================================
                # ======================================================================
                # This phase implements meta-cognitive monitoring and adaptation
                # Similar to how the prefrontal cortex monitors cognitive processes
                
                # --- Concept Drift Detection ---
                # Check if planning assumptions are still valid
                # This implements environmental change detection
                if hasattr(self, 'planner') and self.planner:
                    try:
                        if hasattr(self.planner, '_detect_concept_drift'):
                            drift_detected = await self.planner._detect_concept_drift()
                            if drift_detected:
                                # React to drift by triggering more aggressive adaptation
                                logger.warning(f"Concept drift detected, triggering adaptive response")
                                await self._react_to_concept_drift()
                    except Exception as e:
                        logger.error(f"Error during concept drift detection: {e}", exc_info=True)
                
                # Update learning cycle counter
                learning_cycles += 1
                self.learning_cycles_completed = learning_cycles
                
            except Exception as e:
                logger.error(f"Error in recursive learning loop: {e}", exc_info=True)
                # Apply exponential backoff to prevent rapid failure loops
                # This implements homeostatic regulation (system self-protection)
                await asyncio.sleep(min(error_backoff, max_backoff))
                error_backoff = min(error_backoff * 2, max_backoff)  # Exponential increase
        
        logger.info("Recursive learning loop stopped")
    
    def _weighted_sample(self, items, weights, max_samples):
        """
        Perform weighted random sampling without replacement.
        
        Args:
            items: List of items to sample from
            weights: List of weights for each item
            max_samples: Maximum number of samples to return
            
        Returns:
            List of selected indices
        """
        if not items or not weights or max_samples <= 0:
            return []
            
        # Convert to numpy arrays for efficient sampling
        try:
            import numpy as np
            items_arr = np.array(items)
            weights_arr = np.array(weights)
            
            # Normalize weights if they don't sum to 1
            if abs(np.sum(weights_arr) - 1.0) > 1e-6:
                weights_arr = weights_arr / np.sum(weights_arr)
                
            # Perform weighted sampling without replacement
            selected = np.random.choice(
                len(items_arr), 
                size=min(max_samples, len(items_arr)), 
                replace=False, 
                p=weights_arr
            )
            
            return selected.tolist()
        except ImportError:
            # Fallback if numpy not available
            selected = []
            remaining = list(range(len(items)))
            
            for _ in range(min(max_samples, len(items))):
                if not remaining:
                    break
                    
                # Calculate remaining weights sum
                total = sum(weights[i] for i in remaining)
                if total <= 0:
                    break
                    
                # Generate random value
                r = random.random() * total
                
                # Find selected item
                running_sum = 0
                for i, idx in enumerate(remaining):
                    running_sum += weights[idx]
                    if running_sum >= r:
                        selected.append(idx)
                        remaining.pop(i)
                        break
                        
            return selected
            
    async def _react_to_concept_drift(self):
        """
        React to detected concept drift with adaptive countermeasures.
        
        This method implements multiple strategies to address concept drift:
        1. Increase learning rates temporarily
        2. Invalidate affected planning caches
        3. Accelerate meta-parameter optimization
        4. Adjust the sensitivity of future drift detection
        """
        logger.info("Executing concept drift countermeasures")
        
        try:
            # 1. Temporarily increase plasticity in memory system
            if hasattr(self.memory, 'adjust_learning_parameters'):
                await self.memory.adjust_learning_parameters(
                    hebbian_rate_multiplier=1.5,  # Increase Hebbian learning rate
                    temporal_decay_multiplier=0.8,  # Reduce temporal decay
                    duration=3600  # Apply for 1 hour
                )
            
            # 2. Invalidate planning caches in affected regions
            if hasattr(self.planner, 'invalidate_drifted_cache'):
                invalidated = await self.planner.invalidate_drifted_cache()
                logger.info(f"Invalidated {invalidated} cached plans due to concept drift")
            
            # 3. Accelerate meta-learning update cycle
            if hasattr(self.rl_agent, 'meta_controller'):
                # Force immediate parameter recalculation
                await self.rl_agent.meta_controller.force_parameter_update()
                
                # Increase parameter sensitivity temporarily
                if hasattr(self.rl_agent.meta_controller, 'increase_sensitivity'):
                    self.rl_agent.meta_controller.increase_sensitivity(
                        factor=2.0,
                        duration=3600
                    )
            
            # 4. Adjust threshold for future drift detection based on history
            if hasattr(self.planner, 'adjust_drift_threshold'):
                adjusted = await self.planner.adjust_drift_threshold()
                if adjusted:
                    logger.info(f"Adjusted concept drift threshold to {adjusted:.3f}")
                
        except Exception as e:
            logger.error(f"Error in concept drift response: {e}", exc_info=True)
            
    async def _evaluate_system_performance(self) -> Dict[str, Any]:
        """
        Comprehensively evaluate system performance across all components.
        Returns a dictionary of performance metrics and an overall score.
        """
        try:
            # Gather performance metrics from all components in parallel
            memory_metrics, planner_metrics, rl_metrics, transfer_metrics = await asyncio.gather(
                self._evaluate_memory_performance(),
                self._evaluate_planner_performance(),
                self._evaluate_rl_performance(),
                self._evaluate_transfer_learning()
            )
            
            # Combine all metrics into a single dictionary
            metrics = {
                'memory': memory_metrics,
                'planner': planner_metrics,
                'rl_agent': rl_metrics,
                'transfer_learning': transfer_metrics
            }
            
            # Calculate overall performance score (weighted average)
            component_weights = {
                'memory': 0.3,
                'planner': 0.3,
                'rl_agent': 0.25,
                'transfer_learning': 0.15
            }
            
            overall_score = 0.0
            for component, weight in component_weights.items():
                if component in metrics:
                    overall_score += metrics[component].get('overall_score', 0.0) * weight
            
            # Add overall score to metrics
            metrics['overall_score'] = overall_score
            
            # Track performance history for trend analysis
            if not hasattr(self, 'performance_history'):
                self.performance_history = []
                
            # Add current metrics to history with timestamp
            self.performance_history.append({
                'timestamp': time.time(),
                'overall_score': overall_score,
                'component_scores': {c: m.get('overall_score', 0.0) for c, m in metrics.items() if c != 'overall_score'}
            })
            
            # Trim history if it grows too large
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating system performance: {e}")
            return {'overall_score': 0.5}  # Default score on error
    
    async def _evaluate_memory_performance(self) -> Dict[str, Any]:
        """
        Evaluate memory system performance
        
        Returns:
            Dictionary of memory performance metrics
        """
        if not hasattr(self, 'memory') or self.memory is None:
            return {'overall_score': 0.0, 'error': 'Memory not available'}
            
        try:
            metrics = {}
            
            # Retrieval precision (if available)
            retrieval_precision = 0.0
            if hasattr(self.memory, 'get_retrieval_precision'):
                retrieval_precision = await self.memory.get_retrieval_precision()
            metrics['retrieval_precision'] = retrieval_precision
            
            # Calculate memory efficiency (size vs. quality tradeoff)
            memory_count = len(self.memory.memories) if hasattr(self.memory, 'memories') else 0
            metrics['memory_count'] = memory_count
            
            # Episodic buffer metrics
            buffer_stats = await self.memory.get_episodic_buffer_stats() if hasattr(self.memory, 'get_episodic_buffer_stats') else {}
            metrics['episodic_buffer'] = buffer_stats
            
            # Connection graph metrics if using Zettelkasten
            if hasattr(self.memory, '_calculate_graph_metrics'):
                graph_metrics = await self.memory._calculate_graph_metrics()
                metrics['graph_metrics'] = graph_metrics
            
            # Calculate overall memory score (weighted combination of metrics)
            # Higher is better (0.0 to 1.0)
            overall_score = 0.5  # Default moderate score
            
            # Factor in retrieval precision (heavily weighted)
            if retrieval_precision > 0:
                overall_score = 0.7 * retrieval_precision
            
            # Factor in memory utilization (penalize if too empty or too full)
            if hasattr(self.memory, 'max_items') and self.memory.max_items > 0:
                utilization = memory_count / self.memory.max_items
                # Ideal utilization is around 40-80%
                if 0.4 <= utilization <= 0.8:
                    overall_score += 0.15
                elif 0.2 <= utilization < 0.4 or 0.8 < utilization <= 0.9:
                    overall_score += 0.1
                else:
                    overall_score += 0.05
            
            # Add graph connectivity score if available
            if 'graph_metrics' in metrics and 'connectivity' in metrics['graph_metrics']:
                connectivity = metrics['graph_metrics']['connectivity']
                overall_score += 0.15 * connectivity
            
            metrics['overall_score'] = min(1.0, overall_score)
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating memory performance: {e}")
            return {'overall_score': 0.3, 'error': str(e)}
    
    async def _evaluate_planner_performance(self) -> Dict[str, Any]:
        """
        Evaluate planning system performance
        
        Returns:
            Dictionary of planner performance metrics
        """
        if not hasattr(self, 'planner') or self.planner is None:
            return {'overall_score': 0.0, 'error': 'Planner not available'}
            
        try:
            metrics = {}
            
            # Plan success rate
            if hasattr(self.planner, 'get_success_rate'):
                metrics['success_rate'] = self.planner.get_success_rate()
            else:
                metrics['success_rate'] = 0.5  # Default moderate score
            
            # Plan complexity (higher is better, but not too high)
            if hasattr(self.planner, 'get_average_plan_depth'):
                avg_depth = self.planner.get_average_plan_depth()
                metrics['average_plan_depth'] = avg_depth
                
                # Ideal plan depth is 3-5 steps (neither too simple nor too complex)
                if 3 <= avg_depth <= 5:
                    metrics['depth_score'] = 1.0
                elif 2 <= avg_depth < 3 or 5 < avg_depth <= 7:
                    metrics['depth_score'] = 0.8
                else:
                    metrics['depth_score'] = 0.6
            else:
                metrics['depth_score'] = 0.5
            
            # Drift detection (planner's adaptability to changing contexts)
            if hasattr(self.planner, 'get_drift_metrics'):
                drift_metrics = self.planner.get_drift_metrics()
                metrics['drift'] = drift_metrics
                
                # Lower drift severity is better
                if 'severity' in drift_metrics:
                    drift_severity = drift_metrics['severity']
                    if drift_severity < 0.3:
                        metrics['drift_score'] = 1.0
                    elif drift_severity < 0.6:
                        metrics['drift_score'] = 0.7
                    else:
                        metrics['drift_score'] = 0.4
                else:
                    metrics['drift_score'] = 0.5
            else:
                metrics['drift_score'] = 0.5
            
            # Domain-specific performance if available
            if hasattr(self.planner, 'get_domain_success_rates'):
                domain_rates = self.planner.get_domain_success_rates()
                metrics['domain_success_rates'] = domain_rates
                
                # Calculate domain diversity score
                if domain_rates:
                    # Higher diversity of successful domains is better
                    successful_domains = sum(1 for rate in domain_rates.values() if rate > 0.7)
                    domain_diversity = min(1.0, successful_domains / max(1, len(domain_rates)))
                    metrics['domain_diversity'] = domain_diversity
                else:
                    metrics['domain_diversity'] = 0.0
            
            # Calculate overall planning score (weighted combination of metrics)
            # Higher is better (0.0 to 1.0)
            overall_score = (
                0.4 * metrics['success_rate'] +
                0.2 * metrics['depth_score'] +
                0.2 * metrics['drift_score'] +
                0.2 * metrics.get('domain_diversity', 0.5)
            )
            
            metrics['overall_score'] = overall_score
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating planner performance: {e}")
            return {'overall_score': 0.3, 'error': str(e)}
    
    async def _evaluate_rl_performance(self) -> Dict[str, Any]:
        """
        Evaluate reinforcement learning agent performance
        
        Returns:
            Dictionary of RL performance metrics
        """
        if not hasattr(self, 'rl_agent') or self.rl_agent is None:
            return {'overall_score': 0.0, 'error': 'RL agent not available'}
            
        try:
            metrics = {}
            
            # Get performance stats from RL agent
            if hasattr(self.rl_agent, 'get_performance_stats'):
                stats = self.rl_agent.get_performance_stats()
                metrics.update(stats)
            
            # Get learning efficiency
            if hasattr(self.rl_agent, 'get_learning_efficiency'):
                learning_efficiency = self.rl_agent.get_learning_efficiency()
                metrics['learning_efficiency'] = learning_efficiency
            else:
                metrics['learning_efficiency'] = 0.5
            
            # Get learning status
            if hasattr(self.rl_agent, 'get_learning_status'):
                learning_status = self.rl_agent.get_learning_status()
                metrics['learning_status'] = learning_status
                
                # Extract additional metrics from learning status
                if 'performance' in learning_status:
                    metrics.update(learning_status['performance'])
            
            # Calculate exploration-exploitation balance
            if hasattr(self.rl_agent, 'model') and hasattr(self.rl_agent.model, 'clip_range'):
                exploration_rate = self.rl_agent.model.clip_range
                metrics['exploration_rate'] = exploration_rate
                
                # Check if exploration rate is appropriate for current performance
                avg_reward = metrics.get('avg_recent_reward', 0.5)
                
                if avg_reward > 0.8 and exploration_rate < 0.2:
                    # Good balance: high performance, low exploration
                    metrics['exploration_balance_score'] = 1.0
                elif avg_reward < 0.4 and exploration_rate > 0.5:
                    # Good balance: low performance, high exploration
                    metrics['exploration_balance_score'] = 0.9
                elif 0.4 <= avg_reward <= 0.8 and 0.2 <= exploration_rate <= 0.5:
                    # Good balance: moderate performance, moderate exploration
                    metrics['exploration_balance_score'] = 0.8
                else:
                    # Suboptimal balance
                    metrics['exploration_balance_score'] = 0.5
            else:
                metrics['exploration_balance_score'] = 0.5
            
            # Calculate overall RL score
            overall_score = (
                0.3 * metrics.get('avg_recent_reward', 0.5) +
                0.4 * metrics.get('learning_efficiency', 0.5) +
                0.3 * metrics.get('exploration_balance_score', 0.5)
            )
            
            metrics['overall_score'] = overall_score
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating RL agent performance: {e}")
            return {'overall_score': 0.3, 'error': str(e)}
    
    async def _evaluate_transfer_learning(self) -> Dict[str, Any]:
        """
        Evaluate transfer learning performance
        
        Returns:
            Dictionary of transfer learning performance metrics
        """
        if not (hasattr(self, 'rl_agent') and self.rl_agent and 
                hasattr(self.rl_agent, 'transfer_bridge') and self.rl_agent.transfer_bridge):
            return {'overall_score': 0.0, 'error': 'Transfer learning not available'}
            
        try:
            metrics = {}
            transfer_bridge = self.rl_agent.transfer_bridge
            
            # Count number of known domains
            if hasattr(transfer_bridge, 'domain_embeddings'):
                domain_count = len(transfer_bridge.domain_embeddings)
                metrics['domain_count'] = domain_count
            else:
                domain_count = 0
                metrics['domain_count'] = 0
            
            # Average similarity between domains
            if hasattr(transfer_bridge, 'domain_similarities') and transfer_bridge.domain_similarities:
                similarities = []
                for domain, sim_dict in transfer_bridge.domain_similarities.items():
                    similarities.extend(sim_dict.values())
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    metrics['average_similarity'] = avg_similarity
                    
                    # Calculate diversity score (lower average similarity means more diverse domains)
                    diversity_score = 1.0 - (avg_similarity - 0.3) / 0.7
                    diversity_score = max(0.0, min(1.0, diversity_score))
                    metrics['diversity_score'] = diversity_score
                else:
                    metrics['average_similarity'] = 0.0
                    metrics['diversity_score'] = 0.0
            else:
                metrics['average_similarity'] = 0.0
                metrics['diversity_score'] = 0.0
            
            # Transfer success rates if available
            if hasattr(transfer_bridge, 'transfer_success_rates'):
                metrics['transfer_success_rates'] = transfer_bridge.transfer_success_rates
                
                # Calculate average transfer success
                if transfer_bridge.transfer_success_rates:
                    avg_success = sum(transfer_bridge.transfer_success_rates.values()) / len(transfer_bridge.transfer_success_rates)
                    metrics['average_transfer_success'] = avg_success
                else:
                    metrics['average_transfer_success'] = 0.0
            else:
                metrics['average_transfer_success'] = 0.0
            
            # Calculate overall transfer learning score
            # Domain count factor (more domains is better, up to a point)
            domain_factor = min(1.0, domain_count / 10)  # Max out at 10 domains
            
            # Combine metrics
            if domain_count >= 2:  # Need at least 2 domains for meaningful transfer
                overall_score = (
                    0.3 * domain_factor +
                    0.3 * metrics.get('diversity_score', 0.5) +
                    0.4 * metrics.get('average_transfer_success', 0.5)
                )
            else:
                # Not enough domains for proper assessment
                overall_score = 0.3 * domain_factor
            
            metrics['overall_score'] = overall_score
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating transfer learning performance: {e}")
            return {'overall_score': 0.3, 'error': str(e)}
    
    def _detect_performance_patterns(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect patterns in performance across components
        
        Args:
            metrics: Performance metrics from all components
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {}
        
        # Need historical data for pattern detection
        if not hasattr(self, 'performance_history') or len(self.performance_history) < 5:
            return {'sufficient_data': False}
            
        try:
            # Extract recent history (last 10 entries)
            recent_history = self.performance_history[-10:]
            
            # Calculate trends for each component
            component_trends = {}
            
            for component in ['memory', 'planner', 'rl_agent', 'transfer_learning']:
                scores = [entry['component_scores'].get(component, 0.5) for entry in recent_history]
                
                if len(scores) >= 3:  # Need at least 3 points for trend
                    # Calculate slope using simple linear regression
                    x = list(range(len(scores)))
                    mean_x = sum(x) / len(x)
                    mean_y = sum(scores) / len(scores)
                    
                    numerator = sum((x[i] - mean_x) * (scores[i] - mean_y) for i in range(len(scores)))
                    denominator = sum((x[i] - mean_x) ** 2 for i in range(len(scores)))
                    
                    if denominator != 0:
                        slope = numerator / denominator
                    else:
                        slope = 0
                    
                    component_trends[component] = {
                        'slope': slope,
                        'direction': 'improving' if slope > 0.01 else ('declining' if slope < -0.01 else 'stable'),
                        'current': scores[-1],
                        'change': scores[-1] - scores[0] if len(scores) > 0 else 0
                    }
            
            patterns['component_trends'] = component_trends
            
            # Detect imbalanced components (one performing much worse than others)
            current_scores = {}
            for component in ['memory', 'planner', 'rl_agent', 'transfer_learning']:
                if component in metrics and 'overall_score' in metrics[component]:
                    current_scores[component] = metrics[component]['overall_score']
            
            if current_scores:
                avg_score = sum(current_scores.values()) / len(current_scores)
                lagging_components = [comp for comp, score in current_scores.items() 
                                    if score < avg_score - 0.15]
                
                if lagging_components:
                    patterns['imbalance'] = {
                        'lagging_components': lagging_components,
                        'avg_score': avg_score
                    }
            
            # Detect oscillating performance (consistent up-down patterns)
            if len(recent_history) >= 6:
                for component in ['memory', 'planner', 'rl_agent', 'transfer_learning']:
                    scores = [entry['component_scores'].get(component, 0.5) for entry in recent_history]
                    
                    # Check for alternating increases and decreases
                    increases = 0
                    decreases = 0
                    
                    for i in range(1, len(scores)):
                        if scores[i] > scores[i-1]:
                            increases += 1
                        elif scores[i] < scores[i-1]:
                            decreases += 1
                    
                    # If we have roughly equal increases and decreases, it might be oscillating
                    if min(increases, decreases) >= 2 and abs(increases - decreases) <= 1:
                        if 'oscillating' not in patterns:
                            patterns['oscillating'] = []
                        patterns['oscillating'].append(component)
            
            # Check for plateauing performance
            if len(recent_history) >= 5:
                for component in ['memory', 'planner', 'rl_agent', 'transfer_learning']:
                    scores = [entry['component_scores'].get(component, 0.5) for entry in recent_history]
                    
                    # Calculate variance in recent scores
                    mean = sum(scores) / len(scores)
                    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
                    
                    # Low variance and high score indicates plateau
                    if variance < 0.01 and mean > 0.7:
                        if 'plateauing' not in patterns:
                            patterns['plateauing'] = []
                        patterns['plateauing'].append(component)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting performance patterns: {e}")
            return {'error': str(e)}
    
    def _detect_concept_drift(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect concept drift across the system
        
        Args:
            metrics: Performance metrics from all components
            
        Returns:
            Dictionary of detected concept drift
        """
        drift_assessment = {'detected': False}
        
        try:
            # Check planner drift metrics (most direct measure of concept drift)
            if ('planner' in metrics and 'drift' in metrics['planner'] and 
                'severity' in metrics['planner']['drift']):
                severity = metrics['planner']['drift']['severity']
                drift_assessment['planner_drift_severity'] = severity
                
                if severity > 0.6:
                    drift_assessment['detected'] = True
                    drift_assessment['level'] = 'high'
                elif severity > 0.3:
                    drift_assessment['detected'] = True
                    drift_assessment['level'] = 'moderate'
                else:
                    drift_assessment['level'] = 'low'
            
            # Check performance trend reversals (another indicator of drift)
            if hasattr(self, 'performance_history') and len(self.performance_history) >= 10:
                recent_history = self.performance_history[-10:]
                overall_scores = [entry.get('overall_score', 0.5) for entry in recent_history]
                
                # Check for sudden drops after improvement
                was_improving = False
                sudden_drop = False
                
                for i in range(1, len(overall_scores)-1):
                    # Check for 3 consecutive improvements
                    if (i >= 3 and
                        overall_scores[i-2] < overall_scores[i-1] < overall_scores[i]):
                        was_improving = True
                    
                    # Then check for significant drop
                    if was_improving and i < len(overall_scores)-1:
                        if overall_scores[i+1] < overall_scores[i] - 0.1:
                            sudden_drop = True
                            break
                
                if sudden_drop:
                    drift_assessment['performance_reversal'] = True
                    drift_assessment['detected'] = True
                    
                    # If not already set by planner drift
                    if 'level' not in drift_assessment:
                        drift_assessment['level'] = 'moderate'
            
            # Check for domain-specific drift in RL agent
            if ('rl_agent' in metrics and 'learning_status' in metrics['rl_agent'] and
                'domain_performance' in metrics['rl_agent']['learning_status']):
                domain_perf = metrics['rl_agent']['learning_status']['domain_performance']
                
                # Identify domains with sudden performance drops
                drifting_domains = []
                
                for domain, perf in domain_perf.items():
                    if 'recent_trend' in perf and perf['recent_trend'] < -0.15:
                        drifting_domains.append(domain)
                
                if drifting_domains:
                    drift_assessment['drifting_domains'] = drifting_domains
                    drift_assessment['detected'] = True
            
            return drift_assessment
            
        except Exception as e:
            logger.error(f"Error detecting concept drift: {e}")
            return {'detected': False, 'error': str(e)}
    
    def _identify_improvement_opportunities(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify specific opportunities for system improvement
        
        Args:
            metrics: Performance metrics from all components
            
        Returns:
            List of improvement opportunities
        """
        opportunities = []
        
        try:
            # Check for memory optimization opportunities
            if 'memory' in metrics:
                memory_metrics = metrics['memory']
                
                # Check retrieval precision
                if ('retrieval_precision' in memory_metrics and 
                    memory_metrics['retrieval_precision'] < 0.7):
                    opportunities.append({
                        'component': 'memory',
                        'aspect': 'retrieval_precision',
                        'current_value': memory_metrics['retrieval_precision'],
                        'target_value': 0.8,
                        'priority': 'high' if memory_metrics['retrieval_precision'] < 0.5 else 'medium',
                        'suggested_action': 'Improve memory indexing and similarity threshold tuning'
                    })
                
                # Check graph connectivity
                if ('graph_metrics' in memory_metrics and 
                    'connectivity' in memory_metrics['graph_metrics'] and
                    memory_metrics['graph_metrics']['connectivity'] < 0.5):
                    opportunities.append({
                        'component': 'memory',
                        'aspect': 'graph_connectivity',
                        'current_value': memory_metrics['graph_metrics']['connectivity'],
                        'target_value': 0.6,
                        'priority': 'medium',
                        'suggested_action': 'Enhance concept linking in the Zettelkasten system'
                    })
            
            # Check for planner optimization opportunities
            if 'planner' in metrics:
                planner_metrics = metrics['planner']
                
                # Check success rate
                if ('success_rate' in planner_metrics and
                    planner_metrics['success_rate'] < 0.7):
                    opportunities.append({
                        'component': 'planner',
                        'aspect': 'success_rate',
                        'current_value': planner_metrics['success_rate'],
                        'target_value': 0.8,
                        'priority': 'high' if planner_metrics['success_rate'] < 0.5 else 'medium',
                        'suggested_action': 'Improve planning algorithms and beam search parameters'
                    })
                
                # Check drift handling
                if ('drift_score' in planner_metrics and
                    planner_metrics['drift_score'] < 0.6):
                    opportunities.append({
                        'component': 'planner',
                        'aspect': 'drift_handling',
                        'current_value': planner_metrics['drift_score'],
                        'target_value': 0.8,
                        'priority': 'high',
                        'suggested_action': 'Enhance concept drift detection and adaptive planning'
                    })
                
                # Check domain coverage
                if ('domain_diversity' in planner_metrics and
                    planner_metrics['domain_diversity'] < 0.5):
                    opportunities.append({
                        'component': 'planner',
                        'aspect': 'domain_diversity',
                        'current_value': planner_metrics['domain_diversity'],
                        'target_value': 0.7,
                        'priority': 'medium',
                        'suggested_action': 'Expand planning capabilities to more domains'
                    })
            
            # Check for RL agent optimization opportunities
            if 'rl_agent' in metrics:
                rl_metrics = metrics['rl_agent']
                
                # Check learning efficiency
                if ('learning_efficiency' in rl_metrics and
                    rl_metrics['learning_efficiency'] < 0.7):
                    opportunities.append({
                        'component': 'rl_agent',
                        'aspect': 'learning_efficiency',
                        'current_value': rl_metrics['learning_efficiency'],
                        'target_value': 0.8,
                        'priority': 'high' if rl_metrics['learning_efficiency'] < 0.5 else 'medium',
                        'suggested_action': 'Optimize experience replay and learning parameters'
                    })
                
                # Check exploration balance
                if ('exploration_balance_score' in rl_metrics and
                    rl_metrics['exploration_balance_score'] < 0.7):
                    opportunities.append({
                        'component': 'rl_agent',
                        'aspect': 'exploration_balance',
                        'current_value': rl_metrics['exploration_balance_score'],
                        'target_value': 0.8,
                        'priority': 'medium',
                        'suggested_action': 'Adjust exploration rate based on performance stability'
                    })
            
            # Check for transfer learning optimization opportunities
            if 'transfer_learning' in metrics:
                transfer_metrics = metrics['transfer_learning']
                
                # Check domain count
                if ('domain_count' in transfer_metrics and
                    transfer_metrics['domain_count'] < 5):
                    opportunities.append({
                        'component': 'transfer_learning',
                        'aspect': 'domain_coverage',
                        'current_value': transfer_metrics['domain_count'],
                        'target_value': 8,
                        'priority': 'low',
                        'suggested_action': 'Expand domain knowledge through more diverse training'
                    })
                
                # Check transfer success
                if ('average_transfer_success' in transfer_metrics and
                    transfer_metrics['average_transfer_success'] < 0.6):
                    opportunities.append({
                        'component': 'transfer_learning',
                        'aspect': 'transfer_success',
                        'current_value': transfer_metrics['average_transfer_success'],
                        'target_value': 0.7,
                        'priority': 'medium',
                        'suggested_action': 'Improve knowledge transfer mechanisms between domains'
                    })
            
            # Check for cross-component imbalances
            if ('component_trends' in metrics.get('performance_patterns', {}) and
                'imbalance' in metrics.get('performance_patterns', {})):
                
                imbalance = metrics['performance_patterns']['imbalance']
                for component in imbalance.get('lagging_components', []):
                    opportunities.append({
                        'component': 'system',
                        'aspect': 'component_balance',
                        'details': f"The {component} component is underperforming",
                        'priority': 'high',
                        'suggested_action': f"Focus on optimizing the {component} component"
                    })
            
            # Prioritize opportunities (sort by priority)
            priority_map = {'high': 0, 'medium': 1, 'low': 2}
            opportunities.sort(key=lambda x: priority_map.get(x['priority'], 3))
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying improvement opportunities: {e}")
            return [{
                'component': 'system',
                'aspect': 'error_handling',
                'priority': 'high',
                'suggested_action': 'Fix error in improvement identification'
            }]
    
    async def _optimize_for_performance(self):
        """
        Apply performance optimizations based on system evaluation.
        Implements a systematic approach to improve underperforming components.
        """
        try:
            # 1. Memory optimization: Rebuild indices for better retrieval
            if hasattr(self.memory, '_rebuild_index'):
                self.memory._rebuild_index()
                logger.info("Rebuilt memory indices for performance optimization")
            
            # 2. Planning optimization: Adjust beam width and depth
            if hasattr(self.planner, 'optimize_beam_parameters'):
                await self.planner.optimize_beam_parameters()
                
            # 3. Learning optimization: Balance exploration/exploitation
            if hasattr(self.rl_agent, 'adjust_exploration_rate'):
                self.rl_agent.adjust_exploration_rate(auto_tune=True)
                
            # 4. Chat optimization: Adjust context size dynamically
            await self._optimize_chat_parameters()
                
        except Exception as e:
            logger.error(f"Error during performance optimization: {e}", exc_info=True)
    
    async def _optimize_chat_parameters(self):
        """
        Optimize chat-related parameters based on conversation patterns.
        """
        if len(self.chat_history) < 5:
            return
            
        try:
            # Calculate conversational coherence
            coherence = self._calculate_conversation_coherence()
            
            # Calculate content complexity
            complexity = self._calculate_query_complexity()
            
            # Optimize context size based on these factors
            if coherence > 0.8 and complexity > 0.7:
                # High coherence, high complexity: maximize context
                if self.chat_context_turns < self.max_chat_history:
                    self.chat_context_turns = min(self.max_chat_history, self.chat_context_turns + 2)
                    logger.info(f"Increased chat context to {self.chat_context_turns} for complex coherent conversation")
            elif coherence < 0.4:
                # Low coherence: minimize context to focus on current topic
                if self.chat_context_turns > 2:
                    self.chat_context_turns = max(2, self.chat_context_turns - 1)
                    logger.info(f"Reduced chat context to {self.chat_context_turns} for low coherence conversation")
                    
        except Exception as e:
            logger.error(f"Error optimizing chat parameters: {e}", exc_info=True)
    
    async def _apply_fast_adaptation(self) -> None:
        """
        Apply fast adaptation cycle (30s) for quick adjustments to system parameters.
        
        This high-frequency adaptation handles short-term parameter tuning,
        episodic buffer processing, quick RL updates, and goal-directed adaptation.
        It allows the system to rapidly adjust to immediate changes in the environment
        while working toward internal improvement goals.
        """
        try:
            current_time = time.time()
            
            # 1. Performance evaluation and goal management - run every 5 cycles
            if not hasattr(self, '_fast_cycle_count'):
                self._fast_cycle_count = 0
            
            self._fast_cycle_count += 1
            
            # Every 5 fast cycles, perform system evaluation and goal updates
            if self._fast_cycle_count >= 5:
                self._fast_cycle_count = 0
                
                # Evaluate current system performance across components
                performance_metrics = await self._evaluate_system_performance()
                
                # Generate new goals if we don't have enough active ones
                if not hasattr(self, 'internal_goals'):
                    self.internal_goals = []
                
                active_goals = [g for g in self.internal_goals if g['status'] == 'active']
                if len(active_goals) < 3:
                    # Generate new internal improvement goals
                    new_goals = await self._generate_internal_goals(performance_metrics)
                    if new_goals:
                        logger.info(f"Generated {len(new_goals)} new internal improvement goals")
                
                # Update progress of existing goals
                await self._update_goal_progress(performance_metrics)
                
                # Generate adaptation actions based on active goals
                active_goals = [g for g in self.internal_goals if g['status'] == 'active']
                if active_goals:
                    adaptations = await self._adapt_for_goals(active_goals)
                    
                    # Apply the highest priority adaptation from each component
                    for component, actions in adaptations.items():
                        if component == 'memory' and actions and self.memory:
                            top_action = actions[0]
                            if top_action['type'] == 'parameter_adjustment':
                                param = top_action['parameter']
                                if hasattr(self.memory, param):
                                    current_value = getattr(self.memory, param)
                                    if top_action['direction'] == 'increase':
                                        new_value = current_value + top_action.get('amount', 0.05)
                                        setattr(self.memory, param, new_value)
                                        logger.info(f"Applied fast memory adaptation: {top_action['description']}")
                            
                        elif component == 'rl_agent' and actions and self.rl_agent:
                            top_action = actions[0]
                            if top_action['type'] == 'parameter_adjustment':
                                param = top_action['parameter']
                                if hasattr(self.rl_agent, param):
                                    current_value = getattr(self.rl_agent, param)
                                    if param == 'exploration_rate' or param == 'clip_range':
                                        if top_action['direction'] == 'increase':
                                            new_value = min(0.9, current_value + top_action.get('amount', 0.1))
                                        else:
                                            new_value = max(0.1, current_value - top_action.get('amount', 0.1))
                                        setattr(self.rl_agent, param, new_value)
                                        logger.info(f"Applied fast RL adaptation: {top_action['description']}")
                            
                            # Dynamically adjust RL reward weights based on active goals
                            if self.rl_agent and hasattr(self.rl_agent, 'reward_weights'):
                                default_weights = {
                                    'accuracy': 0.4,
                                    'efficiency': 0.2,
                                    'novelty': 0.2,
                                    'consistency': 0.2
                                }
                                
                                # Initialize with defaults if needed
                                if not hasattr(self.rl_agent, 'reward_weights'):
                                    self.rl_agent.reward_weights = default_weights.copy()
                                
                                # Adjust weights based on active goals
                                weights = self.rl_agent.reward_weights.copy()
                                adjusted = False
                                
                                for goal in active_goals:
                                    if goal['component'] == 'rl_agent':
                                        if goal['metric'] == 'learning_efficiency':
                                            # Increase efficiency weight
                                            weights['efficiency'] = min(0.5, weights['efficiency'] + 0.1)
                                            adjusted = True
                                            
                                        elif goal['metric'] == 'exploration_balance_score':
                                            # Increase novelty weight for exploration
                                            weights['novelty'] = min(0.5, weights['novelty'] + 0.1)
                                            adjusted = True
                                            
                                    elif goal['component'] == 'memory' and goal['metric'] == 'retrieval_precision':
                                        # Memory precision impacts accuracy
                                        weights['accuracy'] = min(0.6, weights['accuracy'] + 0.1)
                                        adjusted = True
                                        
                                    elif goal['component'] == 'planner':
                                        if goal['metric'] == 'drift_score':
                                            # Drift impacts consistency
                                            weights['consistency'] = min(0.5, weights['consistency'] + 0.1)
                                            adjusted = True
                                
                                # Normalize weights
                                if adjusted:
                                    total = sum(weights.values())
                                    normalized = {k: v/total for k, v in weights.items()}
                                    self.rl_agent.reward_weights = normalized
                                    logger.info(f"Adjusted RL reward weights based on goals: {normalized}")
            
            # 2. Apply quick RL updates if agent is available
            if self.rl_agent:
                # Apply small hyperparameter adjustments through meta-learning
                if hasattr(self.rl_agent, 'meta_controller'):
                    current_params = {
                        'learning_rate': self.rl_agent.learning_rate,
                        'gamma': self.rl_agent.gamma,
                        'clip_range': 0.2,  # Default PPO clip range
                        'exploration_factor': 0.1  # Exploration factor
                    }
                    
                    # Get active goals for goal-driven adaptation
                    active_goals = [g for g in self.internal_goals] if hasattr(self, 'internal_goals') else []
                    
                    # Get adjusted parameters with domain awareness and goal-driven adaptation
                    adjusted_params = await self.rl_agent.meta_controller.adjust_hyperparameters(
                        current_params, self.current_domain, active_goals
                    )
                    
                    # Apply minor adjustments only (conservative fast cycle)
                    for param, value in adjusted_params.items():
                        if param == 'learning_rate':
                            # Gradual learning rate adjustment (max 2% change)
                            current_lr = self.rl_agent.learning_rate
                            max_change = current_lr * 0.02
                            new_lr = current_lr + min(max_change, max(-max_change, value - current_lr))
                            self.rl_agent.learning_rate = new_lr
                            
                        elif param == 'exploration_factor' and hasattr(self.rl_agent, 'exploration_factor'):
                            # Quick exploration adjustment based on performance
                            self.rl_agent.exploration_factor = value
            
            # 3. Process episodic buffer for memory consolidation
            if self.memory and hasattr(self.memory, 'episodic_buffer'):
                # Process a limited number of items from the priority queue
                if hasattr(self.memory, '_process_priority_queue'):
                    await self.memory._process_priority_queue(max_transfers=1)
            
            # 4. Adjust search parameters based on recent query complexity
            complexity = self._calculate_query_complexity()
            if complexity > 0.7:  # High complexity queries
                # Increase search depth
                self.search_parameters['top_k'] = min(10, self.search_parameters['top_k'] + 1)
                # Decrease threshold for broader results
                self.search_parameters['threshold'] = max(0.6, self.search_parameters['threshold'] - 0.02)
            elif complexity < 0.3:  # Low complexity queries
                # Decrease search depth for efficiency
                self.search_parameters['top_k'] = max(3, self.search_parameters['top_k'] - 1)
                # Increase threshold for more precise results
                self.search_parameters['threshold'] = min(0.85, self.search_parameters['threshold'] + 0.02)
            
            # 5. Update chat context size based on recent interaction patterns
            if len(self.chat_history) >= 3:
                # Check if recent interactions are related
                if self._calculate_conversation_coherence() > 0.7:
                    # Conversations with high coherence benefit from more context
                    if self.chat_context_turns < self.max_chat_history:
                        self.chat_context_turns = min(self.max_chat_history, 
                                                     self.chat_context_turns + 1)
                else:
                    # Conversations with low coherence benefit from focused context
                    if self.chat_context_turns > 3:
                        self.chat_context_turns = max(3, self.chat_context_turns - 1)
                    
        except Exception as e:
            logger.error(f"Error in fast adaptation cycle: {e}", exc_info=True)
    
    def _calculate_conversation_coherence(self) -> float:
        """
        Calculate the semantic coherence of recent conversation turns.
        
        Returns:
            Coherence score from 0.0 (low coherence) to 1.0 (high coherence)
        """
        if len(self.chat_history) < 3:
            return 0.5  # Default medium coherence
        
        # Get the most recent messages
        recent_messages = self.chat_history[-3:]
        
        # Simple keyword overlap as a proxy for coherence
        all_text = " ".join([msg.get("content", "") for msg in recent_messages])
        words = all_text.lower().split()
        unique_words = set(words)
        
        # Calculate repetition ratio (higher repetition = higher coherence)
        if len(words) > 0:
            repetition_ratio = 1.0 - (len(unique_words) / len(words))
            
            # Adjust coherence based on repetition
            coherence = 0.3 + (repetition_ratio * 0.7)  # Scale to 0.3-1.0 range
            return min(1.0, max(0.0, coherence))
        
        return 0.5  # Default medium coherence
    
    async def _apply_slow_adaptation(self) -> None:
        """
        Apply slow adaptation cycle (24h) for major system-wide adjustments.
        
        This low-frequency adaptation handles structural changes, major memory
        reorganization, model parameter sweeps, and long-term learning. It allows 
        deeper architectural changes that would be too expensive to run frequently.
        Now integrates comprehensive goal-oriented evaluation and adaptation.
        """
        try:
            logger.info("Starting slow adaptation cycle")
            current_time = time.time()
            
            # 1. Comprehensive system performance evaluation
            performance_metrics = await self._evaluate_system_performance()
            logger.info(f"Slow adaptation performance score: {performance_metrics.get('overall_score', 0.0):.2f}")
            
            # 2. Goal management with higher priority on system-level goals
            # Initialize goals if needed
            if not hasattr(self, 'internal_goals'):
                self.internal_goals = []
                
            # Complete review of all goals (including stalled ones)
            # Archive completed and long-term stalled goals
            archived_count = 0
            for goal in list(self.internal_goals):
                if goal['status'] == 'completed':
                    # Archive completed goals after 7 days
                    if current_time - goal.get('completed_at', 0) > 7 * 24 * 3600:
                        goal['status'] = 'archived'
                        archived_count += 1
                elif goal['status'] == 'stalled':
                    # Archive stalled goals after 14 days
                    if current_time - goal.get('last_updated', 0) > 14 * 24 * 3600:
                        goal['status'] = 'archived'
                        archived_count += 1
            
            if archived_count > 0:
                logger.info(f"Archived {archived_count} completed/stalled goals")
            
            # Generate comprehensive set of new goals including system-level improvements
            # For slow cycle, focus on more ambitious, long-term goals
            system_goals = await self._generate_internal_goals(
                performance_metrics,
                focus='system',  # Focus on system-level goals
                min_gap=0.3      # Set more ambitious targets
            )
            
            if system_goals:
                logger.info(f"Generated {len(system_goals)} new system-level improvement goals")
            
            # Update progress for all active goals
            await self._update_goal_progress(performance_metrics)
            
            # 3. Comprehensive goal-driven adaptations across all components
            active_goals = [g for g in self.internal_goals if g['status'] == 'active']
            
            if active_goals:
                adaptations = await self._adapt_for_goals(active_goals)
                
                # Apply more aggressive adaptations across all components
                adaptation_count = 0
                
                for component, actions in adaptations.items():
                    # Apply multiple actions per component in slow cycle
                    for action in actions[:3]:  # Apply top 3 actions per component
                        applied = False
                        
                        # Memory component adaptations
                        if component == 'memory' and self.memory:
                            if action['type'] == 'parameter_adjustment':
                                param = action['parameter']
                                if hasattr(self.memory, param):
                                    current_value = getattr(self.memory, param)
                                    if action['direction'] == 'increase':
                                        new_value = current_value + action.get('amount', 0.1)
                                    else:
                                        new_value = max(0.1, current_value - action.get('amount', 0.1))
                                    setattr(self.memory, param, new_value)
                                    applied = True
                            elif action['type'] == 'operation' and action['operation'] == 'rebuild_index':
                                if hasattr(self.memory, '_rebuild_index'):
                                    self.memory._rebuild_index()
                                    applied = True
                            
                        # RL agent component adaptations        
                        elif component == 'rl_agent' and self.rl_agent:
                            if action['type'] == 'parameter_adjustment':
                                param = action['parameter']
                                if hasattr(self.rl_agent, param):
                                    if param == 'exploration_rate' or param == 'clip_range':
                                        current_value = getattr(self.rl_agent, param)
                                        if action['direction'] == 'increase':
                                            new_value = min(0.9, current_value + action.get('amount', 0.1))
                                        elif action['direction'] == 'decrease':
                                            new_value = max(0.1, current_value - action.get('amount', 0.1))
                                        else:  # optimize
                                            # Use meta-controller for optimization
                                            if hasattr(self.rl_agent, 'meta_controller'):
                                                new_value = self.rl_agent.meta_controller.optimize_parameter(
                                                    param, current_value, active_goals
                                                )
                                        setattr(self.rl_agent, param, new_value)
                                        applied = True
                                    elif param == 'batch_size' and hasattr(self.rl_agent, 'batch_size'):
                                        current_value = getattr(self.rl_agent, 'batch_size')
                                        if action['direction'] == 'increase':
                                            new_value = min(256, current_value + action.get('amount', 8))
                                        else:
                                            new_value = max(16, current_value - action.get('amount', 8))
                                        setattr(self.rl_agent, 'batch_size', new_value)
                                        applied = True
                        
                        # Transfer learning component adaptations
                        elif component == 'transfer_learning' and hasattr(self, 'rl_agent') and self.rl_agent:
                            if action['type'] == 'operation' and action['operation'] == 'expand_domains':
                                # Update domain similarity matrix
                                if hasattr(self.rl_agent, 'transfer_bridge'):
                                    all_domains = list(self.domain_history.keys())
                                    for i, domain1 in enumerate(all_domains):
                                        for domain2 in all_domains[i+1:]:
                                            self.rl_agent.transfer_bridge.update_domain_similarity(
                                                domain1, domain2, force_recalculate=True
                                            )
                                    applied = True
                                    
                        # System-wide adaptations
                        elif component == 'system':
                            if action['type'] == 'resource_allocation' and 'component' in action:
                                weak_comp = action['component']
                                # Implement resource allocation by adjusting update frequencies
                                if weak_comp == 'memory' and self.memory:
                                    # Increase memory operations frequency
                                    logger.info(f"Prioritizing memory component resources")
                                    applied = True
                                elif weak_comp == 'rl_agent' and self.rl_agent:
                                    # Dedicate more training cycles to RL
                                    if hasattr(self.rl_agent, 'training_frequency'):
                                        self.rl_agent.training_frequency *= 0.8  # Decrease interval (train more often)
                                    logger.info(f"Prioritizing RL agent resources")
                                    applied = True
                        
                        if applied:
                            logger.info(f"Applied slow adaptation: {action['description']}")
                            adaptation_count += 1
                
                logger.info(f"Applied {adaptation_count} goal-driven adaptations in slow cycle")
            
            # 4. Major structural memory pruning
            pruned_count = 0
            if hasattr(self.memory, '_prune_old_memories'):
                pruned_count = self.memory._prune_old_memories()
                logger.info(f"Deep maintenance: pruned {pruned_count} old memories")
            
            # 5. Comprehensive synaptic connection pruning
            if hasattr(self.memory, '_prune_synaptic_connections'):
                # More aggressive pruning in slow cycle
                pruned_connections = await self.memory._prune_synaptic_connections(prune_ratio=0.1)  # 10% prune ratio
                logger.info(f"Deep maintenance: pruned {pruned_connections} weak synaptic connections")
            
            # 6. Major RL model retraining with larger batch size
            if self.rl_agent and hasattr(self.rl_agent, 'learn_from_experiences'):
                if len(self.rl_agent.experiences) > 100:
                    # Full model retraining with larger batch
                    batch_size = min(128, len(self.rl_agent.experiences))
                    retraining_successful = self.rl_agent.learn_from_experiences(batch_size=batch_size)
                    if retraining_successful:
                        logger.info(f"Deep learning: retrained RL model with batch size {batch_size}")
                    
                    # Save RL model after retraining
                    if hasattr(self.rl_agent, 'save_model'):
                        self.rl_agent.save_model()
                        logger.info("Saved RL model after retraining")
            
            # 7. Analyze domain distribution and knowledge transfer
            if hasattr(self, 'domain_history') and self.domain_history:
                # Identify most frequently used domains
                sorted_domains = sorted(
                    self.domain_history.items(), 
                    key=lambda x: x[1].get('use_count', 0), 
                    reverse=True
                )
                
                if sorted_domains and self.rl_agent and hasattr(self.rl_agent, 'transfer_bridge'):
                    # Update cross-domain knowledge transfers
                    top_domain = sorted_domains[0][0]
                    for domain_name, stats in sorted_domains[1:5]:  # Transfer to next 4 domains
                        self.rl_agent.transfer_bridge.update_domain_similarity(
                            top_domain, domain_name, force_recalculate=True
                        )
                    logger.info(f"Updated cross-domain knowledge transfers from top domain: {top_domain}")
            
            # 8. Meta-parameter optimization (more aggressive than fast cycle)
            if self.rl_agent and hasattr(self.rl_agent, 'meta_controller'):
                # Get optimization summary to identify areas for improvement
                optimization_summary = self.rl_agent.meta_controller.get_optimization_summary()
                
                # Recalculate all feature weights based on success rates
                if hasattr(self.rl_agent.meta_controller, 'recalculate_feature_weights'):
                    self.rl_agent.meta_controller.recalculate_feature_weights(reset_history=True)
                    
                logger.info("Performed deep meta-parameter optimization")
            
            # 9. Rebuild memory index for better semantic search
            if hasattr(self.memory, '_rebuild_index'):
                self.memory._rebuild_index()
                logger.info("Rebuilt memory index for optimized retrieval")
            
            # 10. Save full system state
            await self._save_learning_state()
            logger.info("Saved complete learning state during slow adaptation cycle")
            
        except Exception as e:
            logger.error(f"Error in slow adaptation cycle: {e}", exc_info=True)
    
    def _calculate_correction_needed(self) -> bool:
        """
        Determine if correction is needed based on recent interactions
        
        Returns:
            Boolean indicating if correction is needed
        """
        # Placeholder implementation - determine based on history or patterns
        # In a real system, this would analyze recent interactions and outcomes
        if not self.chat_history:
            return False
            
        # Check for signs of confusion or correction in recent history
        correction_indicators = [
            "wrong", "incorrect", "mistake", "error", "not right",
            "didn't understand", "misunderstood", "please correct"
        ]
        
        # Check the most recent message
        if len(self.chat_history) > 0:
            last_msg = self.chat_history[-1].get('user_input', '').lower()
            return any(indicator in last_msg for indicator in correction_indicators)
            
        return False
    
    def _calculate_query_complexity(self) -> float:
        """
        Calculate complexity of recent queries on a scale of 0.0 to 1.0
        
        Returns:
            Complexity score from 0.0 (simple) to 1.0 (complex)
        """
        if not self.chat_history:
            return 0.5  # Default medium complexity
            
        # Get the most recent query
        last_query = self.chat_history[-1].get('user_input', '')
        if not last_query:
            return 0.5
            
        # Calculate based on length, question words, special characters
        length_factor = min(1.0, len(last_query) / 200)  # Length up to 200 chars
        
        # Check for question words, technical terms, logical operators
        question_words = ["why", "how", "explain", "compare", "analyze", "difference", "synthesize"]
        question_factor = 0.0
        for word in question_words:
            if word in last_query.lower():
                question_factor += 0.15
        question_factor = min(1.0, question_factor)
        
        # Check for code or structured content (e.g., JSON, XML)
        code_indicators = ["{", "}", "[", "]", "()", ";", "if ", "for ", "while ", "function", "class", "<>"]
        code_factor = 0.0
        for indicator in code_indicators:
            if indicator in last_query:
                code_factor += 0.1
        code_factor = min(1.0, code_factor)
        
        # Combine factors (weighted average)
        complexity = (0.3 * length_factor) + (0.4 * question_factor) + (0.3 * code_factor)
        return complexity
    
    def _calculate_response_length(self) -> float:
        """
        Calculate the normalized predicted response length required
        
        Returns:
            Normalized length score from 0.0 (short) to 1.0 (long)
        """
        if not self.chat_history:
            return 0.5  # Default to medium length
            
        # Calculate based on query length and complexity
        query_complexity = self._calculate_query_complexity()
        
        # Recent query length
        last_query = self.chat_history[-1].get('user_input', '')
        query_length = min(1.0, len(last_query) / 150)  # Normalize to 0-1
        
        # If previous response exists, use it as a hint too
        prev_response_factor = 0.5
        if len(self.chat_history) > 1 and 'response' in self.chat_history[-2]:
            prev_response = self.chat_history[-2]['response']
            prev_length = min(1.0, len(prev_response) / 500)  # Normalize to 0-1
            prev_response_factor = prev_length
        
        # Combine factors - complexity has the biggest influence
        return (0.6 * query_complexity) + (0.2 * query_length) + (0.2 * prev_response_factor)
    
    async def _apply_rl_action(self, action: str) -> bool:
        """
        Apply an action recommended by the RL agent
        
        Args:
            action: Action to apply
        
        Returns:
            True if action was successfully applied, False otherwise
        """
        if not action or not isinstance(action, str):
            return False
            
        try:
            # Common actions
            if action == "increase_context":
                success, _ = self.increase_chat_context()
                return success
                
            elif action == "reduce_context":
                success, _ = self.decrease_chat_context()
                return success
                
            elif action == "focus_search":
                # Reduce broader search by increasing similarity threshold
                self.search_parameters['threshold'] = min(0.95, self.search_parameters['threshold'] + 0.1)
                # Reduce number of results
                self.search_parameters['top_k'] = max(3, self.search_parameters['top_k'] - 1)
                return True
                
            elif action == "broaden_search":
                # Increase search breadth by decreasing similarity threshold
                self.search_parameters['threshold'] = max(0.5, self.search_parameters['threshold'] - 0.1)
                # Increase number of results
                self.search_parameters['top_k'] = min(10, self.search_parameters['top_k'] + 2)
                return True
                
            # Default action - no change
            elif action == "default":
                return True
                
            # Unknown action
            else:
                logger.warning(f"Unknown RL action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying RL action {action}: {e}")
            return False
    
    async def process_user_feedback(self, feedback_score: float, 
                                   response_time: Optional[float] = None,
                                   task_success: Optional[bool] = None,
                                   metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Process explicit user feedback and update RL agent and meta-learning
        
        Args:
            feedback_score: User feedback score (0.0 to 1.0)
            response_time: Optional time taken to generate response
            task_success: Optional boolean indicating if a task was successful
            metrics: Optional additional metrics to track
        """
        if not feedback_score or not isinstance(feedback_score, (int, float)):
            return
            
        try:
            # Normalize feedback score to 0.0-1.0 range
            normalized_feedback = max(0.0, min(1.0, float(feedback_score)))
            
            # Create metrics dictionary if not provided
            if metrics is None:
                metrics = {}
                
            # Add standard metrics
            if response_time is not None:
                # Convert response time to a score (faster is better)
                # Scale: < 1s is excellent (1.0), > 5s is poor (0.0)
                time_score = max(0.0, min(1.0, 1.0 - (response_time - 1.0) / 4.0))
                metrics['response_time'] = time_score
                
            # Add query complexity
            metrics['query_complexity'] = self._calculate_query_complexity()
            
            # Calculate overall reward
            if self.rl_agent:
                reward = self.rl_agent.calculate_reward(
                    user_feedback=normalized_feedback,
                    task_success=task_success,
                    response_time=response_time,
                    metrics=metrics
                )
                
                # Get most recent state and action
                if hasattr(self.rl_agent, 'state_history') and hasattr(self.rl_agent, 'action_history'):
                    if self.rl_agent.state_history and self.rl_agent.action_history:
                        last_state = self.rl_agent.state_history[-1]
                        last_action = list(self.rl_agent.action_history.keys())[-1]
                        
                        # Add experience with this reward
                        self.rl_agent.add_experience(
                            last_state, last_action, reward, last_state, 
                            done=False, success=task_success
                        )
                        
                        logger.info(f"Added experience with reward {reward:.2f} for action {last_action}")
                        
            # Update meta-learning controller
            if hasattr(self.rl_agent, 'meta_controller') and self.rl_agent.meta_controller:
                self.rl_agent.meta_controller.update_performance_metrics(
                    reward=normalized_feedback, 
                    action="chat_response" if self.chat_history else "default",
                    success=task_success if task_success is not None else (normalized_feedback > 0.7)
                )
                
        except Exception as e:
            logger.error(f"Error processing user feedback: {e}")
    
    async def _save_learning_state(self) -> bool:
        """
        Save all learning component states for persistence
        
        Returns:
            True if all components saved successfully, False otherwise
        """
        try:
            success = True
            
            # Save memory
            if hasattr(self, 'memory'):
                try:
                    # Save embeddings and cache
                    await self.memory.save_cache()
                    
                    # Save Hebbian connection weights if enabled
                    if hasattr(self.memory, 'connection_weights') and self.memory.connection_weights:
                        await self.memory.save_connection_weights()
                        
                    logger.info(f"Successfully saved memory with {len(self.memory.metadata)} items")
                except Exception as e:
                    logger.error(f"Error saving memory state: {e}")
                    success = False
            
            # Save RL agent
            if hasattr(self, 'rl_agent') and self.rl_agent:
                try:
                    self.rl_agent.save_model()
                    logger.info("Successfully saved RL agent model")
                except Exception as e:
                    logger.error(f"Error saving RL agent model: {e}")
                    success = False
            
            # Save orchestrator state (preferences, settings, etc.)
            try:
                # Create a dictionary of current settings
                state = {
                    'learning_cycles_completed': self.learning_cycles_completed,
                    'chat_context_turns': self.chat_context_turns,
                    'timestamp': time.time(),
                    'version': '1.0'  # Add version for future migrations
                }
                
                # Save to disk
                with open(f"orchestrator_state.json", 'w') as f:
                    json.dump(state, f, indent=2)
                    
                logger.info("Successfully saved orchestrator state")
            except Exception as e:
                logger.error(f"Error saving orchestrator state: {e}")
                success = False
            
            # Final message depends on overall success
            if success:
                logger.info("Learning state saved successfully")
            else:
                logger.warning("Learning state saved with some errors")
                
            return success
            
        except Exception as e:
            logger.error(f"Error in save_learning_state: {e}", exc_info=True)
            return False
    
    def get_success_rate(self) -> float:
        """
        Calculate success rate from recent interactions.
        This is a placeholder implementation - in a real system, you would
        use actual success metrics.
        
        Returns:
            Success rate as a float between 0.0 and 1.0
        """
        # Placeholder implementation - in a real system, use actual success metrics
        return 0.75  # Default to reasonably successful
        
    async def chat(self, user_input: str) -> str:
        """
        Process a user input and generate a response with contextual awareness
        
        Args:
            user_input: The user's input text
            
        Returns:
            The assistant's response
        """
        try:
            # Get the recent chat context based on configured turns
            context = self._get_chat_context()
            
            # Add current input to history immediately (we'll add the response later)
            self.chat_history.append({"role": "user", "content": user_input})
            
            # Trim history if it exceeds max size
            while len(self.chat_history) > self.max_chat_history:
                self.chat_history.pop(0)
            
            # Analyze user input and history for personalization
            preferences = await self._analyze_user_preferences()
            
            # Retrieve both semantic memories and historically relevant memories
            relevant_memories = await self.memory.semantic_search(
                user_input, 
                top_k=3,
                min_score=0.7
            )
            
            # Get additional memories based on conversation history
            if len(self.chat_history) > 2:
                # Create a condensed summary of the recent conversation
                conversation_summary = " ".join([msg["content"] for msg in self.chat_history[-3:]])
                
                # Search for memories relevant to the ongoing conversation
                historical_memories = await self.memory.semantic_search(
                    conversation_summary,
                    top_k=2,
                    min_score=0.65
                )
                
                # Merge unique memories from both searches
                existing_ids = {m["id"] for m in relevant_memories}
                for memory in historical_memories:
                    if memory["id"] not in existing_ids:
                        relevant_memories.append(memory)
                        existing_ids.add(memory["id"])
            
            # Format memories as context if any were found
            memory_context = ""
            if relevant_memories:
                memory_context = "Relevant information:\n"
                for memory in relevant_memories:
                    memory_context += f"- {memory['text']}\n"
            
            # Apply personalization based on preferences
            personalization_context = ""
            if preferences:
                personalization_context = "User preferences:\n"
                for pref_type, pref_value in preferences.items():
                    personalization_context += f"- {pref_type}: {pref_value}\n"
            
            # Generate the response using the alignment module
            # We combine: previous chat context + memory context + personalization + current input
            full_context = f"{context}\n{memory_context}\n{personalization_context}\nUser: {user_input}\nAssistant:"
            
            response = await self.memory.alignment.generate_response(full_context)
            
            # Add the assistant's response to the chat history
            self.chat_history.append({"role": "assistant", "content": response})
            
            # Update memory with this interaction
            interaction = f"User: {user_input}\nAssistant: {response}"
            await self.memory.add_to_episodic_buffer(user_input, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def _get_chat_context(self) -> str:
        """
        Retrieve and format the recent chat context based on configured turns
        
        Returns:
            Formatted chat context string
        """
        if not self.chat_history or self.chat_context_turns <= 0:
            return ""
        
        # Get only the most recent turns based on configuration
        recent_history = self.chat_history[-min(self.chat_context_turns*2, len(self.chat_history)):]
        
        # Format the context as a conversation
        context = ""
        for message in recent_history:
            role = message["role"].capitalize()  # User or Assistant
            content = message["content"]
            context += f"{role}: {content}\n"
        
        return context.strip()
        
    async def detect_domain(self, text: str) -> Optional[str]:
        """
        Detect the domain of a text using the transfer learning bridge
        
        Args:
            text: The text to analyze
            
        Returns:
            Detected domain name or None if no match
        """
        if not text or not hasattr(self, 'rl_agent') or not self.rl_agent:
            return None
            
        # Use transfer bridge if available
        if (hasattr(self.rl_agent, 'transfer_bridge') and 
            self.rl_agent.transfer_bridge and 
            hasattr(self.rl_agent.transfer_bridge, 'find_similar_domains')):
            
            # Generate an embedding for the text and find similar domains
            try:
                similar_domains = self.rl_agent.transfer_bridge.find_similar_domains(
                    domain_descriptor=text
                )
                
                # If we found similar domains with good confidence
                if similar_domains:
                    best_match, similarity = similar_domains[0]
                    
                    # Only use if similarity is high enough
                    if similarity > 0.75:
                        logger.info(f"Detected domain '{best_match}' with similarity {similarity:.2f}")
                        return best_match
                        
            except Exception as e:
                logger.error(f"Error detecting domain: {e}")
                
        return None
        
    async def set_task_domain(self, domain_name: str, domain_description: str) -> bool:
        """
        Set the current task domain and register it with the transfer learning bridge
        
        Args:
            domain_name: Short name for the domain
            domain_description: Longer description of the domain
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update current domain
            self.current_domain = domain_name
            
            # Register with transfer bridge if available
            if (hasattr(self, 'rl_agent') and 
                self.rl_agent and 
                hasattr(self.rl_agent, 'transfer_bridge')):
                
                # Register with RL agent
                success = self.rl_agent.set_domain(domain_name, domain_description)
                
                if success:
                    logger.info(f"Set current domain to '{domain_name}'")
                    
                    # Add to domain history
                    self.domain_history[domain_name] = {
                        'description': domain_description,
                        'last_used': time.time(),
                        'use_count': self.domain_history.get(domain_name, {}).get('use_count', 0) + 1
                    }
                    
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error setting task domain: {e}")
            return False
    
    async def cleanup(self) -> None:
        """
        Clean up resources and stop background tasks.
        Ensures a proper shutdown sequence with state persistence.
        """
        logger.info("Cleaning up orchestrator resources")
        
        # First, save the current learning state
        await self._save_learning_state()
        
        # Stop the learning loop gracefully
        await self.stop_learning_loop()
        
        # Clean up memory resources
        if hasattr(self, 'memory') and self.memory:
            # Additional cleanup for Hebbian learning connections
            if hasattr(self.memory, 'connection_weights') and self.memory.connection_weights:
                await self.memory.save_connection_weights()
                logger.info(f"Saved {len(self.memory.connection_weights)} Hebbian connection weights")
                
            # General memory cleanup
            if hasattr(self.memory, 'cleanup'):
                self.memory.cleanup()
        
        # Clean up RL agent
        if hasattr(self, 'rl_agent') and self.rl_agent:
            if hasattr(self.rl_agent, 'cleanup'):
                self.rl_agent.cleanup()
        
        logger.info("Orchestrator cleanup complete")
        
    def start_learning_loop(self) -> None:
        """
        Explicitly start the recursive learning loop.
        Safe to call multiple times (will only start if not already running).
        """
        if self.enable_recursive_learning and (self.learning_task is None or self.learning_task.done()):
            logger.info("Starting recursive learning loop task")
            self.learning_task = asyncio.create_task(self._recursive_learning_loop())
            
    async def stop_learning_loop(self) -> None:
        """
        Stop the recursive learning loop.
        Safe to call multiple times.
        """
        if self.learning_task and not self.learning_task.done():
            logger.info("Stopping recursive learning loop")
            self.learning_task.cancel()
            
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
                
            logger.info("Recursive learning loop stopped")
    
    def increase_chat_context(self, amount: int = 1) -> Tuple[bool, str]:
        """
        Increase the number of chat turns included in context.
        
        Args:
            amount: Number of turns to increase by
            
        Returns:
            Tuple of (success, message)
        """
        if amount <= 0:
            return False, "Increase amount must be positive"
            
        new_value = self.chat_context_turns + amount
        
        # Cap at a reasonable limit
        if new_value > self.max_chat_history:
            return False, f"Cannot increase beyond max chat history ({self.max_chat_history})"
            
        self.chat_context_turns = new_value
        return True, f"Chat context increased to {self.chat_context_turns} turns"
    
    def decrease_chat_context(self, amount: int = 1) -> Tuple[bool, str]:
        """
        Decrease the number of chat turns included in context.
        
        Args:
            amount: Number of turns to decrease by
            
        Returns:
            Tuple of (success, message)
        """
        if amount <= 0:
            return False, "Decrease amount must be positive"
            
        new_value = self.chat_context_turns - amount
        self.chat_context_turns = max(1, new_value)  # Minimum of 1 turn
        
        return True, f"Chat context decreased to {self.chat_context_turns} turns"

    async def _analyze_user_preferences(self) -> Dict[str, Any]:
        """
        Analyze user input and history for personalization
        
        Returns:
            Dictionary of personalization preferences
        """
        preferences = {}
        
        # Skip if not enough history
        if len(self.chat_history) < 3:
            return preferences
            
        try:
            # Analyze response length preferences
            response_lengths = []
            for msg in self.chat_history:
                if msg["role"] == "assistant":
                    content = msg["content"]
                    response_lengths.append(len(content.split()))
                    
            if response_lengths:
                avg_length = sum(response_lengths) / len(response_lengths)
                if avg_length < 30:
                    preferences["response_length"] = "concise"
                elif avg_length > 100:
                    preferences["response_length"] = "detailed"
                else:
                    preferences["response_length"] = "moderate"
            
            # Analyze technical level based on vocabulary and questions
            technical_indicators = 0
            total_messages = 0
            
            technical_terms = [
                "algorithm", "function", "code", "programming", "technical",
                "implementation", "architecture", "database", "system",
                "protocol", "api", "interface", "framework", "library"
            ]
            
            for msg in self.chat_history:
                if msg["role"] == "user":
                    content = msg["content"].lower()
                    total_messages += 1
                    
                    # Check for technical terms
                    for term in technical_terms:
                        if term in content:
                            technical_indicators += 1
                            
                    # Check for code or code-like content
                    if "```" in content or "{" in content or "(" in content or ";" in content:
                        technical_indicators += 2
                        
            if total_messages > 0:
                technical_ratio = technical_indicators / total_messages
                if technical_ratio > 1.5:
                    preferences["technical_level"] = "high"
                elif technical_ratio > 0.5:
                    preferences["technical_level"] = "moderate"
                else:
                    preferences["technical_level"] = "low"
            
            # Analyze formality preference
            formality_indicators = 0
            
            formal_markers = ["please", "thank you", "I would like", "could you", "would you"]
            informal_markers = ["hey", "yo", "sup", "cool", "awesome", "thanks"]
            
            for msg in self.chat_history:
                if msg["role"] == "user":
                    content = msg["content"].lower()
                    
                    # Check for formal markers
                    for marker in formal_markers:
                        if marker in content:
                            formality_indicators += 1
                    
                    # Check for informal markers
                    for marker in informal_markers:
                        if marker in content:
                            formality_indicators -= 1
            
            if total_messages > 0:
                if formality_indicators > 1:
                    preferences["formality"] = "formal"
                elif formality_indicators < -1:
                    preferences["formality"] = "casual"
                else:
                    preferences["formality"] = "neutral"
                    
            return preferences
            
        except Exception as e:
            logger.error(f"Error analyzing user preferences: {e}")
            return {}

    async def _fast_cycle(self):
        """
        Execute the fast learning cycle (high frequency updates)
        Runs in parallel for non-blocking operations
        """
        try:
            # Process multiple tasks concurrently
            await asyncio.gather(
                self._process_episodic_buffer(),
                self._apply_minor_rl_adjustments(),
                self._update_hebbian_connections(),
                self._refresh_working_memory()
            )
            
            logger.debug("Completed fast cycle parallel operations")
            return True
        except Exception as e:
            logger.error(f"Error in fast cycle: {e}")
            return False
            
    async def _process_episodic_buffer(self):
        """Process recent interactions in the episodic buffer"""
        if not hasattr(self, 'memory') or self.memory is None:
            return
        
        try:
            # Get buffer contents
            buffer_contents = await self.memory.get_episodic_buffer_contents()
            if not buffer_contents:
                return
                
            # Process buffer contents for insights
            # This is a simplified version - in a full implementation
            # this would analyze patterns, extract concepts, etc.
            for item in buffer_contents:
                # Extract user input and response
                user_input = item.get('user_input', '')
                response = item.get('response', '')
                
                if not user_input or not response:
                    continue
                    
                # Calculate importance score (simplified)
                importance = min(1.0, len(user_input) / 1000 * 0.5 + len(response) / 1000 * 0.5)
                
                # Add important interactions to long-term memory
                if importance > 0.3:  # Threshold
                    await self.memory.add_to_priority_queue(item, importance)
        except Exception as e:
            logger.error(f"Error processing episodic buffer: {e}")
    
    async def _apply_minor_rl_adjustments(self):
        """Apply minor reinforcement learning adjustments"""
        if not hasattr(self, 'rl_agent') or self.rl_agent is None:
            return
            
        try:
            # Apply small batch updates to RL model
            max_samples = 5  # Limit samples for fast cycle
            await self.rl_agent.update_model(max_samples=max_samples, mini_batch=True)
        except Exception as e:
            logger.error(f"Error applying minor RL adjustments: {e}")
    
    async def _update_hebbian_connections(self):
        """Update neural connection strengths based on recent activity"""
        if not hasattr(self, 'memory') or self.memory is None:
            return
            
        try:
            # Update connection strengths for recently used concepts
            # This is a simplified version that would strengthen 
            # connections between related concepts
            await self.memory.update_connection_strengths()
        except Exception as e:
            logger.error(f"Error updating Hebbian connections: {e}")
    
    async def _refresh_working_memory(self):
        """Refresh working memory with relevant context"""
        # Simplified implementation
        pass
            
    async def _medium_cycle(self):
        """
        Execute the medium learning cycle (medium frequency updates)
        Combines serial and parallel operations
        """
        try:
            # First run tasks that can be executed in parallel
            parallel_tasks = await asyncio.gather(
                self._consolidate_memory(),
                self._update_world_model(),
                self._optimize_planning_strategies()
            )
            
            # Then run tasks that need to be sequential
            await self._adjust_hyperparameters()
            
            logger.debug("Completed medium cycle operations")
            return all(parallel_tasks)
        except Exception as e:
            logger.error(f"Error in medium cycle: {e}")
            return False
    
    async def _consolidate_memory(self):
        """Consolidate memory by organizing and pruning"""
        if not hasattr(self, 'memory') or self.memory is None:
            return True
            
        try:
            # Process priority queue
            await self.memory.process_priority_queue()
            
            # Prune redundant memories
            pruned = await self.memory.prune_redundant()
            
            # Rebuild index if needed
            if pruned > 0 and pruned % 10 == 0:  # Every 10 prunes
                await self.memory.rebuild_index()
                
            return True
        except Exception as e:
            logger.error(f"Error consolidating memory: {e}")
            return False
    
    async def _update_world_model(self):
        """Update the world model based on recent experiences"""
        # Simplified implementation
        return True
    
    async def _optimize_planning_strategies(self):
        """Optimize planning strategies based on success rates"""
        if not hasattr(self, 'planner') or self.planner is None:
            return True
            
        try:
            # Update planner strategies
            await self.planner.optimize_strategies()
            return True
        except Exception as e:
            logger.error(f"Error optimizing planning strategies: {e}")
            return False
            
    async def _slow_cycle(self):
        """
        Execute the slow learning cycle (low frequency, compute-intensive updates)
        Uses ProcessPoolExecutor for CPU-intensive tasks
        """
        try:
            # Determine number of workers based on system load
            workers = self._calculate_optimal_workers()
            
            # Create a process pool for CPU-intensive tasks
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                # Start tasks
                future_full_rl = executor.submit(self._run_full_rl_training)
                future_rebuild_index = executor.submit(self._run_rebuild_indices)
                future_analyze_patterns = executor.submit(self._run_pattern_analysis)
                
                # Wait for all tasks to complete
                results = []
                for future in concurrent.futures.as_completed([future_full_rl, future_rebuild_index, future_analyze_patterns]):
                    results.append(future.result())
            
            # Run any remaining tasks that need to be in the main process
            await self._deep_meta_learning()
            
            logger.info("Completed slow cycle operations")
            return all(results)
        except Exception as e:
            logger.error(f"Error in slow cycle: {e}")
            return False
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker processes based on system load"""
        import os
        import psutil
        
        try:
            # Get CPU count
            cpu_count = os.cpu_count() or 4
            
            # Get current system load
            system_load = psutil.cpu_percent(interval=0.1) / 100.0
            
            # Calculate optimal workers
            # - More workers when system load is low
            # - Fewer workers when system load is high
            # - Never use less than 1 or more than cpu_count - 1
            if system_load < 0.3:  # Low load
                workers = max(1, cpu_count - 1)
            elif system_load < 0.7:  # Medium load
                workers = max(1, cpu_count // 2)
            else:  # High load
                workers = max(1, cpu_count // 4)
                
            return workers
        except Exception as e:
            logger.warning(f"Error calculating optimal workers: {e}")
            return 2  # Default to 2 workers
            
    def _run_full_rl_training(self):
        """Run full reinforcement learning training (CPU intensive)"""
        try:
            if not hasattr(self, 'rl_agent') or self.rl_agent is None:
                return True
                
            # Execute RL training
            # Note: In a real implementation, this would call the actual
            # training methods on the RL agent
            
            # Simulate training time
            time.sleep(1)
            
            return True
        except Exception as e:
            logger.error(f"Error in full RL training: {e}")
            return False
    
    def _run_rebuild_indices(self):
        """Rebuild memory indices (CPU intensive)"""
        try:
            if not hasattr(self, 'memory') or self.memory is None:
                return True
                
            # Execute index rebuilding
            # Note: In a real implementation, this would call the actual
            # index rebuilding methods
            
            # Simulate rebuilding time
            time.sleep(0.5)
            
            return True
        except Exception as e:
            logger.error(f"Error rebuilding indices: {e}")
            return False
    
    def _run_pattern_analysis(self):
        """Run pattern analysis across memories (CPU intensive)"""
        try:
            # Execute pattern analysis
            # In a real implementation, this would analyze patterns
            # across stored memories
            
            # Simulate analysis time
            time.sleep(0.7)
            
            return True
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return False
    
    async def _deep_meta_learning(self):
        """Perform deep meta-learning adjustments"""
        if not hasattr(self, 'meta_controller') or self.meta_controller is None:
            return
            
        try:
            # Get components for optimization
            components = {
                'memory': self.memory if hasattr(self, 'memory') else None,
                'planner': self.planner if hasattr(self, 'planner') else None,
                'rl_agent': self.rl_agent if hasattr(self, 'rl_agent') else None
            }
            
            # Skip components that don't exist
            components = {name: comp for name, comp in components.items() if comp is not None}
            
            if not components:
                return
                
            # Define parameter ranges for each component
            param_ranges = {
                'memory': {
                    'importance_threshold': (0.5, 0.9),
                    'hebbian_learning_rate': (0.01, 0.1)
                },
                'planner': {
                    'beam_width': (2, 5),
                    'exploration_factor': (0.1, 0.5)
                },
                'rl_agent': {
                    'learning_rate': (0.0001, 0.01),
                    'gamma': (0.9, 0.999),
                    'entropy_coef': (0.01, 0.2)
                }
            }
            
            # Only include existing components
            param_ranges = {name: ranges for name, ranges in param_ranges.items() if name in components}
            
            # Perform cross-component optimization
            optimized_params = await self.meta_controller.cross_component_optimization(
                components, param_ranges
            )
            
            # Apply optimized parameters
            for comp_name, params in optimized_params.items():
                component = components.get(comp_name)
                if component and hasattr(component, 'update_parameters'):
                    component.update_parameters(params)
        except Exception as e:
            logger.error(f"Error in deep meta-learning: {e}")
            
    async def run_recursive_learning_loop(self):
        """
        Run the main recursive learning loop with multiple timescales and robust error recovery.
        
        Implements a biologically-inspired multi-scale temporal processing system:
        - Fast cycle: Basic perception-action loop (5s, like theta rhythms)
        - Medium cycle: Working memory update and hypothesis testing (60s)
        - Slow cycle: Long-term conceptual model updating (300s, like sleep stages)
        """
        try:
            # Health metrics tracking
            cycle_metrics = {
                'fast': {'success': 0, 'failure': 0, 'timeouts': 0, 'durations': []},
                'medium': {'success': 0, 'failure': 0, 'timeouts': 0, 'durations': []},
                'slow': {'success': 0, 'failure': 0, 'timeouts': 0, 'durations': []}
            }
            
            # Track cycle timing
            last_medium_cycle = time.time()
            last_slow_cycle = time.time()
            last_health_report = time.time()
            
            # Cycle intervals (seconds)
            medium_cycle_interval = 60  # 1 minute
            slow_cycle_interval = 300   # 5 minutes
            health_report_interval = 600  # 10 minutes
            
            # System health state
            system_health = "NORMAL"  # NORMAL, DEGRADED, RECOVERY
            consecutive_failures = 0
            
            # Start the main loop
            logger.info("Starting enhanced recursive learning loop with recovery capabilities")
            
            while True:
                current_time = time.time()
                
                # --- Fast Cycle (Every 5 seconds) ---
                try:
                    start_time = time.time()
                    # Execute with timeout
                    await asyncio.wait_for(self._fast_cycle(), timeout=4.5)
                    cycle_metrics['fast']['success'] += 1
                    cycle_metrics['fast']['durations'].append(time.time() - start_time)
                    # Reset consecutive failures counter on success
                    consecutive_failures = max(0, consecutive_failures - 1)
                    
                except asyncio.TimeoutError:
                    cycle_metrics['fast']['timeouts'] += 1
                    logger.warning("Fast cycle timed out, initiating component reset")
                    await self._reset_cycle_components("fast")
                    consecutive_failures += 1
                    
                except Exception as e:
                    cycle_metrics['fast']['failure'] += 1
                    logger.error(f"Error in fast cycle: {e}")
                    consecutive_failures += 1
                
                # --- Medium Cycle (Every 60 seconds) ---
                if current_time - last_medium_cycle >= medium_cycle_interval:
                    try:
                        start_time = time.time()
                        # Execute with timeout
                        await asyncio.wait_for(self._medium_cycle(), timeout=55)
                        cycle_metrics['medium']['success'] += 1
                        cycle_metrics['medium']['durations'].append(time.time() - start_time)
                        last_medium_cycle = current_time
                    
                    except asyncio.TimeoutError:
                        cycle_metrics['medium']['timeouts'] += 1
                        logger.warning("Medium cycle timed out, partial results will be used")
                        await self._reset_cycle_components("medium")
                        last_medium_cycle = current_time
                        consecutive_failures += 1
                        
                    except Exception as e:
                        cycle_metrics['medium']['failure'] += 1
                        logger.error(f"Error in medium cycle: {e}")
                        last_medium_cycle = current_time
                        consecutive_failures += 1
                
                # --- Slow Cycle (Every 300 seconds) ---
                if current_time - last_slow_cycle >= slow_cycle_interval:
                    try:
                        start_time = time.time()
                        # Execute with timeout
                        await asyncio.wait_for(self._slow_cycle(), timeout=290)
                        cycle_metrics['slow']['success'] += 1
                        cycle_metrics['slow']['durations'].append(time.time() - start_time)
                        last_slow_cycle = current_time
                    
                    except asyncio.TimeoutError:
                        cycle_metrics['slow']['timeouts'] += 1
                        logger.warning("Slow cycle timed out, will retry next interval")
                        await self._reset_cycle_components("slow")
                        last_slow_cycle = current_time
                        consecutive_failures += 1
                        
                    except Exception as e:
                        cycle_metrics['slow']['failure'] += 1
                        logger.error(f"Error in slow cycle: {e}")
                        last_slow_cycle = current_time
                        consecutive_failures += 1
                
                # --- Health monitoring and recovery ---
                if consecutive_failures >= 5:
                    if system_health == "NORMAL":
                        logger.warning("System entering DEGRADED state due to consecutive failures")
                        system_health = "DEGRADED"
                        await self._apply_conservative_settings()
                        
                if consecutive_failures >= 10:
                    if system_health != "RECOVERY":
                        logger.critical("System entering RECOVERY state")
                        system_health = "RECOVERY"
                        await self._emergency_recovery_procedure()
                        consecutive_failures = 5  # Reset but keep in degraded state
                
                # Return to normal if healthy
                if system_health != "NORMAL" and consecutive_failures == 0:
                    logger.info("System health restored to NORMAL")
                    system_health = "NORMAL"
                    await self._restore_optimal_settings()
                
                # --- Health reporting ---
                if current_time - last_health_report >= health_report_interval:
                    self._report_system_health(cycle_metrics)
                    last_health_report = current_time
                    # Trim metrics history to avoid memory bloat
                    self._trim_metrics_history(cycle_metrics)
                
                # --- Pause before next iteration ---
                # Dynamic sleep based on system health
                sleep_time = 5  # Default fast cycle interval
                if system_health == "DEGRADED":
                    sleep_time = 8  # Slower cycling when degraded
                elif system_health == "RECOVERY":
                    sleep_time = 12  # Much slower cycling during recovery
                
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Recursive learning loop cancelled, performing clean shutdown")
            await self._clean_shutdown()
        except Exception as e:
            logger.critical(f"Critical error in recursive learning loop: {e}")
            # Try to save state even on critical errors
            await self._emergency_state_preservation()
            raise  # Re-raise to allow global error handler to act

    async def _reset_cycle_components(self, cycle_type: str) -> None:
        """Reset components that might be in an inconsistent state after a cycle timeout."""
        try:
            logger.info(f"Resetting {cycle_type} cycle components")
            
            if cycle_type == "fast":
                # Reset perception and attention components
                if hasattr(self, 'active_perception'):
                    self.active_perception.reset_state()
                if hasattr(self, 'attention_mechanism'):
                    self.attention_mechanism.clear_focus()
                    
            elif cycle_type == "medium":
                # Reset planning and memory working state
                if hasattr(self, 'working_memory'):
                    self.working_memory.clear_transient_state()
                if hasattr(self, 'planner'):
                    self.planner.cancel_current_planning()
                    
            elif cycle_type == "slow":
                # Reset meta-learning components
                if hasattr(self, 'meta_learner'):
                    self.meta_learner.reset_optimization_state()
                
                # Force garbage collection on slow cycle resets
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Error resetting {cycle_type} cycle components: {e}")

    async def _apply_conservative_settings(self) -> None:
        """Apply more conservative settings when system is degraded."""
        try:
            logger.info("Applying conservative settings due to system degradation")
            
            # Reduce batch sizes and learning rates
            if hasattr(self, 'learning_parameters'):
                for param_name, value in self.learning_parameters.items():
                    if 'rate' in param_name:
                        self.learning_parameters[param_name] = value * 0.5
                    if 'batch' in param_name:
                        self.learning_parameters[param_name] = max(4, int(value * 0.7))
            
            # Reduce prediction horizon
            if hasattr(self, 'prediction_horizon'):
                self.prediction_horizon = max(1, int(self.prediction_horizon * 0.5))
                
            # Save current state in case we need rollback
            await self._save_degraded_state_checkpoint()
            
        except Exception as e:
            logger.error(f"Failed to apply conservative settings: {e}")

    async def _restore_optimal_settings(self) -> None:
        """Restore optimal settings when system health returns to normal."""
        try:
            logger.info("Restoring optimal system settings")
            
            # Reload original parameter values
            if hasattr(self, 'default_learning_parameters'):
                self.learning_parameters = self.default_learning_parameters.copy()
                
            # Restore prediction horizon
            if hasattr(self, 'default_prediction_horizon'):
                self.prediction_horizon = self.default_prediction_horizon
                
            # Delete any emergency state checkpoints
            if os.path.exists('degraded_state_checkpoint.pkl'):
                os.remove('degraded_state_checkpoint.pkl')
                logger.info("Removed degraded state checkpoint")
                
        except Exception as e:
            logger.error(f"Failed to restore optimal settings: {e}")

    async def _emergency_recovery_procedure(self) -> None:
        """Execute emergency recovery for critical system issues."""
        try:
            logger.critical("Initiating emergency recovery procedure")
            
            # Save current state before attempting recovery
            await self._save_learning_state(emergency=True)
            
            # Check if we have a stable checkpoint to restore from
            if os.path.exists(self.stable_checkpoint_path):
                try:
                    await self._load_stable_checkpoint()
                    logger.info("Restored from stable checkpoint")
                    return
                except Exception as e:
                    logger.error(f"Failed to load stable checkpoint: {e}")
            
            # If no checkpoint or loading failed, perform component resets
            for component_name in ['memory', 'agent', 'planner', 'perception']:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    if hasattr(component, 'emergency_reset'):
                        try:
                            await component.emergency_reset()
                            logger.info(f"Reset {component_name} component")
                        except Exception as e:
                            logger.error(f"Failed to reset {component_name}: {e}")
            
            # Reset all active processes
            self._terminate_hanging_processes()
            
            # Clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Emergency recovery completed")
            
        except Exception as e:
            logger.critical(f"Critical failure in emergency recovery: {e}")

    def _report_system_health(self, metrics: dict) -> None:
        """Generate health report with performance metrics."""
        try:
            # Calculate success rates and averages
            health_summary = {}
            
            for cycle in ['fast', 'medium', 'slow']:
                total = metrics[cycle]['success'] + metrics[cycle]['failure'] + metrics[cycle]['timeouts']
                if total > 0:
                    success_rate = (metrics[cycle]['success'] / total) * 100
                else:
                    success_rate = 100.0
                
                avg_duration = 0
                if metrics[cycle]['durations']:
                    avg_duration = sum(metrics[cycle]['durations']) / len(metrics[cycle]['durations'])
                
                health_summary[cycle] = {
                    'success_rate': success_rate,
                    'avg_duration': avg_duration,
                    'timeouts': metrics[cycle]['timeouts'],
                    'failures': metrics[cycle]['failure']
                }
            
            # Log the health report
            logger.info("============ SYSTEM HEALTH REPORT ============")
            for cycle, stats in health_summary.items():
                logger.info(f"{cycle.upper()} CYCLE: {stats['success_rate']:.1f}% success, " 
                           f"{stats['avg_duration']:.3f}s avg duration, "
                           f"{stats['timeouts']} timeouts, {stats['failures']} failures")
            
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory usage: {memory_usage:.1f} MB")
            
            # GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                logger.info(f"GPU memory: {gpu_memory:.1f} MB")
            
            logger.info("=============================================")
            
        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")

    def _trim_metrics_history(self, metrics: dict) -> None:
        """Prevent memory bloat by trimming metrics history."""
        max_samples = 100  # Keep only the last 100 samples
        
        try:
            for cycle in metrics:
                if len(metrics[cycle]['durations']) > max_samples:
                    metrics[cycle]['durations'] = metrics[cycle]['durations'][-max_samples:]
                    
        except Exception as e:
            logger.error(f"Failed to trim metrics history: {e}")

    def _terminate_hanging_processes(self) -> None:
        """Terminate any hanging processes or threads."""
        try:
            # Cancel any pending tasks
            if hasattr(self, '_pending_tasks'):
                for task in self._pending_tasks:
                    if not task.done():
                        task.cancel()
            
            # Reset thread pools
            if hasattr(self, '_thread_executor'):
                self._thread_executor.shutdown(wait=False)
                self._thread_executor = ThreadPoolExecutor(max_workers=self.num_workers)
                
            # Reset process pools
            if hasattr(self, '_process_executor'):
                self._process_executor.shutdown(wait=False)
                self._process_executor = ProcessPoolExecutor(max_workers=self.num_workers)
                
            logger.info("Terminated hanging processes and reset executors")
            
        except Exception as e:
            logger.error(f"Error terminating processes: {e}")

    async def _save_degraded_state_checkpoint(self) -> None:
        """Save checkpoint when entering degraded state."""
        try:
            checkpoint = {
                'timestamp': time.time(),
                'learning_parameters': self.learning_parameters.copy() if hasattr(self, 'learning_parameters') else {},
                'prediction_horizon': self.prediction_horizon if hasattr(self, 'prediction_horizon') else None,
                'system_state': 'DEGRADED'
            }
            
            with open('degraded_state_checkpoint.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
                
            logger.info("Saved degraded state checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to save degraded state: {e}")

    async def _clean_shutdown(self) -> None:
        """Perform a clean shutdown of the recursive learning loop."""
        try:
            logger.info("Initiating clean shutdown of recursive learning loop")
            
            # 1. Save current learning state
            await self._save_learning_state()
            
            # 2. Flush any pending updates
            if hasattr(self, 'memory') and hasattr(self.memory, 'flush_pending_updates'):
                await self.memory.flush_pending_updates()
            
            # 3. Stop any background tasks
            if hasattr(self, '_background_tasks'):
                for task in self._background_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
            
            # 4. Release resources
            self._clear_all_caches()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Clean shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during clean shutdown: {e}")

    async def _emergency_state_preservation(self) -> None:
        """Preserve system state during critical errors."""
        try:
            logger.critical("Attempting emergency state preservation")
            
            # Use a unique filename based on timestamp
            timestamp = int(time.time())
            filename = f"emergency_state_{timestamp}.pkl"
            
            # Create a minimal state dictionary with essential data
            critical_state = {
                'timestamp': timestamp,
                'error_context': traceback.format_exc(),
                'agent_state': self._extract_agent_state() if hasattr(self, 'agent') else None,
                'memory_state': self._extract_memory_state() if hasattr(self, 'memory') else None
            }
            
            # Save to disk
            with open(filename, 'wb') as f:
                pickle.dump(critical_state, f)
                
            logger.critical(f"Emergency state saved to {filename}")
            
        except Exception as e:
            logger.critical(f"Failed to preserve emergency state: {e}")
            
    def start_recursive_learning(self):
        """Start the recursive learning loop in the background"""
        if not hasattr(self, '_recursive_learning_task') or self._recursive_learning_task is None:
            self._recursive_learning_task = asyncio.create_task(self.run_recursive_learning_loop())
            logger.info("Started recursive learning in background")
        else:
            logger.warning("Recursive learning already running")
            
    def stop_recursive_learning(self):
        """Stop the recursive learning loop"""
        if hasattr(self, '_recursive_learning_task') and self._recursive_learning_task is not None:
            self._recursive_learning_task.cancel()
            self._recursive_learning_task = None
            logger.info("Stopped recursive learning")
        else:
            logger.warning("Recursive learning not running") 
    
    async def _generate_internal_goals(self, performance_metrics: Dict[str, Any], 
                                  focus: str = 'balanced', 
                                  min_gap: float = 0.2) -> List[Dict[str, Any]]:
        """
        Generate internal improvement goals based on performance metrics
        
        Args:
            performance_metrics: Current performance metrics from _evaluate_system_performance
            focus: Focus area for goals ('balanced', 'memory', 'planning', 'rl', 'transfer', or 'system')
            min_gap: Minimum gap between current and target values (higher = more ambitious goals)
            
        Returns:
            List of generated goals
        """
        try:
            if not performance_metrics:
                return []
                
            goals = []
            current_time = time.time()
            
            # Extract component metrics
            memory_metrics = performance_metrics.get('memory', {})
            planner_metrics = performance_metrics.get('planner', {})
            rl_metrics = performance_metrics.get('rl_agent', {})
            transfer_metrics = performance_metrics.get('transfer_learning', {})
            
            # Planner component goals - check for domain-specific issues
            if 'planner' in performance_metrics:
                # Drift score improvement
                if 'drift_score' in planner_metrics:
                    drift_score = planner_metrics['drift_score']
                    if drift_score < 0.75:
                        # Check for domain-specific drift issues
                        domain_drift = planner_metrics.get('domain_drift', {})
                        
                        if domain_drift:
                            # Find most problematic domain
                            problematic_domain = min(domain_drift.items(), key=lambda x: x[1])
                            domain_name = problematic_domain[0]
                            domain_score = problematic_domain[1]
                            
                            if domain_score < 0.6:  # Significant drift
                                # Create domain-specific goal
                                goals.append({
                                    'id': str(uuid.uuid4()),
                                    'component': 'planner',
                                    'metric': 'drift_score',
                                    'description': f"Improve planning stability for '{domain_name}' domain, target drift < 1.5 std dev",
                                    'current_value': domain_score,
                                    'target_value': min(0.9, domain_score + min_gap * 1.5),
                                    'status': 'active',
                                    'created_at': current_time,
                                    'priority': 'high' if domain_score < 0.5 else 'medium',
                                    'progress': 0.0,
                                    'specific_domain': domain_name
                                })
                        else:
                            # Create general drift goal
                            goals.append({
                                'id': str(uuid.uuid4()),
                                'component': 'planner',
                                'metric': 'drift_score',
                                'description': "Reduce overall concept drift",
                                'current_value': drift_score,
                                'target_value': min(0.9, drift_score + min_gap),
                                'status': 'active',
                                'created_at': current_time,
                                'priority': 'medium',
                                'progress': 0.0
                            })
                
                # Success rate improvement by domain
                if 'success_rate' in planner_metrics:
                    # Check for domain-specific success rates
                    domain_success = planner_metrics.get('domain_success_rates', {})
                    
                    if domain_success:
                        # Find domain with lowest success rate
                        worst_domain = min(domain_success.items(), key=lambda x: x[1])
                        domain_name = worst_domain[0]
                        success_rate = worst_domain[1]
                        
                        if success_rate < 0.7:  # Needs improvement
                            goals.append({
                                'id': str(uuid.uuid4()),
                                'component': 'planner',
                                'metric': 'success_rate',
                                'description': f"Improve planning success rate for '{domain_name}' domain",
                                'current_value': success_rate,
                                'target_value': min(0.9, success_rate + min_gap * 1.3),
                                'status': 'active',
                                'created_at': current_time,
                                'priority': 'high' if success_rate < 0.5 else 'medium',
                                'progress': 0.0,
                                'specific_domain': domain_name
                            })
            
            # Memory component goals
            if 'memory' in performance_metrics:
                # Retrieval precision goal
                if 'retrieval_precision' in memory_metrics:
                    precision = memory_metrics['retrieval_precision']
                    if precision < 0.8:  # Room for improvement
                        goals.append({
                            'id': str(uuid.uuid4()),
                            'component': 'memory',
                            'metric': 'retrieval_precision',
                            'description': "Improve memory retrieval precision",
                            'current_value': precision,
                            'target_value': min(0.95, precision + min_gap),
                            'status': 'active',
                            'created_at': current_time,
                            'priority': 'high' if precision < 0.7 else 'medium',
                            'progress': 0.0
                        })
            
            # RL agent component goals
            if 'rl_agent' in performance_metrics:
                # Learning efficiency goal
                if 'learning_efficiency' in rl_metrics:
                    efficiency = rl_metrics['learning_efficiency']
                    if efficiency < 0.8:
                        goals.append({
                            'id': str(uuid.uuid4()),
                            'component': 'rl_agent',
                            'metric': 'learning_efficiency',
                            'description': "Improve RL agent learning efficiency",
                            'current_value': efficiency,
                            'target_value': min(0.95, efficiency + min_gap),
                            'status': 'active',
                            'created_at': current_time,
                            'priority': 'high' if efficiency < 0.6 else 'medium',
                            'progress': 0.0
                        })
                        
                # Exploration-exploitation balance
                if 'exploration_balance_score' in rl_metrics:
                    balance = rl_metrics['exploration_balance_score']
                    if balance < 0.7:
                        goals.append({
                            'id': str(uuid.uuid4()),
                            'component': 'rl_agent',
                            'metric': 'exploration_balance_score',
                            'description': "Improve RL agent exploration-exploitation balance",
                            'current_value': balance,
                            'target_value': min(0.9, balance + min_gap),
                            'status': 'active',
                            'created_at': current_time,
                            'priority': 'medium',
                            'progress': 0.0
                        })
            
            # Transfer learning component goals
            if 'transfer_learning' in performance_metrics:
                # Diversity score goal
                if 'diversity_score' in transfer_metrics:
                    diversity = transfer_metrics['diversity_score']
                    if diversity < 0.7:
                        goals.append({
                            'id': str(uuid.uuid4()),
                            'component': 'transfer_learning',
                            'metric': 'diversity_score',
                            'description': "Increase diversity of knowledge transfer across domains",
                            'current_value': diversity,
                            'target_value': min(0.9, diversity + min_gap),
                            'status': 'active',
                            'created_at': current_time,
                            'priority': 'medium',
                            'progress': 0.0
                        })
            
            # System-wide component balance goals
            if 'overall_score' in performance_metrics:
                # Find the weakest component
                component_scores = {}
                for component in ['memory', 'planner', 'rl_agent', 'transfer_learning']:
                    if component in performance_metrics:
                        score = performance_metrics[component].get('overall_score', 0.0)
                        component_scores[component] = score
                
                if component_scores:
                    # Find component with lowest score
                    weakest_component = min(component_scores.items(), key=lambda x: x[1])
                    if weakest_component[1] < 0.65:  # Only if performing poorly
                        # Create system balance goal
                        goals.append({
                            'id': str(uuid.uuid4()),
                            'component': 'system',
                            'metric': 'component_balance',
                            'description': f"Improve {weakest_component[0]} component performance to balance system",
                            'current_value': weakest_component[1],
                            'target_value': min(0.8, weakest_component[1] + min_gap * 1.5),
                            'status': 'active',
                            'created_at': current_time,
                            'priority': 'high' if weakest_component[1] < 0.5 else 'medium',
                            'progress': 0.0,
                            'status': 'active',
                            'weakest_component': weakest_component[0]
                        })
            
            # Remove duplicate goals (same component and metric)
            deduplicated_goals = []
            for goal in goals:
                is_duplicate = False
                # Check current goals list
                for existing_goal in self.internal_goals:
                    if (existing_goal['component'] == goal['component'] and 
                        existing_goal['metric'] == goal['metric'] and
                        existing_goal['status'] == 'active'):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    deduplicated_goals.append(goal)
            
            # Add new goals to the internal goals list
            self.internal_goals.extend(deduplicated_goals)
            
            # Limit number of active goals
            active_goals = [g for g in self.internal_goals if g['status'] == 'active']
            if len(active_goals) > 5:
                # Sort by priority and recency
                priority_map = {'high': 0, 'medium': 1, 'low': 2}
                active_goals.sort(key=lambda g: (priority_map.get(g['priority'], 3), -g['created_at']))
                
                # Keep only the top 5
                for i, goal in enumerate(active_goals):
                    if i >= 5:
                        # Find this goal in the main list and mark as deferred
                        for g in self.internal_goals:
                            if g['id'] == goal['id']:
                                g['status'] = 'deferred'
                                break
            
            # Return the newly created goals
            return deduplicated_goals
            
        except Exception as e:
            logger.error(f"Error generating internal goals: {e}")
            return []
    
    async def _update_goal_progress(self, performance_metrics: Dict[str, Any]) -> None:
        """
        Update progress on active internal goals based on current performance metrics
        
        Args:
            performance_metrics: Current performance metrics
        """
        if not hasattr(self, 'internal_goals'):
            self.internal_goals = []
            return
            
        try:
            for goal in self.internal_goals:
                if goal['status'] != 'active':
                    continue
                    
                # Get current value for the goal's metric
                component = goal['component']
                metric = goal['metric']
                
                if component == 'system' and metric == 'component_balance':
                    # Special case for system balance goals
                    if 'weakest_component' in goal:
                        weakest_comp = goal['weakest_component']
                        if weakest_comp in performance_metrics:
                            current_value = performance_metrics[weakest_comp].get('overall_score', 0.0)
                        else:
                            continue
                    else:
                        continue
                elif component in performance_metrics:
                    comp_metrics = performance_metrics[component]
                    if metric in comp_metrics:
                        current_value = comp_metrics[metric]
                    else:
                        continue
                else:
                    continue
                
                # Update goal with current value
                original_gap = goal['target_value'] - goal['current_value']
                current_gap = goal['target_value'] - current_value
                
                if original_gap <= 0:
                    progress = 1.0  # Goal was already achieved
                else:
                    # Calculate progress percentage
                    progress = 1.0 - (current_gap / original_gap)
                    progress = max(0.0, min(1.0, progress))
                
                # Update goal progress
                goal['progress'] = progress
                goal['last_value'] = current_value
                goal['last_updated'] = time.time()
                
                # Check if goal is achieved
                if progress >= 0.95 or current_value >= goal['target_value']:
                    goal['status'] = 'completed'
                    goal['completed_at'] = time.time()
                    
                    # Log achievement
                    logger.info(f"Internal goal achieved: {goal['description']}")
                
                # Check for stalled goals (no progress for a long time)
                elif ('last_progress' in goal and 
                      abs(goal['progress'] - goal['last_progress']) < 0.05 and
                      time.time() - goal['last_progress_time'] > 3600):  # 1 hour with no progress
                    
                    # Either mark as stalled or adapt the goal
                    if goal['progress'] < 0.3:
                        # Very little progress made, mark as stalled
                        goal['status'] = 'stalled'
                        logger.warning(f"Internal goal stalled: {goal['description']}")
                    else:
                        # Some progress made, adjust target to be more achievable
                        new_target = current_value + (goal['target_value'] - current_value) * 0.7
                        goal['original_target'] = goal['target_value']
                        goal['target_value'] = new_target
                        goal['adjusted_at'] = time.time()
                        logger.info(f"Adjusted internal goal: {goal['description']} - new target: {new_target:.2f}")
                
                # Save current progress for future comparison
                goal['last_progress'] = goal['progress']
                goal['last_progress_time'] = time.time()
                
        except Exception as e:
            logger.error(f"Error updating goal progress: {e}")
    
    async def _adapt_for_goals(self, goals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate adaptation actions to address internal goals
        
        Args:
            goals: List of active internal goals
            
        Returns:
            Dictionary of adaptation actions by component
        """
        actions = {
            'memory': [],
            'planner': [],
            'rl_agent': [],
            'transfer_learning': [],
            'system': []
        }
        
        try:
            if not goals:
                return actions
                
            # Process each active goal to generate concrete actions
            for goal in [g for g in goals if g['status'] == 'active']:
                component = goal['component']
                metric = goal['metric']
                
                # Memory component adaptations
                if component == 'memory':
                    if metric == 'retrieval_precision':
                        actions['memory'].append({
                            'type': 'parameter_adjustment',
                            'parameter': 'importance_threshold',
                            'direction': 'increase',
                            'amount': 0.05,
                            'goal_id': goal['id'],
                            'description': 'Increase memory importance threshold to improve precision'
                        })
                        
                        # Also consider rebuilding index
                        actions['memory'].append({
                            'type': 'operation',
                            'operation': 'rebuild_index',
                            'params': {'distributed': True},
                            'goal_id': goal['id'],
                            'description': 'Rebuild memory index for better precision'
                        })
                
                # Planner component adaptations
                elif component == 'planner':
                    if metric == 'drift_score':
                        # If there's a specific domain
                        if 'specific_domain' in goal and goal['specific_domain']:
                            domain = goal['specific_domain']
                            actions['planner'].append({
                                'type': 'domain_focus',
                                'domain': domain,
                                'goal_id': goal['id'],
                                'description': f"Focus domain adaptation on '{domain}'"
                            })
                            
                            # Add more targeted actions for specific domains
                            # Recommend increasing data for this domain
                            actions['planner'].append({
                                'type': 'data_collection',
                                'domain': domain,
                                'goal_id': goal['id'],
                                'description': f"Collect more training examples for '{domain}' domain"
                            })
                            
                            # Add domain-specific planning strategy
                            actions['planner'].append({
                                'type': 'strategy_adaptation',
                                'domain': domain,
                                'goal_id': goal['id'],
                                'description': f"Create specialized planning strategy for '{domain}' domain"
                            })
                        
                        # General drift handling improvements
                        actions['planner'].append({
                            'type': 'parameter_adjustment',
                            'parameter': 'drift_sensitivity',
                            'direction': 'increase',
                            'amount': 0.1,
                            'goal_id': goal['id'],
                            'description': 'Increase drift detection sensitivity'
                        })
                        
                        actions['planner'].append({
                            'type': 'parameter_adjustment',
                            'parameter': 'adaptation_rate',
                            'direction': 'increase',
                            'amount': 0.1,
                            'goal_id': goal['id'],
                            'description': 'Increase adaptation rate for faster response to drift'
                        })
                    
                    elif metric == 'success_rate' and 'specific_domain' in goal:
                        domain = goal['specific_domain']
                        # Add domain-specific success rate improvements
                        actions['planner'].append({
                            'type': 'domain_focus',
                            'domain': domain,
                            'goal_id': goal['id'],
                            'description': f"Prioritize planning optimization for '{domain}' domain"
                        })
                        
                        # Specialized domain actions
                        actions['planner'].append({
                            'type': 'template_creation',
                            'domain': domain,
                            'goal_id': goal['id'],
                            'description': f"Create high-success planning templates for '{domain}' domain"
                        })
                        
                        # Transfer learning from similar domains
                        if hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'transfer_bridge'):
                            actions['transfer_learning'].append({
                                'type': 'domain_transfer',
                                'target_domain': domain,
                                'goal_id': goal['id'],
                                'description': f"Transfer successful planning patterns from similar domains to '{domain}'"
                            })
                
                # RL agent component adaptations
                elif component == 'rl_agent':
                    if metric == 'learning_efficiency':
                        actions['rl_agent'].append({
                            'type': 'parameter_adjustment',
                            'parameter': 'learning_rate',
                            'direction': 'optimize',
                            'goal_id': goal['id'],
                            'description': 'Optimize learning rate based on recent performance'
                        })
                        
                        # Consider batch size adjustment
                        actions['rl_agent'].append({
                            'type': 'parameter_adjustment',
                            'parameter': 'batch_size',
                            'direction': 'increase',
                            'amount': 8,  # Increase by 8
                            'goal_id': goal['id'],
                            'description': 'Increase batch size for more stable learning'
                        })
                        
                    elif metric == 'exploration_balance_score':
                        # Get current exploration rate if available
                        exploration_rate = 0.3  # Default
                        if (hasattr(self, 'rl_agent') and hasattr(self.rl_agent, 'model') and 
                            hasattr(self.rl_agent.model, 'clip_range')):
                            exploration_rate = self.rl_agent.model.clip_range
                        
                        # Determine adjustment direction
                        if goal.get('last_value', 0.5) < 0.5:
                            # Poor balance, likely needs adjustment
                            if exploration_rate > 0.5:
                                direction = 'decrease'
                                description = 'Decrease exploration rate for better exploitation'
                            else:
                                direction = 'increase'
                                description = 'Increase exploration rate for better exploration'
                                
                            actions['rl_agent'].append({
                                'type': 'parameter_adjustment',
                                'parameter': 'exploration_rate',
                                'direction': direction,
                                'amount': 0.1,
                                'goal_id': goal['id'],
                                'description': description
                            })
                
                # Transfer learning component adaptations
                elif component == 'transfer_learning':
                    if metric == 'diversity_score':
                        actions['transfer_learning'].append({
                            'type': 'operation',
                            'operation': 'expand_domains',
                            'goal_id': goal['id'],
                            'description': 'Actively seek new domains to expand knowledge diversity'
                        })
                
                # System-wide adaptations
                elif component == 'system' and metric == 'component_balance':
                    if 'weakest_component' in goal:
                        weak_comp = goal['weakest_component']
                        actions['system'].append({
                            'type': 'resource_allocation',
                            'component': weak_comp,
                            'allocation': 'increase',
                            'goal_id': goal['id'],
                            'description': f"Allocate more computational resources to {weak_comp} component"
                        })
                        
                        # Also prioritize that component's goals
                        for other_goal in goals:
                            if other_goal['component'] == weak_comp and other_goal['status'] == 'active':
                                other_goal['priority'] = 'high'
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating adaptations for goals: {e}")
            return actions

    async def _synchronize_component_states(self) -> bool:
        """Ensure all components have consistent state."""
        try:
            logger.debug("Synchronizing component states")
            
            # Get state summaries from each component
            state_summaries = {}
            for component_name in ['agent', 'memory', 'planner', 'perception']:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    if hasattr(component, 'get_state_summary'):
                        state_summaries[component_name] = component.get_state_summary()
            
            # Verify consistency and resolve conflicts
            conflicts = self._detect_state_conflicts(state_summaries)
            if conflicts:
                logger.warning(f"Detected {len(conflicts)} state conflicts")
                resolved = self._resolve_state_conflicts(conflicts)
                if not resolved:
                    logger.error("Failed to resolve state conflicts")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"State synchronization failed: {e}")
            return False

    def _adjust_performance_parameters(self) -> None:
        """Dynamically adjust performance parameters based on system load."""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Adjust batch sizes based on available resources
            if cpu_percent > 85 or memory_percent > 80:
                # High load - reduce batch sizes
                self.batch_size_multiplier = 0.5
                logger.info("High system load detected, reducing batch sizes")
            elif cpu_percent < 40 and memory_percent < 60:
                # Low load - increase batch sizes for efficiency
                self.batch_size_multiplier = 1.5
                logger.info("Low system load detected, increasing batch sizes")
            else:
                # Normal load - standard batch sizes
                self.batch_size_multiplier = 1.0
            
            # Apply batch size adjustments to components
            for component_name in ['agent', 'memory', 'planner']:
                if hasattr(self, component_name):
                    component = getattr(self, component_name)
                    if hasattr(component, 'batch_size'):
                        original_batch_size = getattr(component, 'original_batch_size', component.batch_size)
                        component.original_batch_size = original_batch_size
                        component.batch_size = max(4, int(original_batch_size * self.batch_size_multiplier))
        except Exception as e:
            logger.error(f"Error adjusting performance parameters: {e}")

    def _calculate_optimal_cycle_timing(self) -> dict:
        """Dynamically adjust cycle timings based on performance metrics."""
        try:
            timings = {
                'fast': 5,  # Default 5 seconds
                'medium': 60,  # Default 60 seconds
                'slow': 300  # Default 300 seconds
            }
            
            # If we have performance data, adjust timings
            if hasattr(self, 'cycle_metrics'):
                # Fast cycle adjustment based on average duration
                if len(self.cycle_metrics['fast']['durations']) > 0:
                    avg_fast = sum(self.cycle_metrics['fast']['durations']) / len(self.cycle_metrics['fast']['durations'])
                    # Ensure at least 2x headroom for fast cycle
                    timings['fast'] = max(5, min(15, int(avg_fast * 2.5)))
                
                # Medium cycle adjustment
                if len(self.cycle_metrics['medium']['durations']) > 0:
                    avg_medium = sum(self.cycle_metrics['medium']['durations']) / len(self.cycle_metrics['medium']['durations'])
                    # Ensure at least 1.5x headroom for medium cycle
                    timings['medium'] = max(30, min(180, int(avg_medium * 1.5)))
                    
                # Slow cycle adjustment
                if len(self.cycle_metrics['slow']['durations']) > 0:
                    avg_slow = sum(self.cycle_metrics['slow']['durations']) / len(self.cycle_metrics['slow']['durations'])
                    # Ensure at least 1.2x headroom for slow cycle
                    timings['slow'] = max(120, min(600, int(avg_slow * 1.2)))
            
            return timings
        except Exception as e:
            logger.error(f"Error calculating cycle timings: {e}")
            # Return defaults in case of error
            return {'fast': 5, 'medium': 60, 'slow': 300}
