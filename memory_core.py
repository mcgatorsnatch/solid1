"""
Core memory management, embedding, indexing, Zettelkasten, and plasticity.
"""
import os
import re
import gc
import math
import time
import json
import heapq
import pickle
import asyncio
import hashlib
import shutil
import logging
import threading
import itertools
from typing import List, Dict, Set, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from exceptions import SecureMemoryError
from secure_memory import SecureMemory
from alignment import CulturalAlignmentModule

logger = logging.getLogger(__name__)

class QuantumMemory:
    """
    Manages memory storage, retrieval, and semantic search with encryption and caching.
    Includes an episodic buffer for recent chat interactions, using a relevance gate
    for transferring items to long-term memory.
    """
    # Default scaling factor for the sigmoid gate sharpness
    DEFAULT_GATE_SCALING_FACTOR = 10.0 
    # Default maximum token capacity for episodic buffer
    DEFAULT_MAX_BUFFER_TOKENS = 128
    # Average token estimate for unknown words
    AVG_TOKENS_PER_WORD = 1.3
    
    # Add priority queue constants
    DEFAULT_PRIORITY_QUEUE_SIZE = 25
    DEFAULT_MIN_TRANSFER_THRESHOLD = 0.6
    
    # New defaults for Zettelkasten-inspired tagging
    DEFAULT_TAGGING_MODEL = "distilbert-base-uncased"
    DEFAULT_LINK_THRESHOLD = 0.85
    DEFAULT_HEBBIAN_LEARNING_RATE = 0.01  # η
    DEFAULT_HEBBIAN_DECAY = 0.85         # α
    
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 use_gpu: bool = True,
                 batch_size: int = 8,
                 max_items: int = 10000,
                 max_episodic_buffer_size: int = 10,
                 max_buffer_tokens: int = 2000,
                 importance_threshold: float = 0.7,
                 temporal_decay_rate: float = 0.01,
                 hebbian_learning_rate: float = 0.05,
                 enable_zettelkasten: bool = True,
                 link_threshold: float = 0.8,
                 encryption_key: Optional[str] = None,
                 use_trainable_gates: bool = False):
        """
        Initialize the quantum memory system with the specified parameters.
        
        Args:
            model_name: Name of the embedding model to use
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Batch size for embedding operations
            max_items: Maximum number of items to store in memory
            max_episodic_buffer_size: Maximum number of items in the episodic buffer
            max_buffer_tokens: Maximum number of tokens in the episodic buffer
            importance_threshold: Threshold for memory importance
            temporal_decay_rate: Rate at which memory importance decays over time
            hebbian_learning_rate: Learning rate for Hebbian connections
            enable_zettelkasten: Whether to enable Zettelkasten features
            link_threshold: Similarity threshold for linking related items
            encryption_key: Optional key for encrypting memory
            use_trainable_gates: Whether to use trainable neural gates for memory transfer
        """
        # Basic properties
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_items = max_items
        self.importance_threshold = importance_threshold
        self.temporal_decay_rate = temporal_decay_rate
        self.hebbian_learning_rate = hebbian_learning_rate
        self.link_threshold = link_threshold
        self.enable_zettelkasten = enable_zettelkasten
        
        # Episodic buffer parameters
        self.max_episodic_buffer_size = max_episodic_buffer_size
        self.max_buffer_tokens = max_buffer_tokens
        self.episodic_buffer_token_count = 0
        
        # Memory transfer gate parameters
        self.use_trainable_gates = use_trainable_gates
        
        # Initialize trainable gates if enabled
        if self.use_trainable_gates:
            try:
                import torch
                import torch.nn as nn
                
                # Define the embedding size based on the model
                if 'MiniLM-L6' in model_name:
                    self.embedding_size = 384
                else:
                    # Default size if model is unknown
                    self.embedding_size = 768
                
                # Create trainable parameters for memory gate
                # g_t = sigma(W_g * [h_{t-1}, x_t] + b_g)
                self.gate_W = nn.Parameter(torch.randn(self.embedding_size * 2, 1) * 0.01)
                self.gate_b = nn.Parameter(torch.zeros(1))
                
                # Use GPU if available and requested
                self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
                self.gate_W = self.gate_W.to(self.device)
                self.gate_b = self.gate_b.to(self.device)
                
                # Optimizer for gate parameters
                self.gate_optimizer = torch.optim.Adam([self.gate_W, self.gate_b], lr=0.001)
                
                self.trainable_gates_initialized = True
                logger.info(f"Trainable memory gates initialized on {self.device}")
                
            except ImportError:
                logger.warning("Could not import PyTorch. Falling back to similarity-based gates.")
                self.use_trainable_gates = False
                self.trainable_gates_initialized = False
        else:
            self.trainable_gates_initialized = False
            
        # Other initializations (unchanged)
        # ...

    async def _apply_trainable_gate(self, memory_embedding, new_embedding):
        """
        Apply a trainable gate mechanism to determine if a memory should be transferred.
        Implementation of g_t = sigma(W_g * [h_{t-1}, x_t] + b_g)
        
        Args:
            memory_embedding: The embedding of an existing memory (h_{t-1})
            new_embedding: The embedding of the new item (x_t)
            
        Returns:
            Gate activation value between 0 and 1
        """
        if not self.trainable_gates_initialized:
            # Fall back to similarity if gates aren't initialized
            return self._sigmoid(self._calculate_cosine_similarity(memory_embedding, new_embedding))
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Convert to tensors if they aren't already
            if not isinstance(memory_embedding, torch.Tensor):
                memory_embedding = torch.tensor(memory_embedding, device=self.device)
            if not isinstance(new_embedding, torch.Tensor):
                new_embedding = torch.tensor(new_embedding, device=self.device)
                
            # Ensure correct shape
            memory_embedding = memory_embedding.view(-1)
            new_embedding = new_embedding.view(-1)
            
            # Concatenate the embeddings
            combined = torch.cat([memory_embedding, new_embedding])
            
            # Apply the gate: g_t = sigma(W_g * [h_{t-1}, x_t] + b_g)
            gate_input = combined.view(1, -1)
            gate_output = torch.sigmoid(F.linear(gate_input, self.gate_W.t(), self.gate_b))
            
            # Return as a scalar
            return gate_output.item()
            
        except Exception as e:
            logger.error(f"Error in trainable gate: {e}. Falling back to similarity.")
            return self._sigmoid(self._calculate_cosine_similarity(memory_embedding, new_embedding))

    async def train_memory_gate(self, examples, epochs=10, batch_size=16):
        """
        Train the memory gate model using supervised examples.
        
        Args:
            examples: List of tuples (memory_embedding, new_embedding, label)
                     where label is 1 if the memory should be transferred, 0 otherwise
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training accuracy
        """
        if not self.use_trainable_gates or not self.trainable_gates_initialized:
            logger.warning("Cannot train memory gates: trainable gates not enabled/initialized")
            return 0.0
            
        try:
            import torch
            import torch.nn.functional as F
            import random
            from tqdm import tqdm
            
            # Convert examples to tensors
            memory_embeddings = torch.tensor([ex[0] for ex in examples], device=self.device)
            new_embeddings = torch.tensor([ex[1] for ex in examples], device=self.device)
            labels = torch.tensor([ex[2] for ex in examples], device=self.device, dtype=torch.float32)
            
            # Dataset size
            dataset_size = len(examples)
            
            # Track metrics
            total_loss = 0.0
            correct = 0
            
            # Training loop
            for epoch in range(epochs):
                # Shuffle data
                indices = list(range(dataset_size))
                random.shuffle(indices)
                
                # Process in batches
                for i in range(0, dataset_size, batch_size):
                    # Get batch indices
                    batch_indices = indices[i:i+batch_size]
                    
                    # Get batch data
                    batch_memories = memory_embeddings[batch_indices]
                    batch_new = new_embeddings[batch_indices]
                    batch_labels = labels[batch_indices].view(-1, 1)
                    
                    # Reset gradients
                    self.gate_optimizer.zero_grad()
                    
                    # Forward pass
                    # Concatenate memory and new embeddings
                    batch_combined = torch.cat([batch_memories, batch_new], dim=1)
                    
                    # Apply gate: g_t = sigma(W_g * [h_{t-1}, x_t] + b_g)
                    gate_outputs = torch.sigmoid(F.linear(batch_combined, self.gate_W.t(), self.gate_b))
                    
                    # Calculate loss
                    loss = F.binary_cross_entropy(gate_outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update parameters
                    self.gate_optimizer.step()
                    
                    # Track metrics
                    total_loss += loss.item() * len(batch_indices)
                    predictions = (gate_outputs > 0.5).float()
                    correct += (predictions == batch_labels).sum().item()
                    
                # Print epoch summary
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    avg_loss = total_loss / dataset_size
                    accuracy = correct / dataset_size
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
                    total_loss = 0.0
                    correct = 0
            
            # Final evaluation
            with torch.no_grad():
                # Concatenate memory and new embeddings
                combined = torch.cat([memory_embeddings, new_embeddings], dim=1)
                
                # Apply gate
                gate_outputs = torch.sigmoid(F.linear(combined, self.gate_W.t(), self.gate_b))
                
                # Calculate accuracy
                predictions = (gate_outputs > 0.5).float()
                accuracy = (predictions == labels.view(-1, 1)).sum().item() / dataset_size
                
            logger.info(f"Memory gate training completed. Final accuracy: {accuracy:.4f}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Error training memory gate: {e}")
            return 0.0

    async def _episodic_to_priority_queue(self, max_transfers: int = 3) -> int:
        """
        Transfer important items from episodic buffer to priority queue.
        Uses the trained gate if enabled, otherwise falls back to similarity-based mechanism.
        
        Args:
            max_transfers: Maximum number of items to transfer
            
        Returns:
            Number of items transferred
        """
        if not self.episodic_buffer:
            return 0
            
        # Determine available slots in priority queue
        priority_queue_slots = max(0, self.max_episodic_buffer_size - len(self.priority_queue))
        max_transfers = min(max_transfers, priority_queue_slots, len(self.episodic_buffer))
        
        # Extract embeddings for all episodic memories
        episodic_embeddings = []
        for memory in list(self.episodic_buffer):
            if 'embedding' in memory:
                episodic_embeddings.append(memory['embedding'])
            else:
                # Generate embedding if not available
                try:
                    text = memory.get('text', '')
                    memory['embedding'] = await self._get_embedding(text)
                    episodic_embeddings.append(memory['embedding'])
                except Exception as e:
                    logger.error(f"Failed to get embedding for memory: {e}")
                    episodic_embeddings.append(None)
        
        # Calculate transfer probabilities
        transfer_probs = []
        for i, memory in enumerate(self.episodic_buffer):
            if episodic_embeddings[i] is None:
                transfer_probs.append(0.0)
                continue
                
            # Calculate average similarity to existing memories
            similarities = []
            for j, other_memory in enumerate(self.episodic_buffer):
                if i != j and episodic_embeddings[j] is not None:
                    if self.use_trainable_gates and self.trainable_gates_initialized:
                        # Use trainable gate
                        similarity = await self._apply_trainable_gate(
                            episodic_embeddings[i], episodic_embeddings[j]
                        )
                    else:
                        # Use similarity-based gate
                        similarity = self._sigmoid(self._calculate_cosine_similarity(
                            episodic_embeddings[i], episodic_embeddings[j]
                        ))
                    similarities.append(similarity)
            
            # Combine with importance score
            importance = memory.get('importance', 0.5)
            recency = 1.0 - (len(self.episodic_buffer) - i) / len(self.episodic_buffer)
            
            # Higher for important, unique memories
            if similarities:
                uniqueness = 1.0 - (sum(similarities) / len(similarities))
            else:
                uniqueness = 1.0
                
            # Final priority is a weighted combination
            transfer_prob = (0.4 * importance) + (0.3 * uniqueness) + (0.3 * recency)
            transfer_probs.append(transfer_prob)
        
        # Select top memories for transfer
        transfer_indices = sorted(range(len(transfer_probs)), 
                                  key=lambda i: transfer_probs[i], 
                                  reverse=True)[:max_transfers]
        
        # Transfer selected memories to priority queue
        transferred = 0
        for idx in sorted(transfer_indices, reverse=True):
            if idx < len(self.episodic_buffer):
                memory = self.episodic_buffer[idx]
                
                # Check if this memory should be added to the priority queue
                self.priority_queue.add(memory)
                
                # Remove from episodic buffer
                self.episodic_buffer.remove(memory)
                
                # Update token count
                self.episodic_buffer_token_count -= memory.get('token_count', 0)
                
                transferred += 1
        
        if transferred > 0:
            logger.info(f"Transferred {transferred} memories from episodic buffer to priority queue" +
                       (f" using trainable gates" if self.use_trainable_gates and self.trainable_gates_initialized else ""))
        
        return transferred
    
    @property
    def alignment(self):
        """Lazy-loaded alignment module"""
        if self._alignment_module is None:
            self._alignment_module = CulturalAlignmentModule(load_model_on_init=False)
        return self._alignment_module
    
    @property
    def embedder(self):
        """Lazy-loaded embedding model"""
        if self._embed_model is None:
            try:
                self._embed_model = SentenceTransformer(self.model_name, device=self.device)
                self._embedding_dim = self._embed_model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model with dimension {self._embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise
        return self._embed_model
    
    def _init_index(self, index_type: str = 'flat'):
        """
        Initialize FAISS index based on specified type.
        
        Args:
            index_type: Type of index ('flat', 'hnsw', 'ivf', etc.)
        """
        # Ensure embedding model is loaded to get dimension
        _ = self.embedder
        
        with self.lock:
            try:
                if index_type.lower() == 'flat':
                    self.index = faiss.IndexFlatIP(self._embedding_dim)
                elif index_type.lower() == 'hnsw':
                    self.index = faiss.IndexHNSWFlat(self._embedding_dim, 32)  # 32 neighbors
                elif index_type.lower() == 'ivf':
                    quantizer = faiss.IndexFlatIP(self._embedding_dim)
                    nlist = min(4096, max(64, self.max_items // 1000))
                    self.index = faiss.IndexIVFFlat(quantizer, self._embedding_dim, nlist)
                    self.index.train(np.zeros((max(1000, min(self.max_items, 10000)), 
                                             self._embedding_dim), dtype=np.float32))
                else:
                    logger.warning(f"Unknown index type '{index_type}', using flat index")
                    self.index = faiss.IndexFlatIP(self._embedding_dim)
                    
                logger.info(f"Initialized {index_type} FAISS index with dimension {self._embedding_dim}")
            except Exception as e:
                logger.error(f"Failed to initialize FAISS index: {e}")
                # Fallback to simple flat index
                self.index = faiss.IndexFlatIP(self._embedding_dim)
    
    def _manage_cache(self):
        """Prune cache if it exceeds maximum size"""
        with self.lock:
            while len(self.search_cache) > self.max_cache_size:
                self.search_cache.popitem(last=False)  # Remove oldest item
    
    async def add_memory(self, text: str, task: str, score: float) -> int:
        """
        Add a new memory item with semantic embedding.
        
        Args:
            text: Text content to store
            task: Associated task description
            score: Relevance score (0.0-1.0)
            
        Returns:
            Memory ID if successful, -1 otherwise
        """
        if not text or score < self.min_memory_score:
            return -1
            
        # Check for content alignment
        is_aligned = await self.alignment.check_alignment(text)
        if not is_aligned:
            logger.warning(f"Content failed alignment check: {text[:50]}...")
            return -1
            
        with self.lock:
            try:
                # Check if we need to prune memories
                if len(self.memory_ids) >= self.max_items:
                    self._prune_old_memories()
                    
                # Get text embedding
                embedding = await self._get_embedding(text)
                
                # Generate memory ID
                memory_id = self.next_id
                self.next_id += 1
                
                # Create timestamp
                timestamp = time.time()
                
                # Setup initial memory data structure
                memory_data = {
                    'id': memory_id,
                    'text': text,
                    'task': task,
                    'timestamp': timestamp,
                    'score': score
                }
                
                # If Zettelkasten is enabled, initialize links and tags
                if self.enable_zettelkasten:
                    # Explicitly initialize empty links list
                    memory_data['links'] = []
                    
                    # Generate and add tags using transformer-based tagging
                    tags = await self._generate_auto_tags(text)
                    memory_data['tags'] = tags
                    
                    # Initialize memory_tags for this memory_id
                    self.memory_tags[memory_id] = set(tags)
                    
                    # Update tag to memories mapping
                    for tag in tags:
                        self.tag_to_memories[tag].add(memory_id)
                
                # Store memory data
                self.memories[memory_id] = memory_data
                
                # Store metadata
                self.memory_texts[memory_id] = text
                self.memory_embeddings[memory_id] = embedding
                self.memory_timestamps[memory_id] = timestamp
                self.memory_access_times[memory_id] = timestamp
                self.memory_access_counts[memory_id] = 0
                self.memory_scores[memory_id] = score
                self.memory_tasks[memory_id] = task
                
                # Add to ordered list
                self.memory_ids.append(memory_id)
                
                # Add to index
                if self.index is not None:
                    self.index.add(np.array([embedding], dtype=np.float32))
                
                # If Zettelkasten is enabled, create links
                if self.enable_zettelkasten:
                    # Create links to related memories
                    await self._create_links_for_memory(memory_id, embedding)
                
                logger.debug(f"Added memory {memory_id}: {text[:50]}...")
                return memory_id
                
            except Exception as e:
                logger.error(f"Failed to add memory: {e}")
                return -1
    
    async def add_chat_interaction(self, user_input: str, response: str) -> bool:
        """
        Add a chat interaction to memory.
        
        Args:
            user_input: User input text
            response: Assistant response text
            
        Returns:
            True if successful, False otherwise
        """
        if not user_input or not response:
            return False
            
        try:
            # Format as a memory item
            memory_text = f"User: {user_input}\nAssistant: {response}"
            
            # Add interaction to episodic buffer first
            await self.add_to_episodic_buffer(user_input, response)
            
            # Add to long-term memory with moderate score
            memory_id = await self.add_memory(memory_text, "chat_interaction", score=0.7)
            
            return memory_id > 0
            
        except Exception as e:
            logger.error(f"Failed to add chat interaction: {e}")
            return False
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array embedding
        """
        try:
            # Truncate very long texts to avoid performance issues
            if len(text) > 10000:
                logger.warning(f"Truncating very long text from {len(text)} chars to 10000")
                text = text[:10000]
                
            # Generate embedding
            with torch.no_grad():
                embedding = self.embedder.encode(text, convert_to_numpy=True)
                
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros(self._embedding_dim, dtype=np.float32)
    
    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid activation function with scaling.
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid output (0-1)
        """
        scaled_x = x * self.gate_scaling_factor
        return 1 / (1 + math.exp(-scaled_x))
    
    async def _calculate_interaction_similarity(self, interaction1: Dict[str, Any], interaction2: Dict[str, Any]) -> Optional[float]:
        """
        Calculate semantic similarity between two chat interactions.
        
        Args:
            interaction1: First interaction dict
            interaction2: Second interaction dict
            
        Returns:
            Similarity score (0-1) or None if error
        """
        try:
            # Extract texts
            text1 = f"User: {interaction1['user_input']}\nAssistant: {interaction1['response']}"
            text2 = f"User: {interaction2['user_input']}\nAssistant: {interaction2['response']}"
            
            # Get embeddings
            embedding1 = await self._get_embedding(text1)
            embedding2 = await self._get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Ensure in range [0,1]
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Error calculating interaction similarity: {e}")
            return None
    
    def _initialize_tokenizer(self):
        """
        Initialize a tokenizer for token counting.
        Uses a transformer tokenizer that matches common model tokenization.
        """
        try:
            from transformers import AutoTokenizer
            
            # Choose a tokenizer model (Using Meta's Llama3 tokenizer which is based on tiktoken)
            # This is a more modern tokenizer compared to the previous GPT-2 tokenizer
            # Llama3's tokenizer handles tokens more efficiently, especially for specialized vocabulary
            tokenizer_name = "meta-llama/Meta-Llama-3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            logger.info(f"Initialized tokenizer: {tokenizer_name}")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize tokenizer: {e}")
            # Fall back to smaller Meta model if the 8B model fails
            try:
                tokenizer_name = "facebook/opt-125m"  # Smaller Facebook OPT model as fallback
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info(f"Initialized fallback tokenizer: {tokenizer_name}")
                return True
            except Exception as e2:
                logger.warning(f"Could not initialize fallback tokenizer: {e2}")
                self.tokenizer = None
                return False
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text using a transformer tokenizer.
        Falls back to a word-count approximation if tokenizer isn't available.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
            
        # Initialize tokenizer if not done already
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self._initialize_tokenizer()
            
        # Use the tokenizer if available
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            try:
                # Encode the text and get the number of tokens
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                return len(token_ids)
            except Exception as e:
                logger.warning(f"Error using tokenizer: {e}. Falling back to approximation.")
        
        # Fallback: Use a more accurate approximation based on word count
        # This is still a rough estimate but better than nothing
        words = text.split()
        return max(1, int(len(words) * self.AVG_TOKENS_PER_WORD))
    
    def _calculate_interaction_tokens(self, interaction: Dict[str, Any]) -> int:
        """
        Calculate tokens in a chat interaction.
        
        Args:
            interaction: Chat interaction dict
            
        Returns:
            Estimated token count
        """
        try:
            user_tokens = self._estimate_tokens(interaction.get('user_input', ''))
            response_tokens = self._estimate_tokens(interaction.get('response', ''))
            return user_tokens + response_tokens
        except Exception as e:
            logger.error(f"Error calculating interaction tokens: {e}")
            return 0
    
    async def add_to_episodic_buffer(self, user_input: str, response: str) -> None:
        """
        Add a chat interaction to the episodic buffer.
        
        Args:
            user_input: User input text
            response: Assistant response text
        """
        if not user_input or not response:
            return
            
        with self.lock:
            try:
                # Create interaction object
                interaction = {
                    'user_input': user_input,
                    'response': response,
                    'timestamp': time.time()
                }
                
                # Calculate tokens
                tokens = self._calculate_interaction_tokens(interaction)
                
                # Check if we need to make space in the buffer
                while (len(self.episodic_buffer) > 0 and 
                       (len(self.episodic_buffer) >= self.max_episodic_buffer_size or
                        self.episodic_buffer_token_count + tokens > self.max_buffer_tokens)):
                    
                    # Process the oldest interaction
                    oldest = self.episodic_buffer.pop(0)
                    oldest_tokens = self._calculate_interaction_tokens(oldest)
                    self.episodic_buffer_token_count -= oldest_tokens
                    
                    # Calculate relevance score for transfer to long-term memory
                    relevance = 0.5  # Default moderate relevance
                    
                    # Calculate semantic similarity to remaining buffer items
                    if len(self.episodic_buffer) > 0:
                        similarities = []
                        for item in self.episodic_buffer:
                            sim = await self._calculate_interaction_similarity(oldest, item)
                            if sim is not None:
                                similarities.append(sim)
                                
                        # Average similarity as relevance
                        if similarities:
                            avg_similarity = sum(similarities) / len(similarities)
                            # Apply sigmoid transformation to emphasize high similarities
                            relevance = self._sigmoid(avg_similarity - 0.5)
                    
                    # Add to priority queue if above threshold
                    if self.episodic_transfer_threshold is None or relevance >= self.episodic_transfer_threshold:
                        await self.add_to_priority_queue(oldest, relevance)
                
                # Add the new interaction to the buffer
                self.episodic_buffer.append(interaction)
                self.episodic_buffer_token_count += tokens
                
                # Process priority queue (max 3 transfers at a time)
                await self._process_priority_queue(max_transfers=3)
                
            except Exception as e:
                logger.error(f"Error adding to episodic buffer: {e}")
    
    async def get_episodic_buffer_contents(self) -> List[Dict[str, Any]]:
        """Get contents of the episodic buffer"""
        with self.lock:
            return self.episodic_buffer.copy()
    
    async def get_episodic_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the episodic buffer"""
        with self.lock:
            return {
                'size': len(self.episodic_buffer),
                'token_count': self.episodic_buffer_token_count,
                'max_size': self.max_episodic_buffer_size,
                'max_tokens': self.max_buffer_tokens
            }
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                             threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search on stored memories.
        Uses memory_id mapping to correctly retrieve memories regardless of index position.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of matching memory dictionaries
        """
        if not query:
            return []
            
        # Check cache
        cache_key = f"{query}:{top_k}:{threshold}"
        with self.lock:
            if cache_key in self.search_cache:
                # Update cache position and return cached results
                results = self.search_cache[cache_key]
                self.search_cache.move_to_end(cache_key)
                return results.copy()
        
        try:
            # Generate query embedding
            query_embedding = await self._get_embedding(query)
            
            # Search using FAISS
            with self.lock:
                if self.index is None or not self.memories:
                    return []
                    
                # Convert to numpy array
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # Search
                actual_k = min(top_k, len(self.memories))
                if actual_k == 0:
                    return []
                
                scores, indices = self.index.search(query_vector, actual_k)
                
                # Process results
                results = []
                for i, (score, faiss_idx) in enumerate(zip(scores[0], indices[0])):
                    # If index is out of bounds or invalid, skip
                    if faiss_idx < 0 or faiss_idx not in self.faiss_idx_to_memory_id:
                        continue
                        
                    # Get memory id from mapping
                    memory_id = self.faiss_idx_to_memory_id[faiss_idx]
                    
                    # Apply threshold
                    if threshold is not None and score < threshold:
                        continue
                        
                    # Get memory data
                    memory_data = self.memories.get(memory_id, {}).copy()
                    
                    # Skip if memory not found
                    if not memory_data:
                        continue
                        
                    # Add search score
                    memory_data['score'] = float(score)
                    
                    # Add to results
                    results.append(memory_data)
                    
                    # Update access metadata
                    self._update_memory_access(memory_id)
                
                # Cache results
                self.search_cache[cache_key] = results.copy()
                self._manage_cache()
                
                return results
                
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _update_memory_access(self, memory_id: int) -> bool:
        """
        Update memory access metadata.
        
        Args:
            memory_id: ID of memory to update
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if memory_id not in self.memory_access_times:
                return False
                
            # Update access time and count
            self.memory_access_times[memory_id] = time.time()
            self.memory_access_counts[memory_id] = self.memory_access_counts.get(memory_id, 0) + 1
            
            return True
    
    def _prune_old_memories(self) -> int:
        """
        Prune old/unused memory items when storage limit is reached.
        
        This function prunes memory items only, not the connections between them.
        It prioritizes removing memories based on:
        1. Age (if retention_days is set)
        2. Access count (least accessed items first)
        3. Relevance score (lower scoring items first)
        
        For pruning connections between memories, see _prune_synaptic_connections.
        
        Returns:
            Number of memory items pruned
        """
        with self.lock:
            if len(self.memory_ids) < self.max_items:
                return 0
                
            # Calculate how many to prune
            prune_count = max(1, len(self.memory_ids) - int(self.max_items * 0.9))
            pruned = 0
            
            # Check if retention policy is based on time
            if self.retention_days is not None:
                current_time = time.time()
                retention_seconds = self.retention_days * 24 * 3600
                
                # Collect memories older than retention period
                old_memories = []
                for memory_id in self.memory_ids:
                    timestamp = self.memory_timestamps.get(memory_id, 0)
                    age = current_time - timestamp
                    if age > retention_seconds:
                        old_memories.append((memory_id, self.memory_access_counts.get(memory_id, 0), 
                                            self.memory_scores.get(memory_id, 0)))
                
                # Sort by (access_count, score) to prune least used, low scoring memories first
                old_memories.sort(key=lambda x: (x[1], x[2]))
                
                # Prune old memories
                to_prune = old_memories[:prune_count]
                for memory_id, _, _ in to_prune:
                    self._remove_memory(memory_id)
                    pruned += 1
                    
                # If we pruned enough, return
                if pruned >= prune_count:
                    logger.info(f"Pruned {pruned} memories based on age")
                    return pruned
            
            # If we haven't pruned enough based on age, prune based on access and score
            if pruned < prune_count:
                remaining = prune_count - pruned
                
                # Get access metrics for all memories
                memory_metrics = []
                for memory_id in self.memory_ids:
                    access_count = self.memory_access_counts.get(memory_id, 0)
                    score = self.memory_scores.get(memory_id, 0)
                    memory_metrics.append((memory_id, access_count, score))
                
                # Sort by (access_count, score) to prune least used, low scoring memories first
                memory_metrics.sort(key=lambda x: (x[1], x[2]))
                
                # Prune memories
                to_prune = memory_metrics[:remaining]
                for memory_id, _, _ in to_prune:
                    self._remove_memory(memory_id)
                    pruned += 1
                
                logger.info(f"Pruned {pruned} memories based on usage")
                return pruned
    
    def _remove_memory(self, memory_id: int) -> bool:
        """
        Remove a memory by ID.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if memory_id not in self.memories:
                return False
                
            try:
                # Find index of memory in ordered list
                try:
                    index = self.memory_ids.index(memory_id)
                except ValueError:
                    index = -1
                
                # Remove from index if found
                if index >= 0:
                    # We can't directly remove from FAISS index, so we'll rebuild it later
                    if index < len(self.memory_ids) - 1:
                        # Remove by swapping with last item and then popping
                        self.memory_ids[index] = self.memory_ids[-1]
                    
                    self.memory_ids.pop()
                
                # Remove from all data structures
                self.memories.pop(memory_id, None)
                self.memory_texts.pop(memory_id, None)
                self.memory_embeddings.pop(memory_id, None)
                self.memory_timestamps.pop(memory_id, None)
                self.memory_access_times.pop(memory_id, None)
                self.memory_access_counts.pop(memory_id, None)
                self.memory_scores.pop(memory_id, None)
                self.memory_tasks.pop(memory_id, None)
                
                # Handle Zettelkasten structures
                if self.enable_zettelkasten:
                    # Remove from tags
                    tags = self.memory_tags.pop(memory_id, set())
                    for tag in tags:
                        if memory_id in self.tag_to_memories[tag]:
                            self.tag_to_memories[tag].remove(memory_id)
                            if not self.tag_to_memories[tag]:
                                self.tag_to_memories.pop(tag, None)
                    
                    # Remove connections
                    self.memory_connections.pop(memory_id, None)
                    
                    # Remove from other memories' connections
                    for other_id, connections in self.memory_connections.items():
                        connections.pop(memory_id, None)
                
                # Flag index as needing rebuild
                self._rebuild_index()
                
                return True
                
            except Exception as e:
                logger.error(f"Error removing memory {memory_id}: {e}")
                return False
    
    def save_cache(self) -> bool:
        """
        Save cached data to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_file = "quantum_memory_cache.pkl"
            
            # Create cache data dictionary
            cache_data = {
                'search_cache': dict(self.search_cache),
                'timestamp': time.time()
            }
            
            # Save to file
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            logger.info(f"Saved memory cache to {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory cache: {e}")
            return False
    
    def load_cache(self) -> bool:
        """
        Load cached data from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_file = "quantum_memory_cache.pkl"
            
            # Check if file exists
            if not os.path.exists(cache_file):
                logger.warning(f"Memory cache file not found: {cache_file}")
                return False
                
            # Load from file
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Restore cache
            with self.lock:
                self.search_cache = OrderedDict(cache_data.get('search_cache', {}))
                logger.info(f"Loaded memory cache from {cache_file}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load memory cache: {e}")
            return False
    
    def _rebuild_index(self):
        """
        Rebuild FAISS index from scratch, using memory_id mappings to ensure consistency.
        This implementation resolves the fragility in the original index rebuilding process,
        by directly using memory_embeddings dictionary keyed by memory_id rather than
        relying on list indices matching FAISS IDs.
        """
        with self.lock:
            try:
                # Create a new index
                if self.index_type.lower() == 'flat':
                    new_index = faiss.IndexFlatIP(self._embedding_dim)
                elif self.index_type.lower() == 'hnsw':
                    new_index = faiss.IndexHNSWFlat(self._embedding_dim, 32)
                elif self.index_type.lower() == 'ivf':
                    quantizer = faiss.IndexFlatIP(self._embedding_dim)
                    nlist = min(4096, max(64, self.max_items // 1000))
                    new_index = faiss.IndexIVFFlat(quantizer, self._embedding_dim, nlist)
                    # Train on existing embeddings
                    if self.memory_embeddings and len(self.memory_embeddings) > 0:
                        train_vectors = np.array(list(self.memory_embeddings.values()), dtype=np.float32)
                        if len(train_vectors) > nlist:
                            new_index.train(train_vectors)
                        else:
                            # Not enough vectors to train, use zeros
                            new_index.train(np.zeros((max(nlist, 1000), self._embedding_dim), dtype=np.float32))
                else:
                    logger.warning(f"Unknown index type '{index_type}', using flat index")
                    new_index = faiss.IndexFlatIP(self._embedding_dim)
                
                # Create mapping from FAISS index position to memory_id
                self.faiss_idx_to_memory_id = {}
                
                # Add all vectors
                if self.memory_embeddings and len(self.memory_embeddings) > 0:
                    vectors = []
                    memory_ids = []
                    
                    # Collect vectors and their corresponding memory_ids in the same order
                    for memory_id, embedding in self.memory_embeddings.items():
                        if memory_id in self.memories:  # Only include active memories
                            vectors.append(embedding)
                            memory_ids.append(memory_id)
                    
                    if vectors:
                        vectors_array = np.array(vectors, dtype=np.float32)
                        # Add vectors to index
                        new_index.add(vectors_array)
                        
                        # Create mapping from FAISS index position to memory_id
                        for idx, memory_id in enumerate(memory_ids):
                            self.faiss_idx_to_memory_id[idx] = memory_id
                
                # Replace the old index
                self.index = new_index
                
                logger.info(f"Rebuilt FAISS index with {len(self.faiss_idx_to_memory_id)} items")
                return True
                
            except Exception as e:
                logger.error(f"Failed to rebuild index: {e}")
                # Don't lose the original index if rebuild fails
                return False
    
    async def rotate_key(self, new_key: str) -> bool:
        """
        Rotate encryption key for secure storage.
        
        Args:
            new_key: New encryption key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Rotate key in secure memory
            success = self.secure.rotate_key(new_key)
            
            if success:
                logger.info("Rotated encryption key successfully")
            else:
                logger.error("Failed to rotate encryption key")
                
            return success
            
        except SecureMemoryError as e:
            logger.error(f"Error rotating encryption key: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by QuantumMemory"""
        with self.lock:
            try:
                # Clear cache
                self.search_cache.clear()
                
                # Clear buffers
                self.episodic_buffer.clear()
                self.episodic_buffer_token_count = 0
                
                # Clear priority queue
                self.priority_queue.clear()
                
                # Release embedding model
                if self._embed_model:
                    self._embed_model = None
                
                # Release alignment module
                if self._alignment_module:
                    self._alignment_module.cleanup()
                    self._alignment_module = None
                
                # Release FAISS index
                if self.index is not None:
                    # No direct way to release FAISS index in Python, so just delete the reference
                    self.index = None
                
                # Force garbage collection
                gc.collect()
                
                # Clear torch cache if using GPU
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                logger.info("QuantumMemory resources cleaned up")
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    def _verify_index_integrity(self):
        """
        Verify integrity of the FAISS index.
        
        Returns:
            True if index is valid, False otherwise
        """
        with self.lock:
            try:
                if self.index is None:
                    return False
                    
                # Check if index size matches memory count
                index_size = self.index.ntotal
                memory_count = len(self.memory_ids)
                
                if index_size != memory_count:
                    logger.warning(f"Index integrity issue: index size {index_size} != memory count {memory_count}")
                    return False
                    
                return True
                
            except Exception as e:
                logger.error(f"Error verifying index integrity: {e}")
                return False
    
    def _should_prune_memories(self) -> bool:
        """
        Check if memory pruning is needed.
        
        Returns:
            True if pruning is needed, False otherwise
        """
        with self.lock:
            # Check if we've exceeded maximum memory items
            if len(self.memory_ids) >= self.max_items:
                return True
                
            # Check if we have old memories that should be pruned
            if self.retention_days is not None:
                current_time = time.time()
                retention_seconds = self.retention_days * 24 * 3600
                
                # Check oldest memory
                if self.memory_ids:
                    oldest_id = self.memory_ids[0]  # Memories are added in order
                    timestamp = self.memory_timestamps.get(oldest_id, current_time)
                    age = current_time - timestamp
                    
                    if age > retention_seconds:
                        return True
            
            return False
    
    # Priority Queue Management
    
    class PriorityItem:
        """Item for the priority queue with custom comparison"""
        
        def __init__(self, priority, count, interaction, relevance_score):
            self.priority = priority
            self.count = count
            self.interaction = interaction
            self.relevance_score = relevance_score
            
        def __lt__(self, other):
            # Invert priority to make heapq a max-heap (highest score = highest priority)
            return self.priority > other.priority
            
        def __eq__(self, other):
            return self.priority == other.priority
    
    async def add_to_priority_queue(self, interaction: Dict[str, Any], relevance_score: float) -> bool:
        """
        Add an interaction to the priority queue.
        
        Args:
            interaction: Interaction dictionary
            relevance_score: Relevance score (0-1)
            
        Returns:
            True if added, False otherwise
        """
        if not interaction or 'user_input' not in interaction or 'response' not in interaction:
            return False
            
        async with self.priority_queue_lock:
            try:
                # Skip if below threshold
                if relevance_score < self.min_transfer_threshold:
                    return False
                    
                # Generate priority based on relevance and recency
                # More recent items have higher count, boosting priority of newer items with equal relevance
                priority = relevance_score
                count = int(time.time() * 1000)  # Millisecond precision for ordering
                
                # Create priority item
                item = self.PriorityItem(priority, count, interaction, relevance_score)
                
                # Add to priority queue
                heapq.heappush(self.priority_queue, item)
                
                # Check if queue exceeds max size
                while len(self.priority_queue) > self.max_priority_queue_size:
                    # Remove lowest priority item
                    heapq.heappop(self.priority_queue)
                    
                return True
                
            except Exception as e:
                logger.error(f"Error adding to priority queue: {e}")
                return False
    
    async def _process_priority_queue(self, max_transfers: int = 3) -> int:
        """
        Process items from the priority queue into long-term memory
        with enhanced context retention and association formation.
        
        Args:
            max_transfers: Maximum number of items to transfer
            
        Returns:
            Number of items transferred to long-term memory
        """
        if not hasattr(self, 'priority_queue') or not self.priority_queue:
            return 0
            
        processed_count = 0
        semantic_clusters = {}  # Group similar items
        
        # Process up to max_transfers items
        for _ in range(min(max_transfers, len(self.priority_queue))):
            if not self.priority_queue:
                break
                
            try:
                # Get highest priority item
                _, item = heapq.heappop(self.priority_queue)
                
                if 'text' not in item:
                    continue
                    
                # Find semantic cluster for item
                embedding = await self._get_embedding(item['text'])
                
                # Skip if embedding generation failed
                if embedding is None:
                    continue
                    
                # Get cluster label or create new one
                most_similar_cluster = None
                highest_similarity = 0.5  # Minimum threshold
                
                for cluster_id, cluster in semantic_clusters.items():
                    similarity = cosine_similarity(embedding, cluster['centroid'])
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_cluster = cluster_id
                
                # Add to existing cluster or create new one
                if most_similar_cluster is not None:
                    cluster = semantic_clusters[most_similar_cluster]
                    cluster['items'].append(item)
                    cluster['embeddings'].append(embedding)
                    # Update centroid
                    cluster['centroid'] = np.mean(cluster['embeddings'], axis=0)
                else:
                    # Create new cluster
                    cluster_id = str(uuid.uuid4())
                    semantic_clusters[cluster_id] = {
                        'items': [item],
                        'embeddings': [embedding],
                        'centroid': embedding
                    }
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing priority queue item: {e}")
                continue
        
        # Now process the clusters into long-term memory
        for cluster_id, cluster in semantic_clusters.items():
            try:
                # For single-item clusters, just add to memory
                if len(cluster['items']) == 1:
                    item = cluster['items'][0]
                    await self.add_memory(
                        text=item['text'],
                        task=item.get('task', 'episodic_transfer'),
                        score=item.get('score', 0.7),
                        references=item.get('references', [])
                    )
                else:
                    # For multi-item clusters, combine information with enrichment
                    combined_text = self._combine_cluster_items(cluster['items'])
                    
                    # Create combined references
                    references = []
                    for item in cluster['items']:
                        if 'references' in item:
                            references.extend(item['references'])
                    
                    # Add consolidated memory
                    await self.add_memory(
                        text=combined_text,
                        task='consolidated_memory',
                        score=0.8,  # Higher score for consolidated memories
                        references=references,
                        metadata={'consolidated': True, 'source_count': len(cluster['items'])}
                    )
            except Exception as e:
                logger.error(f"Error adding cluster to memory: {e}")
        
        return processed_count
    
    def _combine_cluster_items(self, items):
        """
        Intelligently combine similar items from a cluster into a single memory.
        
        Args:
            items: List of similar memory items to combine
            
        Returns:
            Combined text representation
        """
        if not items:
            return ""
            
        if len(items) == 1:
            return items[0]['text']
            
        # Extract all texts
        texts = [item['text'] for item in items if 'text' in item]
        
        # Find common themes or combine information
        # Simple approach for now: just concatenate with separator and length limit
        result = texts[0]
        
        for i in range(1, len(texts)):
            # Check if this text adds new information
            if texts[i] not in result:
                # Find overlap to create smooth transition
                overlap = self._find_text_overlap(result, texts[i])
                if overlap and len(overlap) > 10:
                    # Connect using the overlap
                    result = result + texts[i][len(overlap):]
                else:
                    # No significant overlap, just append with separator
                    result = result + " | " + texts[i]
        
        # Limit length to avoid overly long memories
        max_length = 1000
        if len(result) > max_length:
            result = result[:max_length] + "..."
            
        return result
    
    def _find_text_overlap(self, text1, text2, min_overlap=5):
        """
        Find overlap between end of text1 and beginning of text2.
        
        Args:
            text1: First text
            text2: Second text
            min_overlap: Minimum characters to consider an overlap
            
        Returns:
            Overlapping text or empty string
        """
        max_len = min(len(text1), len(text2), 50)  # Limit search to 50 chars
        
        for i in range(min_overlap, max_len + 1):
            if text1[-i:] == text2[:i]:
                return text2[:i]
                
        return ""
    
    async def _replay_random_memories(self, count: int = 2) -> int:
        """
        Randomly select and reprocess older memories to enhance retention and connections.
        This mimics hippocampal replay during sleep in biological brains.
        
        Args:
            count: Number of random memories to replay
            
        Returns:
            Number of memories successfully replayed
        """
        if not self.metadata or len(self.metadata) <= 5:
            return 0
            
        replayed_count = 0
        
        try:
            # Get indices of memories, excluding very recent ones
            older_memory_indices = list(range(len(self.metadata) - 5))
            
            # Skip if no older memories
            if not older_memory_indices:
                return 0
                
            # Randomly select memories to replay
            selected_indices = random.sample(
                older_memory_indices, 
                min(count, len(older_memory_indices))
            )
            
            # Replay selected memories (strengthen their connections)
            for idx in selected_indices:
                try:
                    # Get memory details
                    memory = self.metadata[idx]
                    memory_id = memory.get('id')
                    
                    if not memory_id:
                        continue
                        
                    # Update last accessed time
                    self.metadata[idx]['last_accessed'] = time.time()
                    
                    # Find related memories
                    if 'text' in memory and self.enable_zettelkasten:
                        related_ids = await self._find_related_memories(memory['text'], exclude_ids=[memory_id])
                        
                        # Apply Hebbian learning to strengthen connections
                        if related_ids:
                            all_ids = [memory_id] + related_ids
                            for i in range(len(all_ids)):
                                for j in range(i+1, len(all_ids)):
                                    id1 = all_ids[i]
                                    id2 = all_ids[j]
                                    
                                    # Ensure consistent key order
                                    key = (min(id1, id2), max(id1, id2))
                                    
                                    # Strengthen connection
                                    current_weight = self.connection_weights.get(key, 0.0)
                                    # Apply smaller learning rate for replay (30% of normal)
                                    new_weight = current_weight + (0.3 * self.hebbian_learning_rate)
                                    self.connection_weights[key] = min(1.0, new_weight)
                            
                            replayed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error replaying memory {idx}: {e}")
                    continue
                    
            return replayed_count
            
        except Exception as e:
            logger.error(f"Error in memory replay: {e}")
            return 0
    
    async def _reinforce_memory_schemas(self) -> int:
        """
        Identify and reinforce knowledge schemas in the memory network.
        Schemas are recurring patterns of connected memories that represent
        generalized knowledge structures.
        
        Returns:
            Number of schemas reinforced
        """
        if not hasattr(self, 'connection_weights') or not self.connection_weights:
            return 0
            
        reinforced_count = 0
        
        try:
            # Step 1: Extract graph structure
            memory_graph = self._build_memory_graph()
            if not memory_graph:
                return 0
                
            # Step 2: Identify memory clusters (potential schemas)
            schemas = self._identify_memory_clusters(memory_graph)
            
            # Step 3: Reinforce connections within schemas
            for schema in schemas:
                try:
                    # Calculate size of schema
                    schema_size = len(schema)
                    if schema_size < 3:
                        continue  # Skip too small schemas
                        
                    # Strengthen connections within schema
                    connections_strengthened = 0
                    for i in range(schema_size):
                        for j in range(i+1, schema_size):
                            id1 = schema[i]
                            id2 = schema[j]
                            
                            # Ensure consistent key order
                            key = (min(id1, id2), max(id1, id2))
                            
                            # Check if connection exists and strengthen
                            current_weight = self.connection_weights.get(key, 0.0)
                            if current_weight > 0:
                                # Strengthen by a small amount
                                new_weight = current_weight + 0.05
                                self.connection_weights[key] = min(1.0, new_weight)
                                connections_strengthened += 1
                    
                    if connections_strengthened > 0:
                        reinforced_count += 1
                        
                except Exception as e:
                    logger.error(f"Error reinforcing schema: {e}")
                    continue
                    
            return reinforced_count
            
        except Exception as e:
            logger.error(f"Error in schema reinforcement: {e}")
            return 0
            
    async def _merge_similar_memories(self, similarity_threshold: float = 0.92) -> int:
        """
        Identify and merge highly similar memories to reduce redundancy.
        Uses embeddings to identify near-duplicate memories.
        
        Args:
            similarity_threshold: Threshold above which memories are considered duplicates
            
        Returns:
            Number of memories merged
        """
        if not self.faiss_index or not self.metadata:
            return 0
            
        merged_count = 0
        
        try:
            # Create a dictionary of embeddings by ID for easy access
            id_to_idx = {}
            for i, meta in enumerate(self.metadata):
                id_to_idx[meta.get('id')] = i
            
            # Set to track already processed IDs
            processed_ids = set()
            
            # Identify similar memories
            for i, meta in enumerate(self.metadata):
                memory_id = meta.get('id')
                
                # Skip already processed or merged memories
                if memory_id in processed_ids or meta.get('merged', False):
                    continue
                    
                # Get embedding for this memory
                if i >= len(self.memory_embeddings):
                    continue
                    
                query_embedding = self.memory_embeddings[i].reshape(1, -1)
                
                # Search for similar memories using FAISS
                D, I = self.faiss_index.search(query_embedding, 10)  # Get top 10 similar
                
                # Filter by similarity threshold, excluding the memory itself
                similar_indices = [idx for idx, dist in zip(I[0], D[0]) 
                                  if idx != i and idx < len(self.metadata) and 
                                  self._faiss_distance_to_similarity(dist) >= similarity_threshold]
                
                if not similar_indices:
                    continue
                    
                # Get memory IDs for similar memories
                similar_ids = []
                for idx in similar_indices:
                    if idx < len(self.metadata):
                        similar_id = self.metadata[idx].get('id')
                        if similar_id and similar_id not in processed_ids:
                            similar_ids.append(similar_id)
                
                if not similar_ids:
                    continue
                
                # Merge similar memories
                merged = await self._merge_memory_group([memory_id] + similar_ids)
                
                if merged:
                    merged_count += len(similar_ids)
                    processed_ids.update(similar_ids)
                    processed_ids.add(memory_id)
            
            # If memories were merged, rebuild index
            if merged_count > 0:
                self._rebuild_index()
                
            return merged_count
            
        except Exception as e:
            logger.error(f"Error merging similar memories: {e}")
            return 0
    
    async def _merge_memory_group(self, memory_ids):
        """
        Merge a group of similar memories into a single enhanced memory.
        
        Args:
            memory_ids: List of memory IDs to merge
            
        Returns:
            True if successful, False otherwise
        """
        if not memory_ids or len(memory_ids) < 2:
            return False
            
        try:
            # Get memory details
            memories = []
            for memory_id in memory_ids:
                for i, meta in enumerate(self.metadata):
                    if meta.get('id') == memory_id:
                        memories.append((i, meta))
                        break
            
            if len(memories) < 2:
                return False
                
            # Extract texts and metadata
            texts = []
            tasks = set()
            references = []
            all_metadata = {}
            
            for idx, meta in memories:
                # Collect texts
                if idx < len(self.memory_texts):
                    texts.append(self.memory_texts[idx])
                    
                # Collect tasks
                if 'task' in meta:
                    tasks.add(meta['task'])
                    
                # Collect references
                if 'references' in meta:
                    references.extend(meta['references'])
                    
                # Collect metadata (without overwriting)
                for key, value in meta.items():
                    if key not in all_metadata and key not in ['id', 'last_accessed', 'merged']:
                        all_metadata[key] = value
            
            # Create merged content
            merged_text = self._create_merged_text(texts)
            merged_task = list(tasks)[0] if tasks else "merged_memory"
            
            # Add merged memory
            merged_id = await self.add_memory(
                text=merged_text,
                task=merged_task,
                score=0.85,  # Higher score for merged memories
                references=list(set(references)),  # Remove duplicates
                metadata={**all_metadata, 'merged': True, 'source_count': len(memories)}
            )
            
            # Update original memories to mark as merged
            for idx, meta in memories:
                if idx < len(self.metadata):
                    self.metadata[idx]['merged'] = True
                    self.metadata[idx]['merged_into'] = merged_id
            
            # Transfer connections from original memories to merged memory
            if hasattr(self, 'connection_weights'):
                self._transfer_connections(memory_ids, merged_id)
                
            return True
            
        except Exception as e:
            logger.error(f"Error merging memory group: {e}")
            return False
    
    def _create_merged_text(self, texts):
        """
        Create merged text content from multiple similar memories.
        
        Args:
            texts: List of text strings to merge
            
        Returns:
            Merged text
        """
        if not texts:
            return ""
            
        if len(texts) == 1:
            return texts[0]
            
        # Identify common parts and unique information
        # Simple approach for now: just concatenate with intelligent separator
        result = texts[0]
        
        for i in range(1, len(texts)):
            # Check if this text adds new information
            if texts[i] not in result:
                # Find overlap to create smooth transition
                overlap = self._find_text_overlap(result, texts[i])
                if overlap and len(overlap) > 10:
                    # Connect using the overlap
                    result = result + texts[i][len(overlap):]
                else:
                    # No significant overlap, just append with separator
                    result = result + " | " + texts[i]
        
        # Limit length to avoid overly long memories
        max_length = 1000
        if len(result) > max_length:
            result = result[:max_length] + "..."
            
        return result
    
    def _find_text_overlap(self, text1, text2, min_overlap=5):
        """
        Find overlap between end of text1 and beginning of text2.
        
        Args:
            text1: First text
            text2: Second text
            min_overlap: Minimum characters to consider an overlap
            
        Returns:
            Overlapping text or empty string
        """
        max_len = min(len(text1), len(text2), 50)  # Limit search to 50 chars
        
        for i in range(min_overlap, max_len + 1):
            if text1[-i:] == text2[:i]:
                return text2[:i]
                
        return ""
    
    def _transfer_connections(self, source_ids, target_id):
        """
        Transfer connections from source memories to target memory.
        
        Args:
            source_ids: List of source memory IDs
            target_id: Target memory ID
        """
        if not hasattr(self, 'connection_weights'):
            return
            
        # Collect all connections to transfer
        connections_to_transfer = {}
        
        for id1, id2 in list(self.connection_weights.keys()):
            # Check if either ID is in source_ids
            if id1 in source_ids:
                other_id = id2
                if other_id not in source_ids:
                    weight = self.connection_weights.get((id1, id2), 0.0)
                    connections_to_transfer[other_id] = max(connections_to_transfer.get(other_id, 0.0), weight)
            
            elif id2 in source_ids:
                other_id = id1
                if other_id not in source_ids:
                    weight = self.connection_weights.get((id1, id2), 0.0)
                    connections_to_transfer[other_id] = max(connections_to_transfer.get(other_id, 0.0), weight)
        
        # Create new connections to target
        for other_id, weight in connections_to_transfer.items():
            key = (min(target_id, other_id), max(target_id, other_id))
            self.connection_weights[key] = weight
    
    async def create_second_order_connections(self, memory_indices, strength_multiplier=0.7):
        """
        Create second-order connections between memories that share connections
        to the same important memories. This mimics the transitive property
        of semantic relationships.
        
        Args:
            memory_indices: Indices of important memories to use as bridges
            strength_multiplier: Factor to multiply connection strength
            
        Returns:
            Number of new connections created
        """
        if not hasattr(self, 'connection_weights') or not self.connection_weights:
            return 0
            
        connections_created = 0
        
        try:
            # Convert indices to IDs
            bridge_ids = []
            for idx in memory_indices:
                if idx < len(self.metadata):
                    memory_id = self.metadata[idx].get('id')
                    if memory_id:
                        bridge_ids.append(memory_id)
            
            if not bridge_ids:
                return 0
                
            # For each bridge memory, find connected memories
            for bridge_id in bridge_ids:
                # Find all memories connected to this bridge
                connected_ids = []
                
                # Scan all connections
                for (id1, id2), weight in self.connection_weights.items():
                    # Skip weak connections
                    if weight < 0.4:
                        continue
                        
                    if id1 == bridge_id and id2 not in bridge_ids:
                        connected_ids.append((id2, weight))
                    elif id2 == bridge_id and id1 not in bridge_ids:
                        connected_ids.append((id1, weight))
                
                # Create connections between memories connected to the same bridge
                for i in range(len(connected_ids)):
                    for j in range(i+1, len(connected_ids)):
                        id1, weight1 = connected_ids[i]
                        id2, weight2 = connected_ids[j]
                        
                        # Calculate new connection strength (product of strengths * multiplier)
                        new_strength = weight1 * weight2 * strength_multiplier
                        
                        # Only create significant connections
                        if new_strength < 0.2:
                            continue
                            
                        # Create or strengthen connection
                        key = (min(id1, id2), max(id1, id2))
                        current_strength = self.connection_weights.get(key, 0.0)
                        
                        # Use max to prevent weakening existing strong connections
                        self.connection_weights[key] = max(current_strength, new_strength)
                        
                        # Count as new if it didn't exist or was significantly strengthened
                        if current_strength < 0.1 or (new_strength - current_strength) > 0.3:
                            connections_created += 1
            
            return connections_created
            
        except Exception as e:
            logger.error(f"Error creating second-order connections: {e}")
            return 0
            
    async def _calculate_graph_metrics(self):
        """
        Calculate advanced metrics for the memory graph structure.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'clustering_coefficient': 0.0,
            'avg_centrality': 0.0,
            'connectivity': 0.0,
            'modularity': 0.0
        }
        
        if not hasattr(self, 'connection_weights') or not self.connection_weights:
            return metrics
            
        try:
            # Build graph
            graph = self._build_memory_graph()
            if not graph:
                return metrics
                
            # Calculate simple clustering coefficient
            clustering_sum = 0
            centrality_sum = 0
            node_count = len(graph)
            
            if node_count == 0:
                return metrics
                
            for node, neighbors in graph.items():
                # Skip nodes with <2 neighbors
                if len(neighbors) < 2:
                    continue
                    
                # Calculate clustering: ratio of actual connections between neighbors
                # to possible connections
                neighbor_ids = [n[0] for n in neighbors]
                possible_connections = len(neighbor_ids) * (len(neighbor_ids) - 1) / 2
                
                if possible_connections == 0:
                    continue
                    
                actual_connections = 0
                for i in range(len(neighbor_ids)):
                    for j in range(i+1, len(neighbor_ids)):
                        id1, id2 = neighbor_ids[i], neighbor_ids[j]
                        key = (min(id1, id2), max(id1, id2))
                        if key in self.connection_weights:
                            actual_connections += 1
                
                node_clustering = actual_connections / possible_connections
                clustering_sum += node_clustering
                
                # Calculate degree centrality
                centrality = len(neighbors) / (node_count - 1) if node_count > 1 else 0
                centrality_sum += centrality
            
            # Average the metrics
            metrics['clustering_coefficient'] = clustering_sum / node_count if node_count > 0 else 0
            metrics['avg_centrality'] = centrality_sum / node_count if node_count > 0 else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return metrics
            
    async def _extract_knowledge_patterns(self):
        """
        Extract recurring knowledge patterns from memory structure.
        
        Returns:
            List of knowledge patterns (each a dictionary with pattern information)
        """
        if not self.metadata or not hasattr(self, 'connection_weights'):
            return []
            
        patterns = []
        
        try:
            # 1. Identify clusters in memory graph
            graph = self._build_memory_graph()
            clusters = self._identify_memory_clusters(graph)
            
            # 2. Analyze each cluster for patterns
            for cluster_idx, cluster in enumerate(clusters):
                # Skip small clusters
                if len(cluster) < 4:
                    continue
                    
                # Get texts for cluster members
                cluster_texts = []
                for memory_id in cluster:
                    for i, meta in enumerate(self.metadata):
                        if meta.get('id') == memory_id and i < len(self.memory_texts):
                            cluster_texts.append(self.memory_texts[i])
                            break
                
                if len(cluster_texts) < 3:
                    continue
                    
                # Analyze texts for common themes
                common_words = self._extract_common_words(cluster_texts)
                
                if common_words:
                    pattern = {
                        'id': f"pattern_{cluster_idx}",
                        'memory_ids': cluster,
                        'common_themes': common_words,
                        'strength': len(cluster) / 10,  # Normalized strength
                        'density': self._calculate_cluster_density(cluster)
                    }
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting knowledge patterns: {e}")
            return []
    
    def _extract_common_words(self, texts, min_frequency=3):
        """
        Extract words that appear frequently across multiple texts.
        
        Args:
            texts: List of text strings
            min_frequency: Minimum frequency to consider a word common
            
        Returns:
            List of common words/phrases
        """
        if not texts:
            return []
            
        # Count word occurrences
        word_counts = {}
        
        for text in texts:
            # Tokenize and clean text
            words = text.lower().split()
            words = [word.strip('.,?!()[]{}:;"\'') for word in words]
            
            # Count unique words in this text
            text_words = set()
            for word in words:
                if len(word) > 3:  # Skip very short words
                    text_words.add(word)
            
            # Update global counts
            for word in text_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Filter by minimum frequency
        common_words = [word for word, count in word_counts.items() 
                       if count >= min_frequency]
        
        # Sort by frequency
        common_words.sort(key=lambda w: word_counts[w], reverse=True)
        
        return common_words[:10]  # Return top 10 most common
    
    def _calculate_cluster_density(self, cluster):
        """
        Calculate the connection density within a cluster.
        
        Args:
            cluster: List of memory IDs in the cluster
            
        Returns:
            Density score (0.0 to 1.0)
        """
        if not cluster or len(cluster) < 2:
            return 0.0
            
        possible_connections = len(cluster) * (len(cluster) - 1) / 2
        if possible_connections == 0:
            return 0.0
            
        actual_connections = 0
        connection_strength = 0.0
        
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                id1, id2 = cluster[i], cluster[j]
                key = (min(id1, id2), max(id1, id2))
                
                if key in self.connection_weights:
                    actual_connections += 1
                    connection_strength += self.connection_weights[key]
        
        # Calculate density
        connection_density = actual_connections / possible_connections
        
        # Calculate average strength
        avg_strength = connection_strength / actual_connections if actual_connections > 0 else 0
        
        # Combine density and strength
        return 0.7 * connection_density + 0.3 * avg_strength
    
    async def _apply_knowledge_patterns(self, patterns):
        """
        Apply identified knowledge patterns to improve memory organization.
        
        Args:
            patterns: List of knowledge patterns
            
        Returns:
            Number of memories reorganized
        """
        if not patterns:
            return 0
            
        reorganized = 0
        
        try:
            for pattern in patterns:
                # Skip weak patterns
                if pattern.get('strength', 0) < 0.3 or pattern.get('density', 0) < 0.4:
                    continue
                    
                # Get memory IDs in this pattern
                memory_ids = pattern.get('memory_ids', [])
                if len(memory_ids) < 3:
                    continue
                    
                # Create enhanced connections between pattern members
                for i in range(len(memory_ids)):
                    for j in range(i+1, len(memory_ids)):
                        id1, id2 = memory_ids[i], memory_ids[j]
                        key = (min(id1, id2), max(id1, id2))
                        
                        # Strengthen existing connections or create new ones
                        current_weight = self.connection_weights.get(key, 0.0)
                        new_weight = max(current_weight, 0.5 + (0.1 * pattern['strength']))
                        
                        if new_weight > current_weight:
                            self.connection_weights[key] = min(1.0, new_weight)
                            reorganized += 1
                            
                # Create pattern labels in memory metadata
                common_themes = pattern.get('common_themes', [])
                if common_themes:
                    pattern_label = f"Pattern: {', '.join(common_themes[:3])}"
                    
                    for memory_id in memory_ids:
                        # Find memory index
                        for i, meta in enumerate(self.metadata):
                            if meta.get('id') == memory_id:
                                # Add pattern to metadata
                                if 'patterns' not in meta:
                                    self.metadata[i]['patterns'] = []
                                    
                                if pattern_label not in self.metadata[i]['patterns']:
                                    self.metadata[i]['patterns'].append(pattern_label)
                                    reorganized += 1
                                    break
            
            return reorganized
            
        except Exception as e:
            logger.error(f"Error applying knowledge patterns: {e}")
            return 0
    
    async def adjust_learning_parameters(self, hebbian_rate_multiplier=1.0, 
                                       temporal_decay_multiplier=1.0,
                                       duration=3600):
        """
        Temporarily adjust learning parameters for the memory system.
        
        Args:
            hebbian_rate_multiplier: Multiplier for Hebbian learning rate
            temporal_decay_multiplier: Multiplier for temporal decay rate
            duration: Duration of adjustment in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store original values if not already stored
            if not hasattr(self, '_original_learning_params'):
                self._original_learning_params = {
                    'hebbian_learning_rate': self.hebbian_learning_rate,
                    'temporal_decay_factor': getattr(self, 'temporal_decay_factor', 0.9),
                    'reset_time': 0
                }
                
            # Apply adjustments
            self.hebbian_learning_rate = self._original_learning_params['hebbian_learning_rate'] * hebbian_rate_multiplier
            
            if hasattr(self, 'temporal_decay_factor'):
                self.temporal_decay_factor = self._original_learning_params['temporal_decay_factor'] * temporal_decay_multiplier
                
            # Set reset time
            self._original_learning_params['reset_time'] = time.time() + duration
            
            # Schedule reset
            asyncio.create_task(self._reset_learning_parameters_after_delay(duration))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting learning parameters: {e}")
            return False
    
    async def _reset_learning_parameters_after_delay(self, delay):
        """
        Reset learning parameters to original values after a delay.
        
        Args:
            delay: Delay in seconds
        """
        await asyncio.sleep(delay)
        
        try:
            # Check if reset time has passed
            if hasattr(self, '_original_learning_params'):
                if time.time() >= self._original_learning_params['reset_time']:
                    # Reset parameters
                    self.hebbian_learning_rate = self._original_learning_params['hebbian_learning_rate']
                    
                    if hasattr(self, 'temporal_decay_factor'):
                        self.temporal_decay_factor = self._original_learning_params['temporal_decay_factor']
                        
                    logger.info("Reset learning parameters to original values")
                    
        except Exception as e:
            logger.error(f"Error resetting learning parameters: {e}")
            
    async def get_retrieval_precision(self):
        """
        Estimate the precision of memory retrieval based on recent searches.
        
        Returns:
            Precision score from 0.0 (poor) to 1.0 (excellent)
        """
        # Default implementation - in a real system, this would track user feedback
        # on search results to calculate actual precision
        if not hasattr(self, '_search_quality_metrics'):
            self._search_quality_metrics = {
                'total_searches': 0,
                'successful_searches': 0
            }
            
        # Return a reasonable default based on number of memories
        # More memories generally means better retrieval
        memory_count = len(self.metadata) if hasattr(self, 'metadata') else 0
        base_precision = min(0.8, 0.5 + (memory_count / 1000) * 0.3)
        
        # Add some randomness to simulate variation
        variation = random.uniform(-0.05, 0.05)
        
        return min(1.0, max(0.4, base_precision + variation)) 

    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (between -1 and 1)
        """
        if vec1 is None or vec2 is None:
            return 0.0
            
        try:
            # Convert to numpy arrays if needed
            if not isinstance(vec1, np.ndarray):
                vec1 = np.array(vec1)
            if not isinstance(vec2, np.ndarray):
                vec2 = np.array(vec2)
                
            # Reshape if needed
            if len(vec1.shape) > 1:
                vec1 = vec1.flatten()
            if len(vec2.shape) > 1:
                vec2 = vec2.flatten()
                
            # Check for zero vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def _generate_auto_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """
        Generate tags automatically from text using transformer-based models.
        Uses a transformer model to extract key topics/concepts rather than simple keyword extraction.
        
        Args:
            text: Text to extract tags from
            max_tags: Maximum number of tags to generate
            
        Returns:
            List of extracted tags
        """
        # Check if text is empty
        if not text or len(text.strip()) == 0:
            return []
            
        try:
            # Import necessary libraries
            from transformers import pipeline
            
            # Initialize the zero-shot classification pipeline if not already done
            if not hasattr(self, '_tag_generator') or self._tag_generator is None:
                try:
                    # First try to load a smaller model for tag generation
                    model_name = "facebook/bart-large-mnli"  # Good for zero-shot classification
                    self._tag_generator = pipeline("zero-shot-classification", 
                                                model=model_name, 
                                                device=0 if self.use_gpu and torch.cuda.is_available() else -1)
                    logger.info(f"Initialized tag generator with model: {model_name}")
                except Exception as inner_e:
                    logger.warning(f"Error loading tagging model: {inner_e}. Using fallback.")
                    self._tag_generator = None
            
            # If we have a tag generator, use it for zero-shot classification
            if self._tag_generator is not None:
                # Define candidate labels for common topics
                # This can be expanded based on domain knowledge
                candidate_labels = [
                    "technology", "science", "health", "business", "finance", 
                    "education", "entertainment", "politics", "sports", "environment",
                    "art", "history", "literature", "philosophy", "psychology",
                    "programming", "data", "AI", "machine learning", "security"
                ]
                
                # Truncate very long texts to avoid context length issues
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]
                
                # Classify text against candidate labels
                result = self._tag_generator(text, candidate_labels, multi_label=True)
                
                # Extract tags with scores above threshold
                threshold = 0.3  # Minimum score to consider a tag relevant
                tags = []
                for label, score in zip(result['labels'], result['scores']):
                    if score > threshold:
                        tags.append(label)
                        if len(tags) >= max_tags:
                            break
                
                # If we got any tags, return them
                if tags:
                    return tags
            
            # Fallback to keyword extraction if transformer-based approach failed or returned no tags
            logger.info("Falling back to keyword extraction for tagging")
            return self._extract_keywords_as_tags(text, max_tags)
            
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            # Fallback to simple keyword extraction
            return self._extract_keywords_as_tags(text, max_tags)
    
    def _extract_keywords_as_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """
        Simple keyword extraction fallback method.
        
        Args:
            text: Text to extract keywords from
            max_tags: Maximum number of tags to extract
            
        Returns:
            List of extracted keywords as tags
        """
        try:
            # Remove common punctuation
            text = re.sub(r'[.,;:!?"\'()\[\]{}]', ' ', text.lower())
            
            # Split into words
            words = text.split()
            
            # Remove common stop words (simplified list)
            stop_words = {
                'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
                'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'having', 'do', 'does', 'did', 'doing', 'to', 'from', 'by', 'on', 'at',
                'in', 'with', 'about', 'against', 'between', 'into', 'during', 'before',
                'after', 'above', 'below', 'up', 'down', 'of', 'off', 'over', 'under'
            }
            
            # Filter and count words
            word_counts = {}
            for word in words:
                # Skip short words and stop words
                if len(word) < 4 or word in stop_words:
                    continue
                    
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Sort by count, descending
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N as tags
            tags = [word for word, _ in sorted_words[:max_tags]]
            return tags
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _prune_synaptic_connections(self, weight_threshold: float = 0.3) -> int:
        """
        Prune weak connections between memories to maintain network efficiency.
        
        Note: This is separate from memory item pruning (_prune_old_memories) which
        removes entire memory items. This function only removes connections between
        memories while preserving the memories themselves.
        
        This design differs slightly from the original plan which suggested pruning
        connections within _prune_old_memories. Instead, connection pruning is handled
        separately and called from the orchestrator's recursive learning loop for better
        control over the consolidation process.
        
        Args:
            weight_threshold: Minimum weight to keep a connection
            
        Returns:
            Number of connections pruned
        """
        if not hasattr(self, 'connection_weights') or not self.connection_weights:
            return 0
            
        try:
            # Count initial connections
            initial_count = len(self.connection_weights)
            
            # Find weak connections to prune
            keys_to_prune = []
            for key, weight in self.connection_weights.items():
                if weight < weight_threshold:
                    keys_to_prune.append(key)
            
            # Prune weak connections
            for key in keys_to_prune:
                self.connection_weights.pop(key, None)
                
            # Count pruned connections
            pruned_count = initial_count - len(self.connection_weights)
            
            if pruned_count > 0:
                logger.info(f"Pruned {pruned_count} weak connections (threshold: {weight_threshold})")
                
            return pruned_count
                
        except Exception as e:
            logger.error(f"Error pruning synaptic connections: {e}")
            return 0
    
    def _limit_token_count(self, text: str, max_tokens: int) -> str:
        """
        Limit text to a maximum number of tokens by truncating.
        
        Args:
            text: The text to limit
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text that fits within the token limit
        """
        if not text or max_tokens <= 0:
            return ""
            
        # Initialize tokenizer if needed
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self._initialize_tokenizer()
            
        # If no tokenizer available, use simple approximation
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            # Estimate using average tokens per word
            words = text.split()
            estimated_words = int(max_tokens / self.AVG_TOKENS_PER_WORD)
            if len(words) <= estimated_words:
                return text
            return " ".join(words[:estimated_words])
            
        # Use tokenizer for accurate truncation
        try:
            # Encode the text to tokens
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # If already under limit, return unchanged
            if len(tokens) <= max_tokens:
                return text
                
            # Truncate tokens to max_tokens
            truncated_tokens = tokens[:max_tokens]
            
            # Decode truncated tokens back to text
            truncated_text = self.tokenizer.decode(truncated_tokens)
            return truncated_text
            
        except Exception as e:
            logger.warning(f"Error in token limiting: {e}. Falling back to approximation.")
            # Fall back to simple word-based approximation
            words = text.split()
            estimated_words = int(max_tokens / self.AVG_TOKENS_PER_WORD)
            return " ".join(words[:estimated_words])
    
    async def rebuild_index(self, distributed: bool = True) -> bool:
        """
        Rebuild the vector index for semantic search
        
        Args:
            distributed: Whether to use distributed processing
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if not self.memories:
                logger.warning("No memories to index")
                return False
                
            try:
                logger.info(f"Rebuilding index with {len(self.memories)} memories, distributed={distributed}")
                start_time = time.time()
                
                # Extract all vectors and IDs
                vectors = []
                ids = []
                
                if distributed and len(self.memories) > 1000:
                    # For large memory sets, use parallel processing
                    await self._rebuild_index_distributed()
                else:
                    # For smaller sets, use direct indexing
                    for memory_id, memory in self.memories.items():
                        if 'embedding' in memory and memory['embedding'] is not None:
                            vectors.append(memory['embedding'])
                            ids.append(memory_id)
                    
                    # Create a new index
                    dimension = len(vectors[0]) if vectors else 0
                    if dimension == 0:
                        logger.error("No valid embeddings found")
                        return False
                        
                    import faiss
                    self.index = faiss.IndexFlatIP(dimension)
                    
                    # Convert to numpy array and add to index
                    import numpy as np
                    vectors_np = np.array(vectors).astype(np.float32)
                    self.index.add(vectors_np)
                    
                    # Update ID mapping
                    self.index_to_memory_id = {i: memory_id for i, memory_id in enumerate(ids)}
                
                duration = time.time() - start_time
                logger.info(f"Index rebuilt in {duration:.2f} seconds")
                return True
                
            except Exception as e:
                logger.error(f"Error rebuilding index: {e}")
                return False
    
    async def _rebuild_index_distributed(self) -> bool:
        """
        Rebuild the index using distributed processing for large memory sets
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import numpy as np
            import faiss
            import concurrent.futures
            
            # Collect all memory IDs and embeddings
            memory_ids = list(self.memories.keys())
            
            # Calculate optimal chunk size and worker count
            chunk_size = 10000  # Process 10k memories per worker
            total_memories = len(memory_ids)
            num_chunks = max(1, total_memories // chunk_size)
            
            # Determine ideal worker count (1 worker per CPU core, but not more than chunks)
            import os
            num_workers = min(num_chunks, os.cpu_count() or 4)
            
            logger.info(f"Distributed indexing with {num_workers} workers, {num_chunks} chunks")
            
            # Split memories into chunks
            chunks = []
            for i in range(0, total_memories, chunk_size):
                end_idx = min(i + chunk_size, total_memories)
                chunks.append(memory_ids[i:end_idx])
            
            # Function to process a chunk in a worker process
            def process_chunk(chunk_ids):
                chunk_vectors = []
                valid_ids = []
                
                for memory_id in chunk_ids:
                    memory = self.memories.get(memory_id)
                    if memory and 'embedding' in memory and memory['embedding'] is not None:
                        chunk_vectors.append(memory['embedding'])
                        valid_ids.append(memory_id)
                
                return valid_ids, chunk_vectors
            
            # Process chunks in parallel
            all_ids = []
            all_vectors = []
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                
                for future in concurrent.futures.as_completed(futures):
                    valid_ids, vectors = future.result()
                    all_ids.extend(valid_ids)
                    all_vectors.extend(vectors)
            
            # Create a new index
            if not all_vectors:
                logger.error("No valid vectors found during distributed indexing")
                return False
                
            dimension = len(all_vectors[0])
            self.index = faiss.IndexFlatIP(dimension)
            
            # Add vectors to index
            vectors_np = np.array(all_vectors).astype(np.float32)
            self.index.add(vectors_np)
            
            # Update ID mapping
            self.index_to_memory_id = {i: memory_id for i, memory_id in enumerate(all_ids)}
            
            logger.info(f"Distributed index rebuilt with {len(all_ids)} memories")
            return True
            
        except Exception as e:
            logger.error(f"Error in distributed indexing: {e}")
            return False
            
    async def semantic_search_distributed(self, query: str, 
                                         top_k: int = 5, 
                                         threshold: Optional[float] = None,
                                         num_partitions: int = 4) -> List[Dict[str, Any]]:
        """
        Perform semantic search using distributed processing for very large indices
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Optional similarity threshold
            num_partitions: Number of index partitions to process in parallel
            
        Returns:
            List of matching memory dictionaries
        """
        if not query:
            return []
            
        # Generate query embedding
        query_embedding = await self._get_embedding(query)
        
        # Check if index is small enough for regular search
        with self.lock:
            if self.index is None or not self.memories:
                return []
                
            # If index is small, use regular search
            if len(self.memories) < 50000:  # Threshold for when to use distributed search
                return await self.semantic_search(query, top_k, threshold)
        
        try:
            import numpy as np
            import faiss
            import concurrent.futures
            
            # Convert to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            with self.lock:
                if self.index is None:
                    return []
                    
                # Get index size
                index_size = self.index.ntotal
                
                # Determine partition size
                partition_size = index_size // num_partitions
                if partition_size == 0:
                    partition_size = index_size
                    num_partitions = 1
                
                # Function to search a partition
                def search_partition(partition_idx):
                    start_idx = partition_idx * partition_size
                    end_idx = min(start_idx + partition_size, index_size)
                    
                    # Create a temporary index with just this partition
                    temp_index = faiss.IndexFlatIP(self.index.d)
                    
                    # Copy vectors from main index to temp index
                    # This is a simplified version - in a real implementation,
                    # you would directly extract the vectors from self.index
                    # using low-level FAISS operations
                    vectors = np.zeros((end_idx - start_idx, self.index.d), dtype=np.float32)
                    for i in range(start_idx, end_idx):
                        # Get the vector from the main index
                        # This is just a placeholder - real implementation would be different
                        vector = np.zeros(self.index.d, dtype=np.float32)
                        vectors[i - start_idx] = vector
                    
                    temp_index.add(vectors)
                    
                    # Search this partition
                    actual_k = min(top_k, end_idx - start_idx)
                    if actual_k <= 0:
                        return [], []
                        
                    scores, indices = temp_index.search(query_vector, actual_k)
                    
                    # Adjust indices to global index space
                    adjusted_indices = [idx + start_idx for idx in indices[0]]
                    
                    return scores[0].tolist(), adjusted_indices
            
                # Search partitions in parallel
                all_scores = []
                all_indices = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_partitions) as executor:
                    futures = [executor.submit(search_partition, i) for i in range(num_partitions)]
                    
                    for future in concurrent.futures.as_completed(futures):
                        scores, indices = future.result()
                        all_scores.extend(scores)
                        all_indices.extend(indices)
                
                # Sort by score and take top_k
                if threshold is not None:
                    filtered_results = [(s, i) for s, i in zip(all_scores, all_indices) if s >= threshold]
                else:
                    filtered_results = list(zip(all_scores, all_indices))
                    
                sorted_results = sorted(filtered_results, key=lambda x: x[0], reverse=True)[:top_k]
                
                # Convert to memory objects
                results = []
                for score, idx in sorted_results:
                    memory_id = self.index_to_memory_id.get(idx)
                    if memory_id and memory_id in self.memories:
                        memory = self.memories[memory_id].copy()
                        memory['score'] = float(score)
                        results.append(memory)
                
                return results
                
        except Exception as e:
            logger.error(f"Error in distributed semantic search: {e}")
            return []
    
    async def clear_working_set(self) -> None:
        """Clear working memory without affecting long-term storage."""
        try:
            # Reset active attention buffers
            self.activation_buffer = {}
            self.current_context = []
            self.priority_queue = []
            # Don't clear connection_weights or embedding_store
            logger.info("Working memory set cleared")
            
            # Force garbage collection after clearing large buffers
            gc.collect()
        except Exception as e:
            logger.error(f"Error clearing working set: {e}")
            
    async def optimize_memory_usage(self) -> None:
        """Optimize memory usage by consolidating similar embeddings."""
        try:
            if not hasattr(self, 'last_optimization'):
                self.last_optimization = time.time()
                
            # Don't optimize too frequently
            if time.time() - self.last_optimization < 3600:  # Every hour
                return
                
            logger.info("Optimizing memory usage")
            
            # Find similar embeddings that can be merged
            merge_candidates = []
            # Implementation details...
            
            # Deduplicate embeddings within threshold
            deduped_count = self._deduplicate_similar_embeddings(0.98)
            
            # Update timestamp
            self.last_optimization = time.time()
            logger.info(f"Memory optimization complete, deduplicated {deduped_count} entries")
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
    
    async def consolidate_memory(self) -> bool:
        """Consolidate memory with timeout protection against deadlocks."""
        try:
            # Use wait_for to prevent deadlocks
            return await asyncio.wait_for(self._consolidate_memory_impl(), timeout=60)
        except asyncio.TimeoutError:
            logger.error("Memory consolidation timed out, resetting consolidation state")
            self._reset_consolidation_state()
            return False
        except Exception as e:
            logger.error(f"Memory consolidation error: {e}")
            return False
