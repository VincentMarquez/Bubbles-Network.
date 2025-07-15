import asyncio
import json
import logging
import time
import numpy as np
import hashlib
from typing import Dict, Optional, List, Tuple, Any
from collections import deque, OrderedDict
from bubbles_core import UniversalBubble, Actions, Event, UniversalCode, Tags, SystemContext, logger, EventService

try:
    from transformers import AutoTokenizer, AutoModel
    import faiss
    import torch
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    AutoTokenizer = object
    AutoModel = object
    faiss = None
    torch = None
    logger.warning("transformers, faiss-python, or torch not found. RAGBubble will be disabled.")

class RAGBubble(UniversalBubble):
    """Enhanced Retrieval-Augmented Generation with transformer-based embeddings."""
    
    def __init__(self, 
                 object_id: str, 
                 context: SystemContext, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 max_contexts: int = 1000,
                 embedding_batch_size: int = 32,
                 cache_embeddings: bool = True,
                 relevance_threshold: float = 0.7,
                 max_cache_size: int = 500,
                 context_max_age_hours: float = 2.0,
                 deduplication_threshold: float = 0.95,
                 **kwargs):
        
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Configuration
        self.max_contexts = max_contexts
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.cache_embeddings = cache_embeddings
        self.relevance_threshold = relevance_threshold
        self.max_cache_size = max_cache_size
        self.context_max_age_hours = context_max_age_hours
        self.deduplication_threshold = deduplication_threshold
        
        # Core components
        self.contexts = []  # List of (text, metadata) tuples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if RAG_AVAILABLE else None
        self.tokenizer = None
        self.model = None
        self.index = None
        self.dimension = None  # To be set dynamically
        
        # Performance optimization
        self.embedding_cache = OrderedDict() if cache_embeddings else None
        self._pending_correlations = {}  # Track correlation IDs for response forwarding
        self.context_history = deque(maxlen=max_contexts)
        
        # Metrics tracking
        self.metrics = {
            "queries_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_contexts_used": 0.0,
            "total_augmentations": 0,
            "embedding_time_ms": deque(maxlen=100),
            "search_time_ms": deque(maxlen=100),
            "query_errors": 0,
            "contexts_stored": 0,
            "contexts_removed": 0,
            "duplicates_removed": 0
        }
        
        # Initialize transformer model
        if RAG_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
                self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
                self.dimension = self.model.config.hidden_size
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info(f"{self.object_id}: Initialized RAGBubble with {embedding_model} transformer on {self.device} (dimension: {self.dimension}).")
            except Exception as e:
                logger.error(f"{self.object_id}: Failed to initialize transformer: {e}", exc_info=True)
                self.device = None
                self.tokenizer = None
                self.model = None
                self.index = None
                self.dimension = None
        else:
            logger.error(f"{self.object_id}: RAG dependencies unavailable, switching to placeholder mode.")
        
        # Start background tasks
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Enhanced RAGBubble initialized with max_contexts={max_contexts}, relevance_threshold={relevance_threshold}")

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate transformer embeddings for a list of texts."""
        if not RAG_AVAILABLE or not self.model:
            raise ValueError("RAG dependencies or model not available.")
        
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"{self.object_id}: Error embedding texts: {e}", exc_info=True)
            raise

    async def _get_embedding_cached(self, text: str) -> np.ndarray:
        """Get embedding with caching support."""
        if not self.cache_embeddings:
            return self._embed_text([text])[0]
        
        # Generate cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache and move to end if hit (LRU)
        if cache_key in self.embedding_cache:
            self.metrics["cache_hits"] += 1
            embedding = self.embedding_cache.pop(cache_key)
            self.embedding_cache[cache_key] = embedding  # Move to end
            return embedding
        
        self.metrics["cache_misses"] += 1
        
        # Generate embedding
        embedding = self._embed_text([text])[0]
        
        # Cache with LRU eviction
        if len(self.embedding_cache) >= self.max_cache_size:
            self.embedding_cache.popitem(last=False)  # Remove least recently used
        self.embedding_cache[cache_key] = embedding
        
        return embedding

    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.ACTION_TAKEN, self.handle_event)
            await EventService.subscribe(Actions.LLM_QUERY, self.handle_event)
            await EventService.subscribe(Actions.LLM_RESPONSE, self.handle_event)
            await EventService.subscribe(Actions.GET_STATUS, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, ACTION_TAKEN, LLM_QUERY, LLM_RESPONSE, GET_STATUS")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    async def handle_event(self, event: Event):
        """Handle events to store contexts or augment LLM queries."""
        try:
            if event.type == Actions.SYSTEM_STATE_UPDATE and isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
                await self.store_state_context(event.data.value, event.data.metadata)
            elif event.type == Actions.ACTION_TAKEN and isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
                await self.store_action_context(event.data.value, event.data.metadata)
            elif event.type == Actions.LLM_QUERY and isinstance(event.data, UniversalCode) and event.data.tag == Tags.STRING:
                await self.augment_llm_query(event)
            elif event.type == Actions.LLM_RESPONSE:
                await self._handle_llm_response(event)
            elif event.type == Actions.GET_STATUS and event.origin == "system":
                await self._handle_status_request(event)
            
            await super().handle_event(event)
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event {event.type}: {e}", exc_info=True)
            self.metrics["query_errors"] += 1

    async def store_state_context(self, state: Dict, metadata: Dict):
        """Store system state as a context for retrieval."""
        if not RAG_AVAILABLE or not self.model or not self.index:
            return
        
        try:
            state_text = (
                f"System State (ts: {state.get('timestamp', 0):.2f}): "
                f"Energy={state.get('energy', 0):.1f}, "
                f"CPU={state.get('cpu_percent', 0):.1f}%, "
                f"Memory={state.get('memory_percent', 0):.1f}%, "
                f"Bubbles={state.get('num_bubbles', 0)}, "
                f"LLM Response Time={state.get('metrics', {}).get('avg_llm_response_time_ms', 0):.1f}ms"
            )
            
            # Get embedding (potentially cached)
            embedding = await self._get_embedding_cached(state_text)
            
            # Store context
            self.contexts.append((state_text, {"type": "state", "timestamp": state.get("timestamp", 0)}))
            self.index.add(np.array([embedding], dtype=np.float32))
            self.context_history.append(state_text)
            self.metrics["contexts_stored"] += 1
            
            # Manage size
            if len(self.contexts) > self.max_contexts:
                removed = self.contexts.pop(0)
                self.metrics["contexts_removed"] += 1
                await self._rebuild_index_async()
                
            logger.debug(f"{self.object_id}: Stored state context (ts: {state.get('timestamp', 0):.2f})")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error storing state context: {e}", exc_info=True)

    async def store_action_context(self, action: Dict, metadata: Dict):
        """Store action as a context for retrieval."""
        if not RAG_AVAILABLE or not self.model or not self.index:
            return
        
        try:
            action_text = (
                f"Action Taken (ts: {metadata.get('timestamp', 0):.2f}): "
                f"Type={action.get('action_type', 'UNKNOWN')}, "
                f"Payload={json.dumps(action.get('payload', {}))[:200]}"  # Limit payload size
            )
            
            # Get embedding (potentially cached)
            embedding = await self._get_embedding_cached(action_text)
            
            # Store context
            self.contexts.append((action_text, {"type": "action", "timestamp": metadata.get("timestamp", 0)}))
            self.index.add(np.array([embedding], dtype=np.float32))
            self.context_history.append(action_text)
            self.metrics["contexts_stored"] += 1
            
            # Manage size
            if len(self.contexts) > self.max_contexts:
                removed = self.contexts.pop(0)
                self.metrics["contexts_removed"] += 1
                await self._rebuild_index_async()
                
            logger.debug(f"{self.object_id}: Stored action context (ts: {metadata.get('timestamp', 0):.2f})")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error storing action context: {e}", exc_info=True)

    async def augment_llm_query(self, event: Event):
        """Enhanced LLM query augmentation with intelligent context retrieval."""
        if not RAG_AVAILABLE or not self.model or not self.index or not self.contexts:
            await self.context.dispatch_event(event)
            return

        query = event.data.value
        
        # Check for correlation_id in multiple places
        correlation_id = None
        if hasattr(event, 'correlation_id') and event.correlation_id:
            correlation_id = event.correlation_id
        elif hasattr(event.data, 'metadata') and event.data.metadata:
            correlation_id = event.data.metadata.get("correlation_id")
        
        # Generate fallback if missing
        if not correlation_id:
            correlation_id = hashlib.md5(query.encode()).hexdigest()[:8]
            logger.debug(f"{self.object_id}: Generated fallback correlation_id: {correlation_id}")

        # Log processing
        logger.debug(f"{self.object_id}: Processing LLM_QUERY with correlation_id: {correlation_id}")

        # Check flood control if available
        if hasattr(self.context, '_llm_request_controller'):
            permission = self.context._llm_request_controller.should_allow_request(
                query, 
                self.object_id, 
                priority=3
            )
            if not permission["allowed"]:
                logger.warning(f"{self.object_id}: Query blocked by flood control: {permission['reason']}")
                await self.context.dispatch_event(event)
                return

        try:
            start_time = time.time()
            
            # Get query embedding (with caching)
            query_embedding = await self._get_embedding_cached(query)
            embedding_time = (time.time() - start_time) * 1000
            self.metrics["embedding_time_ms"].append(embedding_time)
            
            # Determine number of contexts based on query complexity
            k = self._determine_k_value(query)
            
            # Search for relevant contexts
            search_start = time.time()
            distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k=min(k, len(self.contexts)))
            search_time = (time.time() - search_start) * 1000
            self.metrics["search_time_ms"].append(search_time)
            
            # Enhanced relevance filtering with scoring
            relevant_contexts = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.contexts):
                    context_text, metadata = self.contexts[idx]
                    
                    # Convert L2 distance to similarity score
                    similarity = 1 / (1 + dist)
                    
                    if similarity >= self.relevance_threshold:
                        relevant_contexts.append({
                            'text': context_text,
                            'metadata': metadata,
                            'similarity': similarity,
                            'recency_score': self._calculate_recency_score(metadata)
                        })
            
            # Sort by combined score (relevance + recency)
            relevant_contexts.sort(
                key=lambda x: x['similarity'] * 0.7 + x['recency_score'] * 0.3, 
                reverse=True
            )
            
            # Take top contexts
            top_contexts = relevant_contexts[:3]
            
            # Format augmented query
            if top_contexts:
                context_texts = []
                for ctx in top_contexts:
                    context_texts.append(f"[Relevance: {ctx['similarity']:.2f}] {ctx['text']}")
                
                augmented_query = (
                    f"System Context:\n{chr(10).join(context_texts)}\n\n"
                    f"Query:\n{query}"
                )
                
                logger.debug(f"{self.object_id}: Augmented query with {len(top_contexts)} contexts (avg relevance: {np.mean([c['similarity'] for c in top_contexts]):.2f})")
            else:
                logger.info(f"{self.object_id}: No relevant context found (threshold: {self.relevance_threshold})")
                augmented_query = query
            
            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["total_augmentations"] += len(top_contexts)
            self.metrics["average_contexts_used"] = self.metrics["total_augmentations"] / max(1, self.metrics["queries_processed"])
            
            # Check cache for augmented query
            if hasattr(self.context, 'response_cache') and self.context.response_cache:
                cached_response = await self.context.response_cache.get(augmented_query)
                if cached_response:
                    if correlation_id:
                        await self.context.send_response(correlation_id, cached_response)
                        logger.info(f"{self.object_id}: Sent cached response via correlation_id")
                    else:
                        response_data = {"response": cached_response, "cached": True}
                        response_uc = UniversalCode(Tags.DICT, response_data)
                        response_event = Event(
                            type=Actions.LLM_RESPONSE, 
                            data=response_uc, 
                            origin=self.object_id, 
                            priority=2,
                            metadata={"original_query": query, "query_origin": event.origin}
                        )
                        await self.context.dispatch_event(response_event)
                        logger.info(f"{self.object_id}: Dispatched cached LLM_RESPONSE event")
                    return

            # Create augmented event
            augmented_metadata = event.data.metadata.copy() if event.data.metadata else {}
            augmented_metadata["contexts_used"] = len(top_contexts)
            augmented_metadata["augmentation_time_ms"] = (time.time() - start_time) * 1000
            
            augmented_uc = UniversalCode(Tags.STRING, augmented_query, metadata=augmented_metadata)
            
            augmented_event = Event(
                type=Actions.LLM_QUERY, 
                data=augmented_uc, 
                origin=event.origin, 
                priority=event.priority,
                correlation_id=correlation_id
            )
            
            # Setup response handling if needed
            if correlation_id:
                self._setup_response_handler(correlation_id, event.origin, query, augmented_query)
            
            await self.context.dispatch_event(augmented_event)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error augmenting LLM query: {e}", exc_info=True)
            self.metrics["query_errors"] += 1
            await self.context.dispatch_event(event)

    def _determine_k_value(self, query: str) -> int:
        """Dynamically determine number of contexts based on query complexity."""
        query_lower = query.lower()
        query_length = len(query.split())
        
        # Check for comparison queries
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs", "between"]):
            return 5
        
        # Check for complex queries
        if any(word in query_lower for word in ["explain", "analyze", "describe", "how", "why"]):
            return 4
        
        # Based on query length
        if query_length < 5:
            return 2  # Simple query
        elif query_length < 15:
            return 3  # Medium query
        else:
            return 4  # Complex query

    def _calculate_recency_score(self, metadata: dict) -> float:
        """Calculate recency score for context ranking."""
        current_time = time.time()
        age_minutes = (current_time - metadata.get('timestamp', 0)) / 60
        
        # Exponential decay - contexts lose half their recency value every 30 minutes
        return 0.5 ** (age_minutes / 30)

    def _setup_response_handler(self, correlation_id: str, original_origin: str, original_query: str, augmented_query: str):
        """Set up handling for the LLM response to forward it back."""
        self._pending_correlations[correlation_id] = {
            'origin': original_origin,
            'timestamp': time.time(),
            'original_query': original_query,
            'augmented_query': augmented_query
        }
        
        # Clean up old correlations (older than 60 seconds)
        current_time = time.time()
        self._pending_correlations = {
            cid: info for cid, info in self._pending_correlations.items() 
            if current_time - info['timestamp'] < 60
        }

    async def _handle_llm_response(self, event: Event):
        """Handle LLM responses to forward them to waiting callers and cache results."""
        if not hasattr(event, 'correlation_id') or not event.correlation_id:
            return
        
        if event.correlation_id in self._pending_correlations:
            correlation_info = self._pending_correlations.pop(event.correlation_id)
            
            # Extract the response
            response = None
            if hasattr(event.data, 'value'):
                if isinstance(event.data.value, dict):
                    response = event.data.value.get('response', str(event.data.value))
                else:
                    response = str(event.data.value)
            
            # Cache the response
            if response and hasattr(self.context, 'response_cache') and self.context.response_cache and 'augmented_query' in correlation_info:
                try:
                    await self.context.response_cache.put(correlation_info['augmented_query'], response)
                    logger.debug(f"{self.object_id}: Cached augmented query response")
                except Exception as e:
                    logger.warning(f"{self.object_id}: Failed to cache response: {e}")
            
            # Forward the response
            if response:
                await self.context.send_response(event.correlation_id, response)
                logger.debug(f"{self.object_id}: Forwarded LLM response for correlation {event.correlation_id[:8]}")

    async def _handle_status_request(self, event: Event):
        """Handle status requests with performance report."""
        report = self.get_performance_report()
        await self.send_response_to_event(event, report)

    def get_performance_report(self) -> dict:
        """Get detailed performance metrics."""
        avg_embedding_time = np.mean(self.metrics["embedding_time_ms"]) if self.metrics["embedding_time_ms"] else 0
        avg_search_time = np.mean(self.metrics["search_time_ms"]) if self.metrics["search_time_ms"] else 0
        cache_hit_rate = self.metrics["cache_hits"] / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
        
        return {
            "bubble_id": self.object_id,
            "queries_processed": self.metrics["queries_processed"],
            "query_errors": self.metrics["query_errors"],
            "error_rate": self.metrics["query_errors"] / max(1, self.metrics["queries_processed"]),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "average_contexts_per_query": round(self.metrics["average_contexts_used"], 2),
            "average_embedding_time_ms": round(avg_embedding_time, 2),
            "average_search_time_ms": round(avg_search_time, 2),
            "total_contexts_stored": len(self.contexts),
            "contexts_added": self.metrics["contexts_stored"],
            "contexts_removed": self.metrics["contexts_removed"],
            "duplicates_removed": self.metrics["duplicates_removed"],
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_cache_size": len(self.embedding_cache) if self.embedding_cache else 0,
            "pending_correlations": len(self._pending_correlations),
            "relevance_threshold": self.relevance_threshold,
            "max_contexts": self.max_contexts
        }

    async def _rebuild_index_async(self):
        """Asynchronously rebuild the FAISS index with error handling."""
        if not self.contexts or not RAG_AVAILABLE:
            return
        
        try:
            start_time = time.time()
            new_index = faiss.IndexFlatL2(self.dimension)
            
            # Process in batches to avoid memory issues
            for i in range(0, len(self.contexts), self.embedding_batch_size):
                batch_texts = [ctx[0] for ctx in self.contexts[i:i + self.embedding_batch_size]]
                embeddings = self._embed_text(batch_texts)
                new_index.add(embeddings)
            
            self.index = new_index
            rebuild_time = time.time() - start_time
            logger.debug(f"{self.object_id}: Index rebuilt with {len(self.contexts)} contexts in {rebuild_time:.2f}s")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to rebuild index: {e}")

    async def _incremental_maintenance(self):
        """Lightweight incremental maintenance."""
        if not RAG_AVAILABLE or not self.contexts:
            return
        
        current_time = time.time()
        max_age_seconds = self.context_max_age_hours * 3600
        
        # Remove old contexts (check all)
        contexts_to_remove = [i for i, (text, meta) in enumerate(self.contexts) if current_time - meta['timestamp'] > max_age_seconds]
        
        if contexts_to_remove:
            for i in sorted(contexts_to_remove, reverse=True):
                del self.contexts[i]
                self.metrics["contexts_removed"] += 1
            
            await self._rebuild_index_async()
            logger.debug(f"{self.object_id}: Removed {len(contexts_to_remove)} old contexts")

    async def _full_maintenance(self):
        """Comprehensive maintenance with deduplication and optimization."""
        if not RAG_AVAILABLE or not self.contexts:
            return
        
        current_time = time.time()
        max_age_seconds = self.context_max_age_hours * 3600
        
        # Remove old contexts first
        self.contexts = [(text, meta) for text, meta in self.contexts if current_time - meta['timestamp'] <= max_age_seconds]
        self.metrics["contexts_removed"] += len(self.contexts) - len(self.contexts)  # Update count
        
        # Deduplicate using FAISS
        dedup_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        unique_contexts = []
        duplicates_found = 0
        
        for text, meta in self.contexts:
            embedding = await self._get_embedding_cached(text)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize for cosine
            D, I = dedup_index.search(np.array([embedding]), k=1)
            
            if D[0][0] > self.deduplication_threshold:
                duplicates_found += 1
                continue
            
            unique_contexts.append((text, meta))
            dedup_index.add(np.array([embedding]))
        
        # Update contexts if changed
        if len(unique_contexts) < len(self.contexts):
            self.contexts = unique_contexts
            self.metrics["duplicates_removed"] += duplicates_found
            await self._rebuild_index_async()
            logger.info(f"{self.object_id}: Full maintenance complete. Contexts: {len(self.contexts)}, Duplicates removed: {duplicates_found}")

    async def autonomous_step(self):
        """Enhanced maintenance with intelligent pruning."""
        await super().autonomous_step()
        
        # More frequent but lighter maintenance
        if self.execution_count % 30 == 0:
            await self._incremental_maintenance()
        
        # Full maintenance less frequently
        if self.execution_count % 300 == 0:
            await self._full_maintenance()
        
        # Clean up pending correlations periodically
        if self.execution_count % 60 == 0:
            current_time = time.time()
            old_count = len(self._pending_correlations)
            self._pending_correlations = {
                cid: info for cid, info in self._pending_correlations.items() 
                if current_time - info['timestamp'] < 60
            }
            if old_count > len(self._pending_correlations):
                logger.debug(f"{self.object_id}: Cleaned {old_count - len(self._pending_correlations)} expired correlations")
        
        # Log performance metrics periodically
        if self.execution_count % 600 == 0:  # Every ~5 minutes
            report = self.get_performance_report()
            logger.info(f"{self.object_id}: Performance Report - "
                       f"Queries: {report['queries_processed']}, "
                       f"Cache Hit Rate: {report['cache_hit_rate']:.1%}, "
                       f"Avg Contexts: {report['average_contexts_per_query']:.1f}, "
                       f"Contexts Stored: {report['total_contexts_stored']}")
        
        await asyncio.sleep(1.0)  # Adaptive: could be made configurable
