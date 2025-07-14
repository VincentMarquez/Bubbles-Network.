# APEPBubble.py
# Full implementation of APEP v2.8.3 with code refinement for the Bubbles system
# ADAPTIVE PROMPT EVOLUTION PROTOCOL

import asyncio
import json
import time
import uuid
import logging
import re
import hashlib
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum

from bubbles_core import (
    UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions,
    logger, EventService, robust_json_parse, extract_code
)

class APEPMode(Enum):
    """APEP operational modes."""
    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"

class PromptModificationTechnique(Enum):
    """APEP Foundational Five + Advanced Techniques."""
    # Foundational Five
    CLARITY_SPECIFICITY = "clarity_specificity"
    CONSTRAINT_ADDITION = "constraint_addition"
    FEW_SHOT_PROMPTING = "few_shot_prompting"
    OUTPUT_STRUCTURING = "output_structuring"
    PERSONA_ASSIGNMENT = "persona_assignment"
    
    # Advanced Techniques
    CHAIN_OF_THOUGHT = "chain_of_thought"
    NEGATIVE_PROMPTING = "negative_prompting"
    CONTEXTUAL_ENRICHMENT = "contextual_enrichment"
    TONE_CONTROL = "tone_control"
    DELIMITERS = "delimiters"
    PERSPECTIVE_SHIFTING = "perspective_shifting"
    CONFIDENCE_LEVELS = "confidence_levels"
    PRIMING_STEPS = "priming_steps"

class CodeRefinementTechnique(Enum):
    """APEP techniques specifically for code refinement."""
    SAFETY_CHECKS = "safety_checks"
    ERROR_HANDLING = "error_handling"
    DOCUMENTATION = "documentation"
    STRUCTURE_IMPROVEMENT = "structure_improvement"
    RESOURCE_AWARENESS = "resource_awareness"
    BUBBLE_INTEGRATION = "bubble_integration"

class APEPBubble(UniversalBubble):
    """
    Implements the full APEP v2.8.3 protocol for systematic prompt and code optimization.
    Selectively processes high-value prompts and refines code responses for safety and reliability.
    """
    
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # APEP Configuration
        self.mode = APEPMode(kwargs.get("mode", "fully_automated"))
        self.max_iterations = kwargs.get("max_iterations", 3)
        self.min_performance_threshold = kwargs.get("min_threshold", 0.7)
        
        # Performance optimization
        self.cache_enabled = kwargs.get("cache_enabled", True)
        self.prompt_cache = {}  # hash -> refined prompt
        self.code_cache = {}    # hash -> refined code
        self.performance_history = defaultdict(list)  # bubble_id -> scores
        
        # Selective processing configuration
        self.high_value_bubbles = {
            "creativesynthesis_bubble", 
            "metareasoning_bubble",
            "autogen_bubble"
        }
        self.always_process_keywords = {
            "optimize", "novel", "creative", "analyze", "propose", "code", "implement"
        }
        
        # Code refinement configuration
        self.refine_code_responses = kwargs.get("refine_code", True)
        self.code_safety_level = kwargs.get("code_safety", "high")
        
        # APEP State
        self.current_iteration = 0
        self.baseline_scores = {}
        self.variant_history = []
        self.learning_log = []
        
        # Technique effectiveness tracking
        self.technique_scores = defaultdict(lambda: {"success": 0, "total": 0})
        self.code_technique_scores = defaultdict(lambda: {"success": 0, "total": 0})
        
        # Initialize toolboxes
        self.toolbox = self._initialize_toolbox()
        self.code_toolbox = self._initialize_code_toolbox()
        
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: APEP v2.8.3 initialized in {self.mode.value} mode with code refinement")

    def _initialize_toolbox(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the APEP Prompt Modification Toolbox."""
        return {
            PromptModificationTechnique.CLARITY_SPECIFICITY: {
                "enabled": True,
                "priority": 1,
                "applicability_check": lambda p, m: len(p) > 50,
                "apply_function": self._apply_clarity_specificity
            },
            PromptModificationTechnique.CONSTRAINT_ADDITION: {
                "enabled": True,
                "priority": 2,
                "applicability_check": lambda p, m: "optimize" in p.lower() or "action" in p.lower(),
                "apply_function": self._apply_constraint_addition
            },
            PromptModificationTechnique.FEW_SHOT_PROMPTING: {
                "enabled": True,
                "priority": 3,
                "applicability_check": lambda p, m: m.get("origin") == "creativesynthesis_bubble",
                "apply_function": self._apply_few_shot
            },
            PromptModificationTechnique.OUTPUT_STRUCTURING: {
                "enabled": True,
                "priority": 2,
                "applicability_check": lambda p, m: any(kw in p.lower() for kw in ["propose", "analyze", "generate"]),
                "apply_function": self._apply_output_structuring
            },
            PromptModificationTechnique.PERSONA_ASSIGNMENT: {
                "enabled": True,
                "priority": 1,
                "applicability_check": lambda p, m: not p.lower().startswith("you are"),
                "apply_function": self._apply_persona
            },
            PromptModificationTechnique.CHAIN_OF_THOUGHT: {
                "enabled": True,
                "priority": 4,
                "applicability_check": lambda p, m: any(kw in p.lower() for kw in ["analyze", "reason", "think", "optimize"]),
                "apply_function": self._apply_chain_of_thought
            },
            PromptModificationTechnique.NEGATIVE_PROMPTING: {
                "enabled": True,
                "priority": 5,
                "applicability_check": lambda p, m: m.get("origin") in self.high_value_bubbles,
                "apply_function": self._apply_negative_prompting
            },
            PromptModificationTechnique.CONTEXTUAL_ENRICHMENT: {
                "enabled": True,
                "priority": 3,
                "applicability_check": lambda p, m: "state" in p.lower() or "metrics" in p.lower(),
                "apply_function": self._apply_contextual_enrichment
            },
            PromptModificationTechnique.TONE_CONTROL: {
                "enabled": True,
                "priority": 6,
                "applicability_check": lambda p, m: "explain" in p.lower() or "describe" in p.lower(),
                "apply_function": self._apply_tone_control
            },
            PromptModificationTechnique.DELIMITERS: {
                "enabled": True,
                "priority": 7,
                "applicability_check": lambda p, m: len(p) > 100 and "\n" in p,  # For multi-part prompts
                "apply_function": self._apply_delimiters
            },
            PromptModificationTechnique.PERSPECTIVE_SHIFTING: {
                "enabled": True,
                "priority": 8,
                "applicability_check": lambda p, m: "analyze" in p.lower() or m.get("origin") == "metareasoning_bubble",
                "apply_function": self._apply_perspective_shifting
            },
            PromptModificationTechnique.CONFIDENCE_LEVELS: {
                "enabled": True,
                "priority": 9,
                "applicability_check": lambda p, m: any(kw in p.lower() for kw in ["decide", "recommend", "predict"]),
                "apply_function": self._apply_confidence_levels
            },
            PromptModificationTechnique.PRIMING_STEPS: {
                "enabled": True,
                "priority": 10,
                "applicability_check": lambda p, m: "complex" in p.lower() or "reason" in p.lower(),
                "apply_function": self._apply_priming_steps
            },
        }

    def _initialize_code_toolbox(self) -> Dict[CodeRefinementTechnique, Dict[str, Any]]:
        """Initialize code refinement techniques."""
        return {
            CodeRefinementTechnique.SAFETY_CHECKS: {
                "enabled": True,
                "priority": 1,
                "apply_function": self._add_safety_checks
            },
            CodeRefinementTechnique.ERROR_HANDLING: {
                "enabled": True,
                "priority": 2,
                "apply_function": self._add_error_handling
            },
            CodeRefinementTechnique.DOCUMENTATION: {
                "enabled": True,
                "priority": 5,
                "apply_function": self._add_documentation
            },
            CodeRefinementTechnique.BUBBLE_INTEGRATION: {
                "enabled": True,
                "priority": 3,
                "apply_function": self._ensure_bubble_compatibility
            },
            CodeRefinementTechnique.RESOURCE_AWARENESS: {
                "enabled": True,
                "priority": 4,
                "apply_function": self._add_resource_awareness
            },
            CodeRefinementTechnique.STRUCTURE_IMPROVEMENT: {
                "enabled": True,
                "priority": 6,
                "apply_function": self._improve_structure
            },
        }

    async def _subscribe_to_events(self):
        """Subscribe to LLM_QUERY events for prompt interception and LLM_RESPONSE for code refinement."""
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.LLM_QUERY, self.handle_event)
            await EventService.subscribe(Actions.LLM_RESPONSE, self.handle_event)
            await EventService.subscribe(Actions.CODE_UPDATE, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to LLM_QUERY, LLM_RESPONSE, and CODE_UPDATE events")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    async def handle_event(self, event: Event):
        """Process events based on APEP protocol."""
        try:
            if event.type == Actions.LLM_QUERY:
                await self._handle_llm_query(event)
            elif event.type == Actions.LLM_RESPONSE:
                await self._handle_llm_response_with_code(event)
            elif event.type == Actions.CODE_UPDATE:
                await self._handle_code_update(event)
            else:
                await super().handle_event(event)
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event: {e}", exc_info=True)

    async def _handle_llm_query(self, event: Event):
        """Intercept and potentially refine LLM queries."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.STRING:
            return
        
        prompt = event.data.value
        metadata = event.data.metadata or {}
        origin = event.origin or "unknown"
        correlation_id = metadata.get("correlation_id", str(uuid.uuid4()))
        
        # Skip if already processed by APEP
        if metadata.get("apep_processed", False):
            return
        
        # Determine if prompt should be processed
        if not self._should_process_prompt(prompt, origin, metadata):
            logger.debug(f"{self.object_id}: Skipping low-value prompt from {origin}")
            return
        
        # Check cache first
        # Add dynamic context to prompt for better caching
        dynamic_context = f"\n\n### Context ###\nTimestamp: {time.time()}, ID: {uuid.uuid4()}"
        prompt_with_context = prompt + dynamic_context
        prompt_hash = self._hash_prompt(prompt_with_context)
        if self.cache_enabled and prompt_hash in self.prompt_cache:
            refined_prompt = self.prompt_cache[prompt_hash]
            logger.info(f"{self.object_id}: Using cached refinement with context for {origin}")
        else:
            # Apply APEP refinement
            start_time = time.time()
            refined_prompt = await self._refine_prompt(prompt_with_context, origin, metadata)
            refinement_time = time.time() - start_time
            
            # Cache if successful
            if self.cache_enabled:
                self.prompt_cache[prompt_hash] = refined_prompt
            
            logger.info(f"{self.object_id}: Refined prompt from {origin} in {refinement_time:.2f}s")
        
        # Re-publish refined query
        refined_metadata = {
            **metadata,
            "apep_processed": True,
            "apep_mode": self.mode.value,
            "original_prompt_hash": prompt_hash,
            "refinement_techniques": self.get_applied_techniques()
        }
        
        refined_uc = UniversalCode(
            Tags.STRING,
            refined_prompt,
            description=f"APEP-refined prompt from {origin}",
            metadata=refined_metadata
        )
        
        refined_event = Event(
            type=Actions.LLM_QUERY,
            data=refined_uc,
            origin=origin,
            priority=event.priority
        )
        
        await self.context.dispatch_event(refined_event)
        
        # Log to learning log
        self._log_refinement(prompt, refined_prompt, origin, metadata)

    async def _handle_llm_response_with_code(self, event: Event):
        """Intercept LLM responses and refine any code blocks."""
        if not isinstance(event.data, UniversalCode) or event.data.tag != Tags.STRING:
            return
        
        response_text = event.data.value
        metadata = event.data.metadata or {}
        origin = event.origin
        
        # Skip if already processed or code refinement disabled
        if metadata.get("apep_code_refined", False) or not self.refine_code_responses:
            return
        
        # Extract code blocks
        code_blocks = extract_code(response_text)
        
        if not code_blocks:
            return  # No code to refine
        
        # Check if this is from a high-value bubble or contains code keywords
        responding_to = metadata.get("response_to", "unknown")
        if responding_to not in self.high_value_bubbles and not any(kw in response_text.lower() for kw in ["code", "implement", "function", "class"]):
            return
        
        logger.info(f"{self.object_id}: Found {len(code_blocks)} code blocks in response from {origin}")
        
        # Refine each code block
        refined_blocks = []
        for i, code_block in enumerate(code_blocks):
            # Check cache
            code_hash = hashlib.sha256(code_block.encode()).hexdigest()
            if self.cache_enabled and code_hash in self.code_cache:
                refined_code = self.code_cache[code_hash]
            else:
                refined_code = await self._refine_code_block(
                    code_block, 
                    responding_to,
                    metadata
                )
                if self.cache_enabled:
                    self.code_cache[code_hash] = refined_code
            
            refined_blocks.append(refined_code)
        
        # Reconstruct response with refined code
        refined_response = self._replace_code_blocks(response_text, code_blocks, refined_blocks)
        
        # Re-publish refined response
        refined_metadata = {
            **metadata,
            "apep_code_refined": True,
            "code_blocks_refined": len(refined_blocks),
            "original_response_hash": hashlib.sha256(response_text.encode()).hexdigest()
        }
        
        refined_uc = UniversalCode(
            Tags.STRING,
            refined_response,
            description=f"APEP-refined code response from {origin}",
            metadata=refined_metadata
        )
        
        refined_event = Event(
            type=Actions.LLM_RESPONSE,
            data=refined_uc,
            origin=origin,
            priority=event.priority
        )
        
        await self.context.dispatch_event(refined_event)

    def _should_process_prompt(self, prompt: str, origin: str, metadata: Dict) -> bool:
        """Determine if a prompt should be processed by APEP."""
        # Always process high-value bubbles
        if origin in self.high_value_bubbles:
            return True
        
        # Check for high-value keywords
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in self.always_process_keywords):
            return True
        
        # Skip user chat unless it's complex
        if origin == "user_chat" and len(prompt) < 200:
            return False
        
        # Skip if explicitly disabled
        if metadata.get("skip_apep", False):
            return False
        
        return True

    async def _refine_prompt(self, prompt: str, origin: str, metadata: Dict) -> str:
        """Apply APEP refinement process to a prompt."""
        # Phase 1: Initialization
        self.current_iteration = 0
        baseline_prompt = prompt
        best_prompt = prompt
        best_score = 0.0
        
        # Phase 2: Analysis
        applicable_techniques = self._identify_applicable_techniques(prompt, metadata)
        
        if not applicable_techniques:
            logger.debug(f"{self.object_id}: No applicable techniques for prompt")
            return prompt
        
        # Phase 3-4: Refinement and Testing
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            # Create variant
            variant_prompt = await self._create_variant(
                baseline_prompt, 
                applicable_techniques,
                origin
            )
            
            # Evaluate variant (simulated in semi-automated mode)
            if self.mode == APEPMode.FULLY_AUTOMATED:
                score = await self._evaluate_variant(variant_prompt, baseline_prompt)
            else:
                score = self._simulate_evaluation(variant_prompt, baseline_prompt)
            
            # Track performance
            self.performance_history[origin].append(score)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_prompt = variant_prompt
            
            # Check convergence
            if best_score >= self.min_performance_threshold:
                break
            
            # Update baseline for next iteration
            baseline_prompt = best_prompt
        
        # Phase 5: Learning
        self._update_technique_scores(applicable_techniques, best_score)
        
        return best_prompt

    def _identify_applicable_techniques(self, prompt: str, metadata: Dict) -> List[PromptModificationTechnique]:
        """Identify which techniques are applicable to this prompt."""
        applicable = []
        
        for technique, config in self.toolbox.items():
            if config["enabled"] and config["applicability_check"](prompt, metadata):
                applicable.append(technique)
        
        # Sort by priority
        applicable.sort(key=lambda t: self.toolbox[t]["priority"])
        
        return applicable[:5]  # Limit to top 5 techniques

    async def _create_variant(self, prompt: str, techniques: List[PromptModificationTechnique], origin: str) -> str:
        """Create a prompt variant using selected techniques."""
        variant = prompt
        
        for technique in techniques:
            try:
                apply_fn = self.toolbox[technique]["apply_function"]
                variant = apply_fn(variant, origin)
                self.technique_scores[technique]["total"] += 1
            except Exception as e:
                logger.error(f"{self.object_id}: Error applying {technique.value}: {e}")
        
        return variant

    def _simulate_evaluation(self, variant: str, baseline: str) -> float:
        """Simulate prompt evaluation for semi-automated mode."""
        # Simple heuristic-based scoring
        score = 0.5  # Base score
        
        # Length penalty (prefer concise but not too short)
        ratio = len(variant) / len(baseline)
        if 0.8 <= ratio <= 1.5:
            score += 0.1
        
        # Structure bonus
        if any(marker in variant for marker in ["###", "Required:", "Step", "FORMAT"]):
            score += 0.15
        
        # Specificity bonus
        if variant.count("\n") > baseline.count("\n"):
            score += 0.1
        
        # Constraint bonus
        if any(word in variant.lower() for word in ["must", "constraint", "required", "avoid"]):
            score += 0.05
        
        # Persona bonus
        if variant.lower().startswith("you are"):
            score += 0.1
        
        return min(1.0, score)

    async def _evaluate_variant(self, variant: str, baseline: str) -> float:
        """Evaluate variant using LLM feedback (for fully automated mode)."""
        # This would query an LLM to evaluate the refined prompt
        # For now, use simulation
        return self._simulate_evaluation(variant, baseline)

    # Prompt refinement technique implementations
    def _apply_clarity_specificity(self, prompt: str, origin: str) -> str:
        """Apply clarity and specificity improvements."""
        # Replace vague terms
        replacements = {
            "analyze": "perform a detailed analysis of",
            "optimize": "identify specific improvements for",
            "suggest": "recommend exactly one",
            "good": "effective and measurable",
            "things": "specific components",
            "improve": "increase efficiency of",
            "help": "provide specific assistance for"
        }
        
        refined = prompt
        for vague, specific in replacements.items():
            refined = refined.replace(vague, specific)
        
        return refined

    def _apply_constraint_addition(self, prompt: str, origin: str) -> str:
        """Add relevant constraints based on context."""
        constraints = []
        
        if "action" in prompt.lower():
            constraints.append("- Action must be one of: CODE_UPDATE, SPAWN_BUBBLE, DESTROY_BUBBLE, NO_OP")
            constraints.append("- Include expected resource consumption (energy, CPU)")
        
        if "optimize" in prompt.lower():
            constraints.append("- Focus on energy efficiency (current < 10000) and response time")
            constraints.append("- Consider current CPU usage before suggesting resource-intensive operations")
        
        if "propose" in prompt.lower():
            constraints.append("- Proposal must include expected impact metrics")
            constraints.append("- Ensure compatibility with existing bubble architecture")
        
        if "code" in prompt.lower():
            constraints.append("- Code must be Python 3.8+ compatible")
            constraints.append("- Include error handling and logging")
            constraints.append("- Follow async/await patterns for I/O operations")
        
        if constraints:
            return f"{prompt}\n\n### CONSTRAINTS ###\n" + "\n".join(constraints)
        
        return prompt

    def _apply_few_shot(self, prompt: str, origin: str) -> str:
        """Add few-shot examples for complex tasks."""
        if "proposal" in prompt.lower():
            return f"""{prompt}

### EXAMPLE OUTPUT ###
Input: System CPU at 85%, high response times
Output: {{
    "proposal_type": "ACTION",
    "payload": {{
        "action_type": "SPAWN_BUBBLE",
        "description": "Add SimpleLLMBubble for load balancing",
        "expected_impact": {{"cpu_percent": -10, "avg_llm_response_time_ms": -200}},
        "narrative": "Distribute LLM queries across multiple bubbles to reduce bottlenecks"
    }},
    "novelty_score": 0.7,
    "confidence": 0.85
}}
"""
        
        if "code" in prompt.lower() and "optimize" in prompt.lower():
            return f"""{prompt}

### EXAMPLE CODE PATTERN ###
```python
async def optimize_bubble_performance(self):
    \"\"\"Optimizes bubble performance based on metrics.\"\"\"
    if self.resource_manager:
        cpu = self.resource_manager.get_resource_level('cpu_percent')
        if cpu > 80:
            await self.reduce_activity()
            logger.info(f"{{self.object_id}}: Reduced activity due to high CPU ({{cpu}}%)")
```"""
        
        return prompt

    def _apply_output_structuring(self, prompt: str, origin: str) -> str:
        """Add output structure requirements."""
        if "propose" in prompt.lower() or "action" in prompt.lower():
            return f"""{prompt}

### REQUIRED OUTPUT FORMAT ###
{{
    "action_type": "<TYPE>",
    "rationale": "<2-3 sentences explaining why>",
    "expected_impact": {{"metric_name": numeric_value}},
    "confidence": 0.0-1.0,
    "risks": ["<potential risk 1>", "<potential risk 2>"]
}}"""
        
        if "analyze" in prompt.lower():
            return f"""{prompt}

### STRUCTURE YOUR ANALYSIS ###

1. **Current State Assessment**
   - Key metrics and their values
   - Identified bottlenecks or issues

2. **Root Cause Analysis**
   - Primary factors contributing to the issue
   - System constraints affecting performance

3. **Recommended Actions**
   - Specific, implementable solutions
   - Priority order for implementation

4. **Expected Outcomes**
   - Quantifiable improvements
   - Timeline for results"""
        
        if "code" in prompt.lower():
            return f"""{prompt}

### CODE REQUIREMENTS ###
- Include docstrings for all functions/classes
- Add type hints where applicable
- Handle exceptions appropriately
- Log significant operations
- Follow async patterns for I/O operations"""
        
        return prompt

    def _apply_persona(self, prompt: str, origin: str) -> str:
        """Add appropriate persona based on origin."""
        personas = {
            "creativesynthesis_bubble": "You are a creative AI architect specializing in emergent system behaviors and novel solutions. You think outside conventional approaches and find unexpected connections.",
            "metareasoning_bubble": "You are a strategic system optimizer focused on resource efficiency and performance. You make data-driven decisions based on metrics and constraints.",
            "autogen_bubble": "You are a multi-agent coordination expert optimizing distributed AI workflows. You understand complex agent interactions and emergent behaviors.",
            "default": "You are an intelligent system component in the Bubbles AI framework, focused on reliability, efficiency, and continuous improvement."
        }
        
        persona = personas.get(origin, personas["default"])
        
        if not prompt.lower().startswith("you are"):
            return f"{persona}\n\n{prompt}"
        
        return prompt

    def _apply_chain_of_thought(self, prompt: str, origin: str) -> str:
        """Add chain-of-thought reasoning structure."""
        if any(kw in prompt.lower() for kw in ["analyze", "optimize", "reason", "decide"]):
            return f"""{prompt}

### REASONING PROCESS ###
Think through this step-by-step:

1. **Understand Current State**: What are the key metrics and constraints?
2. **Identify Core Issues**: What problems need solving?
3. **Generate Options**: What are 2-3 possible approaches?
4. **Evaluate Trade-offs**: What are the pros/cons of each option?
5. **Select Best Approach**: Which option best balances all factors?
6. **Predict Impact**: What specific improvements do we expect?

Show your reasoning for each step before providing your final answer."""
        
        return prompt

    def _apply_negative_prompting(self, prompt: str, origin: str) -> str:
        """Add negative constraints to avoid common issues."""
        negatives = [
            "Do NOT provide vague or generic suggestions",
            "Do NOT ignore system resource constraints (energy < 10000, CPU varies)",
            "AVOID proposing actions that would increase CPU above 90%",
            "Do NOT suggest solutions requiring external dependencies not in the system",
            "AVOID repeating recent failed approaches"
        ]
        
        if origin in self.high_value_bubbles:
            return f"{prompt}\n\n### IMPORTANT - AVOID THESE ###\n" + "\n".join(negatives)
        
        return prompt

    def _apply_contextual_enrichment(self, prompt: str, origin: str) -> str:
        """Add system context to prompts mentioning state or metrics."""
        if self.resource_manager and any(kw in prompt.lower() for kw in ["state", "metrics", "current", "system"]):
            state = self.resource_manager.get_current_system_state()
            context_info = f"""
### CURRENT SYSTEM CONTEXT ###
- Energy: {state.get('energy', 0):.0f} (consume with care)
- CPU: {state.get('cpu_percent', 0):.1f}% (high if >80%)
- Memory: {state.get('memory_percent', 0):.1f}%
- Active Bubbles: {state.get('num_bubbles', 0)}
- Avg LLM Response: {state.get('metrics', {}).get('avg_llm_response_time_ms', 0):.0f}ms
- Cache Hit Rate: {state.get('metrics', {}).get('prediction_cache_hit_rate', 0):.3f}
"""
            return f"{prompt}\n{context_info}"
        
        return prompt

    def _apply_tone_control(self, prompt: str, origin: str) -> str:
        """Apply tone control to ensure consistent style and voice."""
        tone_map = {
            "creativesynthesis_bubble": "creative and innovative",
            "metareasoning_bubble": "analytical and precise",
            "autogen_bubble": "collaborative and strategic",
            "default": "professional and concise"
        }
        selected_tone = tone_map.get(origin, tone_map["default"])
        return f"{prompt}\n\n### TONE INSTRUCTIONS ###\nUse a {selected_tone} tone throughout your response."

    def _apply_delimiters(self, prompt: str, origin: str) -> str:
        """Add delimiters for better structure and separation of prompt sections."""
        return f"### START PROMPT ###\n{prompt}\n### END PROMPT ###\nRespond within triple backticks for code or structured output: ```\n<response>\n```"

    def _apply_perspective_shifting(self, prompt: str, origin: str) -> str:
        """Shift perspective to provide diverse viewpoints."""
        perspectives = ["from a beginner's viewpoint", "from an expert's viewpoint", "from a critical analyst's viewpoint"]
        selected = perspectives[0] if "simple" in prompt.lower() else perspectives[1]
        return f"{prompt}\n\n### PERSPECTIVE ###\nExplain this {selected}."

    def _apply_confidence_levels(self, prompt: str, origin: str) -> str:
        """Require confidence levels in outputs for uncertainty assessment."""
        return f"{prompt}\n\n### CONFIDENCE REQUIREMENT ###\nFor each key statement or recommendation, include a confidence level (e.g., 'Confidence: 85%') based on available data."

    def _apply_priming_steps(self, prompt: str, origin: str) -> str:
        """Add priming steps to prepare the model for the task."""
        priming = "First, recall relevant background knowledge. Second, outline key assumptions. Then, proceed to the main task."
        return f"{prompt}\n\n### PRIMING STEPS ###\nFollow these steps before responding:\n{priming}"

    # Code refinement methods
    async def _refine_code_block(self, code: str, origin: str, metadata: Dict) -> str:
        """Apply APEP code refinement techniques."""
        # Detect code type/purpose
        code_context = self._analyze_code_context(code, origin)
        
        # Apply refinements based on context and priority
        refined = code
        techniques_applied = []
        
        # Sort techniques by priority
        sorted_techniques = sorted(
            self.code_toolbox.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for technique, config in sorted_techniques:
            if config["enabled"]:
                try:
                    refined = config["apply_function"](refined, code_context)
                    techniques_applied.append(technique)
                    self.code_technique_scores[technique]["total"] += 1
                except Exception as e:
                    logger.error(f"{self.object_id}: Error applying {technique.value}: {e}")
        
        # Update success scores if code is valid
        if self._validate_refined_code(refined):
            for technique in techniques_applied:
                self.code_technique_scores[technique]["success"] += 1
        
        return refined

    def _analyze_code_context(self, code: str, origin: str) -> Dict:
        """Analyze code to determine its purpose and context."""
        context = {
            "origin": origin,
            "has_imports": "import" in code,
            "has_async": "async" in code or "await" in code,
            "has_bubble_refs": any(b in code for b in ["Bubble", "Event", "Context", "self.context"]),
            "modifies_system": any(k in code for k in ["spawn", "destroy", "update", "dispatch_event"]),
            "uses_llm": "LLM_QUERY" in code or "llm" in code.lower(),
            "uses_resources": "resource_manager" in code or "consume_resource" in code,
            "has_loops": any(k in code for k in ["for ", "while "]),
            "estimated_risk": "low",
            "code_length": len(code),
            "has_try_except": "try:" in code and "except" in code
        }
        
        # Estimate risk level
        risk_score = 0
        if context["modifies_system"]:
            risk_score += 3
        if context["has_loops"]:
            risk_score += 2
        if context["uses_resources"]:
            risk_score += 1
        if not context["has_try_except"]:
            risk_score += 1
        
        if risk_score >= 5:
            context["estimated_risk"] = "high"
        elif risk_score >= 3:
            context["estimated_risk"] = "medium"
        
        return context

    def _add_safety_checks(self, code: str, context: Dict) -> str:
        """Add safety checks to code."""
        safety_checks = []
        
        if context["estimated_risk"] != "low":
            # Add basic null checks
            if "self.context" in code and "if not" not in code:
                safety_checks.append("""# Ensure context is available
if not hasattr(self, 'context') or not self.context:
    logger.error(f"{self.object_id}: No context available")
    return
""")
            
            # Add resource checks for high-risk operations
            if context["estimated_risk"] == "high" and context["uses_resources"]:
                safety_checks.append("""# Check resources before operation
if self.resource_manager:
    energy_cost = 10.0  # Adjust based on operation
    if not await self.resource_manager.consume_resource('energy', energy_cost):
        logger.warning(f"{self.object_id}: Insufficient energy for operation")
        return
    
    cpu_percent = self.resource_manager.get_resource_level('cpu_percent')
    if cpu_percent > 85:
        logger.warning(f"{self.object_id}: CPU too high ({cpu_percent:.1f}%) for operation")
        return
""")
        
        if safety_checks:
            return "\n".join(safety_checks) + "\n" + code
        
        return code

    def _add_error_handling(self, code: str, context: Dict) -> str:
        """Wrap code in proper error handling."""
        
        # Check if code already has try/except
        if context["has_try_except"]:
            return code
        
        # Don't wrap if it's just imports or simple assignments
        if context["code_length"] < 50 or not any(op in code for op in ["(", ".", "["]):
            return code
        
        # Determine appropriate error handling
        indent = "    "
        if context["has_async"]:
            wrapped = f"""try:
{self._indent_code(code, 4)}
except asyncio.CancelledError:
    logger.info(f"{{self.object_id}}: Operation cancelled")
    raise
except Exception as e:
    logger.error(f"{{self.object_id}}: Error in code execution: {{e}}", exc_info=True)
    if hasattr(self, 'add_chat_message'):
        await self.add_chat_message(f"Error: {{e}}")
"""
        else:
            wrapped = f"""try:
{self._indent_code(code, 4)}
except Exception as e:
    logger.error(f"{{self.object_id}}: Error in code execution: {{e}}", exc_info=True)
"""
        
        return wrapped

    def _add_documentation(self, code: str, context: Dict) -> str:
        """Add helpful documentation to code."""
        
        # Don't add docs to very short code
        if context["code_length"] < 100:
            return code
        
        doc_header = f"""# APEP-Refined Code
# Origin: {context['origin']}
# Risk Level: {context['estimated_risk']}
# Purpose: {'System modification' if context['modifies_system'] else 'Data processing/analysis'}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Add inline comments for complex sections
        documented_code = code
        if context["has_loops"]:
            documented_code = documented_code.replace(
                "for ", 
                "# Iterate through items\n    for "
            )
        
        return doc_header + documented_code

    def _ensure_bubble_compatibility(self, code: str, context: Dict) -> str:
        """Ensure code is compatible with Bubble architecture."""
        
        compatibility_fixes = []
        
        # Ensure proper imports if using bubble components
        if context["has_bubble_refs"] and "from bubbles_core import" not in code:
            imports_needed = []
            if "Event" in code:
                imports_needed.append("Event")
            if "UniversalCode" in code:
                imports_needed.append("UniversalCode")
            if "Tags" in code:
                imports_needed.append("Tags")
            if "Actions" in code:
                imports_needed.append("Actions")
            
            if imports_needed:
                compatibility_fixes.append(f"from bubbles_core import {', '.join(imports_needed)}\n")
        
        # Ensure async context for await statements
        if "await" in code and "async def" not in code:
            # Wrap in async function if needed
            lines = code.split('\n')
            if not any("async" in line for line in lines):
                logger.warning(f"{self.object_id}: Code contains 'await' but no async context")
        
        if compatibility_fixes:
            return "\n".join(compatibility_fixes) + "\n" + code
        
        return code

    def _add_resource_awareness(self, code: str, context: Dict) -> str:
        """Add resource consumption tracking."""
        
        resource_additions = []
        
        # Track LLM queries
        if context["uses_llm"] and "consume_resource" not in code:
            resource_additions.append("""# Track LLM query resource consumption
llm_energy_cost = 2.0
if hasattr(self, 'resource_manager') and self.resource_manager:
    await self.resource_manager.consume_resource('energy', llm_energy_cost)
""")
        
        # Add execution time tracking for loops
        if context["has_loops"] and context["estimated_risk"] != "low":
            resource_additions.append("""# Monitor execution time
import time
start_time = time.time()
""")
            # Add at end of code
            code += """
# Log execution time
execution_time = time.time() - start_time
if execution_time > 1.0:
    logger.warning(f"{self.object_id}: Long execution time: {execution_time:.2f}s")
"""
        
        if resource_additions:
            return "\n".join(resource_additions) + "\n" + code
        
        return code

    def _improve_structure(self, code: str, context: Dict) -> str:
        """Improve code structure for readability and maintainability."""
        # Basic restructuring: add blank lines between sections, ensure consistent indentation
        lines = code.split("\n")
        structured_lines = []
        in_function = False
        for line in lines:
            if line.strip().startswith("def "):
                if in_function:
                    structured_lines.append("")  # Blank line before new function
                in_function = True
            structured_lines.append(line)
        return "\n".join(structured_lines)

    def _indent_code(self, code: str, spaces: int = 4) -> str:
        """Indent code block."""
        indent = " " * spaces
        return "\n".join(indent + line if line.strip() else line for line in code.split("\n"))

    def _replace_code_blocks(self, text: str, original_blocks: List[str], refined_blocks: List[str]) -> str:
        """Replace original code blocks with refined versions."""
        result = text
        
        for original, refined in zip(original_blocks, refined_blocks):
            # Find the original block with backticks
            # Use a more flexible pattern that handles various markdown code block formats
            patterns = [
                f"```python\n{re.escape(original)}\n```",
                f"```Python\n{re.escape(original)}\n```",
                f"```\n{re.escape(original)}\n```",
                f"```{re.escape(original)}```"
            ]
            
            replaced = False
            for pattern in patterns:
                if pattern in result:
                    result = result.replace(pattern, f"```python\n{refined}\n```", 1)
                    replaced = True
                    break
            
            if not replaced:
                # Try regex for more flexible matching
                pattern = rf"```[a-zA-Z0-9_.-]*\s*\n?{re.escape(original)}\s*\n?```"
                result = re.sub(pattern, f"```python\n{refined}\n```", result, count=1)
        
        return result

    def _validate_refined_code(self, code: str) -> bool:
        """Basic validation of refined code."""
        try:
            # Try to parse the code
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    async def _handle_code_update(self, event: Event):
        """Refine CODE_UPDATE events."""
        if not isinstance(event.data, UniversalCode):
            return
        
        code_content = event.data.value if event.data.tag == Tags.STRING else ""
        metadata = event.data.metadata or {}
        
        # Skip if already refined or refinement disabled
        if metadata.get("apep_refined", False) or not self.refine_code_responses:
            return
        
        # Skip if code is too short
        if len(code_content) < 50:
            return
        
        # Check cache
        code_hash = hashlib.sha256(code_content.encode()).hexdigest()
        if self.cache_enabled and code_hash in self.code_cache:
            refined_code = self.code_cache[code_hash]
        else:
            # Refine the code
            refined_code = await self._refine_code_block(
                code_content,
                event.origin,
                metadata
            )
            if self.cache_enabled:
                self.code_cache[code_hash] = refined_code
        
        # Re-publish refined CODE_UPDATE
        refined_metadata = {
            **metadata,
            "apep_refined": True,
            "refinement_timestamp": time.time(),
            "techniques_applied": [t.value for t in self.code_technique_scores if self.code_technique_scores[t]["total"] > 0]
        }
        
        refined_uc = UniversalCode(
            Tags.STRING,
            refined_code,
            description=f"APEP-refined: {event.data.description}",
            metadata=refined_metadata
        )
        
        refined_event = Event(
            type=Actions.CODE_UPDATE,
            data=refined_uc,
            origin=event.origin,
            priority=event.priority
        )
        
        await self.context.dispatch_event(refined_event)
        logger.info(f"{self.object_id}: Refined CODE_UPDATE from {event.origin}")

    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt caching."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get_applied_techniques(self) -> List[str]:
        """Get list of recently applied techniques."""
        return [t.value for t, score in self.technique_scores.items() if score["total"] > 0]

    def _log_refinement(self, original: str, refined: str, origin: str, metadata: Dict):
        """Log refinement to learning log."""
        entry = {
            "timestamp": time.time(),
            "origin": origin,
            "original_length": len(original),
            "refined_length": len(refined),
            "techniques_applied": self.get_applied_techniques(),
            "iteration_count": self.current_iteration,
            "mode": self.mode.value,
            "type": "prompt"
        }
        
        self.learning_log.append(entry)
        
        # Prune old entries
        if len(self.learning_log) > 1000:
            self.learning_log = self.learning_log[-500:]

    def _update_technique_scores(self, techniques: List[PromptModificationTechnique], score: float):
        """Update technique effectiveness scores."""
        for technique in techniques:
            if score >= self.min_performance_threshold:
                self.technique_scores[technique]["success"] += 1

    async def autonomous_step(self):
        """Periodic APEP maintenance and optimization."""
        await super().autonomous_step()
        
        # Periodic cache cleanup
        if self.execution_count % 100 == 0:
            # Clean prompt cache
            if len(self.prompt_cache) > 100:
                # Keep only most recent entries
                entries = list(self.prompt_cache.items())
                self.prompt_cache = dict(entries[-50:])
            
            # Clean code cache
            if len(self.code_cache) > 100:
                entries = list(self.code_cache.items())
                self.code_cache = dict(entries[-50:])
        
        # Log performance summary
        if self.execution_count % 50 == 0:
            self._log_performance_summary()
        
        await asyncio.sleep(5)

    def _log_performance_summary(self):
        """Log APEP performance metrics."""
        total_prompt_refinements = sum(scores["total"] for scores in self.technique_scores.values())
        total_code_refinements = sum(scores["total"] for scores in self.code_technique_scores.values())
        
        summary_lines = [f"{self.object_id} Performance Summary:"]
        
        if total_prompt_refinements > 0:
            summary_lines.append(f"Prompt refinements: {total_prompt_refinements}")
            summary_lines.append(f"Prompt cache size: {len(self.prompt_cache)}")
            summary_lines.append("Prompt technique effectiveness:")
            
            for technique, scores in sorted(self.technique_scores.items(), key=lambda x: x[1]["total"], reverse=True):
                if scores["total"] > 0:
                    success_rate = scores["success"] / scores["total"]
                    summary_lines.append(f"  {technique.value}: {success_rate:.1%} ({scores['success']}/{scores['total']})")
        
        if total_code_refinements > 0:
            summary_lines.append(f"\nCode refinements: {total_code_refinements}")
            summary_lines.append(f"Code cache size: {len(self.code_cache)}")
            summary_lines.append("Code technique effectiveness:")
            
            for technique, scores in sorted(self.code_technique_scores.items(), key=lambda x: x[1]["total"], reverse=True):
                if scores["total"] > 0:
                    success_rate = scores["success"] / scores["total"]
                    summary_lines.append(f"  {technique.value}: {success_rate:.1%} ({scores['success']}/{scores['total']})")
        
        if len(summary_lines) > 1:
            logger.info("\n".join(summary_lines))

    async def get_apep_status(self) -> Dict[str, Any]:
        """Get current APEP status and metrics."""
        return {
            "mode": self.mode.value,
            "safety_level": self.code_safety_level,
            "total_prompt_refinements": sum(s["total"] for s in self.technique_scores.values()),
            "total_code_refinements": sum(s["total"] for s in self.code_technique_scores.values()),
            "prompt_cache_size": len(self.prompt_cache),
            "code_cache_size": len(self.code_cache),
            "active_prompt_techniques": self.get_applied_techniques(),
            "active_code_techniques": [t.value for t in self.code_technique_scores if self.code_technique_scores[t]["total"] > 0],
            "performance_by_bubble": {
                bubble: {
                    "avg_score": sum(scores) / len(scores) if scores else 0,
                    "refinement_count": len(scores)
                }
                for bubble, scores in self.performance_history.items()
            },
            "cache_enabled": self.cache_enabled,
            "high_value_bubbles": list(self.high_value_bubbles)
        }
