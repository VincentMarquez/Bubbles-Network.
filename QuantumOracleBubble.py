# QuantumOracleBubble.py: Prophetic AI bubble for the Bubbles Framework
# Uses quantum fractal memory from QMLBubble to generate visionary proposals
# Integrates with LLM for creative, emergent system evolution
# UPDATED: Added flood control to prevent LLM request overload

import asyncio
import logging
import json
import uuid
from typing import Dict, Optional
from collections import deque
from bubbles_core import UniversalBubble, Actions, Event, UniversalCode, Tags, SystemContext, logger, EventService

# Configure logging for detailed debugging
logger.setLevel(logging.DEBUG)

def robust_json_parse(data: str) -> Dict:
    """Safely parse JSON strings, returning an empty dict on failure."""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON: {data[:50]}")
        return {}

class QuantumOracleBubble(UniversalBubble):
    """Quantum-inspired prophetic AI that generates visionary proposals using quantum fractal memory."""
    def __init__(self, object_id: str, context: SystemContext, max_archive: int = 100, **kwargs):
        """Initialize QuantumOracleBubble with event-driven setup."""
        logger.debug(f"{object_id}: Starting QuantumOracleBubble initialization")
        try:
            # Initialize parent UniversalBubble class
            super().__init__(object_id=object_id, context=context, **kwargs)
            # Archive for storing prophecies
            self.oracle_archive = deque(maxlen=max_archive)
            # ID of QMLBubble for quantum requests
            self.qml_bubble_id = "qml_bubble"
            # Threshold for executing high-impact prophecies
            self.fitness_threshold = 0.7
            # Start async event subscription
            asyncio.create_task(self._subscribe_to_events())
            logger.info(f"{self.object_id}: Initialized QuantumOracleBubble with max_archive={max_archive}")
        except Exception as e:
            logger.error(f"{object_id}: Initialization failed: {e}", exc_info=True)
            raise

    async def _subscribe_to_events(self):
        """Subscribe to relevant framework events."""
        logger.debug(f"{self.object_id}: Starting event subscription")
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.SYSTEM_STATE_UPDATE, self.handle_event)
            await EventService.subscribe(Actions.QUANTUM_RESULT, self.handle_event)
            await EventService.subscribe(Actions.LLM_RESPONSE, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to SYSTEM_STATE_UPDATE, QUANTUM_RESULT, LLM_RESPONSE")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    async def generate_oracular_pattern(self, state: Dict):
        """Request quantum fractal memory encoding from QMLBubble."""
        logger.debug(f"{self.object_id}: Generating oracular pattern")
        try:
            qml_request = {
                "task_type": "memory_store",
                "metrics": state,
                "correlation_id": str(uuid.uuid4())
            }
            qml_uc = UniversalCode(Tags.DICT, qml_request, description="Quantum oracular pattern")
            qml_event = Event(type=Actions.QML_REQUEST, data=qml_uc, origin=self.object_id, priority=3)
            await self.context.dispatch_event(qml_event)
            logger.info(f"{self.object_id}: Requested quantum oracular pattern (cid: {qml_request['correlation_id'][:8]})")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to generate oracular pattern: {e}", exc_info=True)

    async def craft_prophetic_proposal(self, qml_result: Dict):
        """Generate LLM-driven prophetic proposal based on quantum pattern using flood control."""
        logger.debug(f"{self.object_id}: Crafting prophetic proposal with flood control")
        try:
            # Import the flood control function
            from flood_control import query_llm_with_flood_control
            
            prompt = (
                f"Quantum oracular pattern: {qml_result.get('qml_output', 0.0):.2f}, "
                f"Metrics: {qml_result.get('metrics', {})}\n"
                f"Act as a cosmic oracle: craft a prophetic vision for system evolution. "
                f"Return JSON: {{'oracle_type': str, 'prophecy': str, 'actions': list, 'certainty': float}}"
            )
            
            # Use flood-controlled LLM query instead of direct dispatch
            result = await query_llm_with_flood_control(
                prompt=prompt,
                system_context=self.context,
                origin_bubble=self.object_id,
                priority=3  # Oracle analysis is medium priority
            )
            
            # Handle blocked requests gracefully
            if result.get("blocked"):
                reason = result.get("block_reason", "unknown")
                logger.info(f"{self.object_id}: Prophetic proposal blocked - {reason}")
                
                # If it's a duplicate, we can skip safely
                if "duplicate" in reason.lower():
                    logger.debug(f"{self.object_id}: Skipping duplicate prophetic proposal")
                    return
                
                # For rate limits, schedule retry later
                retry_after = result.get("retry_after", 60)
                logger.info(f"{self.object_id}: Oracle will retry proposal in {retry_after}s")
                
                # Store the QML result for retry
                self._pending_qml_result = qml_result
                return
            
            # Process successful response directly
            if result.get("response"):
                logger.info(f"{self.object_id}: Received flood-controlled prophetic response")
                await self._process_prophetic_response(result["response"])
            else:
                logger.warning(f"{self.object_id}: No response in flood control result")
                
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to craft prophetic proposal: {e}", exc_info=True)

    async def _process_prophetic_response(self, response_text: str):
        """Process the prophetic response from flood-controlled LLM."""
        try:
            # Improved JSON parsing for Creative Bubble responses
            if response_text.startswith("```json"):
                # Extract JSON from code blocks
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx:end_idx+1]
            
            prophecy = robust_json_parse(response_text)
            
            # Handle both legacy and new format
            if prophecy and isinstance(prophecy, dict):
                # Check if it's a Creative Bubble proposal
                if "proposal_type" in prophecy and "payload" in prophecy:
                    logger.info(f"{self.object_id}: Processing Creative Bubble proposal: {prophecy.get('proposal_type')}")
                    
                    # Score the Creative Bubble proposal
                    fitness = await self.score_prophecy(prophecy)
                    prophecy["fitness"] = fitness
                    self.oracle_archive.append(prophecy)
                    
                    logger.info(f"{self.object_id}: Proposal fitness: {fitness:.2f} (threshold: {self.fitness_threshold})")
                    
                    if fitness > self.fitness_threshold:
                        logger.info(f"{self.object_id}: Executing high-fitness proposal")
                        await self.execute_prophecy(prophecy)
                    else:
                        logger.debug(f"{self.object_id}: Proposal below threshold, archived for later")
                        
                # Handle legacy Oracle format
                elif "oracle_type" in prophecy or "prophecy" in prophecy:
                    logger.info(f"{self.object_id}: Processing legacy Oracle prophecy")
                    fitness = await self.score_prophecy(prophecy)
                    prophecy["fitness"] = fitness
                    self.oracle_archive.append(prophecy)
                    
                    if fitness > self.fitness_threshold:
                        await self.execute_prophecy(prophecy)
                        
                else:
                    logger.debug(f"{self.object_id}: Received unrecognized response format, keys: {list(prophecy.keys())}")
                    
            else:
                logger.warning(f"{self.object_id}: Failed to parse LLM response as JSON")
                
        except Exception as e:
            logger.error(f"{self.object_id}: Error processing prophetic response: {e}", exc_info=True)

    async def score_prophecy(self, prophecy: Dict) -> float:
        """Score prophecy using DreamerV3Bubble's state prediction."""
        logger.debug(f"{self.object_id}: Scoring prophecy")
        try:
            correlation_id = str(uuid.uuid4())
            
            # Extract actions from different prophecy formats
            actions = []
            if "payload" in prophecy:
                payload = prophecy["payload"]
                actions = payload.get("actions", [])
                if not actions and "experiment_details" in payload:
                    exp_details = payload["experiment_details"]
                    if "methodology" in exp_details:
                        actions = exp_details["methodology"]
                    elif "setup" in exp_details:
                        actions = [exp_details["setup"]]
            else:
                actions = prophecy.get("actions", [])
            
            query_data = {
                "current_state": self.resource_manager.get_current_system_state(),
                "action": {"action_type": "ORACLE_ACTION", "payload": actions}
            }
            query_uc = UniversalCode(Tags.DICT, query_data, metadata={"correlation_id": correlation_id})
            query_event = Event(type=Actions.PREDICT_STATE_QUERY, data=query_uc, origin=self.object_id, priority=2)
            await self.context.dispatch_event(query_event)
            
            # Calculate fitness based on prophecy content and expected impact
            base_fitness = prophecy.get("certainty", prophecy.get("confidence", 0.5))
            novelty_bonus = prophecy.get("novelty_score", 0.0) * 0.3
            fitness = min((base_fitness * 1.6) + novelty_bonus, 1.0)
            
            logger.debug(f"{self.object_id}: Prophecy scored with fitness: {fitness:.2f}")
            return fitness
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to score prophecy: {e}", exc_info=True)
            return 0.0

    async def execute_prophecy(self, prophecy: Dict):
        """Execute high-impact prophecy actions and share with OverseerBubble."""
        logger.debug(f"{self.object_id}: Executing prophecy")
        try:
            # Handle different prophecy formats safely
            description = ""
            actions = []
            
            if "payload" in prophecy:
                # Creative Bubble format
                payload = prophecy["payload"]
                description = payload.get("description", "")
                actions = payload.get("actions", [])
                
                # Also check experiment_details for actions
                if not actions and "experiment_details" in payload:
                    exp_details = payload["experiment_details"]
                    if "methodology" in exp_details:
                        methodology = exp_details["methodology"]
                        # Handle both list and dict formats
                        if isinstance(methodology, list):
                            actions = methodology
                        elif isinstance(methodology, dict):
                            actions = list(methodology.values()) if methodology else []
                    elif "setup" in exp_details:
                        setup = exp_details["setup"]
                        actions = [setup] if setup else []
                        
                # Extract narrative as fallback description
                if not description:
                    description = payload.get("narrative", "")
                    
            else:
                # Legacy Oracle format
                description = prophecy.get("prophecy", prophecy.get("description", "Unknown prophecy"))
                actions = prophecy.get("actions", [])
            
            # Execute actions if any
            executed_actions = 0
            for action in actions:
                try:
                    action_data = action if isinstance(action, dict) else {"action": str(action)}
                    action_uc = UniversalCode(Tags.DICT, action_data, description=f"Prophecy: {description[:50]}")
                    action_event = Event(type=Actions.ACTION_TAKEN, data=action_uc, origin=self.object_id, priority=4)
                    await self.context.dispatch_event(action_event)
                    executed_actions += 1
                except Exception as action_error:
                    logger.error(f"{self.object_id}: Failed to execute action {action}: {action_error}")
            
            # Share prophecy with OverseerBubble for meta-RL
            report_data = {
                "prophecy": prophecy, 
                "performance": prophecy.get("fitness", prophecy.get("confidence", 0.0)),
                "actions_executed": executed_actions,
                "description": description
            }
            report_uc = UniversalCode(Tags.DICT, report_data)
            report_event = Event(type=Actions.OVERSEER_REPORT, data=report_uc, origin=self.object_id, priority=3)
            await self.context.dispatch_event(report_event)
            
            logger.info(f"{self.object_id}: Executed prophetic proposal: {description[:50]}... ({executed_actions} actions)")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to execute prophecy: {e}", exc_info=True)

    async def handle_event(self, event: Event):
        """Handle incoming framework events with flood control integration."""
        logger.debug(f"{self.object_id}: Handling event: {event.type}")
        try:
            if event.type == Actions.SYSTEM_STATE_UPDATE and event.data.tag == Tags.DICT:
                # Check if this event should be analyzed by specialized routing
                if hasattr(self.context, 'analysis_router') and self.context.analysis_router:
                    routing_result = await self.context.analysis_router.route_for_analysis(event)
                    if routing_result.get("analysis_performed"):
                        logger.info(f"{self.object_id}: Analysis handled by specialized routing")
                        return
                
                # Generate quantum pattern for oracle analysis
                await self.generate_oracular_pattern(event.data.value)
                
            elif event.type == Actions.QUANTUM_RESULT and event.data.tag == Tags.DICT:
                qml_result = event.data.value
                if qml_result.get("qml_output"):
                    await self.craft_prophetic_proposal(qml_result)
                    
            elif event.type == Actions.LLM_RESPONSE and event.data.tag == Tags.STRING:
                # This should now be handled by flood control, but keep for backwards compatibility
                if not hasattr(self, '_using_flood_control') or not self._using_flood_control:
                    logger.warning(f"{self.object_id}: Received direct LLM response - should be using flood control")
                    await self._process_prophetic_response(event.data.value)
                    
            await super().handle_event(event)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event: {e}", exc_info=True)

    async def autonomous_step(self):
        """Periodically review archived prophecies for re-evaluation."""
        logger.debug(f"{self.object_id}: Running autonomous step")
        try:
            await super().autonomous_step()
            
            # Retry any pending QML results that were blocked
            if hasattr(self, '_pending_qml_result') and self._pending_qml_result:
                logger.info(f"{self.object_id}: Retrying pending prophetic proposal")
                await self.craft_prophetic_proposal(self._pending_qml_result)
                self._pending_qml_result = None
            
            # Every 10 minutes (600 iterations), review archive
            if self.execution_count % 600 == 0 and self.oracle_archive:
                logger.info(f"{self.object_id}: Reviewing oracle archive, size: {len(self.oracle_archive)}")
                
                # Re-evaluate recent prophecies (last 5)
                for prophecy in list(self.oracle_archive)[-5:]:
                    try:
                        fitness = await self.score_prophecy(prophecy)
                        original_fitness = prophecy.get("fitness", 0.0)
                        
                        # If fitness has improved significantly, consider execution
                        if fitness > self.fitness_threshold and fitness > original_fitness + 0.1:
                            logger.info(f"{self.object_id}: Re-evaluating prophecy - new fitness: {fitness:.2f}")
                            prophecy["fitness"] = fitness
                            await self.execute_prophecy(prophecy)
                            
                    except Exception as eval_error:
                        logger.error(f"{self.object_id}: Error re-evaluating prophecy: {eval_error}")
                        
            # Every hour (3600 iterations), clean up old entries
            if self.execution_count % 3600 == 0:
                original_size = len(self.oracle_archive)
                # The deque automatically manages size, but we can log the cleanup
                if original_size > 0:
                    logger.info(f"{self.object_id}: Archive maintenance - {original_size} prophecies stored")
                    
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"{self.object_id}: Autonomous step failed: {e}", exc_info=True)

    def get_archive_stats(self) -> Dict:
        """Get statistics about the prophecy archive."""
        try:
            if not self.oracle_archive:
                return {"total": 0, "high_fitness": 0, "avg_fitness": 0.0}
                
            fitnesses = [p.get("fitness", 0.0) for p in self.oracle_archive]
            high_fitness_count = sum(1 for f in fitnesses if f > self.fitness_threshold)
            
            # Count by proposal type
            proposal_types = {}
            for prophecy in self.oracle_archive:
                ptype = prophecy.get("proposal_type", "legacy")
                proposal_types[ptype] = proposal_types.get(ptype, 0) + 1
                
            return {
                "total": len(self.oracle_archive),
                "high_fitness": high_fitness_count,
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "max_fitness": max(fitnesses),
                "proposal_types": proposal_types,
                "fitness_threshold": self.fitness_threshold
            }
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to get archive stats: {e}")
            return {"error": str(e)}

    async def adjust_fitness_threshold(self, new_threshold: float):
        """Dynamically adjust the fitness threshold for prophecy execution."""
        try:
            old_threshold = self.fitness_threshold
            self.fitness_threshold = max(0.0, min(1.0, new_threshold))  # Clamp to [0,1]
            
            logger.info(f"{self.object_id}: Fitness threshold adjusted from {old_threshold:.2f} to {self.fitness_threshold:.2f}")
            
            # If threshold was lowered, check if any archived prophecies now qualify
            if self.fitness_threshold < old_threshold:
                qualifying_prophecies = [
                    p for p in self.oracle_archive 
                    if p.get("fitness", 0.0) > self.fitness_threshold
                ]
                
                if qualifying_prophecies:
                    logger.info(f"{self.object_id}: {len(qualifying_prophecies)} archived prophecies now qualify for execution")
                    
                    # Execute the top 3 qualifying prophecies
                    for prophecy in sorted(qualifying_prophecies, key=lambda x: x.get("fitness", 0.0), reverse=True)[:3]:
                        await self.execute_prophecy(prophecy)
                        
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to adjust fitness threshold: {e}")

    async def emergency_execute_all(self):
        """Emergency function to execute all high-fitness prophecies (for crisis situations)."""
        try:
            logger.warning(f"{self.object_id}: EMERGENCY EXECUTION - Processing all high-fitness prophecies")
            
            high_fitness_prophecies = [
                p for p in self.oracle_archive 
                if p.get("fitness", 0.0) > 0.6  # Lower threshold for emergency
            ]
            
            if not high_fitness_prophecies:
                logger.warning(f"{self.object_id}: No high-fitness prophecies available for emergency execution")
                return
                
            logger.info(f"{self.object_id}: Executing {len(high_fitness_prophecies)} emergency prophecies")
            
            for i, prophecy in enumerate(high_fitness_prophecies):
                try:
                    logger.info(f"{self.object_id}: Emergency execution {i+1}/{len(high_fitness_prophecies)} - fitness: {prophecy.get('fitness', 0.0):.2f}")
                    await self.execute_prophecy(prophecy)
                    await asyncio.sleep(0.1)  # Small delay between executions
                except Exception as exec_error:
                    logger.error(f"{self.object_id}: Emergency execution failed for prophecy {i+1}: {exec_error}")
                    
            logger.warning(f"{self.object_id}: Emergency execution completed")
            
        except Exception as e:
            logger.error(f"{self.object_id}: Emergency execution failed: {e}", exc_info=True)
