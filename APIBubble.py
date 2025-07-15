import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Optional
from bubbles_core import UniversalBubble, Actions, Event, UniversalCode, Tags, SystemContext, logger, EventService

class APIBubble(UniversalBubble):
    """Manages external API interactions within the Bubbles system."""
    def __init__(self, object_id: str, context: SystemContext, api_configs: Dict[str, Dict] = None, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        self.api_configs = api_configs or {}
        self.session = None
        self.request_timeout = 30
        self.retry_attempts = 3
        self.retry_delay = 5
        self.api_call_success_count = 0
        self.api_call_total_count = 0
        asyncio.create_task(self._initialize_session())
        asyncio.create_task(self._subscribe_to_events())
        logger.info(f"{self.object_id}: Initialized APIBubble with {len(self.api_configs)} API configurations.")

    async def _initialize_session(self):
        try:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.request_timeout))
            logger.debug(f"{self.object_id}: aiohttp session initialized.")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to initialize aiohttp session: {e}", exc_info=True)
            self.session = None

    async def _subscribe_to_events(self):
        await asyncio.sleep(0.1)
        try:
            await EventService.subscribe(Actions.API_CALL, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to API_CALL events.")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}", exc_info=True)

    async def handle_event(self, event: Event):
        if event.type == Actions.API_CALL and isinstance(event.data, UniversalCode) and event.data.tag == Tags.DICT:
            await self.process_api_call(event)
        await super().handle_event(event)

    async def process_api_call(self, event: Event):
        self.api_call_total_count += 1
        start_time = time.time()
        if not self.session:
            await self._send_api_response(event.origin, event.data.metadata.get("correlation_id", f"fallback_{self.api_call_total_count}"), error="API session not initialized")
            self._update_metrics(start_time, success=False)
            return

        request_data = event.data.value
        api_name = request_data.get("api_name")
        endpoint = request_data.get("endpoint")
        method = request_data.get("method", "GET").upper()
        params = request_data.get("params", {})
        headers = request_data.get("headers", {})
        body = request_data.get("body")
        correlation_id = event.data.metadata.get("correlation_id", f"fallback_{self.api_call_total_count}")

        if not api_name or not endpoint:
            await self._send_api_response(event.origin, correlation_id, error="Missing api_name or endpoint")
            self._update_metrics(start_time, success=False)
            return

        config = self.api_configs.get(api_name)
        if not config:
            await self._send_api_response(event.origin, correlation_id, error=f"Unknown API: {api_name}")
            self._update_metrics(start_time, success=False)
            return

        base_url = config.get("url", "")
        auth = config.get("auth")
        if auth:
            headers.update(auth.get("headers", {}))

        full_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        logger.info(f"{self.object_id}: Processing API call: {method} {full_url}")

        for attempt in range(self.retry_attempts):
            try:
                async with self.session.request(
                    method=method,
                    url=full_url,
                    params=params,
                    headers=headers,
                    json=body if method in ["POST", "PUT"] else None
                ) as response:
                    status = response.status
                    content = await response.text()
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        data = content
                    response_data = {
                        "status": status,
                        "data": data,
                        "headers": dict(response.headers),
                        "error": None if 200 <= status < 300 else f"HTTP {status}"
                    }
                    await self._send_api_response(event.origin, correlation_id, response=response_data)
                    self._update_metrics(start_time, success=200 <= status < 300)
                    return
            except aiohttp.ClientError as e:
                error_msg = f"API call failed (attempt {attempt + 1}/{self.retry_attempts}): {e}"
                logger.warning(f"{self.object_id}: {error_msg}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    await self._send_api_response(event.origin, correlation_id, error=error_msg)
                    self._update_metrics(start_time, success=False)
            except Exception as e:
                logger.error(f"{self.object_id}: Unexpected error in API call: {e}", exc_info=True)
                await self._send_api_response(event.origin, correlation_id, error=str(e))
                self._update_metrics(start_time, success=False)
                return

    async def _send_api_response(self, requester_id: Optional[str], correlation_id: Optional[str], response: Optional[Dict] = None, error: Optional[str] = None):
        """Send an API_RESPONSE event with the result or error."""
        if not requester_id:
            logger.error(f"{self.object_id}: Cannot send API response - missing requester_id.")
            return
        if not correlation_id:
            correlation_id = f"fallback_{self.api_call_total_count}"
            logger.warning(f"{self.object_id}: Generated fallback correlation_id: {correlation_id}")
        if not self.dispatcher:
            logger.error(f"{self.object_id}: Cannot send API response, dispatcher unavailable.")
            return

        response_payload = {"correlation_id": correlation_id}
        if response and not error:
            response_payload["response"] = response
            response_payload["error"] = None
            status = "SUCCESS"
        else:
            response_payload["response"] = None
            response_payload["error"] = error if error else "Unknown API error"
            status = "ERROR"

        response_uc = UniversalCode(Tags.DICT, response_payload, description=f"API response ({status})")
        try:
            response_event = Event(type=Actions.API_RESPONSE, data=response_uc, origin=self.object_id, priority=2)
            await self.context.dispatch_event(response_event)
            logger.info(f"{self.object_id}: Sent API_RESPONSE ({status}) for {correlation_id[:8]} to {requester_id}")
        except AttributeError as e:
            logger.error(f"{self.object_id}: Failed to create API_RESPONSE event: {e}", exc_info=True)
            logger.warning(f"{self.object_id}: Skipping API_RESPONSE dispatch for {correlation_id[:8]}")

    def _update_metrics(self, start_time: float, success: bool):
        duration = time.time() - start_time
        if success:
            self.api_call_success_count += 1
        if hasattr(self.context, 'resource_manager') and self.context.resource_manager:
            try:
                self.context.resource_manager.metrics["api_call_count"] = self.api_call_total_count
                self.context.resource_manager.metrics["api_call_success_rate"] = (
                    self.api_call_success_count / self.api_call_total_count if self.api_call_total_count > 0 else 0.0
                )
                self.context.resource_manager.metrics["api_call_duration"] = duration
                logger.debug(f"{self.object_id}: Updated metrics: success_rate={self.context.resource_manager.metrics['api_call_success_rate']:.2f}, duration={duration:.3f}s")
            except Exception as e:
                logger.error(f"{self.object_id}: Failed to update metrics: {e}", exc_info=True)

    async def autonomous_step(self):
        await super().autonomous_step()
        if self.execution_count % 60 == 0:
            for api_name, config in self.api_configs.items():
                try:
                    health_endpoint = config.get("health_endpoint", "/health")
                    await self.process_api_call(Event(
                        type=Actions.API_CALL,
                        data=UniversalCode(Tags.DICT, {
                            "api_name": api_name,
                            "endpoint": health_endpoint,
                            "method": "GET",
                            "params": {},
                            "headers": {}
                        }, metadata={"correlation_id": f"health_{api_name}_{int(time.time())}"}),
                        origin=self.object_id,
                        priority=1
                    ))
                except Exception as e:
                    logger.error(f"{self.object_id}: Error in health check for {api_name}: {e}", exc_info=True)
        await asyncio.sleep(0.5)

    async def stop_autonomous_loop(self):
        await super().stop_autonomous_loop()
        if self.session:
            await self.session.close()
            logger.info(f"{self.object_id}: aiohttp session closed.")
