import asyncio
import aiohttp
import logging
import os
from typing import Dict, Any, Optional
from bubbles_core import UniversalBubble, Event, Actions, UniversalCode, Tags, SystemContext
from tv_control import TVControl
from light_control import LightControl
from pool_control import PoolControlBubble

logger = logging.getLogger(__name__)

class HomeAssistantBubble(UniversalBubble):
    def __init__(self, object_id: str, context: SystemContext, ha_url: str = "http://10.0.0.XXXXX", ha_token: str = None):
        super().__init__(object_id=object_id, context=context)
        # Validate SystemContext
        if not hasattr(context, 'event_service'):
            logger.warning(f"{self.object_id}: SystemContext lacks event_service. Event subscription may fail.")
        
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token or os.getenv("HA_TOKEN")
        if not self.ha_token:
            logger.critical(f"{self.object_id}: HA_TOKEN not provided.")
            raise ValueError("HA_TOKEN is required for HomeAssistantBubble.")
        self.session: Optional[aiohttp.ClientSession] = None
        self.device_bubbles = [
            TVControl(
                object_id="tv_bubble",
                context=context,
                ha_url=self.ha_url,
                ha_token=self.ha_token,
                entity_id="media_player.living_room_tv",
                mac_address="aa:bb:cc:dd:ee:ff"  # Replace with your TV's MAC address
            ),
            LightControl(
                object_id="light_bubble",
                context=context,
                ha_url=self.ha_url,
                ha_token=self.ha_token,
                entity_id="light.living_room"
            ),
            PoolControlBubble(
                object_id="pool_bubble",
                context=context,
                ha_url=self.ha_url,
                ha_token=self.ha_token,
                config={
                    "chlorinator_entity": "switch.my_pool_chlorinator",
                    "liquid_chlorine_feed_entity": "switch.liquid_chlorine_feed",
                    "acid_feed_entity": "switch.omnilogic_acid_feed",
                    "alkalinity_feed_entity": "switch.alkalinity_feed",
                    "calcium_feed_entity": "switch.calcium_feed",
                    "pump_entity": "number.my_pool_pump_speed",
                    "heater_entity": "climate.my_pool_heater",
                    "lights_entity": "light.my_pool_lights",
                    "cleaner_entity": "switch.aiper_scuba_x1_pro_max",
                    "door_alarm_entity": "binary_sensor.pool_door_alarm",
                    "ph_sensor_entity": "sensor.my_pool_ph",
                    "orp_sensor_entity": "sensor.my_pool_orp",
                    "water_temp_sensor_entity": "sensor.my_pool_temperature",
                    "target_temp_entity": "input_number.pool_target_temperature",
                    "target_chlorine_entity": "input_number.pool_target_chlorine",
                    "target_ph_entity": "input_number.pool_target_ph",
                    "target_orp_entity": "input_number.pool_target_orp",
                    "target_alkalinity_entity": "input_number.pool_target_alkalinity",
                    "target_hardness_entity": "input_number.pool_target_hardness",
                    "weather_url": f"http://api.openweathermap.org/data/2.5/weather?q=Macomb&appid={os.getenv('OPENWEATHERMAP_API_KEY')}&units=imperial",
                    "forecast_url": f"http://api.openweathermap.org/data/2.5/forecast?q=Macomb&appid={os.getenv('OPENWEATHERMAP_API_KEY')}&units=imperial",
                    "openweathermap_api_key": os.getenv("OPENWEATHERMAP_API_KEY", "GET YOUR OPENWEATHERMAP API KEY"),
                    "use_mock_weather": True,
                    "rain_condition_codes": [500, 501, 502, 503, 504, 511, 520, 521, 522, 531],
                    "peak_hours_start": 12,
                    "peak_hours_end": 18,
                    "rain_reduction_temp": 83.0,
                    "target_temp_high": 87.0,
                    "target_temp_low": 85.0,
                    "target_chlorine": 3.0,
                    "target_ph": 7.4,
                    "target_orp": 700.0,
                    "target_alkalinity": 100.0,
                    "target_hardness": 250.0
                }
            )
        ]
        self.device_map = {
            "media_player.living_room_tv": ("TV_CONTROL", self.device_bubbles[0]),
            "light.living_room": ("LIGHT_CONTROL", self.device_bubbles[1]),
            "switch.my_pool_chlorinator": ("POOL_CONTROL", self.device_bubbles[2])
        }
        logger.info(f"{self.object_id}: Initialized with Home Assistant URL {self.ha_url}")

    async def _subscribe_to_events(self):
        try:
            if not hasattr(self.context, 'event_service'):
                logger.error(f"{self.object_id}: SystemContext lacks event_service. Skipping event subscription.")
                return False
            await self.context.event_service.subscribe(Actions.HA_CONTROL, self.handle_event)
            logger.debug(f"{self.object_id}: Subscribed to HA_CONTROL")
            return True
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to HA_CONTROL: {e}", exc_info=True)
            return False

    async def on_start(self):
        try:
            self.session = aiohttp.ClientSession()
            logger.debug(f"{self.object_id}: aiohttp session initialized.")
            for bubble in self.device_bubbles:
                await bubble.on_start()
            # Subscribe to events after session and bubbles are initialized
            await self._subscribe_to_events()
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to start: {e}", exc_info=True)
        await super().on_start()

    async def on_stop(self):
        try:
            for bubble in self.device_bubbles:
                await bubble.on_stop()
            if self.session and not self.session.closed:
                await self.session.close()
                logger.debug(f"{self.object_id}: Closed aiohttp session")
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to stop: {e}", exc_info=True)
        await super().on_stop()

    async def handle_event(self, event: Event):
        if event.type != Actions.HA_CONTROL or not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            return
        try:
            action_data = event.data.value
            entity_id = action_data.get("entity_id")
            action = action_data.get("action")
            logger.debug(f"{self.object_id}: Processing HA_CONTROL event: entity_id={entity_id}, action={action}")

            if not entity_id or not action:
                logger.error(f"{self.object_id}: Invalid HA_CONTROL event: missing entity_id or action")
                return

            device_info = self.device_map.get(entity_id)
            if device_info:
                event_type, device_bubble = device_info
                device_event = Event(
                    type=event_type,
                    data=event.data,
                    origin=self.object_id,
                    metadata=event.metadata
                )
                await device_bubble.process_single_event(device_event)
                logger.info(f"{self.object_id}: Routed HA_CONTROL to {event_type} for {entity_id}")
            else:
                logger.warning(f"{self.object_id}: No device bubble for entity {entity_id}. Attempting generic service call.")
                domain = entity_id.split(".")[0]
                await self._call_ha_service(domain, action, {"entity_id": entity_id})

        except Exception as e:
            logger.error(f"{self.object_id}: Error processing HA_CONTROL event: {e}", exc_info=True)

    async def _call_ha_service(self, domain: str, service: str, service_data: Dict[str, Any]) -> bool:
        if not self.session:
            logger.error(f"{self.object_id}: aiohttp session not initialized")
            return False
        headers = {"Authorization": f"Bearer {self.ha_token}", "Content-Type": "application/json"}
        url = f"{self.ha_url}/api/services/{domain}/{service}"
        try:
            async with self.session.post(url, json=service_data, headers=headers) as response:
                if response.status == 200:
                    logger.info(f"{self.object_id}: Successfully executed {domain}.{service} for {service_data.get('entity_id')}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"{self.object_id}: Failed to execute {domain}.{service}: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"{self.object_id}: Error calling Home Assistant service {domain}.{service}: {e}", exc_info=True)
            return False

    async def process_single_event(self, event: Event):
        await self.handle_event(event)
