import asyncio
import aiohttp
import logging
import os
import numpy as np
import torch
from typing import Dict, Any, Optional
from bubbles_core import UniversalBubble, Event, Actions, UniversalCode, Tags, SystemContext
from stable_baselines3 import PPO
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)

class PoolControlBubble(UniversalBubble):
    def __init__(self, object_id: str, context: SystemContext, ha_url: str, ha_token: str, config: Dict[str, Any]):
        super().__init__(object_id=object_id, context=context)
        self.ha_url = ha_url.rstrip("/")
        self.ha_token = ha_token
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.models = []
        self.num_models = 3
        self.checkpoints_dir = "/Users/marquez/Desktop/Bubbles2STOA/checkpoints"
        self.model_filenames = [f"{self.checkpoints_dir}/ppo_pool_model_{i}.zip" for i in range(self.num_models)]
        self.state_space = {
            "temperature": 0.0, "chlorine_level": 0.0, "ph": 0.0, "orp": 0.0,
            "total_alkalinity": 0.0, "calcium_hardness": 0.0, "pump_speed": 0.0,
            "heater_on": False, "lights_on": False, "cleaner_on": False,
            "door_alarm": False, "hour": 0, "outdoor_temp": 0.0, "forecast_temp": 0.0,
            "day_of_week": 0, "is_raining": 0.0
        }
        self.state_buffer = {key: [] for key in self.state_space.keys()}
        self.smoothing_window = 3
        self.energy_usage = 0.0
        self.last_energy_update = datetime.now()
        self.dosing_counters = {
            "chlorinator": 0, "liquid_chlorine": 0, "acid": 0, "alkalinity": 0, "calcium": 0
        }
        self.last_dosing_reset = datetime.now()
        self.cleaner_schedule = {
            "last_run": datetime.now(),
            "interval_days": 2,
            "run_hour": 2
        }
        self.cleaner_running = False
        self.cleaner_run_duration = 3600
        self.initialize_ppo_models()

    def initialize_ppo_models(self):
        try:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            logger.info(f"{self.object_id}: Ensured checkpoints directory exists at {self.checkpoints_dir}")
            for i in range(self.num_models):
                try:
                    logger.debug(f"Attempting to load model from {self.model_filenames[i]}")
                    model = PPO.load(self.model_filenames[i])
                    logger.info(f"{self.object_id}: Loaded PPO model {i}")
                except FileNotFoundError:
                    logger.info(f"{self.object_id}: Training new PPO model {i}")
                    from pool_env import PoolEnv
                    try:
                        logger.debug(f"Initializing PoolEnv with config: {self.config}")
                        env = PoolEnv(config=self.config)
                        logger.debug(f"Creating PPO model with MlpPolicy, state_dim={len(self.state_space)}, action_dim=9")
                        model = PPO(
                            "MlpPolicy", env, verbose=2, learning_rate=3e-4,
                            n_steps=2048, batch_size=64, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01
                        )
                        logger.info(f"{self.object_id}: Starting PPO training for model {i} with 10,000 timesteps")
                        model.learn(total_timesteps=10_000)
                        logger.debug(f"Saving model to {self.model_filenames[i]}")
                        model.save(self.model_filenames[i])
                        logger.info(f"{self.object_id}: PPO model {i} trained and saved")
                    except Exception as train_e:
                        logger.error(f"{self.object_id}: Failed to train PPO model {i}: {train_e}", exc_info=True)
                        logger.warning(f"{self.object_id}: Using rule-based policy for model {i}")
                        model = lambda state: np.zeros(9, dtype=np.float32)
                self.models.append(model)
        except Exception as e:
            logger.error(f"{self.object_id}: Error initializing PPO models: {e}", exc_info=True)
            self.models = [lambda state: np.zeros(9, dtype=np.float32)] * self.num_models

    async def on_start(self):
        try:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
                logger.debug(f"{self.object_id}: aiohttp session initialized.")
            logger.debug(f"{self.object_id}: Starting control_pool_loop")
            asyncio.create_task(self.control_pool_loop())
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to initialize session or start control loop: {e}", exc_info=True)
        await super().on_start()

    async def on_stop(self):
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug(f"{self.object_id}: Closed aiohttp session")
        self.session = None
        await super().on_stop()

    async def get_weather_data(self):
        is_currently_raining = 0
        outdoor_temp = 75.0
        forecast_temp = 77.0
        if self.session is None:
            logger.warning(f"{self.object_id}: Session not initialized, using default weather values")
            return outdoor_temp, forecast_temp, is_currently_raining

        try:
            async with self.session.get(self.config["weather_url"]) as response:
                if response.status == 200:
                    current_data = await response.json()
                    outdoor_temp = current_data["main"]["temp"]
                    weather_conditions = current_data.get("weather", [])
                    if weather_conditions:
                        current_weather_code = weather_conditions[0].get("id")
                        if current_weather_code in self.config.get("rain_condition_codes", []):
                            is_currently_raining = 1
                else:
                    logger.warning(f"{self.object_id}: Weather API error (current): {response.status}")
        except Exception as e:
            logger.warning(f"{self.object_id}: Weather API error (current): {e}")

        try:
            async with self.session.get(self.config["forecast_url"]) as response:
                if response.status == 200:
                    forecast_data = await response.json()
                    forecast_temp = forecast_data["list"][0]["main"]["temp"]
                else:
                    logger.warning(f"{self.object_id}: Weather API error (forecast): {response.status}")
        except Exception as e:
            logger.warning(f"{self.object_id}: Weather API error (forecast): {e}")

        return outdoor_temp, forecast_temp, is_currently_raining

    async def get_current_state(self):
        logger.debug(f"{self.object_id}: Fetching current pool state")
        state = self.state_space.copy()
        if self.session is None:
            logger.error(f"{self.object_id}: Session not initialized, returning default state")
            return np.array(list(state.values()), dtype=np.float32)

        headers = {"Authorization": f"Bearer {self.ha_token}"}
        try:
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['water_temp_sensor_entity']}", headers=headers) as resp:
                state["temperature"] = float((await resp.json())["state"]) if resp.status == 200 else 70.0
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['ph_sensor_entity']}", headers=headers) as resp:
                state["ph"] = float((await resp.json())["state"]) if resp.status == 200 else 7.2
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['orp_sensor_entity']}", headers=headers) as resp:
                state["orp"] = float((await resp.json())["state"]) if resp.status == 200 else 650.0
            state["chlorine_level"] = self.infer_chlorine_from_orp(state["orp"], state["ph"])
            async with self.session.get(f"{self.ha_url}/api/states/input_number.pool_total_alkalinity", headers=headers) as resp:
                state["total_alkalinity"] = float((await resp.json())["state"]) if resp.status == 200 else 80.0
            async with self.session.get(f"{self.ha_url}/api/states/input_number.pool_calcium_hardness", headers=headers) as resp:
                state["calcium_hardness"] = float((await resp.json())["state"]) if resp.status == 200 else 200.0
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['pump_entity']}", headers=headers) as resp:
                state["pump_speed"] = float((await resp.json())["state"]) if resp.status == 200 else 0.0
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['heater_entity']}", headers=headers) as resp:
                heater_data = await resp.json() if resp.status == 200 else {}
                state["heater_on"] = heater_data.get("attributes", {}).get("hvac_mode", "off") == "heat"
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['lights_entity']}", headers=headers) as resp:
                state["lights_on"] = (await resp.json())["state"] == "on" if resp.status == 200 else False
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['cleaner_entity']}", headers=headers) as resp:
                state["cleaner_on"] = (await resp.json())["state"] == "on" if resp.status == 200 else False
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['door_alarm_entity']}", headers=headers) as resp:
                state["door_alarm"] = (await resp.json())["state"] == "on" if resp.status == 200 else False
            state["hour"] = float(datetime.now().hour)
            state["day_of_week"] = datetime.now().weekday()
            outdoor_temp, forecast_temp, is_raining = await self.get_weather_data()
            state["outdoor_temp"] = outdoor_temp
            state["forecast_temp"] = forecast_temp
            state["is_raining"] = is_raining
            logger.debug(f"{self.object_id}: State fetched: {state}")
        except Exception as e:
            logger.error(f"{self.object_id}: Error getting pool state: {e}", exc_info=True)
            state = self.state_space.copy()

        state = self.validate_state(state)
        state = self.smooth_state(state)
        return np.array(list(state.values()), dtype=np.float32)

    def infer_chlorine_from_orp(self, orp, ph):
        chlorine = (orp / 200) - (ph - 7.4) * 0.5
        return np.clip(chlorine, 0, 5)

    def validate_state(self, state):
        state["temperature"] = np.clip(state["temperature"], 60, 90)
        state["chlorine_level"] = np.clip(state["chlorine_level"], 0, 5)
        state["ph"] = np.clip(state["ph"], 6.0, 8.0)
        state["orp"] = np.clip(state["orp"], 0, 1000)
        state["total_alkalinity"] = np.clip(state["total_alkalinity"], 50, 150)
        state["calcium_hardness"] = np.clip(state["calcium_hardness"], 100, 400)
        state["pump_speed"] = np.clip(state["pump_speed"], 0, 100)
        state["heater_on"] = bool(state["heater_on"])
        state["lights_on"] = bool(state["lights_on"])
        state["cleaner_on"] = bool(state["cleaner_on"])
        state["door_alarm"] = bool(state["door_alarm"])
        state["hour"] = np.clip(state["hour"], 0, 23)
        state["outdoor_temp"] = np.clip(state["outdoor_temp"], -20, 120)
        state["forecast_temp"] = np.clip(state["forecast_temp"], -20, 120)
        state["day_of_week"] = np.clip(state["day_of_week"], 0, 6)
        state["is_raining"] = np.clip(state["is_raining"], 0, 1)
        return state

    def smooth_state(self, state):
        state_array = np.array(list(state.values()), dtype=np.float32)
        for i, key in enumerate(self.state_space.keys()):
            self.state_buffer[key].append(state_array[i])
            if len(self.state_buffer[key]) > self.smoothing_window:
                self.state_buffer[key].pop(0)
            self.state_space[key] = np.mean(self.state_buffer[key])
        return np.array(list(self.state_space.values()), dtype=np.float32)

    async def update_energy_usage(self, pump_speed, heater_on, lights_on, cleaner_on):
        time_now = datetime.now()
        time_delta = (time_now - self.last_energy_update).total_seconds() / 3600
        energy_increment = (pump_speed / 100 * 0.1 + heater_on * 2.0 + lights_on * 0.05 + cleaner_on * 0.2) * time_delta
        self.energy_usage += energy_increment
        self.last_energy_update = time_now
        logger.info(f"{self.object_id}: Energy increment: {energy_increment:.4f} kWh, Total: {self.energy_usage:.2f} kWh")
        await self.context.dispatch_event(Event(
            type=Actions.SYSTEM_STATE_UPDATE,
            data=UniversalCode(Tags.DICT, {"pool_energy_usage": self.energy_usage, "increment": energy_increment}),
            origin=self.object_id
        ))
        return energy_increment

    async def apply_action(self, service: str, entity_id: str, value=None, min_val=None, max_val=None):
        action_name = service.split("/")[-1]
        log_message = f"{self.object_id}: Applying {action_name} on {entity_id}"
        service_data = {"entity_id": entity_id}
        if value is not None:
            clipped_value = np.clip(value, min_val, max_val) if min_val is not None and max_val is not None else value
            service_data["value"] = clipped_value
            log_message += f" to {clipped_value}"
        logger.debug(log_message)
        return await self._call_ha_service(*service.split("/"), service_data)

    async def apply_heater_action(self, heater_temp, heater_active):
        if self.session is None:
            logger.error(f"{self.object_id}: Session not initialized, cannot apply heater action")
            return

        if heater_active and heater_temp > 0:
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['heater_entity']}", headers={"Authorization": f"Bearer {self.ha_token}"}) as resp:
                heater_data = await resp.json() if resp.status == 200 else {}
            current_hvac_mode = heater_data.get("attributes", {}).get("hvac_mode", "off")
            current_temp = heater_data.get("attributes", {}).get("temperature", 0)
            if current_hvac_mode != "heat" or current_temp != heater_temp:
                logger.debug(f"{self.object_id}: Setting heater to heat at {heater_temp}°F")
                await self._call_ha_service("climate", "set_temperature", {"entity_id": self.config["heater_entity"], "temperature": heater_temp})
                await self._call_ha_service("climate", "set_hvac_mode", {"entity_id": self.config["heater_entity"], "hvac_mode": "heat"})
        else:
            async with self.session.get(f"{self.ha_url}/api/states/{self.config['heater_entity']}", headers={"Authorization": f"Bearer {self.ha_token}"}) as resp:
                heater_data = await resp.json() if resp.status == 200 else {}
            if heater_data.get("attributes", {}).get("hvac_mode", "off") != "off":
                logger.debug(f"{self.object_id}: Turning off heater")
                await self._call_ha_service("climate", "set_hvac_mode", {"entity_id": self.config["heater_entity"], "hvac_mode": "off"})

    async def apply_light_action(self, lights_on):
        if self.session is None:
            logger.error(f"{self.object_id}: Session not initialized, cannot apply light action")
            return

        async with self.session.get(f"{self.ha_url}/api/states/{self.config['door_alarm_entity']}", headers={"Authorization": f"Bearer {self.ha_token}"}) as resp:
            door_state = (await resp.json())["state"] if resp.status == 200 else "off"
        if lights_on and door_state != "on":
            logger.debug(f"{self.object_id}: Turning on pool lights")
            await self._call_ha_service("light", "turn_on", {"entity_id": self.config["lights_entity"]})
        elif not lights_on:
            logger.debug(f"{self.object_id}: Turning off pool lights")
            await self._call_ha_service("light", "turn_off", {"entity_id": self.config["lights_entity"]})

    async def check_cleaner_schedule(self):
        now = datetime.now()
        last_run = self.cleaner_schedule["last_run"]
        interval_days = self.cleaner_schedule["interval_days"]
        days_since_last = (now - last_run).days
        if days_since_last >= interval_days and not self.cleaner_running and now.hour == self.cleaner_schedule["run_hour"]:
            logger.info(f"{self.object_id}: Starting scheduled cleaner run")
            self.cleaner_running = True
            await self._call_ha_service("switch", "turn_on", {"entity_id": self.config["cleaner_entity"]})
            await self.context.dispatch_event(Event(
                type=Actions.SYSTEM_STATE_UPDATE,
                data=UniversalCode(Tags.DICT, {"pool_cleaner": "started", "timestamp": now.isoformat()}),
                origin=self.object_id
            ))
            await asyncio.sleep(self.cleaner_run_duration)
            self.cleaner_running = False
            await self._call_ha_service("switch", "turn_off", {"entity_id": self.config["cleaner_entity"]})
            self.cleaner_schedule["last_run"] = now
            logger.info(f"{self.object_id}: Completed scheduled cleaner run")
            await self.context.dispatch_event(Event(
                type=Actions.SYSTEM_STATE_UPDATE,
                data=UniversalCode(Tags.DICT, {"pool_cleaner": "stopped", "timestamp": now.isoformat()}),
                origin=self.object_id
            ))

    async def handle_event(self, event: Event):
        if event.type != "POOL_CONTROL" or not isinstance(event.data, UniversalCode) or event.data.tag != Tags.DICT:
            return
        try:
            action_data = event.data.value
            action = action_data.get("action")
            logger.debug(f"{self.object_id}: Processing POOL_CONTROL: action={action}")
            if action == "control_pool":
                await self.control_pool_loop()
                return True
            logger.warning(f"{self.object_id}: Unsupported action {action}")
            return False
        except Exception as e:
            logger.error(f"{self.object_id}: Error processing POOL_CONTROL: {e}", exc_info=True)
            return False

    async def control_pool_loop(self):
        if self.session is None:
            logger.error(f"{self.object_id}: Session not initialized, cannot start control loop")
            return

        logger.debug(f"{self.object_id}: Entering control_pool_loop")
        while not self.context.stop_event.is_set():
            try:
                logger.debug(f"{self.object_id}: Starting control loop iteration")
                target_temp_high = float((await self._get_ha_state(self.config["target_temp_entity"]))["state"]) if await self._get_ha_state(self.config["target_temp_entity"]) else 87.0
                target_temp_low = target_temp_high - 2
                target_chlorine = float((await self._get_ha_state(self.config["target_chlorine_entity"]))["state"]) if await self._get_ha_state(self.config["target_chlorine_entity"]) else 3.0
                target_ph = float((await self._get_ha_state(self.config["target_ph_entity"]))["state"]) if await self._get_ha_state(self.config["target_ph_entity"]) else 7.4
                target_orp = float((await self._get_ha_state(self.config["target_orp_entity"]))["state"]) if await self._get_ha_state(self.config["target_orp_entity"]) else 700.0
                target_alkalinity = float((await self._get_ha_state(self.config["target_alkalinity_entity"]))["state"]) if await self._get_ha_state(self.config["target_alkalinity_entity"]) else 100.0
                target_hardness = float((await self._get_ha_state(self.config["target_hardness_entity"]))["state"]) if await self._get_ha_state(self.config["target_hardness_entity"]) else 250.0

                current_day_of_week = datetime.now().weekday()
                is_raining = self.state_space.get("is_raining", 0.0)
                target_temp = target_temp_low if current_day_of_week < 5 else target_temp_high
                if is_raining == 1:
                    target_temp = min(target_temp, self.config.get("rain_reduction_temp", 83.0))

                state = await self.get_current_state()
                logger.debug(f"{self.object_id}: Predicting actions with state: {state}")
                actions = [model.predict(state, deterministic=True)[0] if hasattr(model, "predict") else model(state) for model in self.models]
                action = np.mean([act for act in actions if act is not None], axis=0) if any(actions) else np.zeros(9, dtype=np.float32)
                logger.debug(f"{self.object_id}: Predicted action: {action}")

                pump_speed, chlorinator_on, liquid_chlorine_feed, acid_feed, alkalinity_feed, calcium_feed, heater_temp, lights_on, cleaner_on = action
                if self.cleaner_running:
                    cleaner_on = 1

                now = datetime.now()
                time_since_reset = (now - self.last_dosing_reset).total_seconds() / 60
                if time_since_reset >= 60:
                    self.dosing_counters = {key: 0 for key in self.dosing_counters}
                    self.last_dosing_reset = now
                else:
                    if chlorinator_on > 0.5 and self.dosing_counters["chlorinator"] >= 5:
                        chlorinator_on = 0
                    if liquid_chlorine_feed > 0.5 and self.dosing_counters["liquid_chlorine"] >= 5:
                        liquid_chlorine_feed = 0
                    if acid_feed > 0.5 and self.dosing_counters["acid"] >= 5:
                        acid_feed = 0
                    if alkalinity_feed > 0.5 and self.dosing_counters["alkalinity"] >= 5:
                        alkalinity_feed = 0
                    if calcium_feed > 0.5 and self.dosing_counters["calcium"] >= 5:
                        calcium_feed = 0

                if chlorinator_on > 0.5:
                    self.dosing_counters["chlorinator"] += 10
                if liquid_chlorine_feed > 0.5:
                    self.dosing_counters["liquid_chlorine"] += 10
                if acid_feed > 0.5:
                    self.dosing_counters["acid"] += 10
                if alkalinity_feed > 0.5:
                    self.dosing_counters["alkalinity"] += 10
                if calcium_feed > 0.5:
                    self.dosing_counters["calcium"] += 10

                heater_active = 1 if heater_temp > 0 else 0
                if state[11] >= self.config["peak_hours_start"] and state[11] < self.config["peak_hours_end"]:
                    if state[0] >= target_temp - 2:
                        heater_active = 0
                if state[13] > target_temp + 2:
                    heater_active = 0 if state[0] >= target_temp - 2 else heater_active

                logger.debug(f"{self.object_id}: Applying actions: pump_speed={pump_speed}, chlorinator_on={chlorinator_on}, ...")
                await self.apply_action("number/set_value", self.config["pump_entity"], pump_speed, 0, 100)
                await self.apply_action("switch/turn_on" if chlorinator_on > 0.5 else "switch/turn_off", self.config["chlorinator_entity"])
                await self.apply_action("switch/turn_on" if liquid_chlorine_feed > 0.5 else "switch/turn_off", self.config["liquid_chlorine_feed_entity"])
                await self.apply_action("switch/turn_on" if acid_feed > 0.5 else "switch/turn_off", self.config["acid_feed_entity"])
                await self.apply_action("switch/turn_on" if alkalinity_feed > 0.5 else "switch/turn_off", self.config["alkalinity_feed_entity"])
                await self.apply_action("switch/turn_on" if calcium_feed > 0.5 else "switch/turn_off", self.config["calcium_feed_entity"])
                await self.apply_heater_action(heater_temp, heater_active)
                await self.apply_light_action(lights_on > 0.5)
                await self.apply_action("switch/turn_on" if cleaner_on > 0.5 else "switch/turn_off", self.config["cleaner_entity"])

                await self.update_energy_usage(pump_speed, heater_active, lights_on > 0.5, cleaner_on > 0.5)

                logger.info(f"{self.object_id}: Pool state: Temp={state[0]:.1f}°F, Chlorine={state[1]:.1f} ppm, pH={state[2]:.1f}, ORP={state[3]:.1f} mV, Alkalinity={state[4]:.1f} ppm, Hardness={state[5]:.1f} ppm")
                await self.context.dispatch_event(Event(
                    type=Actions.SYSTEM_STATE_UPDATE,
                    data=UniversalCode(Tags.DICT, {
                        "pool_chemistry": {
                            "temperature": state[0], "chlorine": state[1], "ph": state[2],
                            "orp": state[3], "alkalinity": state[4], "hardness": state[5]
                        }
                    }),
                    origin=self.object_id
                ))

                if abs(state[0] - target_temp) > 5:
                    logger.warning(f"{self.object_id}: Pool temperature out of range: {state[0]:.1f}°F (target: {target_temp:.1f}°F)")
                if abs(state[1] - target_chlorine) > 1:
                    logger.warning(f"{self.object_id}: Pool chlorine out of range: {state[1]:.1f} ppm (target: {target_chlorine:.1f} ppm)")

            except Exception as e:
                logger.error(f"{self.object_id}: Error in pool control loop: {e}", exc_info=True)

            logger.debug(f"{self.object_id}: Control loop iteration complete, sleeping for 600 seconds")
            await asyncio.sleep(600)  # Run every 10 minutes

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
            logger.error(f"{self.object_id}: Error calling service {domain}.{service}: {e}", exc_info=True)
            return False

    async def _get_ha_state(self, entity_id: str) -> Optional[Dict]:
        if not self.session:
            logger.error(f"{self.object_id}: aiohttp session not initialized")
            return None
        headers = {"Authorization": f"Bearer {self.ha_token}", "Content-Type": "application/json"}
        try:
            async with self.session.get(f"{self.ha_url}/api/states/{entity_id}", headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                logger.error(f"{self.object_id}: Failed to get state for {entity_id}: {response.status}")
                return None
        except Exception as e:
            logger.error(f"{self.object_id}: Error getting state for {entity_id}: {e}", exc_info=True)
            return None

    async def process_single_event(self, event: Event):
        await self.handle_event(event)
