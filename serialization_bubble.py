# serialization_bubble.py
"""
Production-ready Serialization Bubble that handles complex types including Enums,
providing robust multi-format serialization for the Bubbles Framework.
Solves enum serialization issues and ensures UniversalCode compatibility.
"""

import json
import asyncio
import time
import struct
import hashlib
import pickle
import gzip
import base64
from typing import Dict, Any, Optional, Union, List, Tuple, Type
from collections import deque, defaultdict, namedtuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict, is_dataclass
import importlib
import logging

# Handle numpy as optional dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# Handle msgpack as optional dependency
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    msgpack = None
    MSGPACK_AVAILABLE = False

from bubbles_core import (
    UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions,
    EventService, logger, InvalidTagError, BinaryConverter
)


class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = "json"
    BINARY = "binary"
    PICKLE = "pickle"
    COMPRESSED = "compressed"
    MSGPACK = "msgpack"


@dataclass
class SerializationMetrics:
    """Tracks serialization performance and statistics"""
    total_serializations: int = 0
    failed_serializations: int = 0
    format_usage: Dict[str, int] = None
    average_size_bytes: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    universal_code_creations: int = 0
    universal_code_failures: int = 0
    
    def __post_init__(self):
        if self.format_usage is None:
            self.format_usage = defaultdict(int)
class SerializationBubble(UniversalBubble):
    """
    Serialization bubble that handles complex types including all system enums,
    provides multi-format serialization, and ensures UniversalCode compatibility.
    
    Key features:
    - Handles Enum serialization (solves HealthStatus problem)
    - Multiple serialization formats (JSON, Binary, Pickle, etc.)
    - Caching for performance
    - UniversalCode integration
    - Comprehensive error handling
    """
    
    def __init__(self, object_id: str, context: SystemContext, 
                 cache_size: int = 1000, default_format: str = "json", **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Configuration
        self.default_format = SerializationFormat(default_format)
        self.cache_size = cache_size
        
        # Serialization cache for performance
        self.serialization_cache = {}
        self.cache_order = deque(maxlen=cache_size)
        
        # Metrics tracking
        self.metrics = SerializationMetrics()
        
        # Custom type handlers - this is where we solve the Enum problem
        self.type_handlers = {
            Enum: self._serialize_enum,
            datetime: self._serialize_datetime,
            type: self._serialize_type,
            bytes: self._serialize_bytes,
            bytearray: self._serialize_bytes,
            memoryview: self._serialize_memoryview,
            set: self._serialize_set,
            frozenset: self._serialize_frozenset,
            Event: self._serialize_event,
            UniversalCode: self._serialize_universal_code,
            Exception: self._serialize_exception
        }
        
        # Add numpy handler if available
        if NUMPY_AVAILABLE:
            self.type_handlers[np.ndarray] = self._serialize_numpy
        
        # Deserialization handlers
        self.deserialize_handlers = {
            "Enum": self._deserialize_enum,
            "datetime": self._deserialize_datetime,
            "ndarray": self._deserialize_numpy,
            "type": self._deserialize_type,
            "bytes": self._deserialize_bytes,
            "bytearray": self._deserialize_bytearray,
            "memoryview": self._deserialize_memoryview,
            "set": self._deserialize_set,
            "frozenset": self._deserialize_frozenset,
            "tuple": self._deserialize_tuple,
            "Event": self._deserialize_event,
            "UniversalCode": self._deserialize_universal_code,
            "Exception": self._deserialize_exception,
            "dataclass": self._deserialize_dataclass,
            "object": self._deserialize_object,
            "fallback": self._deserialize_fallback
        }
        
        # Initialize event subscriptions
        asyncio.create_task(self._subscribe_to_events())
        
        logger.info(f"{self.object_id}: Serialization Bubble initialized with format {self.default_format.value}")
    
    async def _subscribe_to_events(self):
        """Subscribe to serialization-related events"""
        await asyncio.sleep(0.1)
        try:
            # Check if the serialization actions exist in the Actions enum
            if hasattr(Actions, 'SERIALIZE_OBJECT') and hasattr(Actions, 'DESERIALIZE_OBJECT'):
                # Use the proper enum members if they exist
                await EventService.subscribe(Actions.SERIALIZE_OBJECT, self.handle_event)
                await EventService.subscribe(Actions.DESERIALIZE_OBJECT, self.handle_event)
                logger.info(f"{self.object_id}: Subscribed to serialization events (event-driven mode)")
            else:
                # Log that we're running in direct-call mode
                logger.info(f"{self.object_id}: Serialization actions not found in Actions enum. "
                           "Running in direct-call mode. Add SERIALIZE_OBJECT and DESERIALIZE_OBJECT "
                           "to Actions enum in bubbles_core.py to enable event-driven mode.")
                
                # Optional: Subscribe to alternative existing events
                # Uncomment the following lines if you want to use existing events as a workaround
                # if hasattr(Actions, 'API_CALL'):
                #     await EventService.subscribe(Actions.API_CALL, self.handle_event)
                #     logger.debug(f"{self.object_id}: Subscribed to API_CALL as alternative")
                
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to subscribe to events: {e}")
            logger.info(f"{self.object_id}: Will operate in direct-call mode")






    # ==================== Type-specific serialization handlers ====================
    
    def _serialize_enum(self, obj: Enum) -> Dict[str, Any]:
        """Handle Enum serialization - this solves the HealthStatus problem!"""
        return {
            "__type__": "Enum",
            "__module__": obj.__class__.__module__,
            "__class__": obj.__class__.__name__,
            "name": obj.name,
            "value": obj.value
        }
    
    def _serialize_datetime(self, obj: datetime) -> Dict[str, Any]:
        """Handle datetime serialization"""
        return {
            "__type__": "datetime",
            "iso": obj.isoformat(),
            "timestamp": obj.timestamp()
        }
    
    def _serialize_numpy(self, obj: np.ndarray) -> Dict[str, Any]:
        """Handle numpy array serialization"""
        return {
            "__type__": "ndarray",
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "data": obj.tolist()
        }
    
    def _serialize_type(self, obj: type) -> Dict[str, Any]:
        """Handle type object serialization"""
        return {
            "__type__": "type",
            "__module__": obj.__module__,
            "__class__": obj.__name__
        }
    
    def _serialize_bytes(self, obj: Union[bytes, bytearray]) -> Dict[str, Any]:
        """Handle bytes and bytearray serialization"""
        return {
            "__type__": "bytes" if isinstance(obj, bytes) else "bytearray",
            "data": base64.b64encode(bytes(obj)).decode('ascii')
        }
    
    def _serialize_memoryview(self, obj: memoryview) -> Dict[str, Any]:
        """Handle memoryview serialization"""
        return {
            "__type__": "memoryview",
            "data": bytes(obj).hex(),
            "format": obj.format,
            "shape": obj.shape if obj.shape else None
        }
    
    def _serialize_set(self, obj: set) -> Dict[str, Any]:
        """Handle set serialization"""
        return {
            "__type__": "set",
            "items": [self._make_serializable(item) for item in obj]
        }
    
    def _serialize_frozenset(self, obj: frozenset) -> Dict[str, Any]:
        """Handle frozenset serialization"""
        return {
            "__type__": "frozenset",
            "items": [self._make_serializable(item) for item in obj]
        }
    
    def _serialize_event(self, obj: Event) -> Dict[str, Any]:
        """Handle Event serialization"""
        return {
            "__type__": "Event",
            "type": obj.type.value if isinstance(obj.type, Enum) else str(obj.type),
            "data": self._make_serializable(obj.data),
            "origin": obj.origin,
            "priority": obj.priority,
            "timestamp": obj.timestamp
        }
    
    def _serialize_universal_code(self, obj: UniversalCode) -> Dict[str, Any]:
        """Handle UniversalCode serialization"""
        return {
            "__type__": "UniversalCode",
            "tag": obj.tag.value if isinstance(obj.tag, Enum) else str(obj.tag),
            "value": self._make_serializable(obj.value),
            "description": obj.description,
            "metadata": self._make_serializable(obj.metadata) if obj.metadata else None
        }
    
    def _serialize_exception(self, obj: Exception) -> Dict[str, Any]:
        """Handle exception serialization"""
        return {
            "__type__": "Exception",
            "__module__": obj.__class__.__module__,
            "__class__": obj.__class__.__name__,
            "args": obj.args,
            "message": str(obj)
        }
    
    # ==================== Deserialization handlers ====================
    
    def _deserialize_enum(self, data: Dict[str, Any]) -> Any:
        """Deserialize Enum objects"""
        try:
            # Special handling for known system enums
            if data['__module__'] == 'bubbles_core':
                if data['__class__'] == 'Tags':
                    return Tags[data['name']]
                elif data['__class__'] == 'Actions':
                    return Actions[data['name']]
            
            # Try to import and restore the actual enum
            module = importlib.import_module(data['__module__'])
            enum_class = getattr(module, data['__class__'])
            
            # Try to get by name first (more reliable)
            try:
                return enum_class[data['name']]
            except KeyError:
                # Fall back to value
                return enum_class(data['value'])
                
        except Exception as e:
            logger.warning(f"Could not restore enum {data['__class__']}: {e}")
            # Return a named tuple as fallback for compatibility
            EnumProxy = namedtuple('EnumProxy', ['name', 'value', 'enum_type'])
            return EnumProxy(
                name=data['name'],
                value=data['value'],
                enum_type=f"{data['__module__']}.{data['__class__']}"
            )
    
    def _deserialize_datetime(self, data: Dict[str, Any]) -> datetime:
        """Deserialize datetime objects"""
        return datetime.fromisoformat(data["iso"])
    
    def _deserialize_numpy(self, data: Dict[str, Any]) -> Any:
        """Deserialize numpy arrays"""
        if NUMPY_AVAILABLE:
            return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])
        else:
            # Return as list if numpy not available
            return data["data"]
    
    def _deserialize_type(self, data: Dict[str, Any]) -> Any:
        """Deserialize type objects"""
        try:
            module = importlib.import_module(data['__module__'])
            return getattr(module, data['__class__'])
        except Exception:
            # Return string representation if can't import
            return f"{data['__module__']}.{data['__class__']}"
    
    def _deserialize_bytes(self, data: Dict[str, Any]) -> bytes:
        """Deserialize bytes objects"""
        return base64.b64decode(data["data"])
    
    def _deserialize_bytearray(self, data: Dict[str, Any]) -> bytearray:
        """Deserialize bytearray objects"""
        return bytearray(base64.b64decode(data["data"]))
    
    def _deserialize_memoryview(self, data: Dict[str, Any]) -> memoryview:
        """Deserialize memoryview objects"""
        byte_data = bytes.fromhex(data["data"])
        return memoryview(byte_data)
    
    def _deserialize_set(self, data: Dict[str, Any]) -> set:
        """Deserialize set objects"""
        return set(self._restore_from_serializable(item) for item in data["items"])
    
    def _deserialize_frozenset(self, data: Dict[str, Any]) -> frozenset:
        """Deserialize frozenset objects"""
        return frozenset(self._restore_from_serializable(item) for item in data["items"])
    
    def _deserialize_tuple(self, data: Dict[str, Any]) -> tuple:
        """Deserialize tuple objects"""
        return tuple(self._restore_from_serializable(item) for item in data["items"])
    
    def _deserialize_event(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize Event objects (returns dict representation)"""
        return {
            "type": data["type"],
            "data": self._restore_from_serializable(data["data"]),
            "origin": data["origin"],
            "priority": data["priority"],
            "timestamp": data["timestamp"]
        }
    
    def _deserialize_universal_code(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize UniversalCode objects (returns dict representation)"""
        return {
            "tag": data["tag"],
            "value": self._restore_from_serializable(data["value"]),
            "description": data["description"],
            "metadata": self._restore_from_serializable(data["metadata"]) if data["metadata"] else None
        }
    
    def _deserialize_exception(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize exception objects (as dict representation)"""
        return {
            "exception_type": f"{data['__module__']}.{data['__class__']}",
            "args": data["args"],
            "message": data["message"]
        }
    
    def _deserialize_dataclass(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize dataclass objects (returns dict representation)"""
        return data["data"]
    
    def _deserialize_object(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize generic objects (returns attributes dict)"""
        return data["attributes"]
    
    def _deserialize_fallback(self, data: Dict[str, Any]) -> str:
        """Deserialize fallback objects (returns string representation)"""
        return data["string"]
    
    # ==================== Core serialization methods ====================
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert any object to a JSON-serializable form"""
        # Handle None and basic types
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Check custom type handlers
        for type_class, handler in self.type_handlers.items():
            if isinstance(obj, type_class):
                return handler(obj)
        
        # Handle dataclasses
        if is_dataclass(obj):
            return {
                "__type__": "dataclass",
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "data": asdict(obj)
            }
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        
        # Handle lists
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        
        # Handle tuples
        if isinstance(obj, tuple):
            return {
                "__type__": "tuple",
                "items": [self._make_serializable(item) for item in obj]
            }
        
        # Handle objects with __dict__
        if hasattr(obj, "__dict__"):
            return {
                "__type__": "object",
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "attributes": self._make_serializable(obj.__dict__)
            }
        
        # Fallback to string representation
        return {
            "__type__": "fallback",
            "__class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "string": str(obj)
        }
    
    def _restore_from_serializable(self, obj: Any) -> Any:
        """Restore Python objects from serializable form"""
        if isinstance(obj, dict) and "__type__" in obj:
            obj_type = obj["__type__"]
            
            # Use specific deserializer if available
            if obj_type in self.deserialize_handlers:
                return self.deserialize_handlers[obj_type](obj)
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._restore_from_serializable(v) for k, v in obj.items()}
        
        # Handle lists recursively
        if isinstance(obj, list):
            return [self._restore_from_serializable(item) for item in obj]
        
        # Return as-is for basic types
        return obj
    
    def serialize(self, obj: Any, format: SerializationFormat = None) -> Union[str, bytes]:
        """
        Serialize object to specified format.
        Returns string for JSON format, bytes for others.
        """
        format = format or self.default_format
        
        # Check cache
        cache_key = self._get_cache_key(obj, format)
        if cache_key in self.serialization_cache:
            self.metrics.cache_hits += 1
            self.metrics.format_usage[format.value] += 1
            return self.serialization_cache[cache_key]
        
        self.metrics.cache_misses += 1
        
        try:
            # Convert to serializable form
            serializable = self._make_serializable(obj)
            
            # Serialize based on format
            if format == SerializationFormat.JSON:
                result = json.dumps(serializable, indent=2)
            
            elif format == SerializationFormat.BINARY:
                # Simple binary format using JSON + struct
                json_bytes = json.dumps(serializable).encode('utf-8')
                length = struct.pack('>I', len(json_bytes))
                result = length + json_bytes
            
            elif format == SerializationFormat.PICKLE:
                result = pickle.dumps(obj)
            
            elif format == SerializationFormat.COMPRESSED:
                json_str = json.dumps(serializable)
                result = gzip.compress(json_str.encode())
            
            elif format == SerializationFormat.MSGPACK and MSGPACK_AVAILABLE:
                result = msgpack.packb(serializable)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Update metrics
            self.metrics.total_serializations += 1
            self.metrics.format_usage[format.value] += 1
            
            # Add to cache
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.metrics.failed_serializations += 1
            logger.error(f"{self.object_id}: Serialization failed for {type(obj).__name__}: {e}")
            
            # Fallback serialization
            fallback = {"error": str(e), "type": str(type(obj)), "repr": str(obj)}
            return json.dumps(fallback)
    
    def deserialize(self, data: Union[str, bytes], format: SerializationFormat = None) -> Any:
        """Deserialize data back to Python objects"""
        format = format or self._detect_format(data)
        
        try:
            if format == SerializationFormat.JSON:
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                obj = json.loads(data)
                return self._restore_from_serializable(obj)
            
            elif format == SerializationFormat.BINARY:
                # Extract length and JSON data
                length = struct.unpack('>I', data[:4])[0]
                json_bytes = data[4:4+length]
                obj = json.loads(json_bytes.decode('utf-8'))
                return self._restore_from_serializable(obj)
            
            elif format == SerializationFormat.PICKLE:
                return pickle.loads(data)
            
            elif format == SerializationFormat.COMPRESSED:
                decompressed = gzip.decompress(data)
                obj = json.loads(decompressed.decode('utf-8'))
                return self._restore_from_serializable(obj)
            
            elif format == SerializationFormat.MSGPACK and MSGPACK_AVAILABLE:
                obj = msgpack.unpackb(data, raw=False)
                return self._restore_from_serializable(obj)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"{self.object_id}: Deserialization failed: {e}")
            raise
    
    # ==================== Universal Code integration ====================
    
    def prepare_for_universal_code(self, value: Any, tag: Tags) -> Any:
        """
        Prepare any value to be compatible with UniversalCode validation.
        Ensures the value will pass UC's JSON validation for DICT and LIST tags.
        """
        # Validate tag type
        if not isinstance(tag, Tags):
            raise InvalidTagError(f"Invalid tag type: {type(tag)}. Must be a Tags enum member.")
        
        # For DICT and LIST tags, ensure JSON-serializability
        if tag in (Tags.DICT, Tags.LIST):
            # Convert to JSON-serializable form
            serializable = self._make_serializable(value)
            
            # Validate it will pass UC's check
            try:
                json.dumps(serializable, sort_keys=True)
            except (TypeError, ValueError) as e:
                logger.error(f"{self.object_id}: Failed to make value JSON-serializable: {e}")
                # Return a safe fallback
                return {"error": "serialization_failed", "type": str(type(value)), "repr": str(value)}
            
            return serializable
        
        # For BINARY tag, ensure bytes
        elif tag == Tags.BINARY:
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value)
            else:
                # Try to convert to bytes
                try:
                    return str(value).encode('utf-8')
                except Exception as e:
                    logger.error(f"{self.object_id}: Failed to convert to bytes: {e}")
                    return b''
        
        # For other tags, validate type compatibility
        expected_types = {
            Tags.INTEGER: int,
            Tags.FLOAT: float,
            Tags.STRING: str,
            Tags.BOOLEAN: bool,
            Tags.NULL: type(None)
        }
        
        if tag in expected_types:
            expected_type = expected_types[tag]
            if not isinstance(value, expected_type):
                # Try to convert
                try:
                    if tag == Tags.INTEGER:
                        return int(value)
                    elif tag == Tags.FLOAT:
                        return float(value)
                    elif tag == Tags.STRING:
                        return str(value)
                    elif tag == Tags.BOOLEAN:
                        return bool(value)
                    elif tag == Tags.NULL:
                        return None
                except Exception as e:
                    logger.error(f"{self.object_id}: Failed to convert to {expected_type}: {e}")
                    raise ValueError(f"Cannot convert {type(value)} to {tag.name}")
        
        return value
    
    def create_universal_code(self, 
                            tag: Tags, 
                            value: Any, 
                            binary_data: Optional[bytes] = None,
                            description: str = "",
                            metadata: Dict[str, Any] = None) -> UniversalCode:
        """
        Safely create a UniversalCode object, handling any serialization issues.
        """
        try:
            # Prepare the value
            safe_value = self.prepare_for_universal_code(value, tag)
            
            # Prepare metadata if provided
            safe_metadata = None
            if metadata:
                try:
                    # Ensure metadata is JSON-serializable
                    json.dumps(metadata)
                    safe_metadata = metadata
                except (TypeError, ValueError):
                    # Convert metadata to safe form
                    safe_metadata = self._make_serializable(metadata)
            
            # Generate binary data if needed and not provided
            if binary_data is None and tag == Tags.BINARY:
                binary_data = safe_value if isinstance(safe_value, bytes) else b''
            
            # Create the UniversalCode
            uc = UniversalCode(
                tag=tag,
                value=safe_value,
                binary_data=binary_data,
                description=description,
                metadata=safe_metadata
            )
            
            self.metrics.universal_code_creations += 1
            return uc
            
        except Exception as e:
            self.metrics.universal_code_failures += 1
            logger.error(f"{self.object_id}: Failed to create UniversalCode: {e}")
            
            # Return a safe error UC
            return UniversalCode(
                tag=Tags.DICT,
                value={
                    "error": str(e),
                    "original_type": str(type(value)),
                    "tag_requested": tag.name
                },
                description=f"Serialization error: {description}",
                metadata={"serialization_failed": True}
            )
    
    # ==================== Helper methods ====================
    
    def fix_hardware_event(self, event_data: Dict) -> Dict:
        """
        Fix hardware event data by properly serializing all complex types.
        This is the main method to solve your HealthStatus enum problem.
        """
        try:
            # Serialize and deserialize to ensure all types are converted
            serialized = self.serialize(event_data, format=SerializationFormat.JSON)
            fixed_data = self.deserialize(serialized, format=SerializationFormat.JSON)
            
            return fixed_data
            
        except Exception as e:
            logger.error(f"{self.object_id}: Failed to fix hardware event: {e}")
            # Return a safe fallback
            return {"error": "serialization_failed", "original_error": str(e)}
    
    def fix_for_universal_code(self, data: Any) -> Any:
        """
        Fix any data structure to be compatible with UniversalCode.
        This is an alias for fix_hardware_event but with a more general name.
        """
        return self.fix_hardware_event(data)
    
    async def process_hardware_event(self, event_data: Dict) -> Dict:
        """
        Process hardware event data for safe serialization.
        This is an async wrapper around fix_hardware_event for consistency.
        """
        return self.fix_hardware_event(event_data)
    
    def detect_best_tag(self, value: Any) -> Tags:
        """
        Detect the best Tags enum value for a given Python object.
        """
        if value is None:
            return Tags.NULL
        elif isinstance(value, bool):  # Check bool before int!
            return Tags.BOOLEAN
        elif isinstance(value, int):
            return Tags.INTEGER
        elif isinstance(value, float):
            return Tags.FLOAT
        elif isinstance(value, str):
            return Tags.STRING
        elif isinstance(value, (bytes, bytearray, memoryview)):
            return Tags.BINARY
        elif isinstance(value, dict):
            return Tags.DICT
        elif isinstance(value, (list, tuple)):
            return Tags.LIST
        else:
            # Complex objects default to DICT after serialization
            return Tags.DICT
    
    def can_serialize(self, obj: Any, format: SerializationFormat = None) -> bool:
        """Check if an object can be serialized without actually serializing it"""
        try:
            format = format or self.default_format
            cache_key = self._get_cache_key(obj, format)
            
            # Check cache first
            if cache_key in self.serialization_cache:
                return True
            
            # Try to make it serializable
            self._make_serializable(obj)
            return True
        except Exception:
            return False
    
    def get_problem_types(self, obj: Any, path: str = "root") -> List[str]:
        """Identify problematic types in an object structure"""
        problems = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                problems.extend(self.get_problem_types(value, f"{path}.{key}"))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                problems.extend(self.get_problem_types(item, f"{path}[{i}]"))
        else:
            # Check if this type would cause problems
            try:
                json.dumps(obj)
            except TypeError:
                problems.append(f"{path}: {type(obj).__name__}")
        
        return problems
    
    def serialize_batch(self, objects: List[Any], format: SerializationFormat = None) -> List[Union[str, bytes]]:
        """Serialize multiple objects efficiently"""
        format = format or self.default_format
        results = []
        
        for obj in objects:
            try:
                result = self.serialize(obj, format)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch serialization failed for item: {e}")
                results.append(None)
        
        return results
    
    # ==================== Cache management ====================
    
    def _get_cache_key(self, obj: Any, format: SerializationFormat) -> str:
        """Generate cache key for object"""
        # Create a representation of the object for caching
        try:
            # For simple types, use the value directly
            if isinstance(obj, (str, int, float, bool, type(None))):
                obj_repr = f"{type(obj).__name__}:{obj}"
            # For other types, use type name and id
            else:
                obj_repr = f"{type(obj).__name__}:{id(obj)}"
            
            key_str = f"{obj_repr}:{format.value}"
            return hashlib.md5(key_str.encode()).hexdigest()
        except:
            # Fallback for unhashable objects
            return f"{id(obj)}:{format.value}"
    
    def _add_to_cache(self, key: str, value: Any):
        """Add to cache with LRU eviction"""
        if len(self.serialization_cache) >= self.cache_size:
            # Remove oldest entry
            if self.cache_order:
                oldest = self.cache_order.popleft()
                self.serialization_cache.pop(oldest, None)
        
        self.serialization_cache[key] = value
        self.cache_order.append(key)
    
    def clear_cache(self):
        """Clear the serialization cache"""
        self.serialization_cache.clear()
        self.cache_order.clear()
        logger.info(f"{self.object_id}: Cache cleared")
    
    def _detect_format(self, data: Union[str, bytes]) -> SerializationFormat:
        """Detect serialization format from data"""
        if isinstance(data, str):
            return SerializationFormat.JSON
        elif isinstance(data, bytes):
            # Check for gzip magic number
            if data.startswith(b'\x1f\x8b'):
                return SerializationFormat.COMPRESSED
            # Check for pickle protocol
            elif data.startswith(b'\x80'):
                return SerializationFormat.PICKLE
            # Check for msgpack
            elif MSGPACK_AVAILABLE and len(data) > 0:
                try:
                    # Try to unpack first few bytes
                    msgpack.unpackb(data[:10], raw=False)
                    return SerializationFormat.MSGPACK
                except:
                    pass
            # Default to binary
            return SerializationFormat.BINARY
        else:
            raise ValueError(f"Cannot detect format for type {type(data)}")
    
    # ==================== Event handling ====================
    
    async def handle_event(self, event: Event):
        """Handle serialization-related events"""
        try:
            if event.type == Actions.SERIALIZE_OBJECT:
                await self._handle_serialize_request(event)
            
            elif event.type == Actions.DESERIALIZE_OBJECT:
                await self._handle_deserialize_request(event)
                
        except Exception as e:
            logger.error(f"{self.object_id}: Error handling event: {e}")
            
            # Send error response
            error_uc = self.create_universal_code(
                Tags.DICT,
                {"error": str(e), "event_type": str(event.type)},
                description="Serialization error"
            )
            
            await self.context.dispatch_event(
                Event(
                    type=Actions.SERIALIZATION_ERROR,
                    data=error_uc,
                    origin=self.object_id,
                    priority=2
                )
            )
    
    async def _handle_serialize_request(self, event: Event):
        """Handle SERIALIZE_OBJECT event"""
        # Extract parameters from event
        if isinstance(event.data, UniversalCode):
            if event.data.tag == Tags.DICT:
                obj = event.data.value.get("object")
                format_str = event.data.value.get("format", self.default_format.value)
                create_uc = event.data.value.get("create_universal_code", False)
            else:
                obj = event.data.value
                format_str = event.data.metadata.get("format", self.default_format.value)
                create_uc = event.data.metadata.get("create_universal_code", False)
        else:
            obj = event.data
            format_str = self.default_format.value
            create_uc = False
        
        format = SerializationFormat(format_str)
        
        # Serialize the object
        result = self.serialize(obj, format)
        
        # Create response
        if create_uc:
            # Return as UniversalCode
            tag = Tags.STRING if isinstance(result, str) else Tags.BINARY
            response_uc = self.create_universal_code(
                tag=tag,
                value=result,
                description="Serialized object",
                metadata={"format": format.value}
            )
        else:
            # Return raw serialized data in UC
            response_uc = UniversalCode(
                Tags.DICT,
                {
                    "serialized": result,
                    "format": format.value,
                    "original_type": str(type(obj))
                },
                description="Serialization result"
            )
        
        # Send response
        await self.context.dispatch_event(
            Event(
                type=Actions.SERIALIZATION_RESULT,
                data=response_uc,
                origin=self.object_id,
                priority=2
            )
        )
    
    async def _handle_deserialize_request(self, event: Event):
        """Handle DESERIALIZE_OBJECT event"""
        # Extract parameters from event
        if isinstance(event.data, UniversalCode):
            if event.data.tag == Tags.DICT:
                data = event.data.value.get("data")
                format_str = event.data.value.get("format")
            else:
                data = event.data.value
                format_str = event.data.metadata.get("format")
        else:
            data = event.data
            format_str = None
        
        format = SerializationFormat(format_str) if format_str else None
        
        # Deserialize the data
        result = self.deserialize(data, format)
        
        # Create response
        response_uc = UniversalCode(
            Tags.DICT,
            result,
            description="Deserialized object"
        )
        
        # Send response
        await self.context.dispatch_event(
            Event(
                type=Actions.SERIALIZATION_RESULT,
                data=response_uc,
                origin=self.object_id,
                priority=2
            )
        )
    
    # ==================== Metrics and monitoring ====================
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report"""
        total = max(1, self.metrics.total_serializations)
        uc_total = max(1, self.metrics.universal_code_creations)
        
        return {
            "serialization": {
                "total": self.metrics.total_serializations,
                "failed": self.metrics.failed_serializations,
                "success_rate": (total - self.metrics.failed_serializations) / total
            },
            "universal_code": {
                "created": self.metrics.universal_code_creations,
                "failed": self.metrics.universal_code_failures,
                "success_rate": (uc_total - self.metrics.universal_code_failures) / uc_total
            },
            "format_usage": dict(self.metrics.format_usage),
            "cache": {
                "size": len(self.serialization_cache),
                "capacity": self.cache_size,
                "hits": self.metrics.cache_hits,
                "misses": self.metrics.cache_misses,
                "hit_rate": self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses)
            }
        }
    
    async def autonomous_step(self):
        """Periodic maintenance and monitoring"""
        await super().autonomous_step()
        
        # Log metrics periodically
        if self.execution_count % 100 == 0:
            report = self.get_metrics_report()
            logger.info(f"{self.object_id}: Metrics - {report}")
        
        # Clean cache if it's getting full
        if len(self.serialization_cache) > self.cache_size * 0.9:
            # Remove 10% oldest entries
            to_remove = int(self.cache_size * 0.1)
            for _ in range(to_remove):
                if self.cache_order:
                    oldest = self.cache_order.popleft()
                    self.serialization_cache.pop(oldest, None)
            logger.debug(f"{self.object_id}: Cleaned {to_remove} cache entries")
        
        await asyncio.sleep(30)  # Run every 30 seconds


# ==================== Helper functions for integration ====================

def integrate_serialization_bubble(context: SystemContext, **kwargs) -> SerializationBubble:
    """
    Create and integrate the serialization bubble into your system.
    
    Args:
        context: The SystemContext for your bubbles network
        **kwargs: Additional configuration options
    
    Returns:
        The initialized SerializationBubble
    """
    config = {
        "cache_size": 2000,
        "default_format": "json"
    }
    config.update(kwargs)
    
    # Create the serialization bubble
    serialization_bubble = SerializationBubble(
        object_id="serialization_bubble",
        context=context,
        **config
    )
    
    # Make it easily accessible through context
    context.serialization_bubble = serialization_bubble
    
    # Start the bubble
    asyncio.create_task(serialization_bubble.start_autonomous_loop())
    
    logger.info("Serialization Bubble integrated into system")
    
    return serialization_bubble


# ==================== Factory pattern for easier use ====================

class UniversalCodeFactory:
    """Factory for creating UniversalCode objects with automatic serialization fixing"""
    
    def __init__(self, context: SystemContext):
        self.context = context
        self._serializer = None
    
    @property
    def serializer(self):
        """Lazy load serializer from context"""
        if self._serializer is None:
            self._serializer = getattr(self.context, 'serialization_bubble', None)
        return self._serializer
    
    def create(self,
               tag: Tags,
               value: Any,
               binary_data: Optional[bytes] = None,
               description: str = "",
               metadata: Optional[Dict[str, Any]] = None,
               auto_fix: bool = True) -> UniversalCode:
        """
        Create UniversalCode with optional automatic serialization fixing.
        
        Args:
            tag: The data type tag
            value: The value to wrap
            binary_data: Optional binary representation
            description: Human-readable description
            metadata: Additional metadata
            auto_fix: If True and serializer available, auto-fix serialization issues
        
        Returns:
            UniversalCode object
        """
        # If auto_fix is enabled and serializer is available
        if auto_fix and self.serializer:
            try:
                # Try direct creation first (faster)
                return UniversalCode(tag, value, binary_data, description, metadata)
            except ValueError as e:
                # If it fails due to serialization, use serializer to fix it
                if "not JSON-serializable" in str(e):
                    return self.serializer.create_universal_code(
                        tag, value, binary_data, description, metadata
                    )
                else:
                    raise
        else:
            # No auto-fix, create directly
            return UniversalCode(tag, value, binary_data, description, metadata)
    
    def create_from_llm_response(self,
                                response: str,
                                metadata: Optional[Dict[str, Any]] = None) -> UniversalCode:
        """
        Create UniversalCode from LLM text response with smart parsing.
        """
        # Try to parse as JSON first
        try:
            import json
            data = json.loads(response)
            # Detected JSON response
            tag = Tags.DICT if isinstance(data, dict) else Tags.LIST
            return self.create(
                tag=tag,
                value=data,
                description="Parsed LLM JSON response",
                metadata=metadata
            )
        except json.JSONDecodeError:
            # Not JSON, continue
            pass
        
        # Check if it's code
        if response.strip().startswith("```"):
            # Extract code block
            lines = response.strip().split('\n')
            if len(lines) > 2:
                lang = lines[0].replace("```", "").strip()
                code = '\n'.join(lines[1:-1])
                return self.create(
                    tag=Tags.DICT,
                    value={
                        "type": "code",
                        "language": lang,
                        "code": code
                    },
                    description="LLM code response",
                    metadata=metadata
                )
        
        # Default to string
        return self.create(
            tag=Tags.STRING,
            value=response,
            description="LLM text response",
            metadata=metadata
        )


# ==================== Example usage and testing ====================

if __name__ == "__main__":
    import asyncio
    from enum import Enum
    
    # Your problematic enum
    class HealthStatus(Enum):
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        CRITICAL = "critical"
    
    async def test_serialization():
        # Create test context
        from bubbles_core import SystemContext
        context = SystemContext()
        
        # Create serialization bubble
        serializer = SerializationBubble("test_serializer", context)
        
        # Test data with your problematic enum
        test_data = {
            "overall_status": HealthStatus.HEALTHY,
            "timestamp": datetime.now(),
            "system_tags": Tags.DICT,
            "action": Actions.SYSTEM_STATE_UPDATE,
            "check_results": {
                "cpu": {
                    "status": HealthStatus.HEALTHY,
                    "message": "CPU temperature normal",
                    "temp": 45.5
                },
                "memory": {
                    "status": HealthStatus.DEGRADED,
                    "message": "Memory usage high",
                    "value": 85.5
                },
                "thermal": {
                    "status": HealthStatus.CRITICAL,
                    "message": "System running hot",
                    "temperature": 95.0
                }
            },
            "diagnostics": {
                "uptime": 3600.5,
                "last_check": datetime.now(),
                "check_count": 42,
                "errors": set([InvalidTagError("test"), ValueError("another test")])
            }
        }
        
        print("Original data contains enums that cause JSON errors")
        
        # Test JSON serialization
        print("\n1. Testing JSON serialization:")
        json_result = serializer.serialize(test_data, SerializationFormat.JSON)
        print(f"Success! Serialized to {len(json_result)} characters")
        print("Sample:", json_result[:200], "...")
        
        # Test Universal Code creation
        print("\n2. Testing UniversalCode creation:")
        uc = serializer.create_universal_code(
            Tags.DICT,
            test_data,
            description="Health check with enums"
        )
        print(f"Created UC successfully: {uc}")
        
        # Test fix_hardware_event
        print("\n3. Testing fix_hardware_event:")
        fixed_data = serializer.fix_hardware_event(test_data)
        print("Fixed data:", json.dumps(fixed_data, indent=2)[:300], "...")
        
        # Show metrics
        print("\n4. Serialization metrics:")
        print(json.dumps(serializer.get_metrics_report(), indent=2))
        
        # Test factory pattern
        print("\n5. Testing UniversalCodeFactory:")
        factory = UniversalCodeFactory(context)
        context.serialization_bubble = serializer
        
        # This would normally fail but factory fixes it
        uc2 = factory.create(Tags.DICT, test_data, description="Factory test")
        print(f"Factory created UC successfully: {uc2}")
    
    # Run the test
    asyncio.run(test_serialization())
