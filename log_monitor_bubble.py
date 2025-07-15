import asyncio
import logging
import re
import time
from typing import Dict, Any, Optional, List, Pattern
from collections import deque, defaultdict
from dataclasses import dataclass
import json

from bubbles_core import (
    UniversalBubble, SystemContext, Event, UniversalCode, Tags, Actions,
    logger, EventService
)

@dataclass
class LogPattern:
    """Pattern for matching and classifying log entries"""
    pattern: Pattern
    event_type: str
    warning_type: str
    extract_fields: List[str]  # Fields to extract from regex groups

class AsyncLogHandler(logging.Handler):
    """Custom log handler that queues log records for async processing"""
    
    def __init__(self, log_queue: asyncio.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.loop = None
        
    def emit(self, record):
        """Queue log record for async processing"""
        try:
            # Get or create event loop
            if self.loop is None:
                try:
                    self.loop = asyncio.get_running_loop()
                except RuntimeError:
                    return  # No event loop running
            
            # Create serializable log entry
            log_entry = {
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'timestamp': record.created,
                'filename': record.filename,
                'lineno': record.lineno,
                'funcName': record.funcName,
                'exc_info': str(record.exc_info) if record.exc_info else None
            }
            
            # Queue the log entry
            asyncio.run_coroutine_threadsafe(
                self.log_queue.put(log_entry),
                self.loop
            )
        except Exception:
            # Silently fail to avoid infinite recursion
            pass

class LogMonitorBubble(UniversalBubble):
    """Monitors system logs and publishes events for warnings, errors, and patterns"""
    
    def __init__(self, object_id: str, context: SystemContext, **kwargs):
        super().__init__(object_id=object_id, context=context, **kwargs)
        
        # Configuration
        self.monitor_levels = kwargs.get('monitor_levels', ['WARNING', 'ERROR', 'CRITICAL'])
        self.batch_size = kwargs.get('batch_size', 10)
        self.batch_timeout = kwargs.get('batch_timeout', 1.0)  # seconds
        
        # Log processing
        self.log_queue = asyncio.Queue(maxsize=1000)
        self.processed_count = 0
        self.event_count = 0
        
        # Pattern matching for log classification
        self.log_patterns = self._initialize_patterns()
        
        # Rate limiting to prevent event flooding
        self.rate_limiter = defaultdict(lambda: {'count': 0, 'reset_time': time.time()})
        self.rate_limit_window = 60  # seconds
        self.rate_limit_max = 50  # max events per window per type
        
        # Aggregation for similar warnings
        self.warning_aggregator = defaultdict(list)
        self.aggregation_window = 5  # seconds
        self.last_aggregation_flush = time.time()
        
        # Statistics
        self.stats = {
            'logs_processed': 0,
            'events_published': 0,
            'warnings_detected': 0,
            'errors_detected': 0,
            'patterns_matched': 0,
            'rate_limited': 0
        }
        
        # Setup log handler
        self.log_handler = AsyncLogHandler(self.log_queue)
        self.log_handler.setLevel(logging.WARNING)
        
        # Attach to root logger
        logging.getLogger().addHandler(self.log_handler)
        
        # Start monitoring
        asyncio.create_task(self._process_log_queue())
        asyncio.create_task(self._periodic_aggregation_flush())
        asyncio.create_task(self._periodic_stats_report())
        
        logger.info(f"{self.object_id}: LogMonitorBubble initialized, monitoring {self.monitor_levels}")
    
    def _initialize_patterns(self) -> List[LogPattern]:
        """Initialize log patterns for classification"""
        patterns = [
            # Correlation ID warnings
            LogPattern(
                pattern=re.compile(r'LLM_QUERY missing correlation_id'),
                event_type='CORRELATION_WARNING',
                warning_type='MISSING_CORRELATION_ID',
                extract_fields=[]
            ),
            
            # API Errors
            LogPattern(
                pattern=re.compile(r'Provider (\w+) failed: .* API error: (\d+)'),
                event_type='API_ERROR',
                warning_type='API_FAILURE',
                extract_fields=['provider', 'error_code']
            ),
            
            # Performance warnings
            LogPattern(
                pattern=re.compile(r'duration: ([\d.]+)s'),
                event_type='PERFORMANCE_WARNING',
                warning_type='SLOW_OPERATION',
                extract_fields=['duration']
            ),
            
            # Memory warnings
            LogPattern(
                pattern=re.compile(r'memory_percent.*?([\d.]+)'),
                event_type='MEMORY_WARNING',
                warning_type='HIGH_MEMORY',
                extract_fields=['memory_percent']
            ),
            
            # Powermetrics errors
            LogPattern(
                pattern=re.compile(r'powermetrics error: (.+)'),
                event_type='HARDWARE_ERROR',
                warning_type='POWERMETRICS_FAILURE',
                extract_fields=['error_message']
            ),
            
            # No context found
            LogPattern(
                pattern=re.compile(r'No relevant context found \(threshold: ([\d.]+)\)'),
                event_type='RAG_WARNING',
                warning_type='NO_CONTEXT',
                extract_fields=['threshold']
            ),
            
            # Bubble errors
            LogPattern(
                pattern=re.compile(r'(\w+): (.+Error): (.+)'),
                event_type='BUBBLE_ERROR',
                warning_type='RUNTIME_ERROR',
                extract_fields=['bubble_id', 'error_type', 'error_message']
            ),
            
            # Fix detection
            LogPattern(
                pattern=re.compile(r'Applied (\w+) fix to (\w+)'),
                event_type='FIX_APPLIED',
                warning_type='AUTO_FIX',
                extract_fields=['fix_type', 'bubble_id']
            ),
            
            # Recovery success
            LogPattern(
                pattern=re.compile(r'Recovery action.*?was successful'),
                event_type='RECOVERY_SUCCESS',
                warning_type='RECOVERY_COMPLETE',
                extract_fields=[]
            )
        ]
        
        return patterns
    
    async def _process_log_queue(self):
        """Process queued log entries"""
        batch = []
        last_batch_time = time.time()
        
        while True:
            try:
                # Try to get log entry with timeout
                timeout = max(0.1, self.batch_timeout - (time.time() - last_batch_time))
                
                try:
                    log_entry = await asyncio.wait_for(
                        self.log_queue.get(), 
                        timeout=timeout
                    )
                    batch.append(log_entry)
                except asyncio.TimeoutError:
                    pass
                
                # Process batch if full or timeout reached
                should_process = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and time.time() - last_batch_time >= self.batch_timeout)
                )
                
                if should_process and batch:
                    await self._process_log_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                    
            except Exception as e:
                logger.error(f"{self.object_id}: Error processing log queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_log_batch(self, batch: List[Dict]):
        """Process a batch of log entries"""
        for log_entry in batch:
            try:
                self.stats['logs_processed'] += 1
                
                # Skip our own logs to prevent loops
                if log_entry['name'] == 'bubbles_core' and 'LogMonitorBubble' in log_entry['message']:
                    continue
                
                # Check if we should process this log level
                if log_entry['level'] not in self.monitor_levels:
                    continue
                
                # Classify and potentially publish event
                await self._classify_and_publish(log_entry)
                
            except Exception as e:
                # Can't log here - would cause recursion
                pass
    
    async def _classify_and_publish(self, log_entry: Dict):
        """Classify log entry and publish appropriate event"""
        level = log_entry['level']
        message = log_entry['message']
        source = log_entry['name']
        
        # Track basic warnings/errors
        if level == 'WARNING':
            self.stats['warnings_detected'] += 1
        elif level in ['ERROR', 'CRITICAL']:
            self.stats['errors_detected'] += 1
        
        # Try pattern matching first
        matched = False
        for pattern in self.log_patterns:
            match = pattern.pattern.search(message)
            if match:
                matched = True
                self.stats['patterns_matched'] += 1
                
                # Extract fields
                extracted_data = {}
                if pattern.extract_fields:
                    groups = match.groups()
                    for i, field in enumerate(pattern.extract_fields):
                        if i < len(groups):
                            extracted_data[field] = groups[i]
                
                # Create event data
                event_data = {
                    'warning_type': pattern.warning_type,
                    'source': source,
                    'message': message,
                    'level': level,
                    'timestamp': log_entry['timestamp'],
                    'filename': log_entry['filename'],
                    'lineno': log_entry['lineno'],
                    **extracted_data
                }
                
                # Check rate limiting
                if not self._check_rate_limit(pattern.event_type):
                    self.stats['rate_limited'] += 1
                    return
                
                # Aggregate similar warnings
                if pattern.event_type in ['CORRELATION_WARNING', 'API_ERROR', 'RAG_WARNING']:
                    await self._aggregate_warning(pattern.event_type, event_data)
                else:
                    # Publish immediately for critical events
                    await self._publish_event(pattern.event_type, event_data)
                
                break
        
        # If no pattern matched, create generic event based on level
        if not matched and level in ['ERROR', 'CRITICAL']:
            event_data = {
                'warning_type': f'GENERIC_{level}',
                'source': source,
                'message': message,
                'level': level,
                'timestamp': log_entry['timestamp'],
                'filename': log_entry['filename'],
                'lineno': log_entry['lineno']
            }
            
            if self._check_rate_limit('GENERIC_ERROR'):
                await self._publish_event('BUBBLE_ERROR', event_data)
    
    def _check_rate_limit(self, event_type: str) -> bool:
        """Check if we can publish this event type (rate limiting)"""
        now = time.time()
        limiter = self.rate_limiter[event_type]
        
        # Reset window if needed
        if now - limiter['reset_time'] > self.rate_limit_window:
            limiter['count'] = 0
            limiter['reset_time'] = now
        
        # Check limit
        if limiter['count'] >= self.rate_limit_max:
            return False
        
        limiter['count'] += 1
        return True
    
    async def _aggregate_warning(self, event_type: str, event_data: Dict):
        """Aggregate similar warnings to reduce event spam"""
        key = f"{event_type}:{event_data['source']}:{event_data['warning_type']}"
        self.warning_aggregator[key].append(event_data)
    
    async def _periodic_aggregation_flush(self):
        """Periodically flush aggregated warnings"""
        while True:
            await asyncio.sleep(self.aggregation_window)
            await self._flush_aggregated_warnings()
    
    async def _flush_aggregated_warnings(self):
        """Flush aggregated warnings as consolidated events"""
        now = time.time()
        
        for key, warnings in list(self.warning_aggregator.items()):
            if not warnings:
                continue
                
            event_type = key.split(':')[0]
            
            # Create aggregated event
            aggregated_data = {
                'warning_type': warnings[0]['warning_type'],
                'source': warnings[0]['source'],
                'count': len(warnings),
                'first_occurrence': warnings[0]['timestamp'],
                'last_occurrence': warnings[-1]['timestamp'],
                'sample_messages': [w['message'] for w in warnings[:3]],  # First 3 samples
                'aggregated': True
            }
            
            await self._publish_event(event_type, aggregated_data)
        
        # Clear aggregator
        self.warning_aggregator.clear()
        self.last_aggregation_flush = now
    
    async def _publish_event(self, event_type: str, event_data: Dict):
        """Publish event to event system"""
        try:
            # Map event types to Actions
            action_map = {
                'CORRELATION_WARNING': Actions.CORRELATION_WARNING,
                'API_ERROR': Actions.BUBBLE_ERROR,
                'PERFORMANCE_WARNING': Actions.PERFORMANCE_WARNING,
                'MEMORY_WARNING': Actions.MEMORY_WARNING,
                'HARDWARE_ERROR': Actions.BUBBLE_ERROR,
                'RAG_WARNING': Actions.WARNING_EVENT,
                'BUBBLE_ERROR': Actions.BUBBLE_ERROR,
                'FIX_APPLIED': Actions.CODE_UPDATE,
                'RECOVERY_SUCCESS': Actions.OVERSEER_REPORT,
                'GENERIC_ERROR': Actions.BUBBLE_ERROR
            }
            
            action = action_map.get(event_type, Actions.WARNING_EVENT)
            
            event = Event(
                type=action,
                data=UniversalCode(Tags.DICT, event_data),
                origin=self.object_id,
                priority=4 if 'ERROR' in event_type else 3
            )
            
            await self.context.dispatch_event(event)
            
            self.stats['events_published'] += 1
            self.event_count += 1
            
        except Exception as e:
            # Can't log - would cause recursion
            pass
    
    async def _periodic_stats_report(self):
        """Periodically report statistics"""
        while True:
            await asyncio.sleep(60)  # Every minute
            
            try:
                # Flush any pending aggregated warnings
                await self._flush_aggregated_warnings()
                
                # Create stats report
                report = {
                    'timestamp': time.time(),
                    'stats': dict(self.stats),  # Copy to avoid modification during serialization
                    'queue_size': self.log_queue.qsize(),
                    'aggregated_warnings': len(self.warning_aggregator),
                    'rate_limited_types': [k for k, v in self.rate_limiter.items() 
                                          if v['count'] >= self.rate_limit_max]
                }
                
                # Publish stats
                stats_event = Event(
                    type=Actions.OVERSEER_REPORT,
                    data=UniversalCode(Tags.DICT, {
                        'report_type': 'LOG_MONITOR_STATS',
                        'data': report
                    }),
                    origin=self.object_id,
                    priority=2
                )
                
                await self.context.dispatch_event(stats_event)
                
                # Log summary (careful not to create loops)
                if self.stats['events_published'] > 0:
                    # Use print to avoid logging system
                    print(f"LogMonitor: Processed {self.stats['logs_processed']} logs, "
                          f"published {self.stats['events_published']} events")
                
            except Exception:
                # Silently handle errors
                pass
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Remove log handler
            logging.getLogger().removeHandler(self.log_handler)
            
            # Flush remaining warnings
            await self._flush_aggregated_warnings()
            
            # Final stats report
            await self._periodic_stats_report()
            
        except Exception:
            pass
        
        await super().cleanup()

# Example instantiation function
def create_log_monitor(context: SystemContext) -> LogMonitorBubble:
    """Create and configure a LogMonitorBubble instance"""
    return LogMonitorBubble(
        object_id="log_monitor",
        context=context,
        monitor_levels=['WARNING', 'ERROR', 'CRITICAL'],
        batch_size=10,
        batch_timeout=1.0
    )