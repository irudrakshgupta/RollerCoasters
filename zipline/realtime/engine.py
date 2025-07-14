"""
Real-time Processing Engine for Zipline

This module provides real-time data processing, streaming, and live trading
capabilities for low-latency algorithmic trading.
"""

from typing import Any, Callable, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from zipline.algorithm import TradingAlgorithm
from zipline.data.data_portal import DataPortal
from zipline.finance.blotter import SimulationBlotter
from zipline.utils.events import EventManager

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for real-time engine."""
    STREAMING = "streaming"
    EVENT_DRIVEN = "event_driven"
    MICRO_BATCH = "micro_batch"


@dataclass
class MarketEvent:
    """Market event data structure."""
    timestamp: datetime
    symbol: str
    event_type: str
    data: Dict[str, Any]
    source: str


@dataclass
class OrderEvent:
    """Order event data structure."""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    order_type: str
    status: str


class RealTimeEngine:
    """
    Real-time processing engine for live trading and streaming data.
    """
    
    def __init__(self,
                 algorithm: TradingAlgorithm,
                 data_portal: DataPortal,
                 processing_mode: ProcessingMode = ProcessingMode.STREAMING,
                 batch_size: int = 100,
                 max_latency_ms: int = 10,
                 enable_risk_management: bool = True,
                 enable_order_management: bool = True):
        """
        Initialize the real-time engine.
        
        Parameters
        ----------
        algorithm : TradingAlgorithm
            The trading algorithm to run in real-time.
        data_portal : DataPortal
            The data portal for market data access.
        processing_mode : ProcessingMode
            The processing mode for the engine.
        batch_size : int
            Batch size for micro-batch processing.
        max_latency_ms : int
            Maximum allowed latency in milliseconds.
        enable_risk_management : bool
            Whether to enable real-time risk management.
        enable_order_management : bool
            Whether to enable order management.
        """
        self.algorithm = algorithm
        self.data_portal = data_portal
        self.processing_mode = processing_mode
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms
        self.enable_risk_management = enable_risk_management
        self.enable_order_management = enable_order_management
        
        # Event queues
        self.market_event_queue = queue.Queue()
        self.order_event_queue = queue.Queue()
        self.signal_queue = queue.Queue()
        
        # Processing components
        self.event_manager = EventManager()
        self.streaming_engine = StreamingEngine(self)
        self.event_driven_engine = EventDrivenEngine(self)
        self.micro_batch_processor = MicroBatchProcessor(self)
        
        # State management
        self.is_running = False
        self.current_timestamp = None
        self.performance_metrics = {
            'total_events_processed': 0,
            'average_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'events_per_second': 0.0
        }
        
        # Threading
        self.processing_thread = None
        self.lock = threading.Lock()
        
        logger.info(f"Real-time engine initialized with mode: {processing_mode.value}")
    
    def start(self):
        """Start the real-time processing engine."""
        if self.is_running:
            logger.warning("Engine is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time engine started")
    
    def stop(self):
        """Stop the real-time processing engine."""
        if not self.is_running:
            logger.warning("Engine is not running")
            return
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Real-time engine stopped")
    
    def _processing_loop(self):
        """Main processing loop for the real-time engine."""
        start_time = datetime.now()
        event_count = 0
        
        while self.is_running:
            try:
                # Process based on mode
                if self.processing_mode == ProcessingMode.STREAMING:
                    self.streaming_engine.process()
                elif self.processing_mode == ProcessingMode.EVENT_DRIVEN:
                    self.event_driven_engine.process()
                elif self.processing_mode == ProcessingMode.MICRO_BATCH:
                    self.micro_batch_processor.process()
                
                # Update performance metrics
                event_count += 1
                if event_count % 1000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    self.performance_metrics['events_per_second'] = event_count / elapsed
                    self.performance_metrics['total_events_processed'] = event_count
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                # Continue processing despite errors
    
    def add_market_event(self, event: MarketEvent):
        """Add a market event to the processing queue."""
        self.market_event_queue.put(event)
    
    def add_order_event(self, event: OrderEvent):
        """Add an order event to the processing queue."""
        self.order_event_queue.put(event)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()


class StreamingEngine:
    """
    Streaming data processing engine for real-time market data.
    """
    
    def __init__(self, parent_engine: RealTimeEngine):
        """Initialize the streaming engine."""
        self.parent = parent_engine
        self.data_streams = {}
        self.processors = {}
        
    def add_data_stream(self, stream_name: str, stream: 'DataStream'):
        """Add a data stream to the engine."""
        self.data_streams[stream_name] = stream
        logger.info(f"Added data stream: {stream_name}")
    
    def add_processor(self, processor_name: str, processor: Callable):
        """Add a data processor to the engine."""
        self.processors[processor_name] = processor
        logger.info(f"Added processor: {processor_name}")
    
    def process(self):
        """Process streaming data."""
        # Process market events
        while not self.parent.market_event_queue.empty():
            try:
                event = self.parent.market_event_queue.get_nowait()
                self._process_market_event(event)
            except queue.Empty:
                break
        
        # Process order events
        while not self.parent.order_event_queue.empty():
            try:
                event = self.parent.order_event_queue.get_nowait()
                self._process_order_event(event)
            except queue.Empty:
                break
    
    def _process_market_event(self, event: MarketEvent):
        """Process a market event."""
        start_time = datetime.now()
        
        try:
            # Update current timestamp
            self.parent.current_timestamp = event.timestamp
            
            # Apply processors
            for processor_name, processor in self.processors.items():
                try:
                    processor(event)
                except Exception as e:
                    logger.error(f"Error in processor {processor_name}: {e}")
            
            # Update algorithm with new data
            self._update_algorithm(event)
            
            # Calculate latency
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.parent.performance_metrics['average_latency_ms'] = (
                (self.parent.performance_metrics['average_latency_ms'] + latency) / 2
            )
            self.parent.performance_metrics['max_latency_ms'] = max(
                self.parent.performance_metrics['max_latency_ms'], latency
            )
            
        except Exception as e:
            logger.error(f"Error processing market event: {e}")
    
    def _process_order_event(self, event: OrderEvent):
        """Process an order event."""
        try:
            # Update order status in blotter
            if self.parent.enable_order_management:
                self._update_order_status(event)
            
        except Exception as e:
            logger.error(f"Error processing order event: {e}")
    
    def _update_algorithm(self, event: MarketEvent):
        """Update the algorithm with new market data."""
        try:
            # Create a data object for the algorithm
            data = self._create_data_object(event)
            
            # Call algorithm's handle_data method
            self.parent.algorithm.handle_data(data)
            
        except Exception as e:
            logger.error(f"Error updating algorithm: {e}")
    
    def _create_data_object(self, event: MarketEvent):
        """Create a data object for the algorithm."""
        # This is a simplified implementation
        # In practice, you'd need to create a proper BarData object
        class DataObject:
            def __init__(self, event_data):
                self.event_data = event_data
            
            def current(self, asset, field):
                # Return current data for the asset and field
                return self.event_data.get(field, 0.0)
        
        return DataObject(event.data)
    
    def _update_order_status(self, event: OrderEvent):
        """Update order status in the blotter."""
        # This would update the order status in the algorithm's blotter
        pass


class EventDrivenEngine:
    """
    Event-driven processing engine for real-time trading.
    """
    
    def __init__(self, parent_engine: RealTimeEngine):
        """Initialize the event-driven engine."""
        self.parent = parent_engine
        self.event_handlers = {}
        self.event_filters = {}
        
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    def register_event_filter(self, event_type: str, filter_func: Callable):
        """Register an event filter."""
        self.event_filters[event_type] = filter_func
        logger.info(f"Registered filter for event type: {event_type}")
    
    def process(self):
        """Process events in event-driven mode."""
        # Process market events
        while not self.parent.market_event_queue.empty():
            try:
                event = self.parent.market_event_queue.get_nowait()
                self._handle_event(event)
            except queue.Empty:
                break
        
        # Process order events
        while not self.parent.order_event_queue.empty():
            try:
                event = self.parent.order_event_queue.get_nowait()
                self._handle_event(event)
            except queue.Empty:
                break
    
    def _handle_event(self, event: Union[MarketEvent, OrderEvent]):
        """Handle an event using registered handlers."""
        event_type = event.event_type if hasattr(event, 'event_type') else 'order'
        
        # Apply event filter if exists
        if event_type in self.event_filters:
            if not self.event_filters[event_type](event):
                return  # Event filtered out
        
        # Call registered handlers
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")


class MicroBatchProcessor:
    """
    Micro-batch processing engine for real-time data.
    """
    
    def __init__(self, parent_engine: RealTimeEngine):
        """Initialize the micro-batch processor."""
        self.parent = parent_engine
        self.batch_buffer = []
        self.batch_timer = None
        self.batch_timeout = timedelta(milliseconds=parent_engine.max_latency_ms)
        
    def process(self):
        """Process data in micro-batches."""
        current_time = datetime.now()
        
        # Collect events into batch
        self._collect_batch()
        
        # Process batch if ready
        if self._should_process_batch(current_time):
            self._process_batch()
    
    def _collect_batch(self):
        """Collect events into the batch buffer."""
        # Collect market events
        while len(self.batch_buffer) < self.parent.batch_size and not self.parent.market_event_queue.empty():
            try:
                event = self.parent.market_event_queue.get_nowait()
                self.batch_buffer.append(event)
            except queue.Empty:
                break
        
        # Collect order events
        while len(self.batch_buffer) < self.parent.batch_size and not self.parent.order_event_queue.empty():
            try:
                event = self.parent.order_event_queue.get_nowait()
                self.batch_buffer.append(event)
            except queue.Empty:
                break
    
    def _should_process_batch(self, current_time: datetime) -> bool:
        """Determine if the batch should be processed."""
        if len(self.batch_buffer) >= self.parent.batch_size:
            return True
        
        if self.batch_timer and (current_time - self.batch_timer) >= self.batch_timeout:
            return True
        
        return False
    
    def _process_batch(self):
        """Process the current batch."""
        if not self.batch_buffer:
            return
        
        start_time = datetime.now()
        
        try:
            # Sort events by timestamp
            self.batch_buffer.sort(key=lambda x: x.timestamp)
            
            # Process each event in the batch
            for event in self.batch_buffer:
                if isinstance(event, MarketEvent):
                    self.parent.streaming_engine._process_market_event(event)
                elif isinstance(event, OrderEvent):
                    self.parent.streaming_engine._process_order_event(event)
            
            # Clear batch buffer
            self.batch_buffer.clear()
            self.batch_timer = None
            
            # Update performance metrics
            batch_latency = (datetime.now() - start_time).total_seconds() * 1000
            self.parent.performance_metrics['average_latency_ms'] = (
                (self.parent.performance_metrics['average_latency_ms'] + batch_latency) / 2
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.batch_buffer.clear()


class DataStream(ABC):
    """
    Abstract base class for data streams.
    """
    
    @abstractmethod
    def start(self):
        """Start the data stream."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the data stream."""
        pass
    
    @abstractmethod
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest data from the stream."""
        pass


class MarketDataStream(DataStream):
    """
    Market data stream implementation.
    """
    
    def __init__(self, symbols: List[str], data_source: str = "live"):
        """Initialize the market data stream."""
        self.symbols = symbols
        self.data_source = data_source
        self.is_running = False
        self.latest_data = {}
        
    def start(self):
        """Start the market data stream."""
        self.is_running = True
        logger.info(f"Started market data stream for symbols: {self.symbols}")
    
    def stop(self):
        """Stop the market data stream."""
        self.is_running = False
        logger.info("Stopped market data stream")
    
    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest market data."""
        if not self.is_running:
            return None
        
        # This is a placeholder implementation
        # In practice, you'd connect to a real market data feed
        return {
            'timestamp': datetime.now(),
            'symbols': self.symbols,
            'data': self.latest_data
        } 