"""
Real-time Processing for Zipline

This module provides real-time data processing, streaming, and live trading capabilities
for Zipline, enabling low-latency algorithmic trading.
"""

from .engine import (
    RealTimeEngine,
    StreamingEngine,
    EventDrivenEngine,
    MicroBatchProcessor,
)
from .data_streams import (
    DataStream,
    MarketDataStream,
    NewsDataStream,
    SentimentDataStream,
    AlternativeDataStream,
)
from .order_management import (
    SmartOrderRouter,
    OrderManager,
    ExecutionEngine,
    MarketImpactModel,
)
from .risk_management import (
    RealTimeRiskManager,
    PositionMonitor,
    RiskLimits,
    CircuitBreaker,
)
from .messaging import (
    MessageBroker,
    KafkaBroker,
    RedisBroker,
    WebSocketBroker,
)
from .monitoring import (
    PerformanceMonitor,
    LatencyMonitor,
    HealthChecker,
    AlertManager,
)
from .deployment import (
    LiveTradingDeployment,
    StrategyDeployment,
    ModelDeployment,
    InfrastructureManager,
)

__all__ = [
    # Core engines
    'RealTimeEngine',
    'StreamingEngine',
    'EventDrivenEngine',
    'MicroBatchProcessor',
    
    # Data streams
    'DataStream',
    'MarketDataStream',
    'NewsDataStream',
    'SentimentDataStream',
    'AlternativeDataStream',
    
    # Order management
    'SmartOrderRouter',
    'OrderManager',
    'ExecutionEngine',
    'MarketImpactModel',
    
    # Risk management
    'RealTimeRiskManager',
    'PositionMonitor',
    'RiskLimits',
    'CircuitBreaker',
    
    # Messaging
    'MessageBroker',
    'KafkaBroker',
    'RedisBroker',
    'WebSocketBroker',
    
    # Monitoring
    'PerformanceMonitor',
    'LatencyMonitor',
    'HealthChecker',
    'AlertManager',
    
    # Deployment
    'LiveTradingDeployment',
    'StrategyDeployment',
    'ModelDeployment',
    'InfrastructureManager',
] 