"""
Enhanced Zipline Algorithm Example

This example demonstrates the new enhanced features of Zipline including:
- Machine Learning integration with AutoML
- Real-time processing capabilities
- Advanced risk management
- Modern Python features and type hints
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Import enhanced Zipline modules
from zipline.algorithm import TradingAlgorithm
from zipline.api import (
    order_target_percent, 
    record, 
    symbol, 
    symbols,
    attach_pipeline,
    pipeline_output,
    schedule_function,
    date_rules,
    time_rules
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (
    SimpleMovingAverage,
    Returns,
    VWAP,
    AverageDollarVolume
)
from zipline.pipeline.filters import QTradableStocks

# Import new ML modules
from zipline.ml import (
    AutoMLFactor,
    AutoMLClassifier,
    FeatureEngineer,
    ModelRegistry
)

# Import new real-time modules
from zipline.realtime import (
    RealTimeEngine,
    ProcessingMode,
    MarketDataStream
)

# Import new risk management modules
from zipline.risk import (
    VaRCalculator,
    HistoricalVaR,
    MonteCarloVaR,
    ConditionalVaR,
    StressTester,
    RegimeDetector
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTradingAlgorithm(TradingAlgorithm):
    """
    Enhanced trading algorithm demonstrating new Zipline features.
    """
    
    def initialize(self, context):
        """Initialize the enhanced algorithm."""
        # Set basic parameters
        context.lookback = 20
        context.risk_free_rate = 0.02
        context.max_leverage = 1.5
        
        # Set benchmark
        context.benchmark = symbol('SPY')
        
        # Initialize ML components
        self._initialize_ml_components(context)
        
        # Initialize risk management
        self._initialize_risk_management(context)
        
        # Initialize real-time processing
        self._initialize_realtime_processing(context)
        
        # Set up pipeline
        self._setup_pipeline(context)
        
        # Schedule functions
        self._schedule_functions(context)
        
        logger.info("Enhanced algorithm initialized successfully")
    
    def _initialize_ml_components(self, context):
        """Initialize machine learning components."""
        # Create feature engineer
        context.feature_engineer = FeatureEngineer(
            technical_features=True,
            fundamental_features=True,
            sentiment_features=False
        )
        
        # Create model registry
        context.model_registry = ModelRegistry()
        
        # Initialize AutoML models
        context.price_predictor = AutoMLFactor(
            inputs=[
                SimpleMovingAverage(window_length=5),
                SimpleMovingAverage(window_length=20),
                Returns(window_length=1),
                VWAP(window_length=10)
            ],
            target_factor=Returns(window_length=1),
            model_type="regression",
            feature_selection_method="kbest",
            n_features=5,
            hyperparameter_optimization=True,
            cv_folds=5
        )
        
        context.regime_classifier = AutoMLClassifier(
            inputs=[
                Returns(window_length=1),
                Returns(window_length=5),
                Returns(window_length=20)
            ],
            target_classifier=None,  # Will be set dynamically
            feature_selection_method="mutual_info",
            n_features=3,
            hyperparameter_optimization=True,
            cv_folds=3
        )
        
        logger.info("ML components initialized")
    
    def _initialize_risk_management(self, context):
        """Initialize risk management components."""
        # Initialize VaR calculators
        context.historical_var = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        context.monte_carlo_var = MonteCarloVaR(confidence_level=0.95, time_horizon=1, n_simulations=10000)
        context.conditional_var = ConditionalVaR(confidence_level=0.95, time_horizon=1)
        
        # Initialize stress tester
        context.stress_tester = StressTester(
            scenarios=['market_crash', 'volatility_spike', 'correlation_breakdown']
        )
        
        # Initialize regime detector
        context.regime_detector = RegimeDetector(
            n_regimes=3,
            lookback_period=60
        )
        
        # Set risk limits
        context.max_var = 0.02  # 2% VaR limit
        context.max_drawdown = 0.15  # 15% max drawdown
        context.position_limit = 0.1  # 10% max position size
        
        logger.info("Risk management components initialized")
    
    def _initialize_realtime_processing(self, context):
        """Initialize real-time processing components."""
        # Create real-time engine
        context.realtime_engine = RealTimeEngine(
            algorithm=self,
            data_portal=context.data_portal,
            processing_mode=ProcessingMode.STREAMING,
            batch_size=100,
            max_latency_ms=10,
            enable_risk_management=True,
            enable_order_management=True
        )
        
        # Create market data stream
        context.market_stream = MarketDataStream(
            symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
            data_source="live"
        )
        
        # Register event handlers
        context.realtime_engine.event_driven_engine.register_event_handler(
            'price_update',
            self._handle_price_update
        )
        
        context.realtime_engine.event_driven_engine.register_event_handler(
            'risk_alert',
            self._handle_risk_alert
        )
        
        logger.info("Real-time processing components initialized")
    
    def _setup_pipeline(self, context):
        """Set up the data pipeline."""
        # Create pipeline
        pipe = Pipeline()
        
        # Add basic factors
        pipe.add(SimpleMovingAverage(window_length=20), 'sma_20')
        pipe.add(SimpleMovingAverage(window_length=50), 'sma_50')
        pipe.add(Returns(window_length=1), 'returns')
        pipe.add(VWAP(window_length=10), 'vwap')
        pipe.add(AverageDollarVolume(window_length=30), 'adv')
        
        # Add ML factors
        pipe.add(context.price_predictor, 'ml_prediction')
        pipe.add(context.regime_classifier, 'regime')
        
        # Set screen
        pipe.set_screen(QTradableStocks() & (pipe.columns['adv'] > 1000000))
        
        # Attach pipeline
        attach_pipeline(pipe, 'enhanced_pipeline')
        
        logger.info("Pipeline setup completed")
    
    def _schedule_functions(self, context):
        """Schedule regular functions."""
        # Daily risk check
        schedule_function(
            self._daily_risk_check,
            date_rules.every_day(),
            time_rules.market_open(minutes=30)
        )
        
        # Weekly model retraining
        schedule_function(
            self._retrain_models,
            date_rules.week_start(),
            time_rules.market_open(minutes=60)
        )
        
        # Monthly stress testing
        schedule_function(
            self._monthly_stress_test,
            date_rules.month_start(),
            time_rules.market_open(minutes=90)
        )
        
        logger.info("Functions scheduled")
    
    def before_trading_start(self, context, data):
        """Called before trading starts each day."""
        # Get pipeline data
        context.pipeline_data = pipeline_output('enhanced_pipeline')
        
        # Update regime detection
        context.current_regime = context.regime_detector.detect_regime(
            context.pipeline_data['returns']
        )
        
        # Calculate current VaR
        context.current_var = self._calculate_portfolio_var(context)
        
        # Check risk limits
        self._check_risk_limits(context)
        
        logger.info(f"Trading day started. Regime: {context.current_regime}, VaR: {context.current_var:.4f}")
    
    def handle_data(self, context, data):
        """Main trading logic."""
        # Get current positions
        positions = context.portfolio.positions
        
        # Get pipeline data for current assets
        current_data = context.pipeline_data.loc[list(positions.keys())]
        
        # Apply ML predictions
        for asset in positions:
            if asset in current_data.index:
                prediction = current_data.loc[asset, 'ml_prediction']
                regime = current_data.loc[asset, 'regime']
                
                # Adjust position based on ML prediction and regime
                target_weight = self._calculate_target_weight(prediction, regime, context)
                
                # Apply risk limits
                target_weight = self._apply_risk_limits(target_weight, context)
                
                # Place order
                order_target_percent(asset, target_weight)
        
        # Record metrics
        self._record_metrics(context, data)
    
    def _calculate_target_weight(self, prediction: float, regime: int, context) -> float:
        """Calculate target weight based on ML prediction and regime."""
        # Base weight from ML prediction
        base_weight = np.clip(prediction, -0.1, 0.1)
        
        # Adjust based on regime
        regime_adjustments = {
            0: 0.5,  # Low volatility regime
            1: 1.0,  # Normal regime
            2: 0.3   # High volatility regime
        }
        
        regime_multiplier = regime_adjustments.get(regime, 1.0)
        adjusted_weight = base_weight * regime_multiplier
        
        return adjusted_weight
    
    def _apply_risk_limits(self, target_weight: float, context) -> float:
        """Apply risk limits to target weight."""
        # Position size limit
        target_weight = np.clip(target_weight, -context.position_limit, context.position_limit)
        
        # Leverage limit
        current_leverage = context.account.leverage
        if current_leverage > context.max_leverage:
            # Reduce position sizes
            target_weight *= 0.5
        
        return target_weight
    
    def _calculate_portfolio_var(self, context) -> float:
        """Calculate current portfolio VaR."""
        # Get portfolio returns
        portfolio_returns = self._get_portfolio_returns(context)
        
        if len(portfolio_returns) < 30:
            return 0.0
        
        # Calculate VaR using multiple methods
        historical_var = context.historical_var.calculate(portfolio_returns)
        monte_carlo_var = context.monte_carlo_var.calculate(portfolio_returns)
        conditional_var = context.conditional_var.calculate(portfolio_returns)
        
        # Use the maximum of the three methods
        max_var = max(historical_var, monte_carlo_var, conditional_var)
        
        return max_var
    
    def _get_portfolio_returns(self, context) -> pd.Series:
        """Get historical portfolio returns."""
        # This is a simplified implementation
        # In practice, you'd get actual portfolio returns from context.portfolio
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.015, len(dates)), index=dates)
        return returns
    
    def _check_risk_limits(self, context):
        """Check and enforce risk limits."""
        # Check VaR limit
        if context.current_var > context.max_var:
            logger.warning(f"VaR limit exceeded: {context.current_var:.4f} > {context.max_var}")
            self._reduce_risk_exposure(context)
        
        # Check drawdown limit
        current_drawdown = self._calculate_drawdown(context)
        if current_drawdown > context.max_drawdown:
            logger.warning(f"Drawdown limit exceeded: {current_drawdown:.4f} > {context.max_drawdown}")
            self._reduce_risk_exposure(context)
    
    def _calculate_drawdown(self, context) -> float:
        """Calculate current drawdown."""
        portfolio_value = context.portfolio.portfolio_value
        peak_value = context.portfolio.portfolio_value  # Simplified
        
        if peak_value > 0:
            return (peak_value - portfolio_value) / peak_value
        return 0.0
    
    def _reduce_risk_exposure(self, context):
        """Reduce risk exposure by reducing positions."""
        # Reduce all positions by 50%
        for asset in context.portfolio.positions:
            current_position = context.portfolio.positions[asset]
            target_amount = current_position.amount * 0.5
            order_target_percent(asset, target_amount / context.portfolio.portfolio_value)
        
        logger.info("Risk exposure reduced")
    
    def _daily_risk_check(self, context, data):
        """Daily risk management check."""
        # Calculate VaR
        var = self._calculate_portfolio_var(context)
        
        # Update regime detection
        context.regime_detector.update(context.pipeline_data['returns'])
        
        # Record risk metrics
        record(
            var=var,
            regime=context.current_regime,
            leverage=context.account.leverage,
            drawdown=self._calculate_drawdown(context)
        )
        
        logger.info(f"Daily risk check - VaR: {var:.4f}, Regime: {context.current_regime}")
    
    def _retrain_models(self, context, data):
        """Retrain ML models."""
        logger.info("Retraining ML models...")
        
        # Retrain price predictor
        context.price_predictor.model.retrain()
        
        # Retrain regime classifier
        context.regime_classifier.model.retrain()
        
        logger.info("ML models retrained successfully")
    
    def _monthly_stress_test(self, context, data):
        """Monthly stress testing."""
        logger.info("Running monthly stress test...")
        
        # Run stress tests
        stress_results = context.stress_tester.run_stress_tests(
            context.portfolio,
            context.pipeline_data
        )
        
        # Record stress test results
        record(
            stress_test_loss=stress_results['max_loss'],
            stress_test_scenario=stress_results['worst_scenario']
        )
        
        logger.info(f"Stress test completed - Max loss: {stress_results['max_loss']:.4f}")
    
    def _handle_price_update(self, event):
        """Handle real-time price updates."""
        # Process price update
        symbol = event.data['symbol']
        price = event.data['price']
        
        # Update algorithm with new price
        # This would integrate with the main algorithm logic
        
        logger.debug(f"Price update: {symbol} = {price}")
    
    def _handle_risk_alert(self, event):
        """Handle risk alerts."""
        # Process risk alert
        alert_type = event.data['type']
        alert_message = event.data['message']
        
        logger.warning(f"Risk alert: {alert_type} - {alert_message}")
        
        # Take action based on alert type
        if alert_type == 'var_limit_exceeded':
            self._reduce_risk_exposure(self)
    
    def _record_metrics(self, context, data):
        """Record performance metrics."""
        record(
            portfolio_value=context.portfolio.portfolio_value,
            cash=context.portfolio.cash,
            positions=len(context.portfolio.positions),
            returns=context.portfolio.returns,
            var=context.current_var,
            regime=context.current_regime
        )
    
    def analyze(self, context, results):
        """Analyze backtest results."""
        # Calculate additional metrics
        returns = results['returns']
        
        # Calculate Sharpe ratio
        excess_returns = returns - context.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Print results
        print(f"Enhanced Algorithm Results:")
        print(f"Total Return: {cumulative_returns.iloc[-1] - 1:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Final VaR: {results['var'].iloc[-1]:.4f}")
        
        logger.info("Analysis completed")


def run_enhanced_algorithm():
    """Run the enhanced algorithm example."""
    # This would be called with zipline run_algorithm
    # For demonstration purposes, we'll just show the structure
    
    algorithm = EnhancedTradingAlgorithm()
    
    print("Enhanced Zipline Algorithm Example")
    print("This example demonstrates:")
    print("- Machine Learning integration with AutoML")
    print("- Real-time processing capabilities")
    print("- Advanced risk management with VaR")
    print("- Modern Python features and type hints")
    print("- Comprehensive logging and monitoring")
    
    return algorithm


if __name__ == "__main__":
    run_enhanced_algorithm() 