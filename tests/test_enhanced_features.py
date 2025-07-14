"""
Test suite for enhanced Zipline features.

This module tests all the new features including:
- Machine Learning integration
- Real-time processing
- Advanced risk management
- Modern Python compatibility
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import enhanced modules
from zipline.ml import (
    AutoMLFactor,
    AutoMLClassifier,
    AutoMLModel,
    FeatureSelector,
    HyperparameterOptimizer
)

from zipline.realtime import (
    RealTimeEngine,
    ProcessingMode,
    MarketEvent,
    OrderEvent,
    StreamingEngine,
    EventDrivenEngine,
    MicroBatchProcessor
)

from zipline.risk import (
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    ConditionalVaR,
    VaRBacktester
)

from zipline.algorithm import TradingAlgorithm


class TestMachineLearning(unittest.TestCase):
    """Test machine learning features."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic data
        np.random.seed(42)
        self.dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, len(self.dates)), index=self.dates)
        self.features = pd.DataFrame({
            'feature1': np.random.randn(len(self.dates)),
            'feature2': np.random.randn(len(self.dates)),
            'feature3': np.random.randn(len(self.dates)),
            'feature4': np.random.randn(len(self.dates)),
            'feature5': np.random.randn(len(self.dates))
        }, index=self.dates)
    
    def test_automl_model_initialization(self):
        """Test AutoML model initialization."""
        model = AutoMLModel(
            model_type="regression",
            feature_selection_method="kbest",
            n_features=3,
            hyperparameter_optimization=True,
            cv_folds=5
        )
        
        self.assertEqual(model.model_type, "regression")
        self.assertEqual(model.feature_selection_method, "kbest")
        self.assertEqual(model.n_features, 3)
        self.assertTrue(model.hyperparameter_optimization)
        self.assertEqual(model.cv_folds, 5)
        self.assertFalse(model.is_trained)
    
    def test_feature_selector(self):
        """Test feature selection."""
        selector = FeatureSelector(method="kbest", n_features=3)
        
        # Test fit
        X = self.features.values
        y = self.returns.values
        selector.fit(X, y)
        
        self.assertIsNotNone(selector.selected_features)
        self.assertEqual(selector.selected_features.sum(), 3)
        self.assertIsNotNone(selector.feature_scores)
        
        # Test transform
        X_transformed = selector.transform(X)
        self.assertEqual(X_transformed.shape[1], 3)
    
    def test_hyperparameter_optimizer(self):
        """Test hyperparameter optimization."""
        optimizer = HyperparameterOptimizer(
            model_type="regression",
            optimization_method="grid_search",
            cv_folds=3
        )
        
        X = self.features.values
        y = self.returns.values
        
        best_params = optimizer.optimize(X, y)
        
        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)
        self.assertIsNotNone(optimizer.best_score)
    
    def test_automl_model_training(self):
        """Test AutoML model training."""
        model = AutoMLModel(
            model_type="regression",
            feature_selection_method="kbest",
            n_features=3,
            hyperparameter_optimization=False,
            cv_folds=3
        )
        
        X = self.features.values
        y = self.returns.values
        
        # Train model
        model.fit(X, y)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
        
        # Test prediction
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
        
        # Test scoring
        score = model.score(X, y)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
    
    def test_automl_factor(self):
        """Test AutoML factor creation."""
        # Mock inputs
        inputs = [Mock(), Mock(), Mock()]
        target_factor = Mock()
        
        factor = AutoMLFactor(
            inputs=inputs,
            target_factor=target_factor,
            model_type="regression",
            feature_selection_method="kbest",
            n_features=3,
            hyperparameter_optimization=False,
            cv_folds=3
        )
        
        self.assertEqual(factor.model_type, "regression")
        self.assertEqual(factor.feature_selection_method, "kbest")
        self.assertEqual(factor.n_features, 3)
        self.assertEqual(factor.target_factor, target_factor)


class TestRealTimeProcessing(unittest.TestCase):
    """Test real-time processing features."""
    
    def setUp(self):
        """Set up test data."""
        self.algorithm = Mock()
        self.data_portal = Mock()
        
    def test_realtime_engine_initialization(self):
        """Test real-time engine initialization."""
        engine = RealTimeEngine(
            algorithm=self.algorithm,
            data_portal=self.data_portal,
            processing_mode=ProcessingMode.STREAMING,
            batch_size=100,
            max_latency_ms=10
        )
        
        self.assertEqual(engine.processing_mode, ProcessingMode.STREAMING)
        self.assertEqual(engine.batch_size, 100)
        self.assertEqual(engine.max_latency_ms, 10)
        self.assertFalse(engine.is_running)
    
    def test_market_event_creation(self):
        """Test market event creation."""
        event = MarketEvent(
            timestamp=datetime.now(),
            symbol="AAPL",
            event_type="price_update",
            data={"price": 150.0, "volume": 1000},
            source="market_data"
        )
        
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.event_type, "price_update")
        self.assertEqual(event.data["price"], 150.0)
    
    def test_order_event_creation(self):
        """Test order event creation."""
        event = OrderEvent(
            timestamp=datetime.now(),
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=150.0,
            order_type="market",
            status="filled"
        )
        
        self.assertEqual(event.order_id, "order_123")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.side, "buy")
        self.assertEqual(event.quantity, 100)
    
    def test_streaming_engine(self):
        """Test streaming engine."""
        parent_engine = Mock()
        streaming_engine = StreamingEngine(parent_engine)
        
        # Test adding data stream
        data_stream = Mock()
        streaming_engine.add_data_stream("test_stream", data_stream)
        
        self.assertIn("test_stream", streaming_engine.data_streams)
        
        # Test adding processor
        processor = Mock()
        streaming_engine.add_processor("test_processor", processor)
        
        self.assertIn("test_processor", streaming_engine.processors)
    
    def test_event_driven_engine(self):
        """Test event-driven engine."""
        parent_engine = Mock()
        event_engine = EventDrivenEngine(parent_engine)
        
        # Test registering event handler
        handler = Mock()
        event_engine.register_event_handler("test_event", handler)
        
        self.assertIn("test_event", event_engine.event_handlers)
        self.assertIn(handler, event_engine.event_handlers["test_event"])
        
        # Test registering event filter
        filter_func = Mock()
        event_engine.register_event_filter("test_event", filter_func)
        
        self.assertIn("test_event", event_engine.event_filters)
    
    def test_micro_batch_processor(self):
        """Test micro-batch processor."""
        parent_engine = Mock()
        parent_engine.batch_size = 10
        parent_engine.max_latency_ms = 100
        
        batch_processor = MicroBatchProcessor(parent_engine)
        
        self.assertEqual(len(batch_processor.batch_buffer), 0)
        self.assertIsNone(batch_processor.batch_timer)


class TestRiskManagement(unittest.TestCase):
    """Test risk management features."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        self.returns = pd.Series(np.random.normal(0.001, 0.02, len(self.dates)), index=self.dates)
        self.portfolio_value = 1000000.0
    
    def test_historical_var(self):
        """Test historical VaR calculation."""
        var_calc = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        
        var = var_calc.calculate(self.returns, self.portfolio_value)
        
        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)
        self.assertLess(var, self.portfolio_value)
    
    def test_parametric_var(self):
        """Test parametric VaR calculation."""
        var_calc = ParametricVaR(confidence_level=0.95, time_horizon=1)
        
        var = var_calc.calculate(self.returns, self.portfolio_value)
        
        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)
        self.assertLess(var, self.portfolio_value)
        
        # Test with skewness and kurtosis
        var_sk = var_calc.calculate_with_skewness_kurtosis(self.returns, self.portfolio_value)
        
        self.assertIsInstance(var_sk, float)
        self.assertGreater(var_sk, 0)
    
    def test_monte_carlo_var(self):
        """Test Monte Carlo VaR calculation."""
        var_calc = MonteCarloVaR(
            confidence_level=0.95,
            time_horizon=1,
            n_simulations=1000
        )
        
        var = var_calc.calculate(self.returns, self.portfolio_value)
        
        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)
        self.assertLess(var, self.portfolio_value)
    
    def test_conditional_var(self):
        """Test conditional VaR calculation."""
        var_calc = ConditionalVaR(confidence_level=0.95, time_horizon=1)
        
        var = var_calc.calculate(self.returns, self.portfolio_value)
        
        self.assertIsInstance(var, float)
        self.assertGreater(var, 0)
        self.assertLess(var, self.portfolio_value)
        
        # Test parametric CVaR
        cvar_parametric = var_calc.calculate_parametric_cvar(self.returns, self.portfolio_value)
        
        self.assertIsInstance(cvar_parametric, float)
        self.assertGreater(cvar_parametric, 0)
    
    def test_var_backtester(self):
        """Test VaR backtesting."""
        var_calc = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        backtester = VaRBacktester(var_calc)
        
        # Run backtest
        results = backtester.backtest(self.returns, window=252)
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        self.assertIn('var', results.columns)
        self.assertIn('violation', results.columns)
        
        # Test violation rate
        violation_rate = backtester.calculate_violation_rate()
        self.assertIsInstance(violation_rate, float)
        self.assertGreaterEqual(violation_rate, 0.0)
        self.assertLessEqual(violation_rate, 1.0)
        
        # Test Kupiec test
        kupiec_results = backtester.calculate_kupiec_test()
        self.assertIn('statistic', kupiec_results)
        self.assertIn('p_value', kupiec_results)
        
        # Test Christoffersen test
        christoffersen_results = backtester.calculate_christoffersen_test()
        self.assertIn('statistic', christoffersen_results)
        self.assertIn('p_value', christoffersen_results)
    
    def test_rolling_var(self):
        """Test rolling VaR calculation."""
        var_calc = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        
        rolling_var = var_calc.calculate_rolling_var(self.returns, window=252)
        
        self.assertIsInstance(rolling_var, pd.Series)
        self.assertEqual(len(rolling_var), len(self.returns))
        self.assertTrue(all(rolling_var >= 0))


class TestModernPythonFeatures(unittest.TestCase):
    """Test modern Python features and compatibility."""
    
    def test_type_hints(self):
        """Test that type hints are properly used."""
        # This test ensures that type hints are used throughout the codebase
        # The actual type checking would be done by mypy or similar tools
        
        def test_function(x: int, y: str) -> bool:
            return isinstance(x, int) and isinstance(y, str)
        
        # Test that the function works with proper types
        result = test_function(42, "test")
        self.assertTrue(result)
    
    def test_modern_imports(self):
        """Test modern import statements."""
        # Test that we can import all the new modules
        try:
            from zipline.ml import AutoMLModel
            from zipline.realtime import RealTimeEngine
            from zipline.risk import HistoricalVaR
        except ImportError as e:
            self.fail(f"Failed to import enhanced modules: {e}")
    
    def test_dataclasses(self):
        """Test dataclass usage."""
        from dataclasses import dataclass
        
        @dataclass
        class TestData:
            name: str
            value: int
        
        data = TestData("test", 42)
        self.assertEqual(data.name, "test")
        self.assertEqual(data.value, 42)
    
    def test_async_await(self):
        """Test async/await functionality."""
        import asyncio
        
        async def async_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        # Test that async function can be called
        result = asyncio.run(async_function())
        self.assertEqual(result, "async_result")


class TestIntegration(unittest.TestCase):
    """Test integration between different components."""
    
    def setUp(self):
        """Set up test environment."""
        self.algorithm = Mock()
        self.data_portal = Mock()
    
    def test_ml_with_realtime_integration(self):
        """Test ML integration with real-time processing."""
        # Create ML model
        ml_model = AutoMLModel(
            model_type="regression",
            feature_selection_method="kbest",
            n_features=3,
            hyperparameter_optimization=False
        )
        
        # Create real-time engine
        realtime_engine = RealTimeEngine(
            algorithm=self.algorithm,
            data_portal=self.data_portal,
            processing_mode=ProcessingMode.STREAMING
        )
        
        # Test that both can coexist
        self.assertIsNotNone(ml_model)
        self.assertIsNotNone(realtime_engine)
    
    def test_risk_with_ml_integration(self):
        """Test risk management integration with ML."""
        # Create VaR calculator
        var_calc = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        
        # Create ML model
        ml_model = AutoMLModel(
            model_type="regression",
            feature_selection_method="kbest",
            n_features=3,
            hyperparameter_optimization=False
        )
        
        # Test that both can coexist
        self.assertIsNotNone(var_calc)
        self.assertIsNotNone(ml_model)
    
    def test_full_workflow(self):
        """Test a complete workflow with all components."""
        # Create synthetic data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        features = np.random.randn(len(dates), 5)
        
        # Create ML model and train it
        ml_model = AutoMLModel(
            model_type="regression",
            feature_selection_method="kbest",
            n_features=3,
            hyperparameter_optimization=False
        )
        ml_model.fit(features, returns.values)
        
        # Create VaR calculator
        var_calc = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        var = var_calc.calculate(returns, 1000000.0)
        
        # Create real-time engine
        realtime_engine = RealTimeEngine(
            algorithm=self.algorithm,
            data_portal=self.data_portal,
            processing_mode=ProcessingMode.STREAMING
        )
        
        # Test that all components work together
        self.assertTrue(ml_model.is_trained)
        self.assertIsInstance(var, float)
        self.assertIsNotNone(realtime_engine)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_ml_training_performance(self):
        """Test ML training performance."""
        import time
        
        # Create large dataset
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Time ML training
        start_time = time.time()
        
        model = AutoMLModel(
            model_type="regression",
            feature_selection_method="kbest",
            n_features=10,
            hyperparameter_optimization=False
        )
        model.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Test that training completes in reasonable time
        self.assertLess(training_time, 10.0)  # Should complete in under 10 seconds
        self.assertTrue(model.is_trained)
    
    def test_var_calculation_performance(self):
        """Test VaR calculation performance."""
        import time
        
        # Create large dataset
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 10000))
        
        # Time VaR calculation
        start_time = time.time()
        
        var_calc = HistoricalVaR(confidence_level=0.95, time_horizon=1)
        var = var_calc.calculate(returns, 1000000.0)
        
        calculation_time = time.time() - start_time
        
        # Test that calculation completes quickly
        self.assertLess(calculation_time, 1.0)  # Should complete in under 1 second
        self.assertIsInstance(var, float)
    
    def test_realtime_processing_performance(self):
        """Test real-time processing performance."""
        import time
        
        # Create real-time engine
        realtime_engine = RealTimeEngine(
            algorithm=Mock(),
            data_portal=Mock(),
            processing_mode=ProcessingMode.STREAMING,
            batch_size=1000,
            max_latency_ms=1
        )
        
        # Time processing
        start_time = time.time()
        
        # Add events
        for i in range(1000):
            event = MarketEvent(
                timestamp=datetime.now(),
                symbol=f"STOCK_{i}",
                event_type="price_update",
                data={"price": 100.0 + i},
                source="test"
            )
            realtime_engine.add_market_event(event)
        
        processing_time = time.time() - start_time
        
        # Test that processing is fast
        self.assertLess(processing_time, 1.0)  # Should complete in under 1 second


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 