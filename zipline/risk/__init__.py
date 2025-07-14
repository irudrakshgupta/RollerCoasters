"""
Advanced Risk Management for Zipline

This module provides comprehensive risk management capabilities including
VaR calculations, stress testing, regime detection, and dynamic position sizing.
"""

from .var import (
    VaRCalculator,
    MonteCarloVaR,
    HistoricalVaR,
    ParametricVaR,
    ConditionalVaR,
)
from .stress_testing import (
    StressTester,
    HistoricalScenario,
    HypotheticalScenario,
    SensitivityAnalysis,
    ScenarioGenerator,
)
from .regime_detection import (
    RegimeDetector,
    HiddenMarkovModel,
    RegimeClassifier,
    RegimeTransitionModel,
    VolatilityRegime,
)
from .position_sizing import (
    PositionSizer,
    KellyCriterion,
    RiskParity,
    BlackLitterman,
    DynamicSizing,
)
from .limits import (
    RiskLimits,
    PositionLimits,
    LeverageLimits,
    ConcentrationLimits,
    DrawdownLimits,
)
from .monitoring import (
    RiskMonitor,
    PortfolioHeatMap,
    RiskDashboard,
    AlertSystem,
    RiskReport,
)

__all__ = [
    # VaR calculations
    'VaRCalculator',
    'MonteCarloVaR',
    'HistoricalVaR',
    'ParametricVaR',
    'ConditionalVaR',
    
    # Stress testing
    'StressTester',
    'HistoricalScenario',
    'HypotheticalScenario',
    'SensitivityAnalysis',
    'ScenarioGenerator',
    
    # Regime detection
    'RegimeDetector',
    'HiddenMarkovModel',
    'RegimeClassifier',
    'RegimeTransitionModel',
    'VolatilityRegime',
    
    # Position sizing
    'PositionSizer',
    'KellyCriterion',
    'RiskParity',
    'BlackLitterman',
    'DynamicSizing',
    
    # Risk limits
    'RiskLimits',
    'PositionLimits',
    'LeverageLimits',
    'ConcentrationLimits',
    'DrawdownLimits',
    
    # Monitoring
    'RiskMonitor',
    'PortfolioHeatMap',
    'RiskDashboard',
    'AlertSystem',
    'RiskReport',
] 