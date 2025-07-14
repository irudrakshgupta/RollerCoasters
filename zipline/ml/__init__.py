"""
Machine Learning Integration for Zipline

This module provides comprehensive machine learning capabilities for algorithmic trading,
including factor engineering, model training, and prediction.
"""

from .base import (
    MLModel,
    MLFactor,
    MLClassifier,
    MLFilter,
    FeatureStore,
    ModelRegistry,
)
from .automl import (
    AutoMLFactor,
    AutoMLClassifier,
    FeatureSelector,
    HyperparameterOptimizer,
)
from .online_learning import (
    OnlineLearner,
    IncrementalModel,
    AdaptiveModel,
)
from .ensemble import (
    EnsembleModel,
    VotingClassifier,
    StackingModel,
    BaggingModel,
)
from .feature_engineering import (
    FeatureEngineer,
    TechnicalFeatureExtractor,
    FundamentalFeatureExtractor,
    SentimentFeatureExtractor,
)
from .model_management import (
    ModelVersioning,
    ModelDeployment,
    ModelMonitoring,
    A_BTesting,
)
from .deep_learning import (
    DeepLearningModel,
    NeuralNetwork,
    LSTM,
    Transformer,
    AttentionMechanism,
)
from .reinforcement_learning import (
    RLAgent,
    QLearning,
    PolicyGradient,
    ActorCritic,
    MultiAgentRL,
)

__all__ = [
    # Base classes
    'MLModel',
    'MLFactor',
    'MLClassifier',
    'MLFilter',
    'FeatureStore',
    'ModelRegistry',
    
    # AutoML
    'AutoMLFactor',
    'AutoMLClassifier',
    'FeatureSelector',
    'HyperparameterOptimizer',
    
    # Online learning
    'OnlineLearner',
    'IncrementalModel',
    'AdaptiveModel',
    
    # Ensemble methods
    'EnsembleModel',
    'VotingClassifier',
    'StackingModel',
    'BaggingModel',
    
    # Feature engineering
    'FeatureEngineer',
    'TechnicalFeatureExtractor',
    'FundamentalFeatureExtractor',
    'SentimentFeatureExtractor',
    
    # Model management
    'ModelVersioning',
    'ModelDeployment',
    'ModelMonitoring',
    'A_BTesting',
    
    # Deep learning
    'DeepLearningModel',
    'NeuralNetwork',
    'LSTM',
    'Transformer',
    'AttentionMechanism',
    
    # Reinforcement learning
    'RLAgent',
    'QLearning',
    'PolicyGradient',
    'ActorCritic',
    'MultiAgentRL',
] 