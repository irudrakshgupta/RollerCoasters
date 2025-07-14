"""
Base classes for Machine Learning integration in Zipline.

This module provides the foundational classes for ML-powered algorithmic trading.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import json
import logging

from zipline.pipeline import Factor, Classifier, Filter
from zipline.pipeline.term import ComputableTerm
from zipline.utils.numpy_utils import float64_dtype, categorical_dtype, bool_dtype


logger = logging.getLogger(__name__)


class MLModel(ABC):
    """
    Abstract base class for machine learning models in Zipline.
    
    This class provides the interface for all ML models used in algorithmic trading.
    """
    
    def __init__(self, 
                 model_name: str,
                 model_type: str = "regression",
                 hyperparameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML model.
        
        Parameters
        ----------
        model_name : str
            Name of the model for identification and versioning.
        model_type : str
            Type of model ('regression', 'classification', 'clustering').
        hyperparameters : dict, optional
            Model hyperparameters.
        """
        self.model_name = model_name
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.is_trained = False
        self.training_history = []
        self.feature_names = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MLModel':
        """Train the model on the provided data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model performance score."""
        pass
    
    def save(self, filepath: str) -> None:
        """Save the model to disk."""
        model_data = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'model': self.model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> 'MLModel':
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.model_type = model_data['model_type']
        self.hyperparameters = model_data['hyperparameters']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.feature_names = model_data['feature_names']
        self.created_at = model_data['created_at']
        self.last_updated = model_data['last_updated']
        self.model = model_data['model']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        return None
    
    def update(self, X: np.ndarray, y: np.ndarray) -> 'MLModel':
        """Update the model with new data (online learning)."""
        if not self.is_trained:
            return self.fit(X, y)
        
        # For models that support partial_fit
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X, y)
        else:
            # Retrain with all data
            self.fit(X, y)
        
        self.last_updated = datetime.now()
        return self


class MLFactor(Factor):
    """
    A Factor that uses machine learning models for predictions.
    
    This class allows you to create factors based on ML model predictions,
    enabling sophisticated feature engineering and prediction-based strategies.
    """
    
    def __init__(self,
                 model: MLModel,
                 inputs: List[Factor],
                 window_length: int = 1,
                 mask: Optional[Filter] = None,
                 dtype: np.dtype = float64_dtype,
                 missing_value: float = np.nan):
        """
        Initialize the ML Factor.
        
        Parameters
        ----------
        model : MLModel
            The machine learning model to use for predictions.
        inputs : List[Factor]
            List of factors to use as features.
        window_length : int
            Number of periods to look back for feature calculation.
        mask : Filter, optional
            Filter to apply to the factor.
        dtype : np.dtype
            Data type for the factor output.
        missing_value : float
            Value to use for missing data.
        """
        super().__init__(
            inputs=inputs,
            window_length=window_length,
            mask=mask,
            dtype=dtype,
            missing_value=missing_value
        )
        self.model = model
        self.feature_names = [f"feature_{i}" for i in range(len(inputs))]
    
    def _compute(self, arrays, dates, assets, mask):
        """Compute the ML factor values."""
        if not self.model.is_trained:
            # Return missing values if model is not trained
            return np.full(mask.shape, self.missing_value)
        
        # Prepare features
        features = np.column_stack(arrays)
        
        # Make predictions
        try:
            predictions = self.model.predict(features)
            # Ensure predictions match the expected shape
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.full(mask.shape, self.missing_value)


class MLClassifier(Classifier):
    """
    A Classifier that uses machine learning models for classification.
    
    This class enables ML-based classification for grouping assets or
    creating categorical features based on model predictions.
    """
    
    def __init__(self,
                 model: MLModel,
                 inputs: List[Factor],
                 window_length: int = 1,
                 mask: Optional[Filter] = None,
                 dtype: np.dtype = categorical_dtype,
                 missing_value: str = ''):
        """
        Initialize the ML Classifier.
        
        Parameters
        ----------
        model : MLModel
            The machine learning model to use for classification.
        inputs : List[Factor]
            List of factors to use as features.
        window_length : int
            Number of periods to look back for feature calculation.
        mask : Filter, optional
            Filter to apply to the classifier.
        dtype : np.dtype
            Data type for the classifier output.
        missing_value : str
            Value to use for missing data.
        """
        super().__init__(
            inputs=inputs,
            window_length=window_length,
            mask=mask,
            dtype=dtype,
            missing_value=missing_value
        )
        self.model = model
        self.feature_names = [f"feature_{i}" for i in range(len(inputs))]
    
    def _compute(self, arrays, dates, assets, mask):
        """Compute the ML classifier values."""
        if not self.model.is_trained:
            # Return missing values if model is not trained
            return np.full(mask.shape, self.missing_value, dtype=self.dtype)
        
        # Prepare features
        features = np.column_stack(arrays)
        
        # Make predictions
        try:
            predictions = self.model.predict(features)
            # Convert predictions to categorical format
            if isinstance(predictions[0], (int, float)):
                predictions = [str(p) for p in predictions]
            return np.array(predictions, dtype=self.dtype)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.full(mask.shape, self.missing_value, dtype=self.dtype)


class MLFilter(Filter):
    """
    A Filter that uses machine learning models for filtering.
    
    This class enables ML-based filtering for asset selection based on
    model predictions or classifications.
    """
    
    def __init__(self,
                 model: MLModel,
                 inputs: List[Factor],
                 threshold: float = 0.5,
                 window_length: int = 1,
                 mask: Optional[Filter] = None):
        """
        Initialize the ML Filter.
        
        Parameters
        ----------
        model : MLModel
            The machine learning model to use for filtering.
        inputs : List[Factor]
            List of factors to use as features.
        threshold : float
            Threshold for filtering decisions.
        window_length : int
            Number of periods to look back for feature calculation.
        mask : Filter, optional
            Filter to apply to the filter.
        """
        super().__init__(
            inputs=inputs,
            window_length=window_length,
            mask=mask,
            dtype=bool_dtype,
            missing_value=False
        )
        self.model = model
        self.threshold = threshold
        self.feature_names = [f"feature_{i}" for i in range(len(inputs))]
    
    def _compute(self, arrays, dates, assets, mask):
        """Compute the ML filter values."""
        if not self.model.is_trained:
            # Return False if model is not trained
            return np.full(mask.shape, False, dtype=bool)
        
        # Prepare features
        features = np.column_stack(arrays)
        
        # Make predictions
        try:
            predictions = self.model.predict(features)
            # Apply threshold to create boolean filter
            return predictions > self.threshold
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.full(mask.shape, False, dtype=bool)


class FeatureStore:
    """
    A persistent store for features used in machine learning models.
    
    This class provides versioning, caching, and management of features
    used across multiple models and strategies.
    """
    
    def __init__(self, storage_path: str = "./feature_store"):
        """
        Initialize the feature store.
        
        Parameters
        ----------
        storage_path : str
            Path to store features on disk.
        """
        self.storage_path = storage_path
        self.features = {}
        self.metadata = {}
        
    def store(self, 
              feature_name: str, 
              features: np.ndarray, 
              metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store features with metadata."""
        self.features[feature_name] = features
        self.metadata[feature_name] = metadata or {}
        self.metadata[feature_name]['created_at'] = datetime.now()
        
    def retrieve(self, feature_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Retrieve features and metadata."""
        if feature_name not in self.features:
            raise KeyError(f"Feature '{feature_name}' not found in store")
        return self.features[feature_name], self.metadata[feature_name]
    
    def list_features(self) -> List[str]:
        """List all available features."""
        return list(self.features.keys())
    
    def delete(self, feature_name: str) -> None:
        """Delete a feature from the store."""
        if feature_name in self.features:
            del self.features[feature_name]
            del self.metadata[feature_name]


class ModelRegistry:
    """
    A registry for managing machine learning models.
    
    This class provides versioning, deployment, and monitoring of ML models
    used in trading strategies.
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        """
        Initialize the model registry.
        
        Parameters
        ----------
        registry_path : str
            Path to store models on disk.
        """
        self.registry_path = registry_path
        self.models = {}
        self.versions = {}
        
    def register(self, 
                 model: MLModel, 
                 version: str = "latest") -> None:
        """Register a model with a version."""
        model_id = f"{model.model_name}_{version}"
        self.models[model_id] = model
        self.versions[model.model_name] = version
        
    def get_model(self, 
                  model_name: str, 
                  version: str = "latest") -> Optional[MLModel]:
        """Get a model by name and version."""
        model_id = f"{model_name}_{version}"
        return self.models.get(model_id)
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(set(model.split('_')[0] for model in self.models.keys()))
    
    def get_versions(self, model_name: str) -> List[str]:
        """Get all versions of a model."""
        return [model.split('_')[1] for model in self.models.keys() 
                if model.startswith(model_name)]
    
    def deploy_model(self, 
                     model_name: str, 
                     version: str = "latest") -> bool:
        """Deploy a model for production use."""
        model = self.get_model(model_name, version)
        if model and model.is_trained:
            # Mark model as deployed
            model.deployed = True
            return True
        return False 