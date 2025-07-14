"""
AutoML (Automated Machine Learning) for Zipline

This module provides automated machine learning capabilities including
feature engineering, hyperparameter optimization, and model selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline as SklearnPipeline

from .base import MLModel, MLFactor, MLClassifier, MLFilter
from zipline.pipeline import Factor, Classifier, Filter
from zipline.pipeline.term import ComputableTerm
from zipline.utils.numpy_utils import float64_dtype, categorical_dtype, bool_dtype

logger = logging.getLogger(__name__)


class AutoMLFactor(MLFactor):
    """
    Automated Machine Learning Factor that automatically selects features
    and optimizes hyperparameters.
    """
    
    def __init__(self,
                 inputs: List[Factor],
                 target_factor: Factor,
                 model_type: str = "regression",
                 feature_selection_method: str = "kbest",
                 n_features: int = 10,
                 hyperparameter_optimization: bool = True,
                 cv_folds: int = 5,
                 window_length: int = 1,
                 mask: Optional[Filter] = None,
                 dtype: np.dtype = float64_dtype,
                 missing_value: float = np.nan):
        """
        Initialize the AutoML Factor.
        
        Parameters
        ----------
        inputs : List[Factor]
            List of candidate factors to use as features.
        target_factor : Factor
            The target factor to predict.
        model_type : str
            Type of model ('regression' or 'classification').
        feature_selection_method : str
            Method for feature selection ('kbest', 'correlation', 'mutual_info').
        n_features : int
            Number of features to select.
        hyperparameter_optimization : bool
            Whether to perform hyperparameter optimization.
        cv_folds : int
            Number of cross-validation folds.
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
            model=AutoMLModel(
                model_type=model_type,
                feature_selection_method=feature_selection_method,
                n_features=n_features,
                hyperparameter_optimization=hyperparameter_optimization,
                cv_folds=cv_folds
            ),
            inputs=inputs,
            window_length=window_length,
            mask=mask,
            dtype=dtype,
            missing_value=missing_value
        )
        self.target_factor = target_factor
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.hyperparameter_optimization = hyperparameter_optimization
        self.cv_folds = cv_folds
        
    def _compute(self, arrays, dates, assets, mask):
        """Compute the AutoML factor values."""
        if not self.model.is_trained:
            # Train the model if not already trained
            self._train_model(arrays, dates, assets, mask)
        
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
    
    def _train_model(self, arrays, dates, assets, mask):
        """Train the AutoML model."""
        # Prepare training data
        features = np.column_stack(arrays)
        
        # Get target values (this is a simplified approach)
        # In practice, you'd need to get the target factor values
        # This is a placeholder for the actual implementation
        targets = np.random.randn(features.shape[0])  # Placeholder
        
        # Train the model
        self.model.fit(features, targets)
        logger.info("AutoML model trained successfully")


class AutoMLClassifier(MLClassifier):
    """
    Automated Machine Learning Classifier that automatically selects features
    and optimizes hyperparameters for classification tasks.
    """
    
    def __init__(self,
                 inputs: List[Factor],
                 target_classifier: Classifier,
                 feature_selection_method: str = "kbest",
                 n_features: int = 10,
                 hyperparameter_optimization: bool = True,
                 cv_folds: int = 5,
                 window_length: int = 1,
                 mask: Optional[Filter] = None,
                 dtype: np.dtype = categorical_dtype,
                 missing_value: str = ''):
        """
        Initialize the AutoML Classifier.
        
        Parameters
        ----------
        inputs : List[Factor]
            List of candidate factors to use as features.
        target_classifier : Classifier
            The target classifier to predict.
        feature_selection_method : str
            Method for feature selection ('kbest', 'correlation', 'mutual_info').
        n_features : int
            Number of features to select.
        hyperparameter_optimization : bool
            Whether to perform hyperparameter optimization.
        cv_folds : int
            Number of cross-validation folds.
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
            model=AutoMLModel(
                model_type="classification",
                feature_selection_method=feature_selection_method,
                n_features=n_features,
                hyperparameter_optimization=hyperparameter_optimization,
                cv_folds=cv_folds
            ),
            inputs=inputs,
            window_length=window_length,
            mask=mask,
            dtype=dtype,
            missing_value=missing_value
        )
        self.target_classifier = target_classifier
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.hyperparameter_optimization = hyperparameter_optimization
        self.cv_folds = cv_folds


class FeatureSelector:
    """
    Automated feature selection for machine learning models.
    """
    
    def __init__(self, method: str = "kbest", n_features: int = 10):
        """
        Initialize the feature selector.
        
        Parameters
        ----------
        method : str
            Feature selection method ('kbest', 'correlation', 'mutual_info').
        n_features : int
            Number of features to select.
        """
        self.method = method
        self.n_features = n_features
        self.selected_features = None
        self.feature_scores = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector."""
        if self.method == "kbest":
            if len(np.unique(y)) > 2:  # Classification
                selector = SelectKBest(score_func=f_classif, k=self.n_features)
            else:  # Regression
                selector = SelectKBest(score_func=f_regression, k=self.n_features)
            
            selector.fit(X, y)
            self.selected_features = selector.get_support()
            self.feature_scores = selector.scores_
        
        elif self.method == "correlation":
            # Select features based on correlation with target
            correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
            top_indices = np.argsort(correlations)[-self.n_features:]
            self.selected_features = np.zeros(X.shape[1], dtype=bool)
            self.selected_features[top_indices] = True
            self.feature_scores = correlations
        
        elif self.method == "mutual_info":
            # Select features based on mutual information
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            if len(np.unique(y)) > 2:  # Classification
                mi_scores = mutual_info_classif(X, y)
            else:  # Regression
                mi_scores = mutual_info_regression(X, y)
            
            top_indices = np.argsort(mi_scores)[-self.n_features:]
            self.selected_features = np.zeros(X.shape[1], dtype=bool)
            self.selected_features[top_indices] = True
            self.feature_scores = mi_scores
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data to select only the chosen features."""
        if self.selected_features is None:
            raise ValueError("FeatureSelector must be fitted before transform")
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the selector and transform the data."""
        return self.fit(X, y).transform(X)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for machine learning models.
    """
    
    def __init__(self, 
                 model_type: str = "regression",
                 optimization_method: str = "grid_search",
                 cv_folds: int = 5,
                 n_iter: int = 100):
        """
        Initialize the hyperparameter optimizer.
        
        Parameters
        ----------
        model_type : str
            Type of model ('regression' or 'classification').
        optimization_method : str
            Optimization method ('grid_search' or 'random_search').
        cv_folds : int
            Number of cross-validation folds.
        n_iter : int
            Number of iterations for random search.
        """
        self.model_type = model_type
        self.optimization_method = optimization_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.best_params = None
        self.best_score = None
        
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters for the given data."""
        if self.model_type == "regression":
            base_model = RandomForestRegressor()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:  # classification
            base_model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        
        if self.optimization_method == "grid_search":
            optimizer = GridSearchCV(
                base_model, 
                param_grid, 
                cv=self.cv_folds,
                scoring='neg_mean_squared_error' if self.model_type == "regression" else 'accuracy',
                n_jobs=-1
            )
        else:  # random_search
            optimizer = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error' if self.model_type == "regression" else 'accuracy',
                n_jobs=-1,
                random_state=42
            )
        
        optimizer.fit(X, y)
        self.best_params = optimizer.best_params_
        self.best_score = optimizer.best_score_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.best_score}")
        
        return self.best_params


class AutoMLModel(MLModel):
    """
    Automated Machine Learning model that combines feature selection
    and hyperparameter optimization.
    """
    
    def __init__(self,
                 model_type: str = "regression",
                 feature_selection_method: str = "kbest",
                 n_features: int = 10,
                 hyperparameter_optimization: bool = True,
                 cv_folds: int = 5):
        """
        Initialize the AutoML model.
        
        Parameters
        ----------
        model_type : str
            Type of model ('regression' or 'classification').
        feature_selection_method : str
            Method for feature selection.
        n_features : int
            Number of features to select.
        hyperparameter_optimization : bool
            Whether to perform hyperparameter optimization.
        cv_folds : int
            Number of cross-validation folds.
        """
        super().__init__(
            model_name=f"automl_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_type=model_type
        )
        
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.hyperparameter_optimization = hyperparameter_optimization
        self.cv_folds = cv_folds
        
        # Initialize components
        self.feature_selector = FeatureSelector(feature_selection_method, n_features)
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            model_type, "grid_search", cv_folds
        ) if hyperparameter_optimization else None
        
        # Initialize the base model
        if model_type == "regression":
            self.base_model = RandomForestRegressor()
        else:
            self.base_model = RandomForestClassifier()
        
        self.model = None
        self.feature_mask = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'AutoMLModel':
        """Train the AutoML model."""
        logger.info("Starting AutoML training...")
        
        # Step 1: Feature selection
        logger.info("Performing feature selection...")
        X_selected = self.feature_selector.fit_transform(X, y)
        self.feature_mask = self.feature_selector.selected_features
        
        # Step 2: Hyperparameter optimization
        if self.hyperparameter_optimization:
            logger.info("Performing hyperparameter optimization...")
            best_params = self.hyperparameter_optimizer.optimize(X_selected, y)
            self.base_model.set_params(**best_params)
        
        # Step 3: Train the final model
        logger.info("Training final model...")
        self.model = self.base_model.fit(X_selected, y)
        
        self.is_trained = True
        self.training_history.append({
            'timestamp': datetime.now(),
            'n_features_selected': self.n_features,
            'best_score': getattr(self.hyperparameter_optimizer, 'best_score', None)
        })
        
        logger.info("AutoML training completed successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply feature selection
        X_selected = self.feature_selector.transform(X)
        
        # Make predictions
        return self.model.predict(X_selected)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate model performance score."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        # Apply feature selection
        X_selected = self.feature_selector.transform(X)
        
        # Calculate score
        return self.model.score(X_selected, y)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        # Return feature importance only for selected features
        if self.feature_mask is not None:
            importance = np.zeros(len(self.feature_mask))
            importance[self.feature_mask] = self.model.feature_importances_
            return importance
        
        return self.model.feature_importances_
    
    def get_selected_features(self) -> Optional[np.ndarray]:
        """Get the mask of selected features."""
        return self.feature_mask 