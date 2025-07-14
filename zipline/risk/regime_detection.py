"""
Regime Detection for Zipline

This module provides regime detection capabilities using Hidden Markov Models
and other statistical methods for identifying market regimes.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for HMM
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("hmmlearn not available. HMM functionality will be limited.")

logger = logging.getLogger(__name__)


class RegimeDetector(ABC):
    """
    Abstract base class for regime detection.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize the regime detector.
        
        Parameters
        ----------
        n_regimes : int
            Number of regimes to detect.
        """
        self.n_regimes = n_regimes
        self.is_fitted = False
        self.regime_labels = []
        self.regime_probabilities = []
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'RegimeDetector':
        """Fit the regime detector to data."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict regimes for new data."""
        pass
    
    @abstractmethod
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Predict regime probabilities for new data."""
        pass


class HiddenMarkovModel(RegimeDetector):
    """
    Hidden Markov Model for regime detection.
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 covariance_type: str = 'full',
                 n_iter: int = 100,
                 random_state: int = 42):
        """
        Initialize HMM regime detector.
        
        Parameters
        ----------
        n_regimes : int
            Number of regimes to detect.
        covariance_type : str
            Type of covariance matrix ('full', 'tied', 'diag', 'spherical').
        n_iter : int
            Maximum number of iterations for training.
        random_state : int
            Random state for reproducibility.
        """
        super().__init__(n_regimes)
        
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for HMM regime detection")
        
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )
    
    def fit(self, data: np.ndarray) -> 'HiddenMarkovModel':
        """
        Fit the HMM to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        HiddenMarkovModel
            Fitted model.
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Fit the model
        self.model.fit(data)
        self.is_fitted = True
        
        # Get regime labels for training data
        self.regime_labels = self.model.predict(data)
        self.regime_probabilities = self.model.predict_proba(data)
        
        logger.info(f"HMM fitted with {self.n_regimes} regimes")
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return self.model.predict(data)
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Regime probabilities of shape (n_samples, n_regimes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return self.model.predict_proba(data)
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        return self.model.transmat_
    
    def get_emission_parameters(self) -> Dict[str, np.ndarray]:
        """Get emission parameters (means and covariances)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        return {
            'means': self.model.means_,
            'covariances': self.model.covars_,
            'startprob': self.model.startprob_
        }
    
    def get_regime_statistics(self, data: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Get statistics for each regime.
        
        Parameters
        ----------
        data : np.ndarray
            Input data.
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Statistics for each regime.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        regimes = self.predict(data)
        statistics = {}
        
        for regime in range(self.n_regimes):
            regime_data = data[regimes == regime]
            
            if len(regime_data) > 0:
                statistics[regime] = {
                    'count': len(regime_data),
                    'proportion': len(regime_data) / len(data),
                    'mean': np.mean(regime_data, axis=0),
                    'std': np.std(regime_data, axis=0),
                    'min': np.min(regime_data, axis=0),
                    'max': np.max(regime_data, axis=0)
                }
            else:
                statistics[regime] = {
                    'count': 0,
                    'proportion': 0.0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                }
        
        return statistics


class GaussianMixtureRegimeDetector(RegimeDetector):
    """
    Gaussian Mixture Model for regime detection.
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 covariance_type: str = 'full',
                 n_init: int = 10,
                 random_state: int = 42):
        """
        Initialize GMM regime detector.
        
        Parameters
        ----------
        n_regimes : int
            Number of regimes to detect.
        covariance_type : str
            Type of covariance matrix.
        n_init : int
            Number of initializations.
        random_state : int
            Random state for reproducibility.
        """
        super().__init__(n_regimes)
        
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.random_state = random_state
        
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state
        )
    
    def fit(self, data: np.ndarray) -> 'GaussianMixtureRegimeDetector':
        """
        Fit the GMM to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        GaussianMixtureRegimeDetector
            Fitted model.
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Fit the model
        self.model.fit(data)
        self.is_fitted = True
        
        # Get regime labels for training data
        self.regime_labels = self.model.predict(data)
        self.regime_probabilities = self.model.predict_proba(data)
        
        logger.info(f"GMM fitted with {self.n_regimes} regimes")
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return self.model.predict(data)
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Regime probabilities of shape (n_samples, n_regimes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return self.model.predict_proba(data)
    
    def get_regime_statistics(self, data: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Get statistics for each regime.
        
        Parameters
        ----------
        data : np.ndarray
            Input data.
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Statistics for each regime.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        regimes = self.predict(data)
        statistics = {}
        
        for regime in range(self.n_regimes):
            regime_data = data[regimes == regime]
            
            if len(regime_data) > 0:
                statistics[regime] = {
                    'count': len(regime_data),
                    'proportion': len(regime_data) / len(data),
                    'mean': np.mean(regime_data, axis=0),
                    'std': np.std(regime_data, axis=0),
                    'min': np.min(regime_data, axis=0),
                    'max': np.max(regime_data, axis=0)
                }
            else:
                statistics[regime] = {
                    'count': 0,
                    'proportion': 0.0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                }
        
        return statistics


class KMeansRegimeDetector(RegimeDetector):
    """
    K-Means clustering for regime detection.
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 n_init: int = 10,
                 random_state: int = 42):
        """
        Initialize K-Means regime detector.
        
        Parameters
        ----------
        n_regimes : int
            Number of regimes to detect.
        n_init : int
            Number of initializations.
        random_state : int
            Random state for reproducibility.
        """
        super().__init__(n_regimes)
        
        self.n_init = n_init
        self.random_state = random_state
        
        self.model = KMeans(
            n_clusters=n_regimes,
            n_init=n_init,
            random_state=random_state
        )
        self.scaler = StandardScaler()
    
    def fit(self, data: np.ndarray) -> 'KMeansRegimeDetector':
        """
        Fit the K-Means model to data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        KMeansRegimeDetector
            Fitted model.
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale the data
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit the model
        self.model.fit(data_scaled)
        self.is_fitted = True
        
        # Get regime labels for training data
        self.regime_labels = self.model.labels_
        
        # Calculate regime probabilities (distance-based)
        distances = self.model.transform(data_scaled)
        self.regime_probabilities = self._calculate_probabilities(distances)
        
        logger.info(f"K-Means fitted with {self.n_regimes} regimes")
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        return self.model.predict(data_scaled)
    
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Regime probabilities of shape (n_samples, n_regimes).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Scale the data
        data_scaled = self.scaler.transform(data)
        
        # Calculate distances
        distances = self.model.transform(data_scaled)
        
        return self._calculate_probabilities(distances)
    
    def _calculate_probabilities(self, distances: np.ndarray) -> np.ndarray:
        """Calculate probabilities based on distances to cluster centers."""
        # Convert distances to probabilities using softmax
        # Smaller distances should have higher probabilities
        exp_neg_distances = np.exp(-distances)
        probabilities = exp_neg_distances / np.sum(exp_neg_distances, axis=1, keepdims=True)
        return probabilities
    
    def get_regime_statistics(self, data: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Get statistics for each regime.
        
        Parameters
        ----------
        data : np.ndarray
            Input data.
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Statistics for each regime.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        regimes = self.predict(data)
        statistics = {}
        
        for regime in range(self.n_regimes):
            regime_data = data[regimes == regime]
            
            if len(regime_data) > 0:
                statistics[regime] = {
                    'count': len(regime_data),
                    'proportion': len(regime_data) / len(data),
                    'mean': np.mean(regime_data, axis=0),
                    'std': np.std(regime_data, axis=0),
                    'min': np.min(regime_data, axis=0),
                    'max': np.max(regime_data, axis=0)
                }
            else:
                statistics[regime] = {
                    'count': 0,
                    'proportion': 0.0,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                }
        
        return statistics


class RegimeClassifier:
    """
    Classifier for regime prediction using multiple features.
    """
    
    def __init__(self, 
                 method: str = 'hmm',
                 n_regimes: int = 3,
                 features: Optional[List[str]] = None):
        """
        Initialize regime classifier.
        
        Parameters
        ----------
        method : str
            Method for regime detection ('hmm', 'gmm', 'kmeans').
        n_regimes : int
            Number of regimes to detect.
        features : List[str], optional
            List of features to use for regime detection.
        """
        self.method = method
        self.n_regimes = n_regimes
        self.features = features or ['returns', 'volatility', 'volume']
        
        # Initialize the appropriate detector
        if method == 'hmm':
            self.detector = HiddenMarkovModel(n_regimes=n_regimes)
        elif method == 'gmm':
            self.detector = GaussianMixtureRegimeDetector(n_regimes=n_regimes)
        elif method == 'kmeans':
            self.detector = KMeansRegimeDetector(n_regimes=n_regimes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame) -> 'RegimeClassifier':
        """
        Fit the regime classifier.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with features.
            
        Returns
        -------
        RegimeClassifier
            Fitted classifier.
        """
        # Extract features
        feature_data = self._extract_features(data)
        
        # Fit the detector
        self.detector.fit(feature_data)
        self.is_fitted = True
        
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for new data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with features.
            
        Returns
        -------
        np.ndarray
            Predicted regime labels.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        feature_data = self._extract_features(data)
        return self.detector.predict(feature_data)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for new data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with features.
            
        Returns
        -------
        np.ndarray
            Regime probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        feature_data = self._extract_features(data)
        return self.detector.predict_proba(feature_data)
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from data."""
        features = []
        
        for feature in self.features:
            if feature == 'returns':
                if 'close' in data.columns:
                    returns = data['close'].pct_change().fillna(0)
                    features.append(returns.values)
            elif feature == 'volatility':
                if 'close' in data.columns:
                    returns = data['close'].pct_change().fillna(0)
                    volatility = returns.rolling(window=20).std().fillna(0)
                    features.append(volatility.values)
            elif feature == 'volume':
                if 'volume' in data.columns:
                    volume = data['volume'].fillna(0)
                    features.append(volume.values)
            elif feature in data.columns:
                features.append(data[feature].fillna(0).values)
        
        if not features:
            raise ValueError("No valid features found in data")
        
        return np.column_stack(features)


class RegimeTransitionModel:
    """
    Model for predicting regime transitions.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime transition model.
        
        Parameters
        ----------
        n_regimes : int
            Number of regimes.
        """
        self.n_regimes = n_regimes
        self.transition_matrix = None
        self.is_fitted = False
    
    def fit(self, regime_sequence: np.ndarray) -> 'RegimeTransitionModel':
        """
        Fit the transition model to regime sequence.
        
        Parameters
        ----------
        regime_sequence : np.ndarray
            Sequence of regime labels.
            
        Returns
        -------
        RegimeTransitionModel
            Fitted model.
        """
        # Initialize transition matrix
        self.transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        # Count transitions
        for i in range(len(regime_sequence) - 1):
            current_regime = regime_sequence[i]
            next_regime = regime_sequence[i + 1]
            
            if 0 <= current_regime < self.n_regimes and 0 <= next_regime < self.n_regimes:
                self.transition_matrix[current_regime, next_regime] += 1
        
        # Normalize to get probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]
        
        # Handle rows with no transitions
        self.transition_matrix = np.nan_to_num(self.transition_matrix, nan=1.0/self.n_regimes)
        
        self.is_fitted = True
        return self
    
    def predict_next_regime(self, current_regime: int) -> Tuple[int, np.ndarray]:
        """
        Predict the next regime given current regime.
        
        Parameters
        ----------
        current_regime : int
            Current regime.
            
        Returns
        -------
        Tuple[int, np.ndarray]
            Predicted next regime and transition probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if current_regime < 0 or current_regime >= self.n_regimes:
            raise ValueError(f"Invalid regime: {current_regime}")
        
        transition_probs = self.transition_matrix[current_regime]
        next_regime = np.argmax(transition_probs)
        
        return next_regime, transition_probs
    
    def get_regime_persistence(self) -> np.ndarray:
        """Get persistence probabilities for each regime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        return np.diag(self.transition_matrix)
    
    def get_expected_regime_duration(self) -> np.ndarray:
        """Get expected duration for each regime."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        persistence = self.get_regime_persistence()
        # Expected duration = 1 / (1 - persistence)
        return 1 / (1 - persistence)


class VolatilityRegime:
    """
    Volatility regime detection and analysis.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize volatility regime detector.
        
        Parameters
        ----------
        n_regimes : int
            Number of volatility regimes.
        """
        self.n_regimes = n_regimes
        self.detector = HiddenMarkovModel(n_regimes=n_regimes) if HMM_AVAILABLE else GaussianMixtureRegimeDetector(n_regimes=n_regimes)
        self.is_fitted = False
    
    def fit(self, returns: pd.Series) -> 'VolatilityRegime':
        """
        Fit volatility regime detector.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
            
        Returns
        -------
        VolatilityRegime
            Fitted detector.
        """
        # Calculate rolling volatility
        volatility = returns.rolling(window=20).std().fillna(0)
        
        # Fit the detector
        self.detector.fit(volatility.values.reshape(-1, 1))
        self.is_fitted = True
        
        return self
    
    def predict(self, returns: pd.Series) -> np.ndarray:
        """
        Predict volatility regimes.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
            
        Returns
        -------
        np.ndarray
            Predicted volatility regimes.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        volatility = returns.rolling(window=20).std().fillna(0)
        return self.detector.predict(volatility.values.reshape(-1, 1))
    
    def get_volatility_regimes(self, returns: pd.Series) -> Dict[int, Dict[str, float]]:
        """
        Get volatility characteristics for each regime.
        
        Parameters
        ----------
        returns : pd.Series
            Return series.
            
        Returns
        -------
        Dict[int, Dict[str, float]]
            Volatility characteristics for each regime.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted")
        
        regimes = self.predict(returns)
        volatility = returns.rolling(window=20).std().fillna(0)
        
        characteristics = {}
        for regime in range(self.n_regimes):
            regime_vol = volatility[regimes == regime]
            
            if len(regime_vol) > 0:
                characteristics[regime] = {
                    'mean_volatility': np.mean(regime_vol),
                    'std_volatility': np.std(regime_vol),
                    'min_volatility': np.min(regime_vol),
                    'max_volatility': np.max(regime_vol),
                    'regime_duration': np.mean(np.diff(np.where(regimes == regime)[0])) if len(np.where(regimes == regime)[0]) > 1 else np.nan
                }
            else:
                characteristics[regime] = {
                    'mean_volatility': np.nan,
                    'std_volatility': np.nan,
                    'min_volatility': np.nan,
                    'max_volatility': np.nan,
                    'regime_duration': np.nan
                }
        
        return characteristics 