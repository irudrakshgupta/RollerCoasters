"""
Feature Engineering for Zipline

This module provides comprehensive feature engineering capabilities for
machine learning in algorithmic trading.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import talib

from zipline.pipeline import Factor, Classifier, Filter
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import (
    SimpleMovingAverage,
    Returns,
    VWAP,
    AverageDollarVolume,
    BollingerBands,
    RSI,
    MACD
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering for algorithmic trading.
    """
    
    def __init__(self,
                 technical_features: bool = True,
                 fundamental_features: bool = True,
                 sentiment_features: bool = False,
                 alternative_features: bool = False,
                 feature_scaling: str = "standard",
                 feature_selection: bool = True,
                 n_features: int = 50):
        """
        Initialize the feature engineer.
        
        Parameters
        ----------
        technical_features : bool
            Whether to include technical indicators.
        fundamental_features : bool
            Whether to include fundamental data.
        sentiment_features : bool
            Whether to include sentiment data.
        alternative_features : bool
            Whether to include alternative data.
        feature_scaling : str
            Scaling method ('standard', 'minmax', 'robust').
        feature_selection : bool
            Whether to perform feature selection.
        n_features : int
            Number of features to select.
        """
        self.technical_features = technical_features
        self.fundamental_features = fundamental_features
        self.sentiment_features = sentiment_features
        self.alternative_features = alternative_features
        self.feature_scaling = feature_scaling
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        # Initialize components
        self.scaler = self._get_scaler()
        self.feature_selector = None
        self.feature_names = []
        
    def _get_scaler(self):
        """Get the appropriate scaler."""
        if self.feature_scaling == "standard":
            return StandardScaler()
        elif self.feature_scaling == "minmax":
            return MinMaxScaler()
        elif self.feature_scaling == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def create_technical_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators from price data.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data with OHLCV columns.
            
        Returns
        -------
        pd.DataFrame
            Technical features.
        """
        features = pd.DataFrame(index=prices.index)
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{window}'] = prices['close'].rolling(window=window).mean()
            features[f'ema_{window}'] = prices['close'].ewm(span=window).mean()
            features[f'std_{window}'] = prices['close'].rolling(window=window).std()
        
        # Price-based features
        features['returns'] = prices['close'].pct_change()
        features['log_returns'] = np.log(prices['close'] / prices['close'].shift(1))
        features['price_momentum'] = prices['close'] / prices['close'].shift(20) - 1
        features['price_acceleration'] = features['returns'].diff()
        
        # Volume features
        features['volume_sma'] = prices['volume'].rolling(window=20).mean()
        features['volume_ratio'] = prices['volume'] / features['volume_sma']
        features['volume_momentum'] = prices['volume'] / prices['volume'].shift(20) - 1
        
        # Volatility features
        features['volatility_20'] = features['returns'].rolling(window=20).std()
        features['volatility_60'] = features['returns'].rolling(window=60).std()
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_60']
        
        # Technical indicators using TA-Lib
        if 'high' in prices.columns and 'low' in prices.columns:
            try:
                # RSI
                features['rsi_14'] = talib.RSI(prices['close'].values, timeperiod=14)
                features['rsi_30'] = talib.RSI(prices['close'].values, timeperiod=30)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(prices['close'].values)
                features['macd'] = macd
                features['macd_signal'] = macd_signal
                features['macd_hist'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(prices['close'].values)
                features['bb_upper'] = bb_upper
                features['bb_middle'] = bb_middle
                features['bb_lower'] = bb_lower
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                features['bb_position'] = (prices['close'] - bb_lower) / (bb_upper - bb_lower)
                
                # Stochastic
                slowk, slowd = talib.STOCH(prices['high'].values, prices['low'].values, 
                                         prices['close'].values)
                features['stoch_k'] = slowk
                features['stoch_d'] = slowd
                
                # Williams %R
                features['williams_r'] = talib.WILLR(prices['high'].values, prices['low'].values, 
                                                   prices['close'].values)
                
            except Exception as e:
                logger.warning(f"Could not calculate TA-Lib indicators: {e}")
        
        # Remove infinite and NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        return features
    
    def create_fundamental_features(self, fundamental_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create fundamental features from financial data.
        
        Parameters
        ----------
        fundamental_data : Dict[str, pd.DataFrame]
            Dictionary of fundamental data (income, balance, cash flow).
            
        Returns
        -------
        pd.DataFrame
            Fundamental features.
        """
        features = pd.DataFrame()
        
        # This is a placeholder implementation
        # In practice, you'd extract features from actual fundamental data
        
        # Example features (would be calculated from real data)
        features['pe_ratio'] = np.random.normal(15, 5, 1000)  # Placeholder
        features['pb_ratio'] = np.random.normal(2, 1, 1000)   # Placeholder
        features['debt_to_equity'] = np.random.normal(0.5, 0.3, 1000)  # Placeholder
        features['roe'] = np.random.normal(0.1, 0.05, 1000)  # Placeholder
        features['roa'] = np.random.normal(0.05, 0.03, 1000)  # Placeholder
        
        return features
    
    def create_sentiment_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment features from news and social media data.
        
        Parameters
        ----------
        sentiment_data : pd.DataFrame
            Sentiment data from news and social media.
            
        Returns
        -------
        pd.DataFrame
            Sentiment features.
        """
        features = pd.DataFrame()
        
        # This is a placeholder implementation
        # In practice, you'd extract features from actual sentiment data
        
        # Example features (would be calculated from real data)
        features['news_sentiment'] = np.random.normal(0, 0.5, 1000)  # Placeholder
        features['social_sentiment'] = np.random.normal(0, 0.5, 1000)  # Placeholder
        features['news_volume'] = np.random.poisson(10, 1000)  # Placeholder
        features['social_volume'] = np.random.poisson(50, 1000)  # Placeholder
        
        return features
    
    def create_alternative_features(self, alt_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create alternative data features.
        
        Parameters
        ----------
        alt_data : Dict[str, pd.DataFrame]
            Alternative data sources.
            
        Returns
        -------
        pd.DataFrame
            Alternative features.
        """
        features = pd.DataFrame()
        
        # This is a placeholder implementation
        # In practice, you'd extract features from actual alternative data
        
        # Example features (would be calculated from real data)
        features['satellite_data'] = np.random.normal(0, 1, 1000)  # Placeholder
        features['weather_data'] = np.random.normal(0, 1, 1000)   # Placeholder
        features['esg_score'] = np.random.uniform(0, 100, 1000)   # Placeholder
        
        return features
    
    def engineer_features(self, 
                         prices: pd.DataFrame,
                         fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
                         sentiment_data: Optional[pd.DataFrame] = None,
                         alt_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Engineer all features from available data.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        fundamental_data : Dict[str, pd.DataFrame], optional
            Fundamental data.
        sentiment_data : pd.DataFrame, optional
            Sentiment data.
        alt_data : Dict[str, pd.DataFrame], optional
            Alternative data.
            
        Returns
        -------
        pd.DataFrame
            Engineered features.
        """
        all_features = []
        
        # Technical features
        if self.technical_features:
            tech_features = self.create_technical_features(prices)
            all_features.append(tech_features)
        
        # Fundamental features
        if self.fundamental_features and fundamental_data:
            fund_features = self.create_fundamental_features(fundamental_data)
            all_features.append(fund_features)
        
        # Sentiment features
        if self.sentiment_features and sentiment_data:
            sent_features = self.create_sentiment_features(sentiment_data)
            all_features.append(sent_features)
        
        # Alternative features
        if self.alternative_features and alt_data:
            alt_features = self.create_alternative_features(alt_data)
            all_features.append(alt_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
        else:
            # Fallback to basic features
            combined_features = self.create_technical_features(prices)
        
        # Store feature names
        self.feature_names = combined_features.columns.tolist()
        
        return combined_features
    
    def fit_transform(self, 
                     prices: pd.DataFrame,
                     fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
                     sentiment_data: Optional[pd.DataFrame] = None,
                     alt_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Fit the feature engineer and transform the data.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        fundamental_data : Dict[str, pd.DataFrame], optional
            Fundamental data.
        sentiment_data : pd.DataFrame, optional
            Sentiment data.
        alt_data : Dict[str, pd.DataFrame], optional
            Alternative data.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        # Engineer features
        features = self.engineer_features(prices, fundamental_data, sentiment_data, alt_data)
        
        # Remove any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        # Feature selection
        if self.feature_selection and len(features_scaled.columns) > self.n_features:
            self.feature_selector = SelectKBest(score_func=f_regression, k=self.n_features)
            features_selected = self.feature_selector.fit_transform(features_scaled, np.zeros(len(features_scaled)))
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support()
            selected_features = features_scaled.columns[selected_indices]
            
            features_scaled = pd.DataFrame(
                features_selected,
                index=features_scaled.index,
                columns=selected_features
            )
        
        return features_scaled
    
    def transform(self, 
                 prices: pd.DataFrame,
                 fundamental_data: Optional[Dict[str, pd.DataFrame]] = None,
                 sentiment_data: Optional[pd.DataFrame] = None,
                 alt_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Transform new data using fitted parameters.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data.
        fundamental_data : Dict[str, pd.DataFrame], optional
            Fundamental data.
        sentiment_data : pd.DataFrame, optional
            Sentiment data.
        alt_data : Dict[str, pd.DataFrame], optional
            Alternative data.
            
        Returns
        -------
        pd.DataFrame
            Transformed features.
        """
        # Engineer features
        features = self.engineer_features(prices, fundamental_data, sentiment_data, alt_data)
        
        # Remove any remaining NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        # Feature selection
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support()
            selected_features = features_scaled.columns[selected_indices]
            
            features_scaled = pd.DataFrame(
                features_selected,
                index=features_scaled.index,
                columns=selected_features
            )
        
        return features_scaled
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if self.feature_selector is not None:
            return pd.Series(
                self.feature_selector.scores_,
                index=self.feature_selector.get_feature_names_out()
            )
        return pd.Series()
    
    def get_feature_names(self) -> List[str]:
        """Get the names of engineered features."""
        return self.feature_names.copy()


class TechnicalFeatureExtractor:
    """Extract technical features from price data."""
    
    @staticmethod
    def extract_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
        """Extract momentum-based features."""
        features = pd.DataFrame(index=prices.index)
        
        # Price momentum
        for period in [1, 5, 10, 20, 60]:
            features[f'momentum_{period}'] = prices['close'] / prices['close'].shift(period) - 1
        
        # Volume momentum
        for period in [1, 5, 10, 20]:
            features[f'volume_momentum_{period}'] = prices['volume'] / prices['volume'].shift(period) - 1
        
        return features
    
    @staticmethod
    def extract_volatility_features(prices: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility-based features."""
        features = pd.DataFrame(index=prices.index)
        
        returns = prices['close'].pct_change()
        
        # Rolling volatility
        for period in [5, 10, 20, 60]:
            features[f'volatility_{period}'] = returns.rolling(window=period).std()
        
        # Volatility ratios
        features['volatility_ratio_5_20'] = features['volatility_5'] / features['volatility_20']
        features['volatility_ratio_10_60'] = features['volatility_10'] / features['volatility_60']
        
        return features
    
    @staticmethod
    def extract_trend_features(prices: pd.DataFrame) -> pd.DataFrame:
        """Extract trend-based features."""
        features = pd.DataFrame(index=prices.index)
        
        # Moving average trends
        for short, long in [(5, 20), (10, 50), (20, 100)]:
            short_ma = prices['close'].rolling(window=short).mean()
            long_ma = prices['close'].rolling(window=long).mean()
            features[f'trend_{short}_{long}'] = short_ma / long_ma - 1
        
        # Price position relative to moving averages
        for period in [20, 50, 100, 200]:
            ma = prices['close'].rolling(window=period).mean()
            features[f'price_vs_ma_{period}'] = prices['close'] / ma - 1
        
        return features


class FundamentalFeatureExtractor:
    """Extract fundamental features from financial data."""
    
    @staticmethod
    def extract_valuation_features(financial_data: pd.DataFrame) -> pd.DataFrame:
        """Extract valuation ratios."""
        features = pd.DataFrame(index=financial_data.index)
        
        # This would be implemented with real fundamental data
        # Placeholder implementation
        features['pe_ratio'] = np.random.normal(15, 5, len(financial_data))
        features['pb_ratio'] = np.random.normal(2, 1, len(financial_data))
        features['ps_ratio'] = np.random.normal(3, 1, len(financial_data))
        
        return features
    
    @staticmethod
    def extract_profitability_features(financial_data: pd.DataFrame) -> pd.DataFrame:
        """Extract profitability metrics."""
        features = pd.DataFrame(index=financial_data.index)
        
        # This would be implemented with real fundamental data
        # Placeholder implementation
        features['roe'] = np.random.normal(0.1, 0.05, len(financial_data))
        features['roa'] = np.random.normal(0.05, 0.03, len(financial_data))
        features['gross_margin'] = np.random.normal(0.3, 0.1, len(financial_data))
        
        return features


class SentimentFeatureExtractor:
    """Extract sentiment features from news and social media."""
    
    @staticmethod
    def extract_news_sentiment(news_data: pd.DataFrame) -> pd.DataFrame:
        """Extract news sentiment features."""
        features = pd.DataFrame(index=news_data.index)
        
        # This would be implemented with real sentiment data
        # Placeholder implementation
        features['news_sentiment'] = np.random.normal(0, 0.5, len(news_data))
        features['news_volume'] = np.random.poisson(10, len(news_data))
        features['news_polarity'] = np.random.uniform(-1, 1, len(news_data))
        
        return features
    
    @staticmethod
    def extract_social_sentiment(social_data: pd.DataFrame) -> pd.DataFrame:
        """Extract social media sentiment features."""
        features = pd.DataFrame(index=social_data.index)
        
        # This would be implemented with real sentiment data
        # Placeholder implementation
        features['social_sentiment'] = np.random.normal(0, 0.5, len(social_data))
        features['social_volume'] = np.random.poisson(50, len(social_data))
        features['social_engagement'] = np.random.uniform(0, 100, len(social_data))
        
        return features 