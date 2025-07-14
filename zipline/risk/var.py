"""
Value at Risk (VaR) Calculations for Zipline

This module provides comprehensive VaR calculation capabilities including
historical, parametric, Monte Carlo, and conditional VaR methods.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod
from scipy import stats
from scipy.optimize import minimize

from zipline.portfolio import Portfolio
from zipline.assets import Asset

logger = logging.getLogger(__name__)


class VaRCalculator(ABC):
    """
    Abstract base class for VaR calculations.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """
        Initialize the VaR calculator.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for VaR calculation (e.g., 0.95 for 95%).
        time_horizon : int
            Time horizon in days for VaR calculation.
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.alpha = 1 - confidence_level
        
    @abstractmethod
    def calculate(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate VaR for the given returns.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Value at Risk.
        """
        pass
    
    def calculate_portfolio_var(self, portfolio: Portfolio) -> float:
        """
        Calculate VaR for a portfolio.
        
        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to calculate VaR for.
            
        Returns
        -------
        float
            Portfolio VaR.
        """
        # Get portfolio returns
        returns = self._get_portfolio_returns(portfolio)
        portfolio_value = portfolio.portfolio_value
        
        return self.calculate(returns, portfolio_value)
    
    def _get_portfolio_returns(self, portfolio: Portfolio) -> pd.Series:
        """Get historical returns for the portfolio."""
        # This is a simplified implementation
        # In practice, you'd need to get actual historical returns
        # from the portfolio's performance history
        
        # Placeholder: generate synthetic returns
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        return returns


class HistoricalVaR(VaRCalculator):
    """
    Historical VaR calculation using empirical distribution.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """Initialize Historical VaR calculator."""
        super().__init__(confidence_level, time_horizon)
    
    def calculate(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate Historical VaR.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Historical VaR.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data available")
        
        # Calculate the empirical quantile
        var_quantile = np.percentile(returns, self.alpha * 100)
        
        # Scale by time horizon and portfolio value
        var = abs(var_quantile) * np.sqrt(self.time_horizon) * portfolio_value
        
        return var
    
    def calculate_rolling_var(self, returns: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling Historical VaR.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        window : int
            Rolling window size.
            
        Returns
        -------
        pd.Series
            Rolling VaR series.
        """
        rolling_var = returns.rolling(window=window).quantile(self.alpha)
        return abs(rolling_var) * np.sqrt(self.time_horizon)


class ParametricVaR(VaRCalculator):
    """
    Parametric VaR calculation assuming normal distribution.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """Initialize Parametric VaR calculator."""
        super().__init__(confidence_level, time_horizon)
    
    def calculate(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate Parametric VaR.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Parametric VaR.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data available")
        
        # Calculate mean and standard deviation
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Calculate VaR using normal distribution
        z_score = stats.norm.ppf(self.alpha)
        var = abs(z_score * std_return - mean_return) * np.sqrt(self.time_horizon) * portfolio_value
        
        return var
    
    def calculate_with_skewness_kurtosis(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate Parametric VaR with skewness and kurtosis adjustments.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Adjusted Parametric VaR.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data available")
        
        # Calculate moments
        mean_return = returns.mean()
        std_return = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Cornish-Fisher expansion for VaR
        z_score = stats.norm.ppf(self.alpha)
        z_cf = z_score + (z_score**2 - 1) * skewness / 6 + (z_score**3 - 3*z_score) * (kurtosis - 3) / 24
        
        var = abs(z_cf * std_return - mean_return) * np.sqrt(self.time_horizon) * portfolio_value
        
        return var


class MonteCarloVaR(VaRCalculator):
    """
    Monte Carlo VaR calculation using simulation.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1, n_simulations: int = 10000):
        """
        Initialize Monte Carlo VaR calculator.
        
        Parameters
        ----------
        confidence_level : float
            Confidence level for VaR calculation.
        time_horizon : int
            Time horizon in days.
        n_simulations : int
            Number of Monte Carlo simulations.
        """
        super().__init__(confidence_level, time_horizon)
        self.n_simulations = n_simulations
    
    def calculate(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate Monte Carlo VaR.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Monte Carlo VaR.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data available")
        
        # Estimate parameters from historical data
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate Monte Carlo simulations
        simulated_returns = np.random.normal(
            mean_return, 
            std_return, 
            (self.n_simulations, self.time_horizon)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.prod(1 + simulated_returns, axis=1) - 1
        
        # Calculate VaR
        var_quantile = np.percentile(cumulative_returns, self.alpha * 100)
        var = abs(var_quantile) * portfolio_value
        
        return var
    
    def calculate_with_copula(self, returns_dict: Dict[str, pd.Series], portfolio_value: float = 1.0) -> float:
        """
        Calculate Monte Carlo VaR using copula for multiple assets.
        
        Parameters
        ----------
        returns_dict : Dict[str, pd.Series]
            Dictionary of returns series for different assets.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Monte Carlo VaR with copula.
        """
        # This is a simplified implementation
        # In practice, you'd implement proper copula modeling
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = returns_df.corr()
        
        # Generate correlated random variables
        from scipy.stats import multivariate_normal
        
        # Estimate parameters
        means = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Generate simulations
        simulated_returns = multivariate_normal.rvs(
            means, 
            cov_matrix, 
            size=self.n_simulations
        )
        
        # Calculate portfolio returns (assuming equal weights)
        portfolio_returns = simulated_returns.mean(axis=1)
        
        # Calculate VaR
        var_quantile = np.percentile(portfolio_returns, self.alpha * 100)
        var = abs(var_quantile) * portfolio_value
        
        return var


class ConditionalVaR(VaRCalculator):
    """
    Conditional VaR (Expected Shortfall) calculation.
    """
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """Initialize Conditional VaR calculator."""
        super().__init__(confidence_level, time_horizon)
    
    def calculate(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Conditional VaR.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data available")
        
        # Calculate VaR threshold
        var_threshold = np.percentile(returns, self.alpha * 100)
        
        # Calculate expected shortfall
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        expected_shortfall = tail_returns.mean()
        
        # Scale by time horizon and portfolio value
        cvar = abs(expected_shortfall) * np.sqrt(self.time_horizon) * portfolio_value
        
        return cvar
    
    def calculate_parametric_cvar(self, returns: pd.Series, portfolio_value: float = 1.0) -> float:
        """
        Calculate parametric Conditional VaR assuming normal distribution.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        portfolio_value : float
            Current portfolio value.
            
        Returns
        -------
        float
            Parametric Conditional VaR.
        """
        # Remove NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data available")
        
        # Calculate parameters
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Calculate VaR threshold
        z_score = stats.norm.ppf(self.alpha)
        var_threshold = mean_return + z_score * std_return
        
        # Calculate expected shortfall for normal distribution
        expected_shortfall = mean_return - std_return * stats.norm.pdf(z_score) / self.alpha
        
        # Scale by time horizon and portfolio value
        cvar = abs(expected_shortfall) * np.sqrt(self.time_horizon) * portfolio_value
        
        return cvar


class VaRBacktester:
    """
    Backtesting framework for VaR models.
    """
    
    def __init__(self, var_calculator: VaRCalculator):
        """
        Initialize VaR backtester.
        
        Parameters
        ----------
        var_calculator : VaRCalculator
            The VaR calculator to backtest.
        """
        self.var_calculator = var_calculator
        self.backtest_results = []
    
    def backtest(self, returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """
        Perform VaR backtest.
        
        Parameters
        ----------
        returns : pd.Series
            Historical returns series.
        window : int
            Rolling window size for VaR calculation.
            
        Returns
        -------
        pd.DataFrame
            Backtest results.
        """
        results = []
        
        for i in range(window, len(returns)):
            # Get historical data for VaR calculation
            historical_returns = returns.iloc[i-window:i]
            current_return = returns.iloc[i]
            
            # Calculate VaR
            var = self.var_calculator.calculate(historical_returns)
            
            # Check if VaR violation occurred
            violation = current_return < -var
            
            results.append({
                'date': returns.index[i],
                'return': current_return,
                'var': var,
                'violation': violation,
                'excess_loss': max(0, -current_return - var)
            })
        
        self.backtest_results = pd.DataFrame(results)
        return self.backtest_results
    
    def calculate_violation_rate(self) -> float:
        """Calculate the violation rate."""
        if len(self.backtest_results) == 0:
            return 0.0
        
        return self.backtest_results['violation'].mean()
    
    def calculate_kupiec_test(self) -> Dict[str, float]:
        """
        Perform Kupiec test for VaR model validation.
        
        Returns
        -------
        Dict[str, float]
            Test statistics and p-value.
        """
        if len(self.backtest_results) == 0:
            return {'statistic': 0.0, 'p_value': 1.0}
        
        n = len(self.backtest_results)
        x = self.backtest_results['violation'].sum()
        p = self.var_calculator.alpha
        
        # Calculate test statistic
        if x == 0:
            statistic = 0.0
        else:
            statistic = 2 * (np.log((x/n)**x * ((n-x)/n)**(n-x)) - 
                           np.log(p**x * (1-p)**(n-x)))
        
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {'statistic': statistic, 'p_value': p_value}
    
    def calculate_christoffersen_test(self) -> Dict[str, float]:
        """
        Perform Christoffersen test for independence of violations.
        
        Returns
        -------
        Dict[str, float]
            Test statistics and p-value.
        """
        if len(self.backtest_results) < 2:
            return {'statistic': 0.0, 'p_value': 1.0}
        
        violations = self.backtest_results['violation'].values
        
        # Count transitions
        t00 = t01 = t10 = t11 = 0
        
        for i in range(1, len(violations)):
            if violations[i-1] == 0 and violations[i] == 0:
                t00 += 1
            elif violations[i-1] == 0 and violations[i] == 1:
                t01 += 1
            elif violations[i-1] == 1 and violations[i] == 0:
                t10 += 1
            elif violations[i-1] == 1 and violations[i] == 1:
                t11 += 1
        
        # Calculate probabilities
        p01 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0
        p11 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0
        p = (t01 + t11) / (t00 + t01 + t10 + t11)
        
        # Calculate test statistic
        if p01 == 0 or p11 == 0 or p == 0 or p == 1:
            statistic = 0.0
        else:
            statistic = 2 * (t01 * np.log(p01) + t11 * np.log(p11) - 
                           (t01 + t11) * np.log(p))
        
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        return {'statistic': statistic, 'p_value': p_value} 