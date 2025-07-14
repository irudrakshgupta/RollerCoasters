"""
Stress Testing for Zipline

This module provides comprehensive stress testing capabilities including
historical scenarios, hypothetical scenarios, and sensitivity analysis.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
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


class StressTester:
    """
    Comprehensive stress testing framework for portfolios.
    """
    
    def __init__(self, scenarios: Optional[List[str]] = None):
        """
        Initialize the stress tester.
        
        Parameters
        ----------
        scenarios : List[str], optional
            List of scenario names to include.
        """
        self.scenarios = scenarios or ['market_crash', 'volatility_spike', 'correlation_breakdown']
        self.scenario_generators = {
            'market_crash': self._generate_market_crash_scenario,
            'volatility_spike': self._generate_volatility_spike_scenario,
            'correlation_breakdown': self._generate_correlation_breakdown_scenario,
            'interest_rate_shock': self._generate_interest_rate_shock_scenario,
            'currency_crisis': self._generate_currency_crisis_scenario,
            'liquidity_crisis': self._generate_liquidity_crisis_scenario
        }
    
    def run_stress_tests(self, 
                        portfolio: Portfolio,
                        market_data: pd.DataFrame,
                        custom_scenarios: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Run comprehensive stress tests on a portfolio.
        
        Parameters
        ----------
        portfolio : Portfolio
            The portfolio to stress test.
        market_data : pd.DataFrame
            Historical market data.
        custom_scenarios : Dict[str, Dict[str, Any]], optional
            Custom stress scenarios.
            
        Returns
        -------
        Dict[str, Any]
            Stress test results.
        """
        results = {
            'scenarios': {},
            'summary': {},
            'worst_case': None,
            'recommendations': []
        }
        
        # Run built-in scenarios
        for scenario_name in self.scenarios:
            if scenario_name in self.scenario_generators:
                scenario_data = self.scenario_generators[scenario_name](market_data)
                scenario_result = self._apply_scenario(portfolio, scenario_data)
                results['scenarios'][scenario_name] = scenario_result
        
        # Run custom scenarios
        if custom_scenarios:
            for scenario_name, scenario_data in custom_scenarios.items():
                scenario_result = self._apply_scenario(portfolio, scenario_data)
                results['scenarios'][scenario_name] = scenario_result
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary(results['scenarios'])
        
        # Find worst case scenario
        results['worst_case'] = self._find_worst_case(results['scenarios'])
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _generate_market_crash_scenario(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a market crash scenario."""
        # Calculate historical worst drawdown
        returns = market_data['close'].pct_change().dropna()
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        worst_drawdown = drawdown.min()
        
        # Generate crash scenario (worse than historical worst)
        crash_magnitude = worst_drawdown * 1.5  # 50% worse than worst historical
        
        return {
            'type': 'market_crash',
            'magnitude': crash_magnitude,
            'duration': 30,  # days
            'description': f'Market crash scenario with {abs(crash_magnitude):.1%} decline',
            'asset_shocks': {
                'equity': crash_magnitude,
                'bonds': crash_magnitude * 0.3,  # Bonds less affected
                'commodities': crash_magnitude * 0.8
            }
        }
    
    def _generate_volatility_spike_scenario(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a volatility spike scenario."""
        # Calculate historical volatility
        returns = market_data['close'].pct_change().dropna()
        historical_vol = returns.rolling(window=252).std()
        max_vol = historical_vol.max()
        
        # Generate volatility spike (3x historical max)
        vol_spike = max_vol * 3
        
        return {
            'type': 'volatility_spike',
            'magnitude': vol_spike,
            'duration': 10,  # days
            'description': f'Volatility spike scenario with {vol_spike:.1%} daily volatility',
            'volatility_shock': vol_spike,
            'correlation_impact': 0.8  # High correlation during stress
        }
    
    def _generate_correlation_breakdown_scenario(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a correlation breakdown scenario."""
        return {
            'type': 'correlation_breakdown',
            'magnitude': 0.9,  # Correlation increases to 0.9
            'duration': 20,  # days
            'description': 'Correlation breakdown scenario with high asset correlations',
            'correlation_matrix': np.ones((10, 10)) * 0.9,  # High correlation matrix
            'diversification_impact': 0.5  # Diversification effectiveness reduced by 50%
        }
    
    def _generate_interest_rate_shock_scenario(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate an interest rate shock scenario."""
        return {
            'type': 'interest_rate_shock',
            'magnitude': 0.02,  # 200 basis point increase
            'duration': 5,  # days
            'description': 'Interest rate shock scenario with 200bp increase',
            'rate_shock': 0.02,
            'duration_impact': {
                'short_term': 0.01,  # 100bp impact on short-term rates
                'long_term': 0.025   # 250bp impact on long-term rates
            }
        }
    
    def _generate_currency_crisis_scenario(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a currency crisis scenario."""
        return {
            'type': 'currency_crisis',
            'magnitude': 0.15,  # 15% currency depreciation
            'duration': 15,  # days
            'description': 'Currency crisis scenario with 15% depreciation',
            'currency_shock': 0.15,
            'emerging_markets_impact': 0.25,  # 25% impact on emerging markets
            'developed_markets_impact': 0.05   # 5% impact on developed markets
        }
    
    def _generate_liquidity_crisis_scenario(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a liquidity crisis scenario."""
        return {
            'type': 'liquidity_crisis',
            'magnitude': 0.5,  # 50% reduction in liquidity
            'duration': 25,  # days
            'description': 'Liquidity crisis scenario with 50% liquidity reduction',
            'liquidity_shock': 0.5,
            'bid_ask_spread_impact': 3.0,  # 3x increase in bid-ask spreads
            'volume_impact': 0.7  # 70% reduction in trading volume
        }
    
    def _apply_scenario(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a stress scenario to the portfolio."""
        scenario_type = scenario_data['type']
        
        if scenario_type == 'market_crash':
            return self._apply_market_crash(portfolio, scenario_data)
        elif scenario_type == 'volatility_spike':
            return self._apply_volatility_spike(portfolio, scenario_data)
        elif scenario_type == 'correlation_breakdown':
            return self._apply_correlation_breakdown(portfolio, scenario_data)
        elif scenario_type == 'interest_rate_shock':
            return self._apply_interest_rate_shock(portfolio, scenario_data)
        elif scenario_type == 'currency_crisis':
            return self._apply_currency_crisis(portfolio, scenario_data)
        elif scenario_type == 'liquidity_crisis':
            return self._apply_liquidity_crisis(portfolio, scenario_data)
        else:
            return self._apply_generic_scenario(portfolio, scenario_data)
    
    def _apply_market_crash(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply market crash scenario."""
        initial_value = portfolio.portfolio_value
        asset_shocks = scenario_data['asset_shocks']
        
        # Calculate portfolio loss
        total_loss = 0
        position_losses = {}
        
        for asset, position in portfolio.positions.items():
            # Determine asset type and apply appropriate shock
            asset_type = self._classify_asset(asset)
            shock = asset_shocks.get(asset_type, asset_shocks['equity'])
            
            # Calculate position loss
            position_value = position.amount * position.last_sale_price
            position_loss = position_value * abs(shock)
            total_loss += position_loss
            position_losses[asset] = position_loss
        
        final_value = initial_value - total_loss
        portfolio_return = (final_value - initial_value) / initial_value
        
        return {
            'scenario_type': 'market_crash',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'position_losses': position_losses,
            'magnitude': scenario_data['magnitude'],
            'duration': scenario_data['duration']
        }
    
    def _apply_volatility_spike(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply volatility spike scenario."""
        initial_value = portfolio.portfolio_value
        vol_shock = scenario_data['volatility_shock']
        
        # Simulate high volatility returns
        n_simulations = 1000
        simulated_returns = np.random.normal(0, vol_shock, n_simulations)
        
        # Calculate portfolio value distribution
        portfolio_values = initial_value * (1 + simulated_returns)
        final_value = np.percentile(portfolio_values, 5)  # 5th percentile (95% VaR)
        
        total_loss = initial_value - final_value
        portfolio_return = (final_value - initial_value) / initial_value
        
        return {
            'scenario_type': 'volatility_spike',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'volatility_shock': vol_shock,
            'var_95': total_loss,
            'duration': scenario_data['duration']
        }
    
    def _apply_correlation_breakdown(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply correlation breakdown scenario."""
        initial_value = portfolio.portfolio_value
        diversification_impact = scenario_data['diversification_impact']
        
        # Calculate current diversification benefit
        position_weights = []
        position_returns = []
        
        for asset, position in portfolio.positions.items():
            weight = position.amount * position.last_sale_price / initial_value
            position_weights.append(weight)
            # Simulate individual asset returns
            position_returns.append(np.random.normal(-0.05, 0.15))  # Simulated returns
        
        # Calculate portfolio return with reduced diversification
        portfolio_return = np.sum(np.array(position_weights) * np.array(position_returns))
        portfolio_return *= (1 - diversification_impact)  # Reduce diversification benefit
        
        final_value = initial_value * (1 + portfolio_return)
        total_loss = initial_value - final_value
        
        return {
            'scenario_type': 'correlation_breakdown',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'diversification_impact': diversification_impact,
            'duration': scenario_data['duration']
        }
    
    def _apply_interest_rate_shock(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply interest rate shock scenario."""
        initial_value = portfolio.portfolio_value
        rate_shock = scenario_data['rate_shock']
        
        # Calculate bond price impact (simplified)
        # Bond prices fall when rates rise
        bond_impact = -0.05  # 5% decline in bond prices per 100bp rate increase
        total_rate_impact = rate_shock * 100 * bond_impact  # Convert to basis points
        
        # Apply to fixed income positions
        total_loss = 0
        for asset, position in portfolio.positions.items():
            if self._is_fixed_income(asset):
                position_value = position.amount * position.last_sale_price
                position_loss = position_value * abs(total_rate_impact)
                total_loss += position_loss
        
        final_value = initial_value - total_loss
        portfolio_return = (final_value - initial_value) / initial_value
        
        return {
            'scenario_type': 'interest_rate_shock',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'rate_shock': rate_shock,
            'duration': scenario_data['duration']
        }
    
    def _apply_currency_crisis(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply currency crisis scenario."""
        initial_value = portfolio.portfolio_value
        currency_shock = scenario_data['currency_shock']
        
        # Apply currency impact to international positions
        total_loss = 0
        for asset, position in portfolio.positions.items():
            if self._is_international(asset):
                position_value = position.amount * position.last_sale_price
                position_loss = position_value * currency_shock
                total_loss += position_loss
        
        final_value = initial_value - total_loss
        portfolio_return = (final_value - initial_value) / initial_value
        
        return {
            'scenario_type': 'currency_crisis',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'currency_shock': currency_shock,
            'duration': scenario_data['duration']
        }
    
    def _apply_liquidity_crisis(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply liquidity crisis scenario."""
        initial_value = portfolio.portfolio_value
        liquidity_shock = scenario_data['liquidity_shock']
        
        # Calculate liquidity discount
        total_loss = initial_value * liquidity_shock
        
        final_value = initial_value - total_loss
        portfolio_return = (final_value - initial_value) / initial_value
        
        return {
            'scenario_type': 'liquidity_crisis',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'liquidity_shock': liquidity_shock,
            'duration': scenario_data['duration']
        }
    
    def _apply_generic_scenario(self, portfolio: Portfolio, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a generic stress scenario."""
        initial_value = portfolio.portfolio_value
        magnitude = scenario_data.get('magnitude', 0.1)
        
        # Apply uniform shock to all positions
        total_loss = initial_value * abs(magnitude)
        final_value = initial_value - total_loss
        portfolio_return = (final_value - initial_value) / initial_value
        
        return {
            'scenario_type': 'generic',
            'initial_value': initial_value,
            'final_value': final_value,
            'total_loss': total_loss,
            'portfolio_return': portfolio_return,
            'magnitude': magnitude,
            'duration': scenario_data.get('duration', 10)
        }
    
    def _classify_asset(self, asset: Asset) -> str:
        """Classify an asset by type."""
        # This is a simplified classification
        # In practice, you'd use more sophisticated classification logic
        asset_symbol = getattr(asset, 'symbol', '').upper()
        
        if any(bond in asset_symbol for bond in ['BOND', 'TREASURY', 'GOVT']):
            return 'bonds'
        elif any(commodity in asset_symbol for commodity in ['GOLD', 'OIL', 'SILVER']):
            return 'commodities'
        else:
            return 'equity'
    
    def _is_fixed_income(self, asset: Asset) -> bool:
        """Check if asset is fixed income."""
        return self._classify_asset(asset) == 'bonds'
    
    def _is_international(self, asset: Asset) -> bool:
        """Check if asset is international."""
        # Simplified check - in practice, you'd use more sophisticated logic
        asset_symbol = getattr(asset, 'symbol', '').upper()
        return any(international in asset_symbol for international in ['INTL', 'EM', 'EURO'])
    
    def _calculate_summary(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for all scenarios."""
        if not scenario_results:
            return {}
        
        returns = [result['portfolio_return'] for result in scenario_results.values()]
        losses = [result['total_loss'] for result in scenario_results.values()]
        
        return {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'std_return': np.std(returns),
            'mean_loss': np.mean(losses),
            'max_loss': np.max(losses),
            'total_scenarios': len(scenario_results)
        }
    
    def _find_worst_case(self, scenario_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Find the worst case scenario."""
        if not scenario_results:
            return None
        
        worst_scenario = min(scenario_results.items(), 
                           key=lambda x: x[1]['portfolio_return'])
        return worst_scenario[0]
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stress test results."""
        recommendations = []
        
        summary = results.get('summary', {})
        worst_case = results.get('worst_case')
        
        if worst_case:
            worst_loss = results['scenarios'][worst_case]['total_loss']
            worst_return = results['scenarios'][worst_case]['portfolio_return']
            
            recommendations.append(f"Worst case scenario: {worst_case} with {worst_return:.1%} loss")
            
            if worst_return < -0.2:
                recommendations.append("Consider reducing portfolio risk exposure")
            if worst_return < -0.1:
                recommendations.append("Review position sizing and diversification")
        
        if summary.get('mean_return', 0) < -0.05:
            recommendations.append("Portfolio shows high sensitivity to stress scenarios")
        
        if summary.get('std_return', 0) > 0.1:
            recommendations.append("High scenario variability - consider more robust strategies")
        
        return recommendations


class HistoricalScenario:
    """Historical stress scenario generator."""
    
    def __init__(self):
        """Initialize historical scenario generator."""
        self.historical_events = {
            '2008_financial_crisis': {
                'start_date': '2008-09-15',
                'end_date': '2009-03-09',
                'description': 'Global Financial Crisis',
                'market_decline': -0.56,
                'volatility_spike': 0.08,
                'duration_days': 175
            },
            '2020_covid_crash': {
                'start_date': '2020-02-19',
                'end_date': '2020-03-23',
                'description': 'COVID-19 Market Crash',
                'market_decline': -0.34,
                'volatility_spike': 0.06,
                'duration_days': 33
            },
            '1987_black_monday': {
                'start_date': '1987-10-19',
                'end_date': '1987-10-19',
                'description': 'Black Monday Crash',
                'market_decline': -0.23,
                'volatility_spike': 0.15,
                'duration_days': 1
            }
        }
    
    def generate_historical_scenario(self, event_name: str, severity_multiplier: float = 1.0) -> Dict[str, Any]:
        """Generate a historical stress scenario."""
        if event_name not in self.historical_events:
            raise ValueError(f"Unknown historical event: {event_name}")
        
        event = self.historical_events[event_name]
        
        return {
            'type': 'historical',
            'event_name': event_name,
            'description': event['description'],
            'magnitude': event['market_decline'] * severity_multiplier,
            'volatility_shock': event['volatility_spike'] * severity_multiplier,
            'duration': event['duration_days'],
            'start_date': event['start_date'],
            'end_date': event['end_date']
        }


class HypotheticalScenario:
    """Hypothetical stress scenario generator."""
    
    def __init__(self):
        """Initialize hypothetical scenario generator."""
        pass
    
    def generate_extreme_scenario(self, 
                                market_decline: float = -0.5,
                                volatility_spike: float = 0.1,
                                duration_days: int = 30) -> Dict[str, Any]:
        """Generate an extreme hypothetical scenario."""
        return {
            'type': 'hypothetical',
            'description': 'Extreme market stress scenario',
            'magnitude': market_decline,
            'volatility_shock': volatility_spike,
            'duration': duration_days,
            'correlation_impact': 0.9,
            'liquidity_impact': 0.7
        }
    
    def generate_tail_risk_scenario(self, confidence_level: float = 0.99) -> Dict[str, Any]:
        """Generate a tail risk scenario."""
        # Calculate extreme values based on normal distribution
        z_score = stats.norm.ppf(1 - confidence_level)
        market_decline = z_score * 0.02  # Assuming 2% daily volatility
        
        return {
            'type': 'tail_risk',
            'description': f'{confidence_level:.0%} tail risk scenario',
            'magnitude': market_decline,
            'volatility_shock': abs(market_decline) * 2,
            'duration': 5,
            'confidence_level': confidence_level
        }


class SensitivityAnalysis:
    """Sensitivity analysis for stress testing."""
    
    def __init__(self):
        """Initialize sensitivity analysis."""
        pass
    
    def analyze_parameter_sensitivity(self,
                                    portfolio: Portfolio,
                                    parameter_name: str,
                                    parameter_range: List[float]) -> Dict[str, Any]:
        """Analyze sensitivity to a specific parameter."""
        results = {
            'parameter_name': parameter_name,
            'parameter_values': parameter_range,
            'portfolio_returns': [],
            'sensitivity_metrics': {}
        }
        
        for param_value in parameter_range:
            # Create scenario with parameter value
            scenario_data = {
                'type': 'sensitivity',
                'magnitude': param_value,
                'duration': 10
            }
            
            # Apply scenario (simplified)
            initial_value = portfolio.portfolio_value
            final_value = initial_value * (1 + param_value)
            portfolio_return = (final_value - initial_value) / initial_value
            
            results['portfolio_returns'].append(portfolio_return)
        
        # Calculate sensitivity metrics
        returns = np.array(results['portfolio_returns'])
        param_values = np.array(parameter_range)
        
        # Linear sensitivity (slope)
        if len(param_values) > 1:
            slope = np.polyfit(param_values, returns, 1)[0]
            results['sensitivity_metrics']['linear_sensitivity'] = slope
        
        # Maximum sensitivity
        if len(returns) > 1:
            max_sensitivity = np.max(np.abs(np.diff(returns) / np.diff(param_values)))
            results['sensitivity_metrics']['max_sensitivity'] = max_sensitivity
        
        return results 