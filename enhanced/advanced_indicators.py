"""
Advanced SSAF Indicators and Enhancements

This module provides enhanced versions of the SSAF indicator with additional
features like machine learning integration, regime detection, and advanced
filtering techniques.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Any
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Import base SSAF
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators.ssaf import SSAFIndicator


class AdvancedSSAFIndicator(SSAFIndicator):
    """
    Enhanced SSAF indicator with machine learning capabilities
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 min_freq: float = 0.01,
                 max_freq: float = 0.5,
                 slope_threshold: float = 0.1,
                 adaptive_threshold: bool = True,
                 noise_reduction: bool = True,
                 use_ml: bool = True,
                 ml_model=None):
        """
        Initialize advanced SSAF indicator
        
        Args:
            use_ml: Whether to use machine learning for prediction
            ml_model: Custom ML model (defaults to RandomForest)
        """
        super().__init__(window_size, min_freq, max_freq, slope_threshold, 
                        adaptive_threshold, noise_reduction)
        
        self.use_ml = use_ml
        self.ml_model = ml_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.ml_trained = False
        
    def extract_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract features for ML model
        
        Args:
            prices: Price array
            
        Returns:
            Feature array
        """
        if len(prices) < self.window_size:
            return np.array([])
        
        features = []
        
        # Basic price features
        returns = np.diff(prices)
        features.extend([
            np.mean(returns),
            np.std(returns),
            np.min(returns),
            np.max(returns),
            np.percentile(returns, 25),
            np.percentile(returns, 75),
            np.sum(returns > 0) / len(returns),  # Win rate
        ])
        
        # Technical indicators
        sma_5 = np.mean(prices[-5:])
        sma_10 = np.mean(prices[-10:])
        sma_20 = np.mean(prices[-20:])
        
        features.extend([
            sma_5 / sma_10 - 1,  # Short-term momentum
            sma_10 / sma_20 - 1,  # Medium-term momentum
            (prices[-1] - sma_20) / sma_20,  # Distance from SMA
        ])
        
        # Volatility measures
        features.extend([
            np.std(prices[-5:]) / np.mean(prices[-5:]),
            np.std(prices[-10:]) / np.mean(prices[-10:]),
            np.std(prices[-20:]) / np.mean(prices[-20:]),
        ])
        
        return np.array(features)
    
    def train_ml_model(self, prices: pd.Series, lookback: int = 100):
        """
        Train the ML model on historical data
        
        Args:
            prices: Price series
            lookback: Number of periods to use for training
        """
        if len(prices) < lookback + self.window_size:
            return
        
        X = []
        y = []
        
        for i in range(self.window_size, len(prices)):
            if i < lookback:
                continue
                
            price_window = prices.iloc[i-self.window_size:i].values
            features = self.extract_features(price_window)
            
            if len(features) > 0:
                X.append(features)
                
                # Target: next period return
                if i < len(prices) - 1:
                    target = (prices.iloc[i+1] - prices.iloc[i]) / prices.iloc[i]
                    y.append(target)
        
        if len(X) > 10:
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model.fit(X_scaled, y)
            self.ml_trained = True
    
    def predict_slope(self, prices: np.ndarray) -> float:
        """
        Predict slope using ML model
        
        Args:
            prices: Price array
            
        Returns:
            Predicted slope
        """
        if not self.use_ml or not self.ml_trained:
            return 0.0
        
        features = self.extract_features(prices)
        if len(features) == 0:
            return 0.0
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.ml_model.predict(features_scaled)[0]
        
        return float(prediction)


class RegimeAdaptiveSSAF(SSAFIndicator):
    """
    SSAF indicator that adapts to market regimes
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 min_freq: float = 0.01,
                 max_freq: float = 0.5,
                 slope_threshold: float = 0.1,
                 adaptive_threshold: bool = True,
                 noise_reduction: bool = True,
                 regime_windows: Dict[str, int] = None,
                 regime_thresholds: Dict[str, float] = None):
        """
        Initialize regime-adaptive SSAF
        
        Args:
            regime_windows: Window sizes for different regimes
            regime_thresholds: Thresholds for different regimes
        """
        super().__init__(window_size, min_freq, max_freq, slope_threshold,
                        adaptive_threshold, noise_reduction)
        
        self.regime_windows = regime_windows or {
            'trending': 15,
            'ranging': 25,
            'volatile': 10
        }
        
        self.regime_thresholds = regime_thresholds or {
            'trending': 0.15,
            'ranging': 0.05,
            'volatile': 0.20
        }
        
        self.current_regime = 'ranging'
        self.regime_history = []
    
    def detect_regime(self, prices: np.ndarray) -> str:
        """
        Detect current market regime
        
        Args:
            prices: Price array
            
        Returns:
            Current regime ('trending', 'ranging', 'volatile')
        """
        if len(prices) < 30:
            return 'ranging'
        
        # Calculate volatility
        returns = np.diff(prices)
        volatility = np.std(returns)
        
        # Calculate trend strength
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        trend_strength = abs(slope) / np.mean(prices)
        
        # Calculate ADX-like measure
        highs = prices
        lows = prices
        
        # Simple ADX calculation
        tr = np.maximum(highs[1:] - lows[1:], 
                     np.maximum(np.abs(highs[1:] - highs[:-1]), 
                               np.abs(lows[1:] - lows[:-1])))
        
        if len(tr) > 0:
            atr = np.mean(tr)
            adx = trend_strength / (volatility + 1e-10)
        else:
            adx = 0
        
        # Determine regime
        if adx > 0.3 and volatility < 0.02:
            return 'trending'
        elif volatility > 0.05:
            return 'volatile'
        else:
            return 'ranging'
    
    def update(self, prices: pd.Series) -> Dict:
        """
        Update indicator with regime adaptation
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with regime-adapted results
        """
        if len(prices) < self.window_size:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'spectral_slope': 0.0,
                'threshold': self.slope_threshold,
                'regime': 'ranging'
            }
        
        # Detect regime
        recent_prices = prices.tail(50).values
        regime = self.detect_regime(recent_prices)
        self.current_regime = regime
        self.regime_history.append(regime)
        
        # Adapt parameters based on regime
        self.window_size = self.regime_windows[regime]
        self.slope_threshold = self.regime_thresholds[regime]
        
        # Use base update with adapted parameters
        result = super().update(prices)
        result['regime'] = regime
        
        return result


class MultiAssetSSAF:
    """
    SSAF indicator for multiple assets with correlation analysis
    """
    
    def __init__(self, assets: List[str], base_config: Dict = None):
        """
        Initialize multi-asset SSAF
        
        Args:
            assets: List of asset symbols
            base_config: Base configuration for indicators
        """
        self.assets = assets
        self.base_config = base_config or {}
        self.indicators = {}
        
        for asset in assets:
            self.indicators[asset] = RegimeAdaptiveSSAF(**self.base_config)
        
        self.correlations = {}
        self.signals = {}
    
    def update_all(self, data: Dict[str, pd.Series]) -> Dict:
        """
        Update all assets
        
        Args:
            data: Dictionary of price series by asset
            
        Returns:
            Dictionary of results by asset
        """
        results = {}
        
        for asset in self.assets:
            if asset in data:
                results[asset] = self.indicators[asset].update(data[asset])
        
        # Calculate correlations
        self._calculate_correlations(data)
        
        # Generate combined signals
        self.signals = self._generate_combined_signals(results)
        
        return {
            'individual': results,
            'correlations': self.correlations,
            'combined_signals': self.signals
        }
    
    def _calculate_correlations(self, data: Dict[str, pd.Series]):
        """Calculate correlations between assets"""
        if len(data) < 2:
            return
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlation matrix
        self.correlations = df.corr().to_dict()
    
    def _generate_combined_signals(self, results: Dict) -> Dict:
        """Generate combined signals based on correlations"""
        combined = {}
        
        for asset in self.assets:
            if asset in results:
                signal = results[asset]['signal']
                strength = results[asset]['strength']
                
                # Adjust strength based on correlations
                if asset in self.correlations:
                    corr_strength = self._adjust_strength_by_correlation(asset, strength)
                    combined[asset] = {
                        'signal': signal,
                        'strength': corr_strength
                    }
                else:
                    combined[asset] = {
                        'signal': signal,
                        'strength': strength
                    }
        
        return combined
    
    def _adjust_strength_by_correlation(self, asset: str, strength: float) -> float:
        """Adjust signal strength based on correlations"""
        if asset not in self.correlations:
            return strength
        
        # Simple adjustment: reduce strength if highly correlated with others
        correlations = [abs(c) for k, c in self.correlations[asset].items() 
                       if k != asset and not pd.isna(c)]
        
        if correlations:
            avg_corr = np.mean(correlations)
            adjustment = 1 - avg_corr * 0.5  # Reduce strength for high correlations
            return strength * adjustment
        
        return strength


class MLSSAFIndicator(AdvancedSSAFIndicator):
    """
    Machine Learning SSAF with ensemble methods
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': None,  # Could add GradientBoosting
            'nn': None,  # Could add Neural Network
        }
        self.ensemble_weights = {'rf': 0.6, 'gb': 0.3, 'nn': 0.1}
    
    def ensemble_predict(self, prices: np.ndarray) -> float:
        """
        Make ensemble prediction
        
        Args:
            prices: Price array
            
        Returns:
            Ensemble prediction
        """
        if not self.ml_trained:
            return 0.0
        
        features = self.extract_features(prices)
        if len(features) == 0:
            return 0.0
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from all models
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if model is not None:
                pred = model.predict(features_scaled)[0]
                predictions.append(pred)
                weights.append(self.ensemble_weights[name])
        
        if not predictions:
            return 0.0
        
        # Weighted average
        ensemble_pred = np.average(predictions, weights=weights)
        return float(ensemble_pred)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    prices = data['Close']
    
    # Initialize advanced SSAF
    advanced_ssaf = RegimeAdaptiveSSAF(window_size=20)
    
    # Update indicator
    results = []
    for i in range(50, len(prices)):
        result = advanced_ssaf.update(prices.iloc[:i+1])
        if i % 50 == 0:
            print(f"Period {i}: Signal={result['signal']}, Regime={result['regime']}")
        results.append(result)
    
    # Print regime distribution
    regimes = [r['regime'] for r in results]
    print("\nRegime Distribution:")
    for regime in set(regimes):
        count = regimes.count(regime)
        print(f"{regime}: {count/len(regimes)*100:.1f}%")
