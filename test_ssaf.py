"""
Test suite for Spectral Slope Adaptive Filter (SSAF) Indicator
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Import the SSAF indicator
from indicators.ssaf import SSAFIndicator, SSAFMultiTimeframe


class TestSSAFIndicator:
    """Test cases for SSAFIndicator class"""
    
    def test_initialization(self):
        """Test SSAF indicator initialization"""
        indicator = SSAFIndicator(
            window_size=20,
            min_freq=0.01,
            max_freq=0.5,
            slope_threshold=0.1,
            adaptive_threshold=True,
            noise_reduction=True
        )
        
        assert indicator.window_size == 20
        assert indicator.min_freq == 0.01
        assert indicator.max_freq == 0.5
        assert indicator.slope_threshold == 0.1
        assert indicator.adaptive_threshold is True
        assert indicator.noise_reduction is True
        
        # Check initial state
        assert len(indicator.spectral_slopes) == 0
        assert len(indicator.trend_signals) == 0
        assert len(indicator.signal_strengths) == 0
        assert len(indicator.frequency_bands) == 0
    
    def test_compute_spectral_slope_insufficient_data(self):
        """Test spectral slope computation with insufficient data"""
        indicator = SSAFIndicator(window_size=20)
        prices = np.array([1, 2, 3])  # Less than window_size
        
        slope, freq_info = indicator.compute_spectral_slope(prices)
        assert slope == 0.0
        assert freq_info == {}
    
    def test_compute_spectral_slope_sufficient_data(self):
        """Test spectral slope computation with sufficient data"""
        indicator = SSAFIndicator(window_size=20)
        
        # Generate synthetic price data with trend
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(50)) + 100
        
        slope, freq_info = indicator.compute_spectral_slope(prices)
        
        assert isinstance(slope, float)
        assert isinstance(freq_info, dict)
        assert 'frequencies' in freq_info
        assert 'magnitudes' in freq_info
        assert 'slope' in freq_info
        assert 'intercept' in freq_info
        assert 'r_squared' in freq_info
    
    def test_apply_noise_reduction(self):
        """Test noise reduction functionality"""
        indicator = SSAFIndicator()
        
        # Test with sufficient data
        fft_magnitudes = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        smoothed = indicator._apply_noise_reduction(fft_magnitudes)
        
        assert len(smoothed) == len(fft_magnitudes)
        assert isinstance(smoothed, np.ndarray)
        
        # Test with insufficient data
        fft_magnitudes_short = np.array([1.0, 2.0])
        smoothed_short = indicator._apply_noise_reduction(fft_magnitudes_short)
        
        assert len(smoothed_short) == len(fft_magnitudes_short)
    
    def test_calculate_r_squared(self):
        """Test R-squared calculation"""
        indicator = SSAFIndicator()
        
        # Test with sufficient data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # Perfect linear relationship
        slope = 2.0
        intercept = 0.0
        
        r_squared = indicator._calculate_r_squared(x, y, slope, intercept)
        assert abs(r_squared - 1.0) < 0.01  # Should be close to 1 for perfect fit
        
        # Test with insufficient data
        r_squared_short = indicator._calculate_r_squared(np.array([1]), np.array([1]), 1.0, 0.0)
        assert r_squared_short == 0.0
    
    def test_compute_adaptive_threshold(self):
        """Test adaptive threshold computation"""
        indicator = SSAFIndicator(slope_threshold=0.1)
        
        # Test with insufficient data
        threshold = indicator.compute_adaptive_threshold([0.05])
        assert threshold == 0.1
        
        # Test with sufficient data
        np.random.seed(42)
        recent_slopes = np.random.randn(20) * 0.05
        threshold = indicator.compute_adaptive_threshold(recent_slopes.tolist())
        
        assert isinstance(threshold, float)
        assert threshold >= 0.1
    
    def test_generate_signal(self):
        """Test signal generation"""
        indicator = SSAFIndicator(slope_threshold=0.1)
        
        # Test HOLD signal
        signal, strength = indicator.generate_signal(0.05, 0.1)
        assert signal == "HOLD"
        assert strength == 0.0
        
        # Test LONG signal
        signal, strength = indicator.generate_signal(0.15, 0.1)
        assert signal == "LONG"
        assert strength > 0.0
        
        # Test SHORT signal
        signal, strength = indicator.generate_signal(-0.15, 0.1)
        assert signal == "SHORT"
        assert strength > 0.0
    
    def test_update_insufficient_data(self):
        """Test update with insufficient data"""
        indicator = SSAFIndicator(window_size=20)
        prices = pd.Series([1, 2, 3])  # Less than window_size
        
        result = indicator.update(prices)
        
        assert result['signal'] == 'HOLD'
        assert result['strength'] == 0.0
        assert result['spectral_slope'] == 0.0
    
    def test_update_sufficient_data(self):
        """Test update with sufficient data"""
        indicator = SSAFIndicator(window_size=20)
        
        # Generate synthetic price data
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(50)) + 100)
        
        result = indicator.update(prices)
        
        assert isinstance(result['signal'], str)
        assert isinstance(result['strength'], float)
        assert isinstance(result['spectral_slope'], float)
        assert isinstance(result['threshold'], float)
        assert isinstance(result['freq_info'], dict)
    
    def test_get_trend_analysis(self):
        """Test trend analysis"""
        indicator = SSAFIndicator()
        
        # Add some mock data
        indicator.trend_signals = ['LONG', 'LONG', 'SHORT', 'HOLD', 'LONG']
        indicator.signal_strengths = [0.5, 0.7, 0.3, 0.0, 0.6]
        indicator.spectral_slopes = [0.1, 0.2, -0.1, 0.0, 0.15]
        
        analysis = indicator.get_trend_analysis(lookback=3)
        
        assert isinstance(analysis, dict)
        assert 'dominant_trend' in analysis
        assert 'long_signals' in analysis
        assert 'short_signals' in analysis
        assert 'hold_signals' in analysis
        assert 'avg_strength' in analysis
        assert 'avg_slope' in analysis
        assert 'trend_consistency' in analysis
    
    def test_reset(self):
        """Test reset functionality"""
        indicator = SSAFIndicator()
        
        # Add some data
        indicator.spectral_slopes = [0.1, 0.2, 0.3]
        indicator.trend_signals = ['LONG', 'SHORT', 'HOLD']
        indicator.signal_strengths = [0.5, 0.7, 0.3]
        indicator.frequency_bands = [{'test': 'data'}]
        
        indicator.reset()
        
        assert len(indicator.spectral_slopes) == 0
        assert len(indicator.trend_signals) == 0
        assert len(indicator.signal_strengths) == 0
        assert len(indicator.frequency_bands) == 0
    
    @patch('matplotlib.pyplot.show')
    def test_plot_spectral_analysis(self, mock_show):
        """Test plotting functionality"""
        indicator = SSAFIndicator()
        
        # Generate synthetic data
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(50)) + 100, 
                          index=pd.date_range('2023-01-01', periods=50))
        
        # Add some mock analysis data
        indicator.trend_signals = ['LONG', 'SHORT', 'HOLD']
        indicator.signal_strengths = [0.5, 0.7, 0.3]
        indicator.spectral_slopes = [0.1, -0.2, 0.0]
        indicator.frequency_bands = [
            {'frequencies': [0.1, 0.2], 'magnitudes': [1.0, 2.0]}
        ]
        
        # Test plotting (should not raise errors)
        indicator.plot_spectral_analysis(prices)
        mock_show.assert_called_once()


class TestSSAFMultiTimeframe:
    """Test cases for SSAFMultiTimeframe class"""
    
    def test_initialization(self):
        """Test multi-timeframe initialization"""
        mtf = SSAFMultiTimeframe(timeframes=[10, 20, 50])
        
        assert mtf.timeframes == [10, 20, 50]
        assert len(mtf.indicators) == 3
        assert 10 in mtf.indicators
        assert 20 in mtf.indicators
        assert 50 in mtf.indicators
    
    def test_update(self):
        """Test multi-timeframe update"""
        mtf = SSAFMultiTimeframe(timeframes=[10, 20])
        
        # Generate synthetic data
        np.random.seed(42)
        prices = pd.Series(np.cumsum(np.random.randn(50)) + 100)
        
        results = mtf.update(prices)
        
        assert isinstance(results, dict)
        assert 10 in results
        assert 20 in results
        assert 'combined' in results
        
        combined = results['combined']
        assert 'signal' in combined
        assert 'strength' in combined
        assert 'signal_counts' in combined


class TestIntegration:
    """Integration tests for SSAF indicator"""
    
    def test_full_workflow(self):
        """Test complete workflow with synthetic data"""
        # Generate synthetic price data with trend
        np.random.seed(42)
        n_periods = 100
        trend = np.linspace(0, 10, n_periods)
        noise = np.random.randn(n_periods) * 2
        prices = pd.Series(100 + trend + noise, 
                          index=pd.date_range('2023-01-01', periods=n_periods))
        
        # Initialize indicator
        indicator = SSAFIndicator(window_size=20)
        
        # Process data
        results = []
        for i in range(20, len(prices)):
            result = indicator.update(prices.iloc[:i+1])
            results.append(result)
        
        # Verify we got results
        assert len(results) > 0
        assert len(indicator.spectral_slopes) > 0
        assert len(indicator.trend_signals) > 0
        
        # Check trend analysis
        analysis = indicator.get_trend_analysis()
        assert isinstance(analysis, dict)
        assert 'dominant_trend' in analysis
    
    def test_multi_timeframe_integration(self):
        """Test multi-timeframe integration"""
        # Generate synthetic data
        np.random.seed(42)
        n_periods = 100
        prices = pd.Series(np.cumsum(np.random.randn(n_periods)) + 100,
                          index=pd.date_range('2023-01-01', periods=n_periods))
        
        # Initialize multi-timeframe
        mtf = SSAFMultiTimeframe(timeframes=[10, 20, 30])
        
        # Process data
        results = mtf.update(prices)
        
        # Verify structure
        assert isinstance(results, dict)
        assert len(results) == 4  # 3 timeframes + combined
        
        # Check combined signal
        combined = results['combined']
        assert combined['signal'] in ['LONG', 'SHORT', 'HOLD']
        assert 0 <= combined['strength'] <= 2.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
