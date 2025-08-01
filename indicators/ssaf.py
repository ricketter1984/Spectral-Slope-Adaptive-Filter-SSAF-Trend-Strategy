"""
Spectral Slope Adaptive Filter (SSAF) Indicator

This module implements the core SSAF indicator using spectral analysis techniques
to identify trend direction and strength in financial time series.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Any
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns


class SSAFIndicator:
    """
    Spectral Slope Adaptive Filter Indicator
    
    This indicator uses spectral analysis to identify trend direction and strength
    by analyzing the frequency domain characteristics of price movements.
    """
    
    def __init__(self, 
                 window_size: int = 20,
                 min_freq: float = 0.01,
                 max_freq: float = 0.5,
                 slope_threshold: float = 0.1,
                 adaptive_threshold: bool = True,
                 noise_reduction: bool = True):
        """
        Initialize SSAF indicator
        
        Args:
            window_size: Size of the analysis window
            min_freq: Minimum frequency to consider (as fraction of Nyquist)
            max_freq: Maximum frequency to consider (as fraction of Nyquist)
            slope_threshold: Threshold for trend detection
            adaptive_threshold: Whether to use adaptive thresholding
            noise_reduction: Whether to apply noise reduction
        """
        self.window_size = window_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.slope_threshold = slope_threshold
        self.adaptive_threshold = adaptive_threshold
        self.noise_reduction = noise_reduction
        
        # Storage for computed values
        self.spectral_slopes = []
        self.trend_signals = []
        self.signal_strengths = []
        self.frequency_bands = []
        
    def compute_spectral_slope(self, prices: np.ndarray) -> Tuple[float, Dict]:
        """
        Compute spectral slope using FFT analysis
        
        Args:
            prices: Price array for analysis
            
        Returns:
            Tuple of (spectral_slope, analysis_results)
        """
        if len(prices) < self.window_size:
            return 0.0, {}
        
        # Apply window function to reduce spectral leakage
        window = signal.windows.hann(len(prices))
        windowed_prices = prices * window
        
        # Compute FFT
        fft_values = fft(windowed_prices)
        fft_freqs = fftfreq(len(prices))
        
        # Get positive frequencies only
        positive_freqs = fft_freqs[:len(fft_freqs)//2]
        positive_fft = np.abs(fft_values[:len(fft_values)//2])
        
        # Filter frequency range of interest
        freq_mask = (positive_freqs >= self.min_freq) & (positive_freqs <= self.max_freq)
        filtered_freqs = positive_freqs[freq_mask]
        filtered_fft = positive_fft[freq_mask]
        
        if len(filtered_freqs) < 2:
            return 0.0, {}
        
        # Apply noise reduction if enabled
        if self.noise_reduction:
            filtered_fft = self._apply_noise_reduction(filtered_fft)
        
        # Compute spectral slope using linear regression
        if len(filtered_freqs) > 1:
            # Use log-log scale for better slope estimation
            log_freqs = np.log(filtered_freqs + 1e-10)
            log_fft = np.log(filtered_fft + 1e-10)
            
            # Linear regression
            slope, intercept = np.polyfit(log_freqs, log_fft, 1)
        else:
            slope = 0.0
            intercept = 0.0
        
        # Store frequency band information
        freq_band_info = {
            'frequencies': filtered_freqs,
            'magnitudes': filtered_fft,
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(self._calculate_r_squared(log_freqs, log_fft, float(slope), float(intercept)))
        }
        
        return float(slope), freq_band_info
    
    def _apply_noise_reduction(self, fft_magnitudes: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to FFT magnitudes
        
        Args:
            fft_magnitudes: FFT magnitude array
            
        Returns:
            Noise-reduced FFT magnitudes
        """
        # Simple moving average smoothing
        window = 3
        if len(fft_magnitudes) >= window:
            smoothed = np.convolve(fft_magnitudes, np.ones(window)/window, mode='same')
            return smoothed
        else:
            return fft_magnitudes
    
    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, slope: float, intercept: float) -> float:
        """Calculate R-squared value for linear fit"""
        if len(x) < 2:
            return 0.0
        
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def compute_adaptive_threshold(self, recent_slopes: List[float]) -> float:
        """
        Compute adaptive threshold based on recent slope values
        
        Args:
            recent_slopes: List of recent spectral slopes
            
        Returns:
            Adaptive threshold value
        """
        if len(recent_slopes) < 10:
            return self.slope_threshold
        
        # Use rolling standard deviation to adapt threshold
        slopes_array = np.array(recent_slopes[-20:])  # Last 20 values
        std_slope = np.std(slopes_array)
        mean_slope = np.mean(slopes_array)
        
        # Adaptive threshold based on volatility
        adaptive_threshold = self.slope_threshold * (1 + std_slope / (abs(mean_slope) + 1e-10))
        
        return max(adaptive_threshold, self.slope_threshold * 0.5)
    
    def generate_signal(self, spectral_slope: float, threshold: float) -> Tuple[str, float]:
        """
        Generate trading signal based on spectral slope
        
        Args:
            spectral_slope: Computed spectral slope
            threshold: Signal threshold
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if abs(spectral_slope) < threshold:
            return "HOLD", 0.0
        
        # Normalize signal strength
        signal_strength = min(abs(spectral_slope) / threshold, 2.0)
        
        if spectral_slope > 0:
            return "LONG", signal_strength
        else:
            return "SHORT", signal_strength
    
    def update(self, prices: pd.Series) -> Dict:
        """
        Update indicator with new price data
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary containing indicator results
        """
        if len(prices) < self.window_size:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'spectral_slope': 0.0,
                'threshold': self.slope_threshold
            }
        
        # Get recent price window
        recent_prices = prices.tail(self.window_size).values
        
        # Compute spectral slope
        spectral_slope, freq_info = self.compute_spectral_slope(recent_prices)
        
        # Store spectral slope
        self.spectral_slopes.append(spectral_slope)
        
        # Compute adaptive threshold if enabled
        if self.adaptive_threshold and len(self.spectral_slopes) > 5:
            threshold = self.compute_adaptive_threshold(self.spectral_slopes)
        else:
            threshold = self.slope_threshold
        
        # Generate signal
        signal_type, signal_strength = self.generate_signal(spectral_slope, threshold)
        
        # Store results
        self.trend_signals.append(signal_type)
        self.signal_strengths.append(signal_strength)
        self.frequency_bands.append(freq_info)
        
        return {
            'signal': signal_type,
            'strength': signal_strength,
            'spectral_slope': spectral_slope,
            'threshold': threshold,
            'freq_info': freq_info
        }
    
    def get_trend_analysis(self, lookback: int = 50) -> Dict:
        """
        Get trend analysis over recent periods
        
        Args:
            lookback: Number of periods to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        if len(self.trend_signals) < lookback:
            lookback = len(self.trend_signals)
        
        recent_signals = self.trend_signals[-lookback:]
        recent_strengths = self.signal_strengths[-lookback:]
        recent_slopes = self.spectral_slopes[-lookback:]
        
        # Calculate trend statistics
        long_signals = recent_signals.count('LONG')
        short_signals = recent_signals.count('SHORT')
        hold_signals = recent_signals.count('HOLD')
        
        avg_strength = np.mean(recent_strengths) if recent_strengths else 0.0
        avg_slope = np.mean(recent_slopes) if recent_slopes else 0.0
        
        # Determine dominant trend
        if long_signals > short_signals and long_signals > hold_signals:
            dominant_trend = 'LONG'
        elif short_signals > long_signals and short_signals > hold_signals:
            dominant_trend = 'SHORT'
        else:
            dominant_trend = 'NEUTRAL'
        
        return {
            'dominant_trend': dominant_trend,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'hold_signals': hold_signals,
            'avg_strength': avg_strength,
            'avg_slope': avg_slope,
            'trend_consistency': max(long_signals, short_signals) / lookback
        }
    
    def plot_spectral_analysis(self, prices: pd.Series, save_path: Optional[str] = None):
        """
        Plot spectral analysis results
        
        Args:
            prices: Price series
            save_path: Optional path to save the plot
        """
        if len(self.frequency_bands) == 0:
            print("No spectral analysis data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Price and signals
        ax1 = axes[0, 0]
        ax1.plot(prices.index[-len(self.trend_signals):], prices.tail(len(self.trend_signals)), 
                label='Price', color='blue', alpha=0.7)
        
        # Add signal markers
        for i, (signal, strength) in enumerate(zip(self.trend_signals, self.signal_strengths)):
            if signal == 'LONG':
                ax1.scatter(prices.index[-len(self.trend_signals)+i], 
                           prices.iloc[-len(self.trend_signals)+i], 
                           color='green', s=50*strength, alpha=0.7)
            elif signal == 'SHORT':
                ax1.scatter(prices.index[-len(self.trend_signals)+i], 
                           prices.iloc[-len(self.trend_signals)+i], 
                           color='red', s=50*strength, alpha=0.7)
        
        ax1.set_title('Price and SSAF Signals')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        
        # Plot 2: Spectral slopes
        ax2 = axes[0, 1]
        ax2.plot(self.spectral_slopes, label='Spectral Slope', color='purple')
        ax2.axhline(y=self.slope_threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')
        ax2.axhline(y=-self.slope_threshold, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Spectral Slopes Over Time')
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Spectral Slope')
        ax2.legend()
        
        # Plot 3: Signal strengths
        ax3 = axes[1, 0]
        ax3.bar(range(len(self.signal_strengths)), self.signal_strengths, 
               color=['green' if s == 'LONG' else 'red' if s == 'SHORT' else 'gray' 
                     for s in self.trend_signals], alpha=0.7)
        ax3.set_title('Signal Strengths')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Signal Strength')
        
        # Plot 4: Frequency spectrum (latest)
        ax4 = axes[1, 1]
        if self.frequency_bands and self.frequency_bands[-1]:
            freq_info = self.frequency_bands[-1]
            ax4.loglog(freq_info['frequencies'], freq_info['magnitudes'], 
                      label='Frequency Spectrum', color='orange')
            ax4.set_title('Latest Frequency Spectrum')
            ax4.set_xlabel('Frequency')
            ax4.set_ylabel('Magnitude')
            ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def reset(self):
        """Reset indicator state"""
        self.spectral_slopes = []
        self.trend_signals = []
        self.signal_strengths = []
        self.frequency_bands = []


class SSAFMultiTimeframe:
    """
    Multi-timeframe SSAF indicator for enhanced signal generation
    """
    
    def __init__(self, timeframes: List[int] = [10, 20, 50]):
        """
        Initialize multi-timeframe SSAF
        
        Args:
            timeframes: List of window sizes for different timeframes
        """
        self.timeframes = timeframes
        self.indicators = {tf: SSAFIndicator(window_size=tf) for tf in timeframes}
        
    def update(self, prices: pd.Series) -> Dict:
        """
        Update all timeframes with new price data
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary containing multi-timeframe results
        """
        results = {}
        
        for tf, indicator in self.indicators.items():
            results[tf] = indicator.update(prices)
        
        # Combine signals from different timeframes
        combined_signal = self._combine_signals(results)
        
        results['combined'] = combined_signal
        return results
    
    def _combine_signals(self, results: Dict) -> Dict:
        """
        Combine signals from different timeframes
        
        Args:
            results: Results from individual timeframes
            
        Returns:
            Combined signal result
        """
        signals = [results[tf]['signal'] for tf in self.timeframes]
        strengths = [results[tf]['strength'] for tf in self.timeframes]
        
        # Count signal types
        long_count = signals.count('LONG')
        short_count = signals.count('SHORT')
        hold_count = signals.count('HOLD')
        
        # Determine combined signal
        if long_count > short_count and long_count > hold_count:
            combined_signal = 'LONG'
            combined_strength = np.mean([s for s, sig in zip(strengths, signals) if sig == 'LONG'])
        elif short_count > long_count and short_count > hold_count:
            combined_signal = 'SHORT'
            combined_strength = np.mean([s for s, sig in zip(strengths, signals) if sig == 'SHORT'])
        else:
            combined_signal = 'HOLD'
            combined_strength = 0.0
        
        return {
            'signal': combined_signal,
            'strength': combined_strength,
            'signal_counts': {
                'LONG': long_count,
                'SHORT': short_count,
                'HOLD': hold_count
            }
        }


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    prices = data['Close']
    
    # Initialize SSAF indicator
    ssaf = SSAFIndicator(window_size=20)
    
    # Update indicator
    for i in range(20, len(prices)):
        result = ssaf.update(prices.iloc[:i+1])
        if i % 50 == 0:  # Print every 50th result
            print(f"Period {i}: Signal={result['signal']}, Strength={result['strength']:.3f}")
    
    # Plot results
    ssaf.plot_spectral_analysis(prices)
    
    # Get trend analysis
    trend_analysis = ssaf.get_trend_analysis()
    print("\nTrend Analysis:")
    for key, value in trend_analysis.items():
        print(f"{key}: {value}")
