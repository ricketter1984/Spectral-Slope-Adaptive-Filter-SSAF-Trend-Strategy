"""
Simple test script to verify SSAF indicator functionality
"""

import numpy as np
import pandas as pd
from indicators.ssaf import SSAFIndicator

def test_ssaf_indicator():
    """Test the SSAF indicator with synthetic data"""
    
    # Create synthetic trending data
    np.random.seed(42)
    n = 100
    
    # Create upward trend
    trend_up = np.linspace(100, 120, n) + np.random.normal(0, 1, n)
    
    # Create downward trend
    trend_down = np.linspace(120, 100, n) + np.random.normal(0, 1, n)
    
    # Create sideways trend
    trend_sideways = np.full(n, 110) + np.random.normal(0, 1, n)
    
    # Test with upward trend
    prices_up = pd.Series(trend_up)
    ssaf = SSAFIndicator(window_size=20)
    
    print("Testing SSAF with upward trend...")
    for i in range(20, len(prices_up)):
        result = ssaf.update(prices_up.iloc[:i+1])
        if i == len(prices_up) - 1:
            print(f"Final signal: {result['signal']}, strength: {result['strength']:.3f}")
            print(f"Spectral slope: {result['spectral_slope']:.4f}")
    
    # Test with downward trend
    ssaf.reset()
    prices_down = pd.Series(trend_down)
    
    print("\nTesting SSAF with downward trend...")
    for i in range(20, len(prices_down)):
        result = ssaf.update(prices_down.iloc[:i+1])
        if i == len(prices_down) - 1:
            print(f"Final signal: {result['signal']}, strength: {result['strength']:.3f}")
            print(f"Spectral slope: {result['spectral_slope']:.4f}")
    
    # Test with sideways trend
    ssaf.reset()
    prices_sideways = pd.Series(trend_sideways)
    
    print("\nTesting SSAF with sideways trend...")
    for i in range(20, len(prices_sideways)):
        result = ssaf.update(prices_sideways.iloc[:i+1])
        if i == len(prices_sideways) - 1:
            print(f"Final signal: {result['signal']}, strength: {result['strength']:.3f}")
            print(f"Spectral slope: {result['spectral_slope']:.4f}")

if __name__ == "__main__":
    test_ssaf_indicator()
