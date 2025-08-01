"""
Test script for Enhanced SSAF Strategy

This script tests the enhanced SSAF indicators and strategy implementations
to ensure they work correctly with the provided backtesting framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from indicators.ssaf import SSAFIndicator, SSAFMultiTimeframe
from enhanced.advanced_indicators import (
    AdvancedSSAFIndicator, 
    RegimeAdaptiveSSAF, 
    MultiAssetSSAF,
    MLSSAFIndicator
)
from strategy_ssaf import SSAFStrategy, StrategyConfig, SignalType
from backtest import SSAFBacktest

def generate_sample_data(days=252, start_price=100, volatility=0.02):
    """Generate sample price data for testing"""
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate price series with trend
    returns = np.random.normal(0.001, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Add some trend
    trend = np.linspace(0, 0.1, days)
    prices = prices * np.exp(trend)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    
    # Generate OHLC from close
    data['Open'] = prices * (1 + np.random.normal(0, 0.005, days))
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    data['Volume'] = np.random.randint(1000000, 10000000, days)
    
    return data

def test_basic_ssaf():
    """Test basic SSAF indicator"""
    print("Testing Basic SSAF Indicator...")
    
    # Generate sample data
    data = generate_sample_data(days=100)
    prices = data['Close']
    
    # Initialize SSAF
    ssaf = SSAFIndicator(window_size=20)
    
    # Test update
    results = []
    for i in range(20, len(prices)):
        result = ssaf.update(prices.iloc[:i+1])
        results.append(result)
    
    # Check results
    assert len(results) > 0, "No results generated"
    assert all('signal' in r for r in results), "Missing signal in results"
    assert all('strength' in r for r in results), "Missing strength in results"
    
    print(f"✓ Basic SSAF: Generated {len(results)} signals")
    return True

def test_multi_timeframe():
    """Test multi-timeframe SSAF"""
    print("Testing Multi-Timeframe SSAF...")
    
    # Generate sample data
    data = generate_sample_data(days=100)
    prices = data['Close']
    
    # Initialize multi-timeframe SSAF
    mtf_ssaf = SSAFMultiTimeframe(timeframes=[10, 20, 30])
    
    # Test update
    results = []
    for i in range(30, len(prices)):
        result = mtf_ssaf.update(prices.iloc[:i+1])
        results.append(result)
    
    # Check results
    assert len(results) > 0, "No results generated"
    assert all('combined' in r for r in results), "Missing combined signal"
    
    print(f"✓ Multi-Timeframe: Generated {len(results)} combined signals")
    return True

def test_regime_adaptive():
    """Test regime-adaptive SSAF"""
    print("Testing Regime-Adaptive SSAF...")
    
    # Generate sample data
    data = generate_sample_data(days=100)
    prices = data['Close']
    
    # Initialize regime-adaptive SSAF
    regime_ssaf = RegimeAdaptiveSSAF(window_size=20)
    
    # Test update
    results = []
    for i in range(50, len(prices)):
        result = regime_ssaf.update(prices.iloc[:i+1])
        results.append(result)
    
    # Check results
    assert len(results) > 0, "No results generated"
    assert all('regime' in r for r in results), "Missing regime information"
    
    # Check regime distribution
    regimes = [r['regime'] for r in results]
    unique_regimes = set(regimes)
    print(f"✓ Regime-Adaptive: Detected regimes: {unique_regimes}")
    
    return True

def test_advanced_ssaf():
    """Test advanced SSAF with ML"""
    print("Testing Advanced SSAF with ML...")
    
    # Generate sample data
    data = generate_sample_data(days=200)
    prices = data['Close']
    
    # Initialize advanced SSAF
    advanced_ssaf = AdvancedSSAFIndicator(window_size=20, use_ml=True)
    
    # Train ML model
    advanced_ssaf.train_ml_model(prices, lookback=100)
    
    # Test update
    results = []
    for i in range(100, len(prices)):
        result = advanced_ssaf.update(prices.iloc[:i+1])
        results.append(result)
    
    # Check results
    assert len(results) > 0, "No results generated"
    print(f"✓ Advanced SSAF: Generated {len(results)} signals with ML")
    
    return True

def test_strategy_integration():
    """Test strategy integration"""
    print("Testing Strategy Integration...")
    
    # Generate sample data
    data = generate_sample_data(days=100)
    
    # Configure strategy
    config = StrategyConfig(
        window_size=20,
        slope_threshold=0.1,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        use_macd_filter=True,
        use_stoch_filter=True,
        use_vwap_confluence=True
    )
    
    # Initialize strategy
    strategy = SSAFStrategy(config)
    
    # Run strategy
    for i in range(20, len(data)):
        strategy.update(data['Close'].iloc[:i+1], data.index[i])
    
    # Check results
    metrics = strategy.get_performance_metrics()
    assert len(metrics) > 0, "No metrics generated"
    
    print(f"✓ Strategy Integration: Generated {metrics.get('total_trades', 0)} trades")
    return True

def test_backtest_integration():
    """Test backtest integration"""
    print("Testing Backtest Integration...")
    
    # Generate sample data
    data = generate_sample_data(days=252)
    
    # Initialize backtest
    backtest = SSAFBacktest(initial_capital=100000)
    
    # Configure strategy
    config = StrategyConfig(
        window_size=20,
        slope_threshold=0.1,
        max_position_size=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.06,
        use_macd_filter=True,
        use_stoch_filter=True,
        use_vwap_confluence=True
    )
    
    # Run backtest
    results = backtest.run_backtest(data, config, verbose=False)
    
    # Check results
    assert 'metrics' in results, "Missing metrics in results"
    assert 'total_return' in results['metrics'], "Missing total return"
    
    print(f"✓ Backtest Integration: Total return: {results['metrics']['total_return']:.2f}%")
    return True

def test_multi_asset():
    """Test multi-asset SSAF"""
    print("Testing Multi-Asset SSAF...")
    
    # Generate sample data for multiple assets
    assets = ['AAPL', 'GOOGL', 'MSFT']
    data = {}
    
    for asset in assets:
        asset_data = generate_sample_data(days=100, start_price=np.random.uniform(50, 200))
        data[asset] = asset_data['Close']
    
    # Initialize multi-asset SSAF
    multi_asset = MultiAssetSSAF(assets=assets)
    
    # Test update
