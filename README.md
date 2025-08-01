# Spectral Slope Adaptive Filter (SSAF) Trend Strategy

A comprehensive implementation of the Spectral Slope Adaptive Filter strategy for trend detection and algorithmic trading.

## 📊 Overview

The SSAF strategy uses spectral analysis techniques to identify trend direction and strength in financial time series. It combines Fast Fourier Transform (FFT) analysis with adaptive filtering to generate trading signals with enhanced accuracy.

## 🚀 Features

- **Spectral Analysis**: FFT-based trend detection using frequency domain analysis
- **Adaptive Thresholding**: Dynamic parameter adjustment based on market volatility
- **Multi-Filter System**: MACD, Stochastic, and VWAP confluence filters
- **Risk Management**: Stop-loss, take-profit, and trailing stop mechanisms
- **Position Sizing**: Dynamic allocation based on signal strength
- **Multi-Timeframe Analysis**: Signal combination across different timeframes
- **Comprehensive Backtesting**: Full performance analysis and visualization
- **Interactive Dashboard**: Streamlit-based web interface for strategy exploration

## 📁 Project Structure

```
spectral_slope_filter_strategy/
├── strategy_ssaf.py               # Core SSAF strategy logic
├── indicators/ssaf.py             # Spectral Slope Adaptive Filter implementation
├── backtest.py                    # Full backtest runner with performance analysis
├── enhancements.md                # Enhancements: MACD/stoch filters, VWAP confluence
├── streamlit_dashboard.py        # Interactive visual explorer
├── utils/data_loader.py          # Load price data from yfinance/FMP
├── notebooks/
│   └── SSAF_Trend_Analysis.ipynb  # Visual notebook showing filter vs price
├── README.md
└── data/
    └── sample_prices.csv
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy
pip install yfinance streamlit plotly
pip install jupyter notebook
```

### Quick Start

1. **Clone the repository**:
```bash
git clone <repository-url>
cd spectral_slope_filter_strategy
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit dashboard**:
```bash
streamlit run streamlit_dashboard.py
```

## 📈 Usage

### Basic Strategy Usage

```python
from strategy_ssaf import SSAFStrategy, StrategyConfig
from utils.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.get_sample_data("AAPL", days=252)

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

# Initialize and run strategy
strategy = SSAFStrategy(config)
for i in range(20, len(data)):
    strategy.update(data['Close'].iloc[:i+1], data.index[i])

# Get performance metrics
metrics = strategy.get_performance_metrics()
print(metrics)
```

### Backtesting

```python
from backtest import SSAFBacktest

# Initialize backtest
backtest = SSAFBacktest(initial_capital=100000)

# Run backtest
results = backtest.run_backtest(data, config, verbose=True)

# Plot results
backtest.plot_results()

# Generate report
report = backtest.generate_report()
print(report)
```

### Multi-Timeframe Analysis

```python
from indicators.ssaf import SSAFMultiTimeframe

# Initialize multi-timeframe analysis
mtf_ssaf = SSAFMultiTimeframe(timeframes=[10, 20, 50])

# Update with data
results = mtf_ssaf.update(prices)

# Get combined signal
combined_signal = results['combined']['signal']
combined_strength = results['combined']['strength']
```

## 🔧 Configuration

### Strategy Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `window_size` | Analysis window size | 20 | 5-100 |
| `slope_threshold` | Signal threshold | 0.1 | 0.01-0.5 |
| `adaptive_threshold` | Enable adaptive thresholding | True | Boolean |
| `max_position_size` | Maximum position size | 0.1 | 0.01-0.5 |
| `stop_loss_pct` | Stop loss percentage | 0.02 | 0.01-0.1 |
| `take_profit_pct` | Take profit percentage | 0.06 | 0.01-0.2 |
| `trailing_stop` | Enable trailing stop | True | Boolean |
| `use_macd_filter` | Enable MACD filter | True | Boolean |
| `use_stoch_filter` | Enable Stochastic filter | True | Boolean |
| `use_vwap_confluence` | Enable VWAP confluence | True | Boolean |

### Filter Parameters

#### MACD Filter
- `macd_fast`: Fast EMA period (default: 12)
- `macd_slow`: Slow EMA period (default: 26)
- `macd_signal`: Signal line period (default: 9)

#### Stochastic Filter
- `stoch_k_period`: %K period (default: 14)
- `stoch_d_period`: %D period (default: 3)
- `stoch_overbought`: Overbought threshold (default: 80)
- `stoch_oversold`: Oversold threshold (default: 20)

## 📊 Performance Metrics

The strategy provides comprehensive performance analysis including:

- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Volatility**: Annualized standard deviation of returns
- **Trade Statistics**: Number of trades, average win/loss

## 🎯 Strategy Logic

### Spectral Analysis
1. **FFT Computation**: Apply Fast Fourier Transform to price window
2. **Frequency Filtering**: Focus on relevant frequency bands
3. **Slope Calculation**: Compute spectral slope using linear regression
4. **Signal Generation**: Convert slope to trading signals

### Adaptive Filtering
1. **Threshold Adaptation**: Adjust signal threshold based on volatility
2. **Noise Reduction**: Apply smoothing to reduce high-frequency noise
3. **Multi-Filter Confluence**: Combine signals from different indicators

### Risk Management
1. **Position Sizing**: Scale position size based on signal strength
2. **Stop Loss**: Automatic exit on adverse price movements
3. **Take Profit**: Lock in profits at predetermined levels
4. **Trailing Stop**: Dynamic stop loss adjustment

## 📈 Visualization

### Interactive Dashboard
The Streamlit dashboard provides:
- Real-time parameter configuration
- Interactive charts and metrics
- Performance analysis
- Data export capabilities

### Jupyter Notebook
The analysis notebook includes:
- Step-by-step strategy implementation
- Parameter optimization
- Multi-timeframe analysis
- Performance visualization

## 🔬 Advanced Features

### Multi-Timeframe Analysis
Combines signals from different timeframes to improve signal quality and reduce noise.

### Adaptive Thresholding
Dynamically adjusts signal thresholds based on market volatility conditions.

### Noise Reduction
Applies spectral smoothing to filter out high-frequency market noise.

### Enhanced Filters
- **MACD Filter**: Confirms trend direction using moving average convergence
- **Stochastic Filter**: Identifies overbought/oversold conditions
- **VWAP Confluence**: Provides additional price level confirmation

## 📋 Examples

### Basic Example
```python
# Simple SSAF strategy
config = StrategyConfig(window_size=20, slope_threshold=0.1)
strategy = SSAFStrategy(config)
```

### Enhanced Example
```python
# Full-featured SSAF strategy
config = StrategyConfig(
    window_size=20,
    slope_threshold=0.1,
    adaptive_threshold=True,
    noise_reduction=True,
    max_position_size=0.1,
    stop_loss_pct=0.02,
    take_profit_pct=0.06,
    trailing_stop=True,
    use_macd_filter=True,
    use_stoch_filter=True,
    use_vwap_confluence=True
)
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Backtesting
```bash
python backtest.py
```

### Dashboard
```bash
streamlit run streamlit_dashboard.py
```

## 📚 Documentation

- **Strategy Logic**: See `strategy_ssaf.py` for core implementation
- **Indicator Details**: See `indicators/ssaf.py` for spectral analysis
- **Enhancements**: See `enhancements.md` for advanced features
- **Notebook**: See `notebooks/SSAF_Trend_Analysis.ipynb` for examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended for actual trading. Always perform thorough testing and risk management before using any trading strategy.

## 🆘 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `enhancements.md`
- Review the Jupyter notebook examples

## 🔄 Version History

- **v1.0.0**: Initial release with basic SSAF implementation
- **v1.1.0**: Added multi-filter system and risk management
- **v1.2.0**: Enhanced with Streamlit dashboard and backtesting
- **v1.3.0**: Added multi-timeframe analysis and optimization

## 📞 Contact

For questions or collaboration:
- GitHub Issues: [Project Issues](https://github.com/your-repo/issues)
- Email: rick84etter@gmail.com or et1209534@email.ccbcmd.edu

---

**Happy Trading! 📈** 
