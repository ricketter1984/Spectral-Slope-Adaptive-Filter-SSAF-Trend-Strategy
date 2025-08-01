"""
SSAF Strategy Streamlit Dashboard

Interactive dashboard for exploring and backtesting the Spectral Slope Adaptive Filter strategy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import strategy components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_ssaf import SSAFStrategy, StrategyConfig
from indicators.ssaf import SSAFIndicator
from utils.data_loader import DataLoader
from backtest import SSAFBacktest

# Configure page
st.set_page_config(
    page_title="SSAF Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 SSAF Strategy Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Spectral Slope Adaptive Filter Trend Strategy**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Data selection
        st.subheader("Data Source")
        data_source = st.selectbox(
            "Select data source",
            ["Yahoo Finance", "CSV Upload", "Sample Data"]
        )
        
        if data_source == "Yahoo Finance":
            symbol = st.text_input("Symbol", value="AAPL")
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", value=datetime.now())
        elif data_source == "CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        else:
            symbol = st.text_input("Symbol", value="AAPL")
            days = st.slider("Days", min_value=30, max_value=500, value=252)
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider("Window Size", min_value=5, max_value=100, value=20)
            slope_threshold = st.slider("Slope Threshold", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
            max_position_size = st.slider("Max Position Size", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        with col2:
            stop_loss_pct = st.slider("Stop Loss %", min_value=0.01, max_value=0.1, value=0.02, step=0.01)
            take_profit_pct = st.slider("Take Profit %", min_value=0.01, max_value=0.2, value=0.06, step=0.01)
            initial_capital = st.number_input("Initial Capital", min_value=1000, max_value=1000000, value=100000, step=10000)
        
        # Filters
        st.subheader("Filters")
        use_macd_filter = st.checkbox("MACD Filter", value=True)
        use_stoch_filter = st.checkbox("Stochastic Filter", value=True)
        use_vwap_confluence = st.checkbox("VWAP Confluence", value=True)
        
        # Advanced options
        with st.expander("Advanced Options"):
            adaptive_threshold = st.checkbox("Adaptive Threshold", value=True)
            noise_reduction = st.checkbox("Noise Reduction", value=True)
            trailing_stop = st.checkbox("Trailing Stop", value=True)
            
            if use_macd_filter:
                st.write("MACD Parameters")
                macd_fast = st.slider("MACD Fast", min_value=5, max_value=20, value=12)
                macd_slow = st.slider("MACD Slow", min_value=20, max_value=50, value=26)
                macd_signal = st.slider("MACD Signal", min_value=5, max_value=20, value=9)
            
            if use_stoch_filter:
                st.write("Stochastic Parameters")
                stoch_k_period = st.slider("Stoch K Period", min_value=5, max_value=30, value=14)
                stoch_d_period = st.slider("Stoch D Period", min_value=1, max_value=10, value=3)
                stoch_overbought = st.slider("Stoch Overbought", min_value=70, max_value=90, value=80)
                stoch_oversold = st.slider("Stoch Oversold", min_value=10, max_value=30, value=20)
        
        # Run button
        run_backtest = st.button("🚀 Run Backtest", type="primary")
    
    # Main content
    if run_backtest:
        with st.spinner("Loading data and running backtest..."):
            
            # Load data
            loader = DataLoader()
            
            if data_source == "Yahoo Finance":
                data = loader.load_from_yfinance(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
            elif data_source == "CSV Upload":
                if uploaded_file is not None:
                    data = loader.load_from_csv(uploaded_file)
                else:
                    st.error("Please upload a CSV file")
                    return
            else:
                data = loader.get_sample_data(symbol=symbol, days=days)
            
            if data.empty:
                st.error("Failed to load data")
                return
            
            # Configure strategy
            config_params = {
                'window_size': window_size,
                'slope_threshold': slope_threshold,
                'adaptive_threshold': adaptive_threshold,
                'max_position_size': max_position_size,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'trailing_stop': trailing_stop,
                'use_macd_filter': use_macd_filter,
                'use_stoch_filter': use_stoch_filter,
                'use_vwap_confluence': use_vwap_confluence,
                'noise_reduction': noise_reduction
            }
            
            if use_macd_filter:
                config_params.update({
                    'macd_fast': macd_fast,
                    'macd_slow': macd_slow,
                    'macd_signal': macd_signal
                })
            
            if use_stoch_filter:
                config_params.update({
                    'stoch_k_period': stoch_k_period,
                    'stoch_d_period': stoch_d_period,
                    'stoch_overbought': stoch_overbought,
                    'stoch_oversold': stoch_oversold
                })
            
            config = StrategyConfig(**config_params)
            
            # Run backtest
            backtest = SSAFBacktest(initial_capital=initial_capital)
            results = backtest.run_backtest(data, config, verbose=False)
            
            if 'error' in results:
                st.error(f"Backtest error: {results['error']}")
                return
            
            # Display results
            display_results(results, data, config)
    
    else:
        # Show welcome message
        st.markdown("""
        ## Welcome to the SSAF Strategy Dashboard!
        
        This interactive dashboard allows you to:
        
        - **Load market data** from Yahoo Finance or upload CSV files
        - **Configure strategy parameters** including filters and risk management
        - **Run backtests** with real-time performance analysis
        - **Visualize results** with interactive charts and metrics
        
        ### Getting Started
        
        1. Configure your data source in the sidebar
        2. Adjust strategy parameters as needed
        3. Click "Run Backtest" to execute the strategy
        4. Analyze the results and performance metrics
        
        ### Strategy Overview
        
        The **Spectral Slope Adaptive Filter (SSAF)** strategy uses spectral analysis to identify trend direction and strength. Key features include:
        
        - **Spectral Analysis**: FFT-based trend detection
        - **Adaptive Thresholding**: Dynamic parameter adjustment
        - **Multiple Filters**: MACD, Stochastic, and VWAP confluence
        - **Risk Management**: Stop-loss, take-profit, and trailing stops
        - **Position Sizing**: Dynamic allocation based on signal strength
        """)

def display_results(results, data, config):
    """Display backtest results"""
    
    # Header
    st.header("📊 Backtest Results")
    
    # Key metrics
    metrics = results['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
    
    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        performance_data = {
            'Metric': ['Total Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
            'Value': [
                f"{metrics.get('total_return', 0):.2f}%",
                f"{metrics.get('volatility', 0):.2f}%",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('max_drawdown', 0):.2f}%"
            ]
        }
        st.dataframe(pd.DataFrame(performance_data))
    
    with col2:
        st.subheader("Trade Statistics")
        if metrics.get('total_trades', 0) > 0:
            trade_data = {
                'Metric': ['Total Trades', 'Winning Trades', 'Losing Trades', 'Profit Factor'],
                'Value': [
                    metrics.get('total_trades', 0),
                    metrics.get('winning_trades', 0),
                    metrics.get('losing_trades', 0),
                    f"{metrics.get('profit_factor', 0):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(trade_data))
        else:
            st.write("No trades executed")
    
    # Charts
    st.subheader("📈 Performance Charts")
    
    # Create equity curve plot
    equity_df = pd.DataFrame(results['equity_curve'])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Equity Curve', 'Drawdown', 'Trade P&L', 'Price vs Signals'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Drawdown
    equity_df['peak'] = equity_df['portfolio_value'].cummax()
    equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak'] * 100
    
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tonexty'
        ),
        row=1, col=2
    )
    
    # Trade P&L
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(trades_df))),
                y=trades_df['pnl'],
                name='Trade P&L',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    # Price vs Signals
    prices = data['Close']
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="SSAF Strategy Performance Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy analysis
    st.subheader("🔍 Strategy Analysis")
    
    # Run SSAF indicator analysis
    ssaf = SSAFIndicator(window_size=config.window_size)
    
    # Update indicator with data
    for i in range(config.window_size, len(data)):
        ssaf.update(data['Close'].iloc[:i+1])
    
    # Get trend analysis
    trend_analysis = ssaf.get_trend_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Trend Analysis**")
        st.write(f"Dominant Trend: {trend_analysis.get('dominant_trend', 'N/A')}")
        st.write(f"Trend Consistency: {trend_analysis.get('trend_consistency', 0):.2%}")
        st.write(f"Average Signal Strength: {trend_analysis.get('avg_strength', 0):.3f}")
    
    with col2:
        st.write("**Signal Distribution**")
        signal_counts = {
            'Long': trend_analysis.get('long_signals', 0),
            'Short': trend_analysis.get('short_signals', 0),
            'Hold': trend_analysis.get('hold_signals', 0)
        }
        
        fig = px.pie(
            values=list(signal_counts.values()),
            names=list(signal_counts.keys()),
            title="Signal Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Configuration summary
    st.subheader("⚙️ Configuration Summary")
    
    config_summary = {
        'Parameter': [
            'Window Size', 'Slope Threshold', 'Max Position Size',
            'Stop Loss %', 'Take Profit %', 'MACD Filter',
            'Stochastic Filter', 'VWAP Confluence', 'Adaptive Threshold'
        ],
        'Value': [
            config.window_size, f"{config.slope_threshold:.3f}",
            f"{config.max_position_size:.1%}", f"{config.stop_loss_pct:.1%}",
            f"{config.take_profit_pct:.1%}", "Yes" if config.use_macd_filter else "No",
            "Yes" if config.use_stoch_filter else "No",
            "Yes" if config.use_vwap_confluence else "No",
            "Yes" if config.adaptive_threshold else "No"
        ]
    }
    
    st.dataframe(pd.DataFrame(config_summary))
    
    # Download results
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export equity curve
        equity_csv = equity_df.to_csv(index=False)
        st.download_button(
            label="Download Equity Curve",
            data=equity_csv,
            file_name="ssaf_equity_curve.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export trades
        if results['trades']:
            trades_csv = pd.DataFrame(results['trades']).to_csv(index=False)
            st.download_button(
                label="Download Trades",
                data=trades_csv,
                file_name="ssaf_trades.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 