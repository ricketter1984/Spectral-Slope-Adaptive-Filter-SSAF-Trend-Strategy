"""
SSAF Strategy Backtest Runner

This module provides a comprehensive backtesting framework for the Spectral Slope
Adaptive Filter strategy with performance analysis and visualization capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import strategy components
from strategy_ssaf import SSAFStrategy, StrategyConfig
from indicators.ssaf import SSAFIndicator


class SSAFBacktest:
    """
    Comprehensive backtesting framework for SSAF strategy
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Initialize backtest framework
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.positions = []
        self.metrics = {}
        
    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load market data from yfinance
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Ensure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            print(f"Loaded {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    config: StrategyConfig,
                    verbose: bool = True) -> Dict:
        """
        Run backtest with given configuration
        
        Args:
            data: Market data DataFrame
            config: Strategy configuration
            verbose: Whether to print progress
            
        Returns:
            Dictionary with backtest results
        """
        if data.empty:
            return {"error": "No data provided"}
        
        # Initialize strategy
        strategy = SSAFStrategy(config)
        strategy.portfolio_value = self.initial_capital
        strategy.current_cash = self.initial_capital
        
        # Get price series
        prices = data['Close']
        
        # Track performance
        equity_curve = []
        trades_summary = []
        
        if verbose:
            print(f"Running backtest on {len(prices)} data points...")
        
        # Run strategy
        for i in range(config.window_size, len(prices)):
            current_prices = prices.iloc[:i+1]
            current_time = prices.index[i]
            
            # Update strategy
            strategy.update(current_prices, current_time)
            
            # Record equity
            total_value = strategy.current_cash
            for position in strategy.positions:
                if position.signal_type.value == 'LONG':
                    total_value += position.size * prices.iloc[i]
                else:  # SHORT
                    total_value += position.size * (2 * position.entry_price - prices.iloc[i])
            
            equity_curve.append({
                'timestamp': current_time,
                'portfolio_value': total_value,
                'cash': strategy.current_cash,
                'positions': len(strategy.positions),
                'price': prices.iloc[i]
            })
            
            if verbose and i % 100 == 0:
                print(f"Progress: {i}/{len(prices)} ({i/len(prices)*100:.1f}%)")
        
        # Close any remaining positions
        final_price = prices.iloc[-1]
        final_time = prices.index[-1]
        
        for position in strategy.positions[:]:
            strategy._close_position(position, final_price, final_time)
        
        # Calculate performance metrics
        self.equity_curve = equity_curve
        self.trades = strategy.trades_history
        self.metrics = self._calculate_performance_metrics(equity_curve, strategy.trades_history)
        
        return {
            'equity_curve': equity_curve,
            'trades': strategy.trades_history,
            'metrics': self.metrics,
            'final_portfolio_value': strategy.portfolio_value,
            'total_return': ((strategy.portfolio_value - self.initial_capital) / self.initial_capital) * 100
        }
    
    def _calculate_performance_metrics(self, equity_curve: List[Dict], trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: List of equity curve data points
            trades: List of trade records
            
        Returns:
            Dictionary of performance metrics
        """
        if not equity_curve:
            return {}
        
        # Convert to DataFrame for easier analysis
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Basic metrics
        initial_value = equity_df['portfolio_value'].iloc[0]
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()
        
        # Volatility
        volatility = equity_df['returns'].std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = equity_df['returns'] - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        
        # Maximum drawdown
        equity_df['peak'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        # Trade metrics
        trade_metrics = {}
        if not trades_df.empty:
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            trade_metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_pnl': trades_df['pnl'].sum(),
                'avg_trade_return': trades_df['return_pct'].mean()
            }
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'initial_value': initial_value,
            **trade_metrics
        }
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot comprehensive backtest results
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.equity_curve:
            print("No backtest results to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(equity_df['timestamp'], equity_df['portfolio_value'], 
                label='Portfolio Value', color='blue', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2 = axes[0, 1]
        equity_df['peak'] = equity_df['portfolio_value'].cummax()
        equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak'] * 100
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(equity_df['timestamp'], equity_df['drawdown'], color='red', linewidth=1)
        ax2.set_title('Portfolio Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trade P&L
        ax3 = axes[1, 0]
        if not trades_df.empty:
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
            ax3.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.set_title('Trade P&L Distribution')
            ax3.set_xlabel('Trade Number')
            ax3.set_ylabel('P&L ($)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Trade P&L Distribution')
        
        # Plot 4: Performance Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create metrics text
        metrics_text = "Performance Metrics:\n\n"
        for key, value in self.metrics.items():
            if isinstance(value, float):
                if 'rate' in key or 'ratio' in key:
                    metrics_text += f"{key.replace('_', ' ').title()}: {value:.2%}\n"
                elif 'return' in key or 'drawdown' in key:
                    metrics_text += f"{key.replace('_', ' ').title()}: {value:.2f}%\n"
                else:
                    metrics_text += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            else:
                metrics_text += f"{key.replace('_', ' ').title()}: {value}\n"
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive backtest report
        
        Returns:
            Formatted report string
        """
        if not self.metrics:
            return "No backtest results available"
        
        report = []
        report.append("=" * 60)
        report.append("SSAF STRATEGY BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 30)
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"Final Portfolio Value: ${self.metrics.get('final_value', 0):,.2f}")
        report.append(f"Total Return: {self.metrics.get('total_return', 0):.2f}%")
        report.append(f"Volatility: {self.metrics.get('volatility', 0):.2f}%")
        report.append(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Maximum Drawdown: {self.metrics.get('max_drawdown', 0):.2f}%")
        report.append("")
        
        # Trade statistics
        if self.metrics.get('total_trades', 0) > 0:
            report.append("TRADE STATISTICS:")
            report.append("-" * 30)
            report.append(f"Total Trades: {self.metrics.get('total_trades', 0)}")
            report.append(f"Winning Trades: {self.metrics.get('winning_trades', 0)}")
            report.append(f"Losing Trades: {self.metrics.get('losing_trades', 0)}")
            report.append(f"Win Rate: {self.metrics.get('win_rate', 0):.2%}")
            report.append(f"Average Win: ${self.metrics.get('avg_win', 0):.2f}")
            report.append(f"Average Loss: ${self.metrics.get('avg_loss', 0):.2f}")
            report.append(f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
            report.append(f"Total P&L: ${self.metrics.get('total_pnl', 0):.2f}")
            report.append("")
        
        # Risk metrics
        report.append("RISK METRICS:")
        report.append("-" * 30)
        report.append(f"Volatility (Annualized): {self.metrics.get('volatility', 0):.2f}%")
        report.append(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Maximum Drawdown: {self.metrics.get('max_drawdown', 0):.2f}%")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def run_sample_backtest():
    """Run a sample backtest with AAPL data"""
    
    # Initialize backtest
    backtest = SSAFBacktest(initial_capital=100000)
    
    # Load data
    data = backtest.load_data("AAPL", "2022-01-01", "2023-12-31")
    
    if data.empty:
        print("Failed to load data")
        return
    
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
    results = backtest.run_backtest(data, config, verbose=True)
    
    # Print results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            if 'rate' in key or 'ratio' in key:
                print(f"{key}: {value:.2%}")
            elif 'return' in key or 'drawdown' in key:
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Generate report
    report = backtest.generate_report()
    print("\n" + report)
    
    # Plot results
    backtest.plot_results()
    
    return results


if __name__ == "__main__":
    # Run sample backtest
    results = run_sample_backtest() 