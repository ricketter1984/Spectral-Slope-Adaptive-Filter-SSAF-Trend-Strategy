"""
Spectral Slope Adaptive Filter (SSAF) Trend Strategy

This module implements a trend-following strategy using the Spectral Slope Adaptive Filter
to identify and trade directional market movements with adaptive position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for the SSAF strategy"""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Position:
    """Represents a trading position"""
    signal_type: SignalType
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class StrategyConfig:
    """Configuration for the SSAF strategy"""
    # SSAF parameters
    window_size: int = 20
    slope_threshold: float = 0.1
    adaptive_threshold: bool = True
    
    # Risk management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_pct: float = 0.02     # 2% stop loss
    take_profit_pct: float = 0.06   # 6% take profit
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.01  # 1% trailing stop
    
    # Filter parameters
    use_macd_filter: bool = True
    use_stoch_filter: bool = True
    use_vwap_confluence: bool = True
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Stochastic parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80
    stoch_oversold: float = 20


class SSAFStrategy:
    """
    Spectral Slope Adaptive Filter Trend Strategy
    
    This strategy uses the SSAF indicator to identify trend direction and strength,
    with additional filters for signal quality and risk management.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.positions: List[Position] = []
        self.portfolio_value = 100000  # Starting portfolio value
        self.current_cash = self.portfolio_value
        
        # Performance tracking
        self.trades_history = []
        self.equity_curve = []
        
        logger.info(f"SSAF Strategy initialized with config: {config}")
    
    def calculate_ssaf_signal(self, prices: pd.Series) -> Tuple[SignalType, float]:
        """
        Calculate SSAF signal based on spectral slope analysis
        
        Args:
            prices: Price series for analysis
            
        Returns:
            Tuple of (signal_type, signal_strength)
        """
        if len(prices) < self.config.window_size:
            return SignalType.HOLD, 0.0
        
        # Calculate spectral slope (simplified implementation)
        # In practice, this would use FFT or other spectral analysis
        recent_prices = prices.tail(self.config.window_size)
        
        # Calculate linear regression slope
        x = np.arange(len(recent_prices))
        y = recent_prices.values
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Normalize slope by price level
        normalized_slope = slope / recent_prices.iloc[-1]
        
        # Determine signal based on slope threshold
        if normalized_slope > self.config.slope_threshold:
            signal_strength = min(abs(normalized_slope), 1.0)
            return SignalType.LONG, signal_strength
        elif normalized_slope < -self.config.slope_threshold:
            signal_strength = min(abs(normalized_slope), 1.0)
            return SignalType.SHORT, signal_strength
        else:
            return SignalType.HOLD, 0.0
    
    def apply_filters(self, prices: pd.Series, signal: SignalType, signal_strength: float) -> Tuple[SignalType, float]:
        """
        Apply additional filters to improve signal quality
        
        Args:
            prices: Price series
            signal: Original signal
            signal_strength: Original signal strength
            
        Returns:
            Tuple of (filtered_signal, adjusted_strength)
        """
        if signal == SignalType.HOLD:
            return signal, signal_strength
        
        filtered_signal = signal
        adjusted_strength = signal_strength
        
        # MACD Filter
        if self.config.use_macd_filter:
            macd_signal = self._calculate_macd_filter(prices)
            if macd_signal != signal and macd_signal != SignalType.HOLD:
                filtered_signal = SignalType.HOLD
                adjusted_strength *= 0.5
        
        # Stochastic Filter
        if self.config.use_stoch_filter:
            stoch_signal = self._calculate_stoch_filter(prices)
            if stoch_signal != signal and stoch_signal != SignalType.HOLD:
                filtered_signal = SignalType.HOLD
                adjusted_strength *= 0.5
        
        # VWAP Confluence
        if self.config.use_vwap_confluence:
            vwap_signal = self._calculate_vwap_confluence(prices)
            if vwap_signal == signal:
                adjusted_strength *= 1.2  # Boost strength if VWAP confirms
            elif vwap_signal != SignalType.HOLD:
                adjusted_strength *= 0.8  # Reduce strength if VWAP disagrees
        
        return filtered_signal, adjusted_strength
    
    def _calculate_macd_filter(self, prices: pd.Series) -> SignalType:
        """Calculate MACD filter signal"""
        if len(prices) < self.config.macd_slow:
            return SignalType.HOLD
        
        # Calculate EMA
        ema_fast = prices.ewm(span=self.config.macd_fast).mean()
        ema_slow = prices.ewm(span=self.config.macd_slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.config.macd_signal).mean()
        
        # Generate signal
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            return SignalType.LONG
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            return SignalType.SHORT
        
        return SignalType.HOLD
    
    def _calculate_stoch_filter(self, prices: pd.Series) -> SignalType:
        """Calculate stochastic filter signal"""
        if len(prices) < self.config.stoch_k_period:
            return SignalType.HOLD
        
        # Calculate %K
        low_min = prices.rolling(window=self.config.stoch_k_period).min()
        high_max = prices.rolling(window=self.config.stoch_k_period).max()
        
        k_percent = 100 * ((prices - low_min) / (high_max - low_min))
        
        # Calculate %D (SMA of %K)
        d_percent = k_percent.rolling(window=self.config.stoch_d_period).mean()
        
        # Generate signal
        if (k_percent.iloc[-1] < self.config.stoch_oversold and 
            d_percent.iloc[-1] < self.config.stoch_oversold):
            return SignalType.LONG
        elif (k_percent.iloc[-1] > self.config.stoch_overbought and 
              d_percent.iloc[-1] > self.config.stoch_overbought):
            return SignalType.SHORT
        
        return SignalType.HOLD
    
    def _calculate_vwap_confluence(self, prices: pd.Series) -> SignalType:
        """Calculate VWAP confluence signal"""
        if len(prices) < 20:
            return SignalType.HOLD
        
        # Simplified VWAP calculation (using typical price)
        typical_price = prices  # In practice, this would be (H+L+C)/3
        vwap = typical_price.rolling(window=20).mean()
        
        current_price = prices.iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        if current_price > current_vwap:
            return SignalType.LONG
        elif current_price < current_vwap:
            return SignalType.SHORT
        
        return SignalType.HOLD
    
    def calculate_position_size(self, signal_strength: float, current_price: float) -> float:
        """
        Calculate position size based on signal strength and risk management
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            
        Returns:
            Position size in units
        """
        # Base position size as percentage of portfolio
        base_size_pct = self.config.max_position_size * signal_strength
        
        # Calculate position size in units
        position_value = self.portfolio_value * base_size_pct
        position_size = position_value / current_price
        
        return position_size
    
    def update_stop_loss(self, position: Position, current_price: float):
        """Update trailing stop loss for position"""
        if not self.config.trailing_stop:
            return
        
        if position.signal_type == SignalType.LONG:
            new_stop = current_price * (1 - self.config.trailing_stop_pct)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
        elif position.signal_type == SignalType.SHORT:
            new_stop = current_price * (1 + self.config.trailing_stop_pct)
            if new_stop < position.stop_loss or position.stop_loss is None:
                position.stop_loss = new_stop
    
    def check_exit_conditions(self, position: Position, current_price: float) -> bool:
        """
        Check if position should be closed based on stop loss or take profit
        
        Args:
            position: Current position
            current_price: Current asset price
            
        Returns:
            True if position should be closed
        """
        if position.signal_type == SignalType.LONG:
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                return True
            # Check take profit
            if position.take_profit and current_price >= position.take_profit:
                return True
        elif position.signal_type == SignalType.SHORT:
            # Check stop loss
            if position.stop_loss and current_price >= position.stop_loss:
                return True
            # Check take profit
            if position.take_profit and current_price <= position.take_profit:
                return True
        
        return False
    
    def execute_trade(self, signal: SignalType, position_size: float, 
                     current_price: float, timestamp: pd.Timestamp) -> Optional[Position]:
        """
        Execute a trade based on signal
        
        Args:
            signal: Trading signal
            position_size: Size of position to take
            current_price: Current asset price
            timestamp: Trade timestamp
            
        Returns:
            New position if created, None otherwise
        """
        if signal == SignalType.HOLD:
            return None
        
        # Close existing positions if signal is opposite
        for position in self.positions[:]:  # Copy list to avoid modification during iteration
            if ((signal == SignalType.LONG and position.signal_type == SignalType.SHORT) or
                (signal == SignalType.SHORT and position.signal_type == SignalType.LONG)):
                self._close_position(position, current_price, timestamp)
        
        # Create new position
        if signal in [SignalType.LONG, SignalType.SHORT]:
            stop_loss = None
            take_profit = None
            
            if signal == SignalType.LONG:
                stop_loss = current_price * (1 - self.config.stop_loss_pct)
                take_profit = current_price * (1 + self.config.take_profit_pct)
            else:  # SHORT
                stop_loss = current_price * (1 + self.config.stop_loss_pct)
                take_profit = current_price * (1 - self.config.take_profit_pct)
            
            position = Position(
                signal_type=signal,
                entry_price=current_price,
                entry_time=timestamp,
                size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            self.positions.append(position)
            logger.info(f"Opened {signal.value} position: {position_size:.2f} units at {current_price:.2f}")
            
            return position
        
        return None
    
    def _close_position(self, position: Position, current_price: float, timestamp: pd.Timestamp):
        """Close a position and record the trade"""
        # Calculate P&L
        if position.signal_type == SignalType.LONG:
            pnl = (current_price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - current_price) * position.size
        
        # Update portfolio
        self.current_cash += pnl
        self.portfolio_value = self.current_cash
        
        # Record trade
        trade = {
            'entry_time': position.entry_time,
            'exit_time': timestamp,
            'signal_type': position.signal_type.value,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'size': position.size,
            'pnl': pnl,
            'return_pct': (pnl / (position.entry_price * position.size)) * 100
        }
        self.trades_history.append(trade)
        
        # Remove position
        self.positions.remove(position)
        
        logger.info(f"Closed {position.signal_type.value} position: P&L = {pnl:.2f}")
    
    def update(self, prices: pd.Series, timestamp: pd.Timestamp):
        """
        Update strategy with new price data
        
        Args:
            prices: Historical price series
            timestamp: Current timestamp
        """
        if len(prices) < self.config.window_size:
            return
        
        current_price = prices.iloc[-1]
        
        # Update existing positions
        for position in self.positions[:]:
            # Update trailing stop
            self.update_stop_loss(position, current_price)
            
            # Check exit conditions
            if self.check_exit_conditions(position, current_price):
                self._close_position(position, current_price, timestamp)
        
        # Calculate new signal
        signal, signal_strength = self.calculate_ssaf_signal(prices)
        
        # Apply filters
        filtered_signal, adjusted_strength = self.apply_filters(prices, signal, signal_strength)
        
        # Execute trade if signal is valid
        if filtered_signal != SignalType.HOLD:
            position_size = self.calculate_position_size(adjusted_strength, current_price)
            self.execute_trade(filtered_signal, position_size, current_price, timestamp)
        
        # Update equity curve
        total_value = self.current_cash
        for position in self.positions:
            if position.signal_type == SignalType.LONG:
                total_value += position.size * current_price
            else:  # SHORT
                total_value += position.size * (2 * position.entry_price - current_price)
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': total_value,
            'cash': self.current_cash,
            'positions': len(self.positions)
        })
    
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        if not self.trades_history:
            return {}
        
        trades_df = pd.DataFrame(self.trades_history)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Calculate drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df['peak'] = equity_df['portfolio_value'].cummax()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'max_drawdown': max_drawdown,
            'final_portfolio_value': self.portfolio_value,
            'total_return': ((self.portfolio_value - 100000) / 100000) * 100
        }


if __name__ == "__main__":
    # Example usage
    config = StrategyConfig()
    strategy = SSAFStrategy(config)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    # Run strategy
    for i in range(20, len(prices)):
        strategy.update(prices.iloc[:i+1], prices.index[i])
    
    # Print results
    metrics = strategy.get_performance_metrics()
    print("Strategy Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}") 