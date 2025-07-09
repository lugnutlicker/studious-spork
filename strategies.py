"""
Trading strategies for automated Solana trading
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger

from config import config, STRATEGY_CONFIGS
from indicators import TechnicalIndicators


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Position(str, Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class Signal:
    """Trading signal class"""
    
    def __init__(
        self,
        signal_type: SignalType,
        strength: float,
        price: float,
        timestamp: pd.Timestamp,
        reason: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.signal_type = signal_type
        self.strength = strength  # 0-1 confidence level
        self.price = price
        self.timestamp = timestamp
        self.reason = reason
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def __repr__(self):
        return f"Signal({self.signal_type}, {self.strength:.2f}, {self.price:.6f}, {self.reason})"


class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.position = Position.NONE
        self.entry_price = None
        self.entry_time = None
        self.signals_history = []
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal based on data"""
        pass
    
    def should_exit(self, df: pd.DataFrame, current_price: float) -> bool:
        """Check if should exit current position"""
        if self.position == Position.NONE or self.entry_price is None:
            return False
        
        # Stop loss check
        if self.position == Position.LONG:
            stop_loss_price = self.entry_price * (1 - config.STOP_LOSS_PCT)
            take_profit_price = self.entry_price * (1 + config.TAKE_PROFIT_PCT)
            
            if current_price <= stop_loss_price:
                logger.info(f"Stop loss triggered: {current_price} <= {stop_loss_price}")
                return True
            
            if current_price >= take_profit_price:
                logger.info(f"Take profit triggered: {current_price} >= {take_profit_price}")
                return True
        
        return False
    
    def update_position(self, signal: Signal):
        """Update position based on signal"""
        if signal.signal_type == SignalType.BUY:
            self.position = Position.LONG
            self.entry_price = signal.price
            self.entry_time = signal.timestamp
        elif signal.signal_type == SignalType.SELL:
            self.position = Position.NONE
            self.entry_price = None
            self.entry_time = None
        
        self.signals_history.append(signal)


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using RSI and Bollinger Bands"""
    
    def __init__(self, config: Dict = None):
        super().__init__("Mean Reversion", config or STRATEGY_CONFIGS["mean_reversion"])
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate indicators if not present
        if 'rsi' not in df.columns:
            df = TechnicalIndicators.calculate_all_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
        
        signal_strength = 0.0
        signal_type = SignalType.HOLD
        reason = ""
        
        # RSI oversold condition
        if latest['rsi'] < self.config['rsi_oversold']:
            signal_strength += 0.3
            reason += "RSI oversold, "
        
        # RSI overbought condition 
        if latest['rsi'] > self.config['rsi_overbought']:
            signal_strength += 0.3
            signal_type = SignalType.SELL
            reason += "RSI overbought, "
        
        # Bollinger Bands oversold
        if latest['bb_oversold']:
            signal_strength += 0.3
            reason += "BB oversold, "
        
        # Bollinger Bands overbought
        if latest['bb_overbought']:
            signal_strength += 0.3
            signal_type = SignalType.SELL
            reason += "BB overbought, "
        
        # Price near Bollinger Band lower
        if latest['bb_position'] < 0.1:
            signal_strength += 0.2
            reason += "Price near BB lower, "
        
        # Price near Bollinger Band upper
        if latest['bb_position'] > 0.9:
            signal_strength += 0.2
            signal_type = SignalType.SELL
            reason += "Price near BB upper, "
        
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            signal_strength += 0.1
            reason += "High volume, "
        
        # Determine final signal
        if signal_strength >= 0.6:
            if signal_type != SignalType.SELL:
                signal_type = SignalType.BUY
        elif signal_strength >= 0.4 and signal_type == SignalType.SELL:
            pass  # Keep sell signal
        else:
            signal_type = SignalType.HOLD
            signal_strength = 0.0
        
        if signal_type != SignalType.HOLD:
            # Calculate stop loss and take profit
            if signal_type == SignalType.BUY:
                stop_loss = latest['close'] * (1 - config.STOP_LOSS_PCT)
                take_profit = latest['close'] * (1 + config.TAKE_PROFIT_PCT)
            else:
                stop_loss = None
                take_profit = None
            
            return Signal(
                signal_type=signal_type,
                strength=signal_strength,
                price=latest['close'],
                timestamp=latest.name,
                reason=reason.rstrip(", "),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        return None


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using EMA crossovers and MACD"""
    
    def __init__(self, config: Dict = None):
        super().__init__("Trend Following", config or STRATEGY_CONFIGS["trend_following"])
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate indicators if not present
        if 'ema_9' not in df.columns:
            df = TechnicalIndicators.calculate_all_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
        
        signal_strength = 0.0
        signal_type = SignalType.HOLD
        reason = ""
        
        # EMA crossover
        if latest['ema_9'] > latest['ema_21'] and prev['ema_9'] <= prev['ema_21']:
            signal_strength += 0.4
            reason += "EMA bullish crossover, "
        
        if latest['ema_9'] < latest['ema_21'] and prev['ema_9'] >= prev['ema_21']:
            signal_strength += 0.4
            signal_type = SignalType.SELL
            reason += "EMA bearish crossover, "
        
        # MACD confirmation
        if latest['macd_bullish']:
            signal_strength += 0.3
            reason += "MACD bullish, "
        
        if latest['macd_bearish']:
            signal_strength += 0.3
            signal_type = SignalType.SELL
            reason += "MACD bearish, "
        
        # Trend strength
        if latest['trend_strength'] > self.config['trend_strength_threshold']:
            signal_strength += 0.2
            reason += "Strong trend, "
        
        # Price above/below EMAs
        if latest['close'] > latest['ema_9'] > latest['ema_21']:
            signal_strength += 0.2
            reason += "Price above EMAs, "
        
        if latest['close'] < latest['ema_9'] < latest['ema_21']:
            signal_strength += 0.2
            signal_type = SignalType.SELL
            reason += "Price below EMAs, "
        
        # Volume confirmation
        if latest['volume_ratio'] > 1.3:
            signal_strength += 0.1
            reason += "Volume confirmation, "
        
        # Determine final signal
        if signal_strength >= 0.6:
            if signal_type != SignalType.SELL:
                signal_type = SignalType.BUY
        elif signal_strength >= 0.4 and signal_type == SignalType.SELL:
            pass  # Keep sell signal
        else:
            signal_type = SignalType.HOLD
            signal_strength = 0.0
        
        if signal_type != SignalType.HOLD:
            # Calculate stop loss and take profit
            if signal_type == SignalType.BUY:
                stop_loss = latest['close'] * (1 - config.STOP_LOSS_PCT)
                take_profit = latest['close'] * (1 + config.TAKE_PROFIT_PCT)
            else:
                stop_loss = None
                take_profit = None
            
            return Signal(
                signal_type=signal_type,
                strength=signal_strength,
                price=latest['close'],
                timestamp=latest.name,
                reason=reason.rstrip(", "),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        return None


class BreakoutStrategy(BaseStrategy):
    """Breakout strategy using volume and price action"""
    
    def __init__(self, config: Dict = None):
        super().__init__("Breakout", config or STRATEGY_CONFIGS["breakout"])
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        
        # Calculate indicators if not present
        if 'resistance' not in df.columns:
            df = TechnicalIndicators.calculate_all_indicators(df)
            latest = df.iloc[-1]
        
        signal_strength = 0.0
        signal_type = SignalType.HOLD
        reason = ""
        
        # Price breakout above resistance
        lookback_period = 20
        recent_high = df['high'].rolling(lookback_period).max().iloc[-1]
        if latest['close'] > recent_high * 1.02:  # 2% above recent high
            signal_strength += 0.4
            reason += "Price breakout above resistance, "
        
        # Volume breakout
        if latest['volume_ratio'] > self.config['volume_threshold']:
            signal_strength += 0.3
            reason += f"High volume ({latest['volume_ratio']:.1f}x avg), "
        
        # Price change momentum
        price_change = latest['momentum_1']
        if abs(price_change) > self.config['price_change_threshold']:
            signal_strength += 0.2
            if price_change > 0:
                reason += "Strong upward momentum, "
            else:
                signal_type = SignalType.SELL
                reason += "Strong downward momentum, "
        
        # Bollinger Band breakout
        if latest['close'] > latest['bb_upper']:
            signal_strength += 0.2
            reason += "BB upper breakout, "
        
        if latest['close'] < latest['bb_lower']:
            signal_strength += 0.2
            signal_type = SignalType.SELL
            reason += "BB lower breakdown, "
        
        # ATR expansion (volatility increase)
        if latest['high_volatility']:
            signal_strength += 0.1
            reason += "High volatility, "
        
        # Confirmation from multiple candles
        confirmation_count = 0
        for i in range(1, min(self.config['confirmation_candles'] + 1, len(df))):
            if df.iloc[-i]['volume_ratio'] > 1.2 and df.iloc[-i]['momentum_1'] > 0:
                confirmation_count += 1
        
        if confirmation_count >= self.config['confirmation_candles']:
            signal_strength += 0.1
            reason += "Multi-candle confirmation, "
        
        # Determine final signal
        if signal_strength >= 0.7:
            if signal_type != SignalType.SELL:
                signal_type = SignalType.BUY
        elif signal_strength >= 0.5 and signal_type == SignalType.SELL:
            pass  # Keep sell signal
        else:
            signal_type = SignalType.HOLD
            signal_strength = 0.0
        
        if signal_type != SignalType.HOLD:
            # Calculate stop loss and take profit
            if signal_type == SignalType.BUY:
                stop_loss = latest['close'] * (1 - config.STOP_LOSS_PCT * 1.5)  # Wider stop for breakouts
                take_profit = latest['close'] * (1 + config.TAKE_PROFIT_PCT * 1.5)  # Higher target
            else:
                stop_loss = None
                take_profit = None
            
            return Signal(
                signal_type=signal_type,
                strength=signal_strength,
                price=latest['close'],
                timestamp=latest.name,
                reason=reason.rstrip(", "),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        return None


class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining multiple approaches"""
    
    def __init__(self, config: Dict = None):
        super().__init__("Hybrid", config)
        
        # Initialize sub-strategies
        self.mean_reversion = MeanReversionStrategy()
        self.trend_following = TrendFollowingStrategy()
        self.breakout = BreakoutStrategy()
        
        # Weights for each strategy
        self.weights = {
            "mean_reversion": 0.3,
            "trend_following": 0.4,
            "breakout": 0.3
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        if len(df) < 50:
            return None
        
        # Get signals from all strategies
        mr_signal = self.mean_reversion.generate_signal(df)
        tf_signal = self.trend_following.generate_signal(df)
        bo_signal = self.breakout.generate_signal(df)
        
        # Combine signals
        total_buy_strength = 0.0
        total_sell_strength = 0.0
        reasons = []
        
        if mr_signal:
            if mr_signal.signal_type == SignalType.BUY:
                total_buy_strength += mr_signal.strength * self.weights["mean_reversion"]
                reasons.append(f"MR: {mr_signal.reason}")
            elif mr_signal.signal_type == SignalType.SELL:
                total_sell_strength += mr_signal.strength * self.weights["mean_reversion"]
                reasons.append(f"MR: {mr_signal.reason}")
        
        if tf_signal:
            if tf_signal.signal_type == SignalType.BUY:
                total_buy_strength += tf_signal.strength * self.weights["trend_following"]
                reasons.append(f"TF: {tf_signal.reason}")
            elif tf_signal.signal_type == SignalType.SELL:
                total_sell_strength += tf_signal.strength * self.weights["trend_following"]
                reasons.append(f"TF: {tf_signal.reason}")
        
        if bo_signal:
            if bo_signal.signal_type == SignalType.BUY:
                total_buy_strength += bo_signal.strength * self.weights["breakout"]
                reasons.append(f"BO: {bo_signal.reason}")
            elif bo_signal.signal_type == SignalType.SELL:
                total_sell_strength += bo_signal.strength * self.weights["breakout"]
                reasons.append(f"BO: {bo_signal.reason}")
        
        # Determine final signal
        if total_buy_strength > total_sell_strength and total_buy_strength > 0.5:
            signal_type = SignalType.BUY
            strength = total_buy_strength
        elif total_sell_strength > total_buy_strength and total_sell_strength > 0.5:
            signal_type = SignalType.SELL
            strength = total_sell_strength
        else:
            return None
        
        latest = df.iloc[-1]
        
        # Calculate stop loss and take profit
        if signal_type == SignalType.BUY:
            stop_loss = latest['close'] * (1 - config.STOP_LOSS_PCT)
            take_profit = latest['close'] * (1 + config.TAKE_PROFIT_PCT)
        else:
            stop_loss = None
            take_profit = None
        
        return Signal(
            signal_type=signal_type,
            strength=strength,
            price=latest['close'],
            timestamp=latest.name,
            reason=" | ".join(reasons),
            stop_loss=stop_loss,
            take_profit=take_profit
        )


class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self):
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "trend_following": TrendFollowingStrategy(),
            "breakout": BreakoutStrategy(),
            "hybrid": HybridStrategy()
        }
        self.active_strategy = "hybrid"
    
    def set_active_strategy(self, strategy_name: str):
        """Set the active trading strategy"""
        if strategy_name in self.strategies:
            self.active_strategy = strategy_name
            logger.info(f"Active strategy set to: {strategy_name}")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def get_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal from active strategy"""
        strategy = self.strategies[self.active_strategy]
        return strategy.generate_signal(df)
    
    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, Optional[Signal]]:
        """Get signals from all strategies"""
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                signals[name] = strategy.generate_signal(df)
            except Exception as e:
                logger.error(f"Error generating signal for {name}: {e}")
                signals[name] = None
        return signals
    
    def should_exit(self, df: pd.DataFrame) -> bool:
        """Check if should exit current position"""
        strategy = self.strategies[self.active_strategy]
        current_price = df.iloc[-1]['close']
        return strategy.should_exit(df, current_price)
    
    def update_position(self, signal: Signal):
        """Update position for active strategy"""
        strategy = self.strategies[self.active_strategy]
        strategy.update_position(signal)
    
    def get_strategy_stats(self) -> Dict[str, Dict]:
        """Get statistics for all strategies"""
        stats = {}
        for name, strategy in self.strategies.items():
            total_signals = len(strategy.signals_history)
            buy_signals = sum(1 for s in strategy.signals_history if s.signal_type == SignalType.BUY)
            sell_signals = sum(1 for s in strategy.signals_history if s.signal_type == SignalType.SELL)
            
            avg_strength = np.mean([s.strength for s in strategy.signals_history]) if strategy.signals_history else 0
            
            stats[name] = {
                "total_signals": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "avg_strength": avg_strength,
                "current_position": strategy.position.value,
                "entry_price": strategy.entry_price
            }
        
        return stats


if __name__ == "__main__":
    # Test the strategies with sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='5min')
    
    # Generate sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    sample_data = pd.DataFrame({
        'open': close_prices + np.random.randn(200) * 0.1,
        'high': close_prices + abs(np.random.randn(200)) * 0.5,
        'low': close_prices - abs(np.random.randn(200)) * 0.5,
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Add indicators
    sample_data = TechnicalIndicators.calculate_all_indicators(sample_data)
    
    # Test strategy manager
    manager = StrategyManager()
    
    print("Testing all strategies...")
    all_signals = manager.get_all_signals(sample_data)
    
    for strategy_name, signal in all_signals.items():
        print(f"\n{strategy_name.upper()}:")
        if signal:
            print(f"  Signal: {signal}")
        else:
            print(f"  No signal generated")
    
    print(f"\nStrategy Statistics:")
    stats = manager.get_strategy_stats()
    for name, stat in stats.items():
        print(f"{name}: {stat}")