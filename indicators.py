"""
Technical indicators for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from loguru import logger

from config import config


class TechnicalIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the dataframe"""
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for technical indicators")
            return df
        
        result_df = df.copy()
        
        # Price-based indicators
        result_df = TechnicalIndicators.add_moving_averages(result_df)
        result_df = TechnicalIndicators.add_rsi(result_df)
        result_df = TechnicalIndicators.add_macd(result_df)
        result_df = TechnicalIndicators.add_bollinger_bands(result_df)
        result_df = TechnicalIndicators.add_stochastic(result_df)
        result_df = TechnicalIndicators.add_williams_r(result_df)
        
        # Volume indicators
        result_df = TechnicalIndicators.add_volume_indicators(result_df)
        
        # Volatility indicators
        result_df = TechnicalIndicators.add_atr(result_df)
        
        # Support and resistance
        result_df = TechnicalIndicators.add_support_resistance(result_df)
        
        # Custom indicators
        result_df = TechnicalIndicators.add_custom_indicators(result_df)
        
        return result_df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages"""
        try:
            # Simple Moving Averages
            df['sma_9'] = df['close'].rolling(window=9).mean()
            df['sma_21'] = df['close'].rolling(window=21).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['ema_9'] = df['close'].ewm(span=config.EMA_FAST).mean()
            df['ema_21'] = df['close'].ewm(span=config.EMA_SLOW).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # Weighted Moving Average (simplified)
            weights = np.arange(1, 22)
            df['wma_21'] = df['close'].rolling(window=21).apply(lambda x: np.average(x, weights=weights), raw=True)
            
            # Hull Moving Average (simplified approximation)
            half_length = 10
            sqrt_length = int(np.sqrt(21))
            wma_half = df['close'].rolling(window=half_length).apply(lambda x: np.average(x, weights=np.arange(1, half_length+1)), raw=True)
            wma_full = df['close'].rolling(window=21).apply(lambda x: np.average(x, weights=np.arange(1, 22)), raw=True)
            raw_hma = 2 * wma_half - wma_full
            df['hma_21'] = raw_hma.rolling(window=sqrt_length).apply(lambda x: np.average(x, weights=np.arange(1, sqrt_length+1)), raw=True)
            
            # Moving average signals
            df['ma_signal'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
            df['ma_crossover'] = df['ma_signal'].diff()
            
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Add RSI indicator"""
        try:
            if period is None:
                period = config.RSI_PERIOD
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Additional RSI periods
            delta_14 = df['close'].diff()
            gain_14 = (delta_14.where(delta_14 > 0, 0)).rolling(window=14).mean()
            loss_14 = (-delta_14.where(delta_14 < 0, 0)).rolling(window=14).mean()
            rs_14 = gain_14 / loss_14
            df['rsi_14'] = 100 - (100 / (1 + rs_14))
            
            delta_21 = df['close'].diff()
            gain_21 = (delta_21.where(delta_21 > 0, 0)).rolling(window=21).mean()
            loss_21 = (-delta_21.where(delta_21 < 0, 0)).rolling(window=21).mean()
            rs_21 = gain_21 / loss_21
            df['rsi_21'] = 100 - (100 / (1 + rs_21))
            
            # RSI signals
            df['rsi_oversold'] = df['rsi'] < config.RSI_OVERSOLD
            df['rsi_overbought'] = df['rsi'] > config.RSI_OVERBOUGHT
            
            # RSI divergence (simplified)
            df['rsi_divergence'] = TechnicalIndicators._calculate_rsi_divergence(df)
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        try:
            # MACD calculation
            ema_fast = df['close'].ewm(span=config.MACD_FAST).mean()
            ema_slow = df['close'].ewm(span=config.MACD_SLOW).mean()
            
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=config.MACD_SIGNAL).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # MACD signals
            df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd_histogram'] > 0)
            df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd_histogram'] < 0)
            df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1).diff()
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        try:
            # Bollinger Bands
            sma = df['close'].rolling(window=config.BOLLINGER_PERIOD).mean()
            std = df['close'].rolling(window=config.BOLLINGER_PERIOD).std()
            
            df['bb_upper'] = sma + (std * config.BOLLINGER_STD)
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * config.BOLLINGER_STD)
            
            # Bollinger Band signals
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).quantile(0.1)
            df['bb_oversold'] = df['close'] < df['bb_lower']
            df['bb_overbought'] = df['close'] > df['bb_upper']
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
        
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic oscillator"""
        try:
            # Stochastic %K
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            
            # Stochastic %D (moving average of %K)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
            
            # Stochastic signals
            df['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
            df['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
            
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {e}")
        
        return df
    
    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R"""
        try:
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            df['williams_r'] = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            # Williams %R signals
            df['williams_oversold'] = df['williams_r'] < -80
            df['williams_overbought'] = df['williams_r'] > -20
            
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {e}")
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        try:
            # Volume moving averages
            df['volume_sma'] = df['volume'].rolling(window=21).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # On-Balance Volume
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_sma'] = df['obv'].rolling(window=21).mean()
            
            # Volume Price Trend
            df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).fillna(0).cumsum()
            
            # Accumulation/Distribution Line (simplified)
            mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfm = mfm.fillna(0)
            df['ad'] = (mfm * df['volume']).cumsum()
            
            # Chaikin Money Flow (simplified)
            mfm_volume = mfm * df['volume']
            df['cmf'] = mfm_volume.rolling(window=21).sum() / df['volume'].rolling(window=21).sum()
            
            # Volume signals
            df['high_volume'] = df['volume_ratio'] > 2.0
            df['volume_breakout'] = (df['volume_ratio'] > 1.5) & (df['close'] > df['sma_21'])
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (volatility)"""
        try:
            # True Range calculation
            high_low = df['high'] - df['low']
            high_prev_close = np.abs(df['high'] - df['close'].shift(1))
            low_prev_close = np.abs(df['low'] - df['close'].shift(1))
            
            tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
            df['atr'] = tr.rolling(window=period).mean()
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # Volatility signals
            df['high_volatility'] = df['atr_percent'] > df['atr_percent'].rolling(50).quantile(0.8)
            df['low_volatility'] = df['atr_percent'] < df['atr_percent'].rolling(50).quantile(0.2)
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
        
        return df
    
    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add support and resistance levels"""
        try:
            # Simple support/resistance based on rolling min/max
            df['resistance'] = df['high'].rolling(window=window).max()
            df['support'] = df['low'].rolling(window=window).min()
            
            # Distance from support/resistance
            df['dist_from_resistance'] = (df['resistance'] - df['close']) / df['close']
            df['dist_from_support'] = (df['close'] - df['support']) / df['close']
            
            # Near support/resistance signals
            df['near_resistance'] = df['dist_from_resistance'] < 0.02  # Within 2%
            df['near_support'] = df['dist_from_support'] < 0.02  # Within 2%
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
        
        return df
    
    @staticmethod
    def add_custom_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom indicators specific to crypto trading"""
        try:
            # Price momentum
            df['momentum_1'] = df['close'].pct_change(1)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_20'] = df['close'].pct_change(20)
            
            # Volatility-adjusted momentum
            df['vol_adj_momentum'] = df['momentum_1'] / (df['atr_percent'] + 1e-10)  # Add small value to avoid division by zero
            
            # Trend strength
            df['trend_strength'] = abs(df['ema_9'] - df['ema_21']) / df['close']
            
            # Price position in range
            range_diff = df['resistance'] - df['support']
            df['price_position'] = np.where(range_diff != 0, (df['close'] - df['support']) / range_diff, 0.5)
            
            # Multi-timeframe signals (simplified)
            df['bullish_alignment'] = (
                (df['close'] > df['ema_9']) & 
                (df['ema_9'] > df['ema_21']) &
                (df['rsi'] > 50) &
                (df['macd'] > df['macd_signal'])
            ).astype(int)
            
            df['bearish_alignment'] = (
                (df['close'] < df['ema_9']) & 
                (df['ema_9'] < df['ema_21']) &
                (df['rsi'] < 50) &
                (df['macd'] < df['macd_signal'])
            ).astype(int)
            
            # Composite score
            df['bullish_score'] = (
                df['bullish_alignment'] +
                df['rsi_oversold'].astype(int) +
                df['bb_oversold'].astype(int) +
                df['stoch_oversold'].astype(int) +
                df['macd_bullish'].astype(int)
            )
            
            df['bearish_score'] = (
                df['bearish_alignment'] +
                df['rsi_overbought'].astype(int) +
                df['bb_overbought'].astype(int) +
                df['stoch_overbought'].astype(int) +
                df['macd_bearish'].astype(int)
            )
            
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {e}")
        
        return df
    
    @staticmethod
    def _calculate_rsi_divergence(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI divergence (simplified)"""
        try:
            # Find local peaks and troughs
            price_peaks = df['close'].rolling(window).max() == df['close']
            price_troughs = df['close'].rolling(window).min() == df['close']
            
            rsi_peaks = df['rsi'].rolling(window).max() == df['rsi']
            rsi_troughs = df['rsi'].rolling(window).min() == df['rsi']
            
            # Simplified divergence detection
            bullish_divergence = price_troughs & ~rsi_troughs
            bearish_divergence = price_peaks & ~rsi_peaks
            
            divergence = pd.Series(0, index=df.index)
            divergence[bullish_divergence] = 1
            divergence[bearish_divergence] = -1
            
            return divergence
            
        except Exception as e:
            logger.error(f"Error calculating RSI divergence: {e}")
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def get_signal_summary(df: pd.DataFrame) -> dict:
        """Get a summary of all trading signals"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        summary = {
            "timestamp": latest.name,
            "price": latest['close'],
            "volume": latest['volume'],
            
            # Moving averages
            "ma_trend": "bullish" if latest['ma_signal'] > 0 else "bearish",
            "ema_9": latest['ema_9'],
            "ema_21": latest['ema_21'],
            
            # RSI
            "rsi": latest['rsi'],
            "rsi_signal": "oversold" if latest['rsi_oversold'] else "overbought" if latest['rsi_overbought'] else "neutral",
            
            # MACD
            "macd": latest['macd'],
            "macd_signal": "bullish" if latest['macd_bullish'] else "bearish" if latest['macd_bearish'] else "neutral",
            
            # Bollinger Bands
            "bb_position": latest['bb_position'],
            "bb_signal": "oversold" if latest['bb_oversold'] else "overbought" if latest['bb_overbought'] else "neutral",
            
            # Volume
            "volume_ratio": latest['volume_ratio'],
            "volume_signal": "high" if latest['high_volume'] else "normal",
            
            # Composite signals
            "bullish_score": latest['bullish_score'],
            "bearish_score": latest['bearish_score'],
            "overall_signal": "bullish" if latest['bullish_score'] > latest['bearish_score'] else "bearish"
        }
        
        return summary


def add_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add all trading signals"""
    return TechnicalIndicators.calculate_all_indicators(df)


if __name__ == "__main__":
    # Test the indicators with sample data
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
    
    # Add all indicators
    result = add_trading_signals(sample_data)
    
    # Print summary
    summary = TechnicalIndicators.get_signal_summary(result)
    print("Signal Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nDataFrame shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")