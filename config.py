"""
Configuration file for Solana Trading Bot
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from enum import Enum


class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class Config(BaseSettings):
    """Main configuration class for the trading bot"""
    
    # Target token configuration
    TARGET_TOKEN: str = "8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm"
    BASE_TOKEN: str = "So11111111111111111111111111111111111111112"  # SOL
    QUOTE_TOKEN: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
    
    # Solana network configuration
    SOLANA_RPC_URL: str = "https://api.mainnet-beta.solana.com"
    SOLANA_WS_URL: str = "wss://api.mainnet-beta.solana.com"
    
    # Trading configuration
    TRADING_MODE: TradingMode = TradingMode.BACKTEST
    WALLET_PRIVATE_KEY: Optional[str] = None
    
    # Strategy parameters
    INITIAL_CAPITAL: float = 1000.0  # USDC
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio per trade
    STOP_LOSS_PCT: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT: float = 0.15  # 15% take profit
    
    # Technical indicators
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30.0
    RSI_OVERBOUGHT: float = 70.0
    
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    
    # Data fetching
    DATA_PROVIDER: str = "solana_tracker"  # Options: solana_tracker, birdeye, custom
    API_KEY: Optional[str] = None
    
    # Backtesting
    START_DATE: str = "2024-01-01"
    END_DATE: str = "2024-12-31"
    TIMEFRAME: str = "5m"  # 1m, 5m, 15m, 1h, 4h, 1d
    
    # Risk management
    MAX_DRAWDOWN: float = 0.20  # 20% max drawdown
    MAX_CONSECUTIVE_LOSSES: int = 5
    
    # Performance metrics
    BENCHMARK_SYMBOL: str = "SOL"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "trading_bot.log"
    
    # Database
    DATABASE_PATH: str = "trading_data.db"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global config instance
config = Config()


# Trading pairs configuration
TRADING_PAIRS = {
    "target_token": config.TARGET_TOKEN,
    "base_token": config.BASE_TOKEN,
    "quote_token": config.QUOTE_TOKEN,
}

# Data provider endpoints
DATA_PROVIDERS = {
    "solana_tracker": {
        "base_url": "https://data.solanatracker.io",
        "endpoints": {
            "token_info": "/tokens/{token_address}",
            "price_history": "/price/history",
            "ohlcv": "/chart/{token}",
            "trades": "/trades/{token_address}",
        }
    },
    "birdeye": {
        "base_url": "https://public-api.birdeye.so",
        "endpoints": {
            "token_info": "/public/token/{token_address}",
            "price_history": "/public/price_history/{token_address}",
            "ohlcv": "/public/price_history/{token_address}",
        }
    }
}

# Jupiter swap configuration
JUPITER_CONFIG = {
    "base_url": "https://quote-api.jup.ag/v6",
    "slippage": 0.5,  # 0.5% slippage
    "priority_fee": 0.001,  # SOL
}

# Strategy configurations
STRATEGY_CONFIGS = {
    "mean_reversion": {
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "bollinger_entry_threshold": 2.0,
        "bollinger_exit_threshold": 0.5,
    },
    "trend_following": {
        "ema_fast": 9,
        "ema_slow": 21,
        "macd_signal_threshold": 0.0,
        "trend_strength_threshold": 0.02,
    },
    "breakout": {
        "volume_threshold": 2.0,  # 2x average volume
        "price_change_threshold": 0.05,  # 5% price change
        "confirmation_candles": 2,
    }
}