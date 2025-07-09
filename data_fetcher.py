"""
Data fetcher for Solana token price data and trading information
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import sqlite3
import json
from loguru import logger

from config import config, DATA_PROVIDERS


class SolanaDataFetcher:
    """Fetches data from various Solana data providers"""
    
    def __init__(self, provider: str = None, api_key: str = None):
        self.provider = provider or config.DATA_PROVIDER
        self.api_key = api_key or config.API_KEY
        self.base_url = DATA_PROVIDERS[self.provider]["base_url"]
        self.endpoints = DATA_PROVIDERS[self.provider]["endpoints"]
        self.session = None
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for caching data"""
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Create tables for different data types
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT,
                timestamp DATETIME,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume REAL,
                timeframe TEXT,
                provider TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(token_address, timestamp, timeframe, provider)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_info (
                token_address TEXT PRIMARY KEY,
                name TEXT,
                symbol TEXT,
                decimals INTEGER,
                supply REAL,
                market_cap REAL,
                liquidity REAL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_address TEXT,
                timestamp DATETIME,
                price REAL,
                amount REAL,
                volume_usd REAL,
                side TEXT,
                wallet_address TEXT,
                signature TEXT,
                provider TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {
            "Accept": "application/json",
            "User-Agent": "Solana-Trading-Bot/1.0"
        }
        
        if self.api_key:
            if self.provider == "solana_tracker":
                headers["x-api-key"] = self.api_key
            elif self.provider == "birdeye":
                headers["X-API-KEY"] = self.api_key
        
        return headers
    
    async def get_token_info(self, token_address: str) -> Dict:
        """Get basic token information"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{self.endpoints['token_info'].format(token_address=token_address)}"
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache token info
                    self._cache_token_info(token_address, data)
                    
                    return data
                else:
                    logger.error(f"Failed to fetch token info: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching token info: {e}")
            return {}
    
    async def get_ohlcv_data(
        self, 
        token_address: str, 
        timeframe: str = "5m",
        limit: int = 1000,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get OHLCV candlestick data"""
        
        # Check cache first
        cached_data = self._get_cached_ohlcv(token_address, timeframe, start_time, end_time)
        if not cached_data.empty:
            logger.info(f"Using cached OHLCV data for {token_address}")
            return cached_data
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Convert timeframe to API format
        api_timeframe = self._convert_timeframe(timeframe)
        
        url = f"{self.base_url}{self.endpoints['ohlcv'].format(token=token_address)}"
        params = {
            "type": api_timeframe,
            "time_from": int(start_time.timestamp()) if start_time else None,
            "time_to": int(end_time.timestamp()) if end_time else None,
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse OHLCV data
                    df = self._parse_ohlcv_response(data, token_address, timeframe)
                    
                    # Cache the data
                    self._cache_ohlcv_data(df, token_address, timeframe)
                    
                    return df
                else:
                    logger.error(f"Failed to fetch OHLCV data: {response.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {e}")
            return pd.DataFrame()
    
    async def get_recent_trades(self, token_address: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for the token"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{self.endpoints['trades'].format(token_address=token_address)}"
        headers = self._get_headers()
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse trades data
                    df = self._parse_trades_response(data, token_address)
                    
                    # Cache trades
                    self._cache_trades_data(df, token_address)
                    
                    return df
                else:
                    logger.error(f"Failed to fetch trades: {response.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return pd.DataFrame()
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to API format"""
        mapping = {
            "1m": "1m",
            "5m": "5m", 
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d"
        }
        return mapping.get(timeframe, "5m")
    
    def _parse_ohlcv_response(self, data: Dict, token_address: str, timeframe: str) -> pd.DataFrame:
        """Parse OHLCV response from API"""
        if self.provider == "solana_tracker":
            if "oclhv" in data:
                ohlcv_data = data["oclhv"]
                df = pd.DataFrame(ohlcv_data)
                
                # Rename columns to standard format
                if not df.empty:
                    df.rename(columns={
                        "time": "timestamp",
                        "open": "open",
                        "close": "close", 
                        "low": "low",
                        "high": "high",
                        "volume": "volume"
                    }, inplace=True)
                    
                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                    df.set_index("timestamp", inplace=True)
                    
                    # Ensure correct order of columns
                    df = df[["open", "high", "low", "close", "volume"]]
        
        return df if 'df' in locals() else pd.DataFrame()
    
    def _parse_trades_response(self, data: Dict, token_address: str) -> pd.DataFrame:
        """Parse trades response from API"""
        trades_list = []
        
        if self.provider == "solana_tracker":
            if "trades" in data:
                for trade in data["trades"]:
                    trades_list.append({
                        "timestamp": pd.to_datetime(trade.get("time", 0), unit="ms"),
                        "price": trade.get("priceUsd", 0),
                        "amount": trade.get("amount", 0),
                        "volume_usd": trade.get("volume", 0),
                        "side": trade.get("type", "unknown"),
                        "wallet_address": trade.get("wallet", ""),
                        "signature": trade.get("tx", "")
                    })
        
        if trades_list:
            df = pd.DataFrame(trades_list)
            df.set_index("timestamp", inplace=True)
            return df
        
        return pd.DataFrame()
    
    def _cache_token_info(self, token_address: str, data: Dict):
        """Cache token information"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            if self.provider == "solana_tracker":
                token_data = data.get("token", {})
                cursor.execute("""
                    INSERT OR REPLACE INTO token_info 
                    (token_address, name, symbol, decimals, supply, market_cap, liquidity)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_address,
                    token_data.get("name", ""),
                    token_data.get("symbol", ""),
                    token_data.get("decimals", 0),
                    0,  # Supply not directly available
                    0,  # Market cap calculation needed
                    0   # Liquidity calculation needed
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error caching token info: {e}")
    
    def _cache_ohlcv_data(self, df: pd.DataFrame, token_address: str, timeframe: str):
        """Cache OHLCV data"""
        if df.empty:
            return
            
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            for timestamp, row in df.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv_data 
                    (token_address, timestamp, open_price, high_price, low_price, close_price, volume, timeframe, provider)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_address,
                    timestamp,
                    row["open"],
                    row["high"], 
                    row["low"],
                    row["close"],
                    row["volume"],
                    timeframe,
                    self.provider
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error caching OHLCV data: {e}")
    
    def _cache_trades_data(self, df: pd.DataFrame, token_address: str):
        """Cache trades data"""
        if df.empty:
            return
            
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            for timestamp, row in df.iterrows():
                conn.execute("""
                    INSERT OR IGNORE INTO trades 
                    (token_address, timestamp, price, amount, volume_usd, side, wallet_address, signature, provider)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_address,
                    timestamp,
                    row["price"],
                    row["amount"],
                    row["volume_usd"],
                    row["side"],
                    row["wallet_address"],
                    row["signature"],
                    self.provider
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error caching trades data: {e}")
    
    def _get_cached_ohlcv(
        self, 
        token_address: str, 
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get cached OHLCV data"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            
            query = """
                SELECT timestamp, open_price, high_price, low_price, close_price, volume
                FROM ohlcv_data 
                WHERE token_address = ? AND timeframe = ? AND provider = ?
            """
            params = [token_address, timeframe, self.provider]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                df.rename(columns={
                    "open_price": "open",
                    "high_price": "high",
                    "low_price": "low", 
                    "close_price": "close"
                }, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting cached OHLCV: {e}")
            return pd.DataFrame()


# Convenience function to create data fetcher
async def get_token_data(
    token_address: str,
    timeframe: str = "5m",
    days_back: int = 30,
    provider: str = None
) -> pd.DataFrame:
    """Convenience function to get token data"""
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    async with SolanaDataFetcher(provider=provider) as fetcher:
        # Get token info
        token_info = await fetcher.get_token_info(token_address)
        logger.info(f"Token info: {token_info}")
        
        # Get OHLCV data
        df = await fetcher.get_ohlcv_data(
            token_address, 
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get recent trades
        trades = await fetcher.get_recent_trades(token_address)
        logger.info(f"Got {len(trades)} recent trades")
        
        return df


if __name__ == "__main__":
    # Test the data fetcher
    async def test():
        token_address = config.TARGET_TOKEN
        df = await get_token_data(token_address, timeframe="5m", days_back=7)
        print(f"Retrieved {len(df)} candles")
        print(df.head())
        print(df.tail())
    
    asyncio.run(test())