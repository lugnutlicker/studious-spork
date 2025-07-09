"""
Example usage of the Solana Trading Bot components
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta

from config import config
from data_fetcher import get_token_data
from indicators import TechnicalIndicators
from strategies import StrategyManager
from backtester import SolanaBacktester


async def example_basic_analysis():
    """Example 1: Basic token analysis"""
    print("="*60)
    print("EXAMPLE 1: Basic Token Analysis")
    print("="*60)
    
    # Get token data
    token_address = config.TARGET_TOKEN
    print(f"Analyzing token: {token_address}")
    
    # Fetch 7 days of 5-minute data
    data = await get_token_data(
        token_address=token_address,
        timeframe="5m",
        days_back=7
    )
    
    if data.empty:
        print("No data available")
        return
    
    print(f"Retrieved {len(data)} price candles")
    print(f"Price range: ${data['close'].min():.8f} - ${data['close'].max():.8f}")
    print(f"Latest price: ${data['close'].iloc[-1]:.8f}")
    
    # Add technical indicators
    data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
    
    # Get signal summary
    signal_summary = TechnicalIndicators.get_signal_summary(data_with_indicators)
    
    print("\nTechnical Analysis Summary:")
    for key, value in signal_summary.items():
        if key not in ['timestamp']:
            print(f"  {key}: {value}")


async def example_strategy_signals():
    """Example 2: Generate trading signals with different strategies"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Strategy Signal Generation")
    print("="*60)
    
    # Get token data
    token_address = config.TARGET_TOKEN
    data = await get_token_data(token_address, timeframe="5m", days_back=7)
    
    if data.empty:
        print("No data available")
        return
    
    # Initialize strategy manager
    strategy_manager = StrategyManager()
    
    # Get signals from all strategies
    all_signals = strategy_manager.get_all_signals(data)
    
    print("Strategy Signals:")
    for strategy_name, signal in all_signals.items():
        print(f"\n{strategy_name.upper()}:")
        if signal:
            print(f"  Signal Type: {signal.signal_type.value}")
            print(f"  Strength: {signal.strength:.2f}")
            print(f"  Price: ${signal.price:.8f}")
            print(f"  Reason: {signal.reason}")
            if signal.stop_loss:
                print(f"  Stop Loss: ${signal.stop_loss:.8f}")
            if signal.take_profit:
                print(f"  Take Profit: ${signal.take_profit:.8f}")
        else:
            print("  No signal generated")


async def example_backtesting():
    """Example 3: Run a simple backtest"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Backtesting")
    print("="*60)
    
    # Get more historical data for backtesting
    token_address = config.TARGET_TOKEN
    data = await get_token_data(token_address, timeframe="5m", days_back=30)
    
    if len(data) < 100:
        print("Insufficient data for backtesting")
        return
    
    print(f"Running backtest on {len(data)} candles...")
    
    # Initialize backtester
    backtester = SolanaBacktester(initial_capital=1000)
    
    # Run backtest with hybrid strategy
    results = backtester.run_backtest(
        data=data,
        strategy_name="hybrid"
    )
    
    # Print key results
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results.total_return:.2%}")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Win Rate: {results.win_rate:.2%}")
    print(f"  Max Drawdown: {results.max_drawdown:.2%}")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    
    if results.trades:
        print(f"\nSample Trades:")
        for i, trade in enumerate(results.trades[:3]):  # Show first 3 trades
            print(f"  Trade {i+1}:")
            print(f"    Entry: ${trade.entry_price:.8f} at {trade.entry_time}")
            print(f"    Exit: ${trade.exit_price:.8f} at {trade.exit_time}")
            print(f"    P&L: ${trade.pnl:.2f} ({trade.pnl_percent:.2f}%)")
            print(f"    Duration: {trade.duration}")


async def example_strategy_comparison():
    """Example 4: Compare different strategies"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Strategy Comparison")
    print("="*60)
    
    # Get historical data
    token_address = config.TARGET_TOKEN
    data = await get_token_data(token_address, timeframe="5m", days_back=30)
    
    if len(data) < 100:
        print("Insufficient data for comparison")
        return
    
    # Initialize backtester
    backtester = SolanaBacktester(initial_capital=1000)
    
    # Compare all strategies
    comparison = backtester.compare_strategies(data)
    
    print("Strategy Comparison Results:")
    print(comparison.to_string(index=False))


async def example_custom_indicators():
    """Example 5: Working with custom indicators"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Technical Indicators")
    print("="*60)
    
    # Get token data
    token_address = config.TARGET_TOKEN
    data = await get_token_data(token_address, timeframe="5m", days_back=7)
    
    if data.empty:
        print("No data available")
        return
    
    # Add all indicators
    data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
    
    # Show latest values for key indicators
    latest = data_with_indicators.iloc[-1]
    
    print("Latest Technical Indicator Values:")
    print(f"  Price: ${latest['close']:.8f}")
    print(f"  RSI: {latest['rsi']:.2f}")
    print(f"  MACD: {latest['macd']:.6f}")
    print(f"  BB Position: {latest['bb_position']:.2f}")
    print(f"  Volume Ratio: {latest['volume_ratio']:.2f}")
    print(f"  Trend Strength: {latest['trend_strength']:.4f}")
    print(f"  Bullish Score: {latest['bullish_score']}")
    print(f"  Bearish Score: {latest['bearish_score']}")


async def main():
    """Run all examples"""
    try:
        print("Solana Trading Bot - Example Usage")
        print("This will demonstrate the key features of the trading bot\n")
        
        await example_basic_analysis()
        await example_strategy_signals()
        await example_backtesting()
        await example_strategy_comparison()
        await example_custom_indicators()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("Check the generated files for detailed results.")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())