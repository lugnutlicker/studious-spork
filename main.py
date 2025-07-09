"""
Main Solana Trading Bot Application
"""

import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional
from loguru import logger
import pandas as pd

from config import config, TradingMode
from data_fetcher import SolanaDataFetcher, get_token_data
from indicators import TechnicalIndicators
from strategies import StrategyManager
from backtester import SolanaBacktester, BacktestResults


class SolanaTradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, trading_mode: TradingMode = TradingMode.BACKTEST):
        self.trading_mode = trading_mode
        self.strategy_manager = StrategyManager()
        self.data_fetcher = None
        self.backtester = None
        
        # Initialize based on mode
        if trading_mode == TradingMode.BACKTEST:
            self.backtester = SolanaBacktester(config.INITIAL_CAPITAL)
        
        logger.info(f"Initialized Solana Trading Bot in {trading_mode.value} mode")
    
    async def run_backtest(
        self,
        token_address: str,
        strategy_name: str = "hybrid",
        timeframe: str = "5m",
        days_back: int = 30,
        generate_plots: bool = True,
        save_results: bool = True
    ) -> BacktestResults:
        """Run backtest for specified parameters"""
        
        logger.info(f"Starting backtest for {token_address}")
        logger.info(f"Strategy: {strategy_name}, Timeframe: {timeframe}, Days: {days_back}")
        
        # Fetch historical data
        try:
            data = await get_token_data(
                token_address=token_address,
                timeframe=timeframe,
                days_back=days_back
            )
            
            if data.empty or len(data) < 100:
                raise ValueError(f"Insufficient data: only {len(data)} candles retrieved")
            
            logger.info(f"Retrieved {len(data)} candles for backtesting")
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
        
        # Run backtest
        try:
            results = self.backtester.run_backtest(
                data=data,
                strategy_name=strategy_name
            )
            
            # Generate report
            report = self.backtester.generate_report(results)
            logger.info("Backtest completed successfully")
            
            # Save results
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save text report
                report_file = f"backtest_report_{strategy_name}_{timestamp}.txt"
                with open(report_file, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {report_file}")
                
                # Save detailed results
                results_file = f"backtest_results_{strategy_name}_{timestamp}.csv"
                results.portfolio_history.to_csv(results_file)
                logger.info(f"Portfolio history saved to {results_file}")
                
                # Save trades
                if results.trades:
                    trades_df = pd.DataFrame([
                        {
                            'entry_time': trade.entry_time,
                            'exit_time': trade.exit_time,
                            'entry_price': trade.entry_price,
                            'exit_price': trade.exit_price,
                            'quantity': trade.quantity,
                            'pnl': trade.pnl,
                            'pnl_percent': trade.pnl_percent,
                            'duration_hours': trade.duration.total_seconds() / 3600,
                            'entry_reason': trade.entry_reason,
                            'exit_reason': trade.exit_reason
                        }
                        for trade in results.trades
                    ])
                    trades_file = f"backtest_trades_{strategy_name}_{timestamp}.csv"
                    trades_df.to_csv(trades_file, index=False)
                    logger.info(f"Trades saved to {trades_file}")
                
                # Generate plots
                if generate_plots:
                    plot_file = f"backtest_plots_{strategy_name}_{timestamp}.html"
                    self.backtester.plot_results(results, plot_file)
                    logger.info(f"Plots saved to {plot_file}")
            
            # Print summary to console
            print("\n" + "="*50)
            print(report)
            print("="*50)
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def compare_strategies(
        self,
        token_address: str,
        strategies: list = None,
        timeframe: str = "5m",
        days_back: int = 30
    ) -> pd.DataFrame:
        """Compare multiple strategies"""
        
        if strategies is None:
            strategies = list(self.strategy_manager.strategies.keys())
        
        logger.info(f"Comparing strategies: {strategies}")
        
        # Fetch data once
        data = await get_token_data(
            token_address=token_address,
            timeframe=timeframe,
            days_back=days_back
        )
        
        # Compare strategies
        comparison = self.backtester.compare_strategies(data, strategies)
        
        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"strategy_comparison_{timestamp}.csv"
        comparison.to_csv(comparison_file, index=False)
        logger.info(f"Strategy comparison saved to {comparison_file}")
        
        # Print results
        print("\nSTRATEGY COMPARISON RESULTS")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("=" * 80)
        
        return comparison
    
    async def analyze_token(
        self,
        token_address: str,
        timeframe: str = "5m",
        days_back: int = 7
    ) -> Dict:
        """Analyze token and provide current trading signals"""
        
        logger.info(f"Analyzing token: {token_address}")
        
        # Fetch recent data
        data = await get_token_data(
            token_address=token_address,
            timeframe=timeframe,
            days_back=days_back
        )
        
        if data.empty:
            raise ValueError("No data available for analysis")
        
        # Add technical indicators
        data = TechnicalIndicators.calculate_all_indicators(data)
        
        # Get current signals from all strategies
        all_signals = self.strategy_manager.get_all_signals(data)
        
        # Get signal summary
        signal_summary = TechnicalIndicators.get_signal_summary(data)
        
        # Compile analysis
        analysis = {
            'token_address': token_address,
            'analysis_time': datetime.now(),
            'latest_price': data['close'].iloc[-1],
            'price_change_24h': data['close'].pct_change(periods=min(288, len(data))).iloc[-1] * 100,  # Assuming 5m candles
            'volume_ratio': data['volume_ratio'].iloc[-1] if 'volume_ratio' in data.columns else None,
            'signal_summary': signal_summary,
            'strategy_signals': {}
        }
        
        # Add strategy-specific signals
        for strategy_name, signal in all_signals.items():
            if signal:
                analysis['strategy_signals'][strategy_name] = {
                    'signal_type': signal.signal_type.value,
                    'strength': signal.strength,
                    'reason': signal.reason,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit
                }
            else:
                analysis['strategy_signals'][strategy_name] = None
        
        # Print analysis
        self._print_analysis(analysis)
        
        return analysis
    
    def _print_analysis(self, analysis: Dict):
        """Print formatted analysis to console"""
        
        print(f"\nTOKEN ANALYSIS: {analysis['token_address']}")
        print("=" * 80)
        print(f"Analysis Time: {analysis['analysis_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latest Price: ${analysis['latest_price']:.8f}")
        print(f"24h Change: {analysis['price_change_24h']:.2f}%")
        
        if analysis['volume_ratio']:
            print(f"Volume Ratio: {analysis['volume_ratio']:.2f}x")
        
        # Signal summary
        summary = analysis['signal_summary']
        print(f"\nTECHNICAL INDICATORS")
        print("-" * 40)
        print(f"RSI: {summary.get('rsi', 'N/A'):.2f} ({summary.get('rsi_signal', 'N/A')})")
        print(f"MACD: {summary.get('macd_signal', 'N/A')}")
        print(f"BB Position: {summary.get('bb_position', 0):.2f} ({summary.get('bb_signal', 'N/A')})")
        print(f"MA Trend: {summary.get('ma_trend', 'N/A')}")
        print(f"Overall Signal: {summary.get('overall_signal', 'N/A')}")
        
        # Strategy signals
        print(f"\nSTRATEGY SIGNALS")
        print("-" * 40)
        for strategy_name, signal_data in analysis['strategy_signals'].items():
            if signal_data:
                print(f"{strategy_name.upper()}:")
                print(f"  Signal: {signal_data['signal_type'].upper()}")
                print(f"  Strength: {signal_data['strength']:.2f}")
                print(f"  Reason: {signal_data['reason']}")
                if signal_data['stop_loss']:
                    print(f"  Stop Loss: ${signal_data['stop_loss']:.8f}")
                if signal_data['take_profit']:
                    print(f"  Take Profit: ${signal_data['take_profit']:.8f}")
            else:
                print(f"{strategy_name.upper()}: No signal")
            print()
    
    async def monitor_token(
        self,
        token_address: str,
        check_interval: int = 60,
        timeframe: str = "5m"
    ):
        """Monitor token for trading signals in real-time"""
        
        logger.info(f"Starting real-time monitoring for {token_address}")
        logger.info(f"Check interval: {check_interval} seconds")
        
        while True:
            try:
                # Analyze current conditions
                analysis = await self.analyze_token(
                    token_address=token_address,
                    timeframe=timeframe,
                    days_back=7
                )
                
                # Check for any actionable signals
                actionable_signals = [
                    signal for signal in analysis['strategy_signals'].values()
                    if signal and signal['signal_type'] in ['buy', 'sell'] and signal['strength'] > 0.7
                ]
                
                if actionable_signals:
                    logger.warning(f"HIGH CONFIDENCE SIGNALS DETECTED!")
                    for i, signal in enumerate(actionable_signals):
                        logger.warning(f"Signal {i+1}: {signal}")
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error during monitoring: {e}")
                await asyncio.sleep(check_interval)


def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(
        sys.stdout,
        level=config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        config.LOG_FILE,
        level=config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB"
    )


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Solana Trading Bot")
    parser.add_argument("command", choices=[
        "backtest", "compare", "analyze", "monitor"
    ], help="Command to execute")
    
    parser.add_argument("--token", default=config.TARGET_TOKEN,
                       help="Token address to trade")
    parser.add_argument("--strategy", default="hybrid",
                       choices=["mean_reversion", "trend_following", "breakout", "hybrid"],
                       help="Trading strategy to use")
    parser.add_argument("--timeframe", default="5m",
                       choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                       help="Timeframe for analysis")
    parser.add_argument("--days", type=int, default=30,
                       help="Days of historical data for backtesting")
    parser.add_argument("--capital", type=float, default=config.INITIAL_CAPITAL,
                       help="Initial capital for backtesting")
    parser.add_argument("--interval", type=int, default=60,
                       help="Check interval for monitoring (seconds)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--no-save", action="store_true",
                       help="Skip saving results")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Update config with command line arguments
    if args.capital != config.INITIAL_CAPITAL:
        config.INITIAL_CAPITAL = args.capital
    
    # Initialize bot
    bot = SolanaTradingBot(TradingMode.BACKTEST)
    
    try:
        if args.command == "backtest":
            logger.info("Running backtest...")
            await bot.run_backtest(
                token_address=args.token,
                strategy_name=args.strategy,
                timeframe=args.timeframe,
                days_back=args.days,
                generate_plots=not args.no_plots,
                save_results=not args.no_save
            )
        
        elif args.command == "compare":
            logger.info("Comparing strategies...")
            await bot.compare_strategies(
                token_address=args.token,
                timeframe=args.timeframe,
                days_back=args.days
            )
        
        elif args.command == "analyze":
            logger.info("Analyzing token...")
            await bot.analyze_token(
                token_address=args.token,
                timeframe=args.timeframe,
                days_back=7
            )
        
        elif args.command == "monitor":
            logger.info("Starting monitoring...")
            await bot.monitor_token(
                token_address=args.token,
                check_interval=args.interval,
                timeframe=args.timeframe
            )
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())