"""
Backtesting engine for Solana trading strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from config import config
from strategies import StrategyManager, Signal, SignalType, Position
from indicators import TechnicalIndicators


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_percent: float
    entry_reason: str
    exit_reason: str
    duration: timedelta
    fees: float = 0.0


@dataclass
class BacktestResults:
    """Backtest results container"""
    
    # Performance metrics
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_duration: timedelta
    
    # Financial metrics
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Risk metrics
    max_consecutive_losses: int
    max_consecutive_wins: int
    value_at_risk_95: float
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float
    
    # Trade history
    trades: List[Trade]
    portfolio_history: pd.DataFrame
    
    # Strategy-specific metrics
    strategy_name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    final_capital: float


class PositionManager:
    """Manages trading positions during backtesting"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = 0.0
        self.position_value = 0.0
        self.current_position = Position.NONE
        self.entry_price = None
        self.entry_time = None
        self.entry_reason = ""
        
        # Portfolio tracking
        self.portfolio_history = []
        self.trades = []
        
        # Risk management
        self.max_position_size = config.MAX_POSITION_SIZE
        self.transaction_cost = 0.001  # 0.1% transaction cost
    
    def calculate_position_size(self, price: float, signal_strength: float = 1.0) -> float:
        """Calculate position size based on current capital and risk management"""
        
        # Base position size as percentage of portfolio
        base_size = self.current_capital * self.max_position_size
        
        # Adjust by signal strength
        adjusted_size = base_size * signal_strength
        
        # Convert to token quantity
        quantity = adjusted_size / price
        
        return quantity
    
    def open_position(self, signal: Signal) -> bool:
        """Open a new position"""
        if self.current_position != Position.NONE:
            logger.warning("Attempted to open position while already in position")
            return False
        
        # Calculate position size
        quantity = self.calculate_position_size(signal.price, signal.strength)
        position_value = quantity * signal.price
        
        # Check if we have enough capital
        required_capital = position_value * (1 + self.transaction_cost)
        if required_capital > self.current_capital:
            logger.warning(f"Insufficient capital: need {required_capital}, have {self.current_capital}")
            return False
        
        # Open position
        self.position_size = quantity
        self.position_value = position_value
        self.current_position = Position.LONG
        self.entry_price = signal.price
        self.entry_time = signal.timestamp
        self.entry_reason = signal.reason
        
        # Deduct transaction costs
        self.current_capital -= position_value * self.transaction_cost
        
        logger.info(f"Opened position: {quantity:.6f} tokens at {signal.price:.6f}")
        return True
    
    def close_position(self, price: float, timestamp: pd.Timestamp, reason: str) -> Optional[Trade]:
        """Close current position and return trade record"""
        if self.current_position == Position.NONE:
            return None
        
        # Calculate PnL
        exit_value = self.position_size * price
        entry_value = self.position_size * self.entry_price
        
        # Transaction costs on exit
        transaction_costs = exit_value * self.transaction_cost
        
        # Net PnL after costs
        pnl = exit_value - entry_value - transaction_costs
        pnl_percent = (pnl / entry_value) * 100
        
        # Update capital
        self.current_capital += exit_value - transaction_costs
        
        # Create trade record
        trade = Trade(
            entry_time=self.entry_time,
            exit_time=timestamp,
            entry_price=self.entry_price,
            exit_price=price,
            quantity=self.position_size,
            side="long",
            pnl=pnl,
            pnl_percent=pnl_percent,
            entry_reason=self.entry_reason,
            exit_reason=reason,
            duration=timestamp - self.entry_time,
            fees=transaction_costs
        )
        
        self.trades.append(trade)
        
        # Reset position
        self.position_size = 0.0
        self.position_value = 0.0
        self.current_position = Position.NONE
        self.entry_price = None
        self.entry_time = None
        self.entry_reason = ""
        
        logger.info(f"Closed position: PnL {pnl:.2f} ({pnl_percent:.2f}%)")
        return trade
    
    def update_portfolio_value(self, current_price: float, timestamp: pd.Timestamp):
        """Update portfolio value with current market price"""
        if self.current_position == Position.LONG:
            current_position_value = self.position_size * current_price
            total_value = self.current_capital + current_position_value
        else:
            total_value = self.current_capital
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'capital': self.current_capital,
            'position_value': self.position_value if self.current_position == Position.LONG else 0,
            'total_value': total_value,
            'position': self.current_position.value,
            'returns': (total_value - self.initial_capital) / self.initial_capital
        })
    
    def get_portfolio_dataframe(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame"""
        return pd.DataFrame(self.portfolio_history).set_index('timestamp')


class SolanaBacktester:
    """Main backtesting engine for Solana trading strategies"""
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.strategy_manager = StrategyManager()
        self.position_manager = None
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_name: str = "hybrid",
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> BacktestResults:
        """Run backtest on historical data"""
        
        logger.info(f"Starting backtest with {strategy_name} strategy")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) < 100:
            raise ValueError("Insufficient data for backtesting")
        
        # Set active strategy
        self.strategy_manager.set_active_strategy(strategy_name)
        
        # Initialize position manager
        self.position_manager = PositionManager(self.initial_capital)
        
        # Add technical indicators if not present
        if 'rsi' not in data.columns:
            data = TechnicalIndicators.calculate_all_indicators(data)
        
        # Main backtesting loop
        for i in range(50, len(data)):  # Start after enough data for indicators
            current_data = data.iloc[:i+1]
            current_row = data.iloc[i]
            current_price = current_row['close']
            current_time = current_row.name
            
            # Check for exit conditions first
            if self.position_manager.current_position != Position.NONE:
                should_exit = self.strategy_manager.should_exit(current_data)
                
                # Check stop loss and take profit
                strategy = self.strategy_manager.strategies[strategy_name]
                if strategy.should_exit(current_data, current_price):
                    should_exit = True
                
                if should_exit:
                    self.position_manager.close_position(
                        current_price, 
                        current_time, 
                        "Strategy exit signal"
                    )
            
            # Look for entry signals
            if self.position_manager.current_position == Position.NONE:
                signal = self.strategy_manager.get_signal(current_data)
                
                if signal and signal.signal_type == SignalType.BUY:
                    success = self.position_manager.open_position(signal)
                    if success:
                        self.strategy_manager.update_position(signal)
            
            # Update portfolio value
            self.position_manager.update_portfolio_value(current_price, current_time)
        
        # Close any remaining position
        if self.position_manager.current_position != Position.NONE:
            final_price = data.iloc[-1]['close']
            final_time = data.index[-1]
            self.position_manager.close_position(
                final_price, 
                final_time, 
                "End of backtest"
            )
        
        # Calculate results
        results = self._calculate_results(data, strategy_name)
        
        logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
        return results
    
    def _calculate_results(self, data: pd.DataFrame, strategy_name: str) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        portfolio_df = self.position_manager.get_portfolio_dataframe()
        trades = self.position_manager.trades
        
        if portfolio_df.empty:
            raise ValueError("No portfolio history available")
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Time-based metrics
        start_date = portfolio_df.index[0]
        end_date = portfolio_df.index[-1]
        trading_days = (end_date - start_date).days
        annual_return = ((1 + total_return) ** (365 / trading_days)) - 1 if trading_days > 0 else 0
        
        # Drawdown calculation
        portfolio_df['cummax'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['total_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Risk metrics
        returns = portfolio_df['returns'].pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            value_at_risk_95 = np.percentile(returns, 5)
        else:
            sharpe_ratio = sortino_ratio = value_at_risk_95 = 0.0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        if trades:
            winning_trades = sum(1 for t in trades if t.pnl > 0)
            losing_trades = sum(1 for t in trades if t.pnl < 0)
            win_rate = winning_trades / len(trades)
            
            avg_trade_duration = sum([t.duration for t in trades], timedelta()) / len(trades)
            total_pnl = sum(t.pnl for t in trades)
            
            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [abs(t.pnl) for t in trades if t.pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float('inf')
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Consecutive wins/losses
            consecutive_wins = consecutive_losses = 0
            max_consecutive_wins = max_consecutive_losses = 0
            current_streak = 0
            streak_type = None
            
            for trade in trades:
                if trade.pnl > 0:
                    if streak_type == 'win':
                        current_streak += 1
                    else:
                        current_streak = 1
                        streak_type = 'win'
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    if streak_type == 'loss':
                        current_streak += 1
                    else:
                        current_streak = 1
                        streak_type = 'loss'
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            # No trades executed
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0
            total_pnl = 0
            avg_trade_duration = timedelta()
            max_consecutive_wins = max_consecutive_losses = 0
        
        # Benchmark comparison (SOL buy-and-hold)
        benchmark_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        alpha = total_return - benchmark_return
        
        # Beta calculation (simplified)
        if len(returns) > 1:
            benchmark_returns = data['close'].pct_change().dropna()
            if len(benchmark_returns) > 1 and len(returns) == len(benchmark_returns):
                beta = np.cov(returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
            else:
                beta = 1.0
        else:
            beta = 1.0
        
        return BacktestResults(
            # Performance metrics
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            
            # Trade statistics
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_trade_duration=avg_trade_duration,
            
            # Financial metrics
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            
            # Risk metrics
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            value_at_risk_95=value_at_risk_95,
            
            # Benchmark comparison
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            
            # Data
            trades=trades,
            portfolio_history=portfolio_df,
            
            # Metadata
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_value,
            final_capital=final_value
        )
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 365)  # Daily risk-free rate
        return excess_returns / returns.std() * np.sqrt(365) if returns.std() > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns.mean() - (risk_free_rate / 365)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        return excess_returns / downside_std * np.sqrt(365) if downside_std > 0 else float('inf')
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[str] = None) -> pd.DataFrame:
        """Compare multiple strategies"""
        if strategies is None:
            strategies = list(self.strategy_manager.strategies.keys())
        
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(data, strategy)
                results.append({
                    'Strategy': strategy,
                    'Total Return': result.total_return,
                    'Annual Return': result.annual_return,
                    'Max Drawdown': result.max_drawdown,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Win Rate': result.win_rate,
                    'Total Trades': result.total_trades,
                    'Profit Factor': result.profit_factor,
                    'Alpha': result.alpha
                })
            except Exception as e:
                logger.error(f"Error backtesting {strategy}: {e}")
        
        return pd.DataFrame(results)
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """Create comprehensive visualization of backtest results"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Portfolio Value Over Time',
                'Drawdown',
                'Trade P&L Distribution',
                'Monthly Returns Heatmap',
                'Cumulative Returns vs Benchmark',
                'Trade Duration Distribution'
            ],
            specs=[
                [{"secondary_y": True}, {}],
                [{}, {}],
                [{}, {}]
            ]
        )
        
        portfolio_df = results.portfolio_history
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['total_value'],
                name='Portfolio Value',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['drawdown'] * 100,
                name='Drawdown (%)',
                fill='tonexty',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # Trade P&L distribution
        if results.trades:
            pnl_values = [trade.pnl for trade in results.trades]
            fig.add_trace(
                go.Histogram(
                    x=pnl_values,
                    name='Trade P&L',
                    nbinsx=20
                ),
                row=2, col=1
            )
            
            # Trade duration distribution
            durations = [trade.duration.total_seconds() / 3600 for trade in results.trades]  # Hours
            fig.add_trace(
                go.Histogram(
                    x=durations,
                    name='Trade Duration (Hours)',
                    nbinsx=20
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Backtest Results - {results.strategy_name}",
            height=1200,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def generate_report(self, results: BacktestResults) -> str:
        """Generate a comprehensive text report"""
        
        report = f"""
SOLANA TRADING STRATEGY BACKTEST REPORT
=====================================

Strategy: {results.strategy_name}
Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}
Initial Capital: ${results.initial_capital:,.2f}
Final Capital: ${results.final_capital:,.2f}

PERFORMANCE METRICS
------------------
Total Return: {results.total_return:.2%}
Annual Return: {results.annual_return:.2%}
Benchmark Return (SOL): {results.benchmark_return:.2%}
Alpha: {results.alpha:.2%}
Beta: {results.beta:.2f}

RISK METRICS
-----------
Maximum Drawdown: {results.max_drawdown:.2%}
Sharpe Ratio: {results.sharpe_ratio:.2f}
Sortino Ratio: {results.sortino_ratio:.2f}
Calmar Ratio: {results.calmar_ratio:.2f}
Value at Risk (95%): {results.value_at_risk_95:.2%}

TRADE STATISTICS
---------------
Total Trades: {results.total_trades}
Winning Trades: {results.winning_trades}
Losing Trades: {results.losing_trades}
Win Rate: {results.win_rate:.2%}
Average Trade Duration: {results.avg_trade_duration}

FINANCIAL METRICS
----------------
Total P&L: ${results.total_pnl:,.2f}
Average Win: ${results.avg_win:,.2f}
Average Loss: ${results.avg_loss:,.2f}
Profit Factor: {results.profit_factor:.2f}
Expectancy: ${results.expectancy:,.2f}

CONSECUTIVE TRADES
-----------------
Max Consecutive Wins: {results.max_consecutive_wins}
Max Consecutive Losses: {results.max_consecutive_losses}

SUMMARY
-------
"""
        
        if results.total_return > results.benchmark_return:
            report += f"✅ Strategy outperformed SOL by {results.alpha:.2%}\n"
        else:
            report += f"❌ Strategy underperformed SOL by {abs(results.alpha):.2%}\n"
        
        if results.win_rate > 0.5:
            report += f"✅ Good win rate of {results.win_rate:.2%}\n"
        else:
            report += f"⚠️  Low win rate of {results.win_rate:.2%}\n"
        
        if results.sharpe_ratio > 1.0:
            report += f"✅ Excellent Sharpe ratio of {results.sharpe_ratio:.2f}\n"
        elif results.sharpe_ratio > 0.5:
            report += f"✅ Good Sharpe ratio of {results.sharpe_ratio:.2f}\n"
        else:
            report += f"⚠️  Poor Sharpe ratio of {results.sharpe_ratio:.2f}\n"
        
        if abs(results.max_drawdown) < 0.1:
            report += f"✅ Low maximum drawdown of {results.max_drawdown:.2%}\n"
        elif abs(results.max_drawdown) < 0.2:
            report += f"⚠️  Moderate maximum drawdown of {results.max_drawdown:.2%}\n"
        else:
            report += f"❌ High maximum drawdown of {results.max_drawdown:.2%}\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    from data_fetcher import get_token_data
    import asyncio
    
    async def test_backtest():
        # Get sample data
        token_address = config.TARGET_TOKEN
        df = await get_token_data(token_address, timeframe="5m", days_back=30)
        
        if len(df) > 100:
            # Run backtest
            backtester = SolanaBacktester(initial_capital=1000)
            
            # Test single strategy
            results = backtester.run_backtest(df, "hybrid")
            
            # Generate report
            report = backtester.generate_report(results)
            print(report)
            
            # Compare all strategies
            comparison = backtester.compare_strategies(df)
            print("\nStrategy Comparison:")
            print(comparison.to_string(index=False))
            
        else:
            print("Insufficient data for backtesting")
    
    # Run the test
    asyncio.run(test_backtest())