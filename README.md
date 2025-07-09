# Solana Automated Trading Bot & Backtesting Framework

A comprehensive automated trading bot and backtesting framework specifically designed for Solana tokens. This project provides multiple trading strategies, technical analysis, and robust backtesting capabilities for the Solana ecosystem.

## 🚀 Features

- **Multiple Trading Strategies**:
  - Mean Reversion (RSI + Bollinger Bands)
  - Trend Following (EMA crossovers + MACD)
  - Breakout Strategy (Volume + Price action)
  - Hybrid Strategy (Combines all approaches)

- **Technical Analysis**:
  - 20+ Technical indicators (RSI, MACD, Bollinger Bands, EMAs, etc.)
  - Volume analysis
  - Support/Resistance levels
  - Custom crypto-specific indicators

- **Comprehensive Backtesting**:
  - Historical performance analysis
  - Risk metrics (Sharpe ratio, Sortino ratio, max drawdown)
  - Trade statistics and P&L analysis
  - Strategy comparison tools
  - Interactive charts and visualizations

- **Data Sources**:
  - Solana Tracker API integration
  - Birdeye API support
  - Local caching for performance
  - Real-time and historical data

- **Risk Management**:
  - Stop-loss and take-profit orders
  - Position sizing
  - Maximum drawdown limits
  - Consecutive loss protection

## 📋 Prerequisites

- Python 3.9+
- Node.js (for some dependencies)
- Git

## 🛠 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd solana-trading-bot
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install TA-Lib** (required for technical analysis):

On Ubuntu/Debian:
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

On macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

On Windows:
- Download TA-Lib from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- Install with pip: `pip install <downloaded-file>`

4. **Configure environment** (optional):

Create a `.env` file in the project root:
```env
# API Configuration
API_KEY=your_solana_tracker_api_key_here
DATA_PROVIDER=solana_tracker

# Trading Configuration
TARGET_TOKEN=8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm
INITIAL_CAPITAL=1000.0
MAX_POSITION_SIZE=0.1
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.15

# Risk Management
MAX_DRAWDOWN=0.20
MAX_CONSECUTIVE_LOSSES=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=trading_bot.log
```

## 🎯 Usage

The bot provides several commands for different operations:

### 1. Backtesting

Run a backtest on your token with a specific strategy:

```bash
python main.py backtest --token 8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm --strategy hybrid --days 30
```

**Options**:
- `--token`: Token address to backtest
- `--strategy`: Trading strategy (`mean_reversion`, `trend_following`, `breakout`, `hybrid`)
- `--timeframe`: Data timeframe (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`)
- `--days`: Days of historical data
- `--capital`: Initial capital amount
- `--no-plots`: Skip generating charts
- `--no-save`: Skip saving results

### 2. Strategy Comparison

Compare all strategies on the same token:

```bash
python main.py compare --token 8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm --days 30
```

### 3. Token Analysis

Get current technical analysis and trading signals:

```bash
python main.py analyze --token 8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm
```

### 4. Real-time Monitoring

Monitor a token for trading signals:

```bash
python main.py monitor --token 8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm --interval 60
```

## 📊 Example Output

### Backtest Report
```
SOLANA TRADING STRATEGY BACKTEST REPORT
=====================================

Strategy: hybrid
Period: 2024-01-01 to 2024-01-30
Initial Capital: $1,000.00
Final Capital: $1,250.00

PERFORMANCE METRICS
------------------
Total Return: 25.00%
Annual Return: 127.45%
Benchmark Return (SOL): 15.30%
Alpha: 9.70%
Beta: 0.85

RISK METRICS
-----------
Maximum Drawdown: -8.50%
Sharpe Ratio: 1.85
Sortino Ratio: 2.10
Calmar Ratio: 15.00
Value at Risk (95%): -2.15%

TRADE STATISTICS
---------------
Total Trades: 45
Winning Trades: 28
Losing Trades: 17
Win Rate: 62.22%
Average Trade Duration: 4:30:00

SUMMARY
-------
✅ Strategy outperformed SOL by 9.70%
✅ Good win rate of 62.22%
✅ Excellent Sharpe ratio of 1.85
✅ Low maximum drawdown of -8.50%
```

### Token Analysis
```
TOKEN ANALYSIS: 8Cd7wXoPb5Yt9cUGtmHNqAEmpMDrhfcVqnGbLC48b8Qm
================================================================================
Analysis Time: 2024-01-30 15:30:00
Latest Price: $0.00001234
24h Change: +5.67%
Volume Ratio: 2.15x

TECHNICAL INDICATORS
----------------------------------------
RSI: 65.43 (neutral)
MACD: bullish
BB Position: 0.75 (neutral)
MA Trend: bullish
Overall Signal: bullish

STRATEGY SIGNALS
----------------------------------------
MEAN_REVERSION: No signal

TREND_FOLLOWING:
  Signal: BUY
  Strength: 0.72
  Reason: EMA bullish crossover, MACD bullish, Price above EMAs
  Stop Loss: $0.00001172
  Take Profit: $0.00001420

BREAKOUT: No signal

HYBRID:
  Signal: BUY
  Strength: 0.68
  Reason: TF: EMA bullish crossover, MACD bullish, Price above EMAs
  Stop Loss: $0.00001172
  Take Profit: $0.00001420
```

## 🏗 Project Structure

```
solana-trading-bot/
├── config.py              # Configuration settings
├── data_fetcher.py         # Data fetching from APIs
├── indicators.py           # Technical indicators
├── strategies.py           # Trading strategies
├── backtester.py          # Backtesting engine
├── main.py                # Main application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .env                   # Environment variables (optional)
```

## 🔧 Configuration

Key configuration parameters in `config.py`:

- **TARGET_TOKEN**: The Solana token address to trade
- **INITIAL_CAPITAL**: Starting capital for backtesting
- **MAX_POSITION_SIZE**: Maximum position size as % of portfolio
- **STOP_LOSS_PCT**: Stop loss percentage
- **TAKE_PROFIT_PCT**: Take profit percentage
- **RSI_OVERSOLD/OVERBOUGHT**: RSI thresholds
- **EMA_FAST/SLOW**: EMA periods
- **DATA_PROVIDER**: Data source (solana_tracker, birdeye)

## 📈 Strategies Explained

### 1. Mean Reversion Strategy
- Uses RSI and Bollinger Bands
- Buys when oversold, sells when overbought
- Best for ranging markets

### 2. Trend Following Strategy
- Uses EMA crossovers and MACD
- Follows momentum and trends
- Best for trending markets

### 3. Breakout Strategy
- Uses volume and price action
- Trades breakouts from consolidation
- Best for volatile markets

### 4. Hybrid Strategy
- Combines all three approaches
- Weighted signal aggregation
- Adaptive to different market conditions

## 📊 Performance Metrics

The backtester calculates comprehensive metrics:

- **Return Metrics**: Total return, annual return, alpha, beta
- **Risk Metrics**: Sharpe ratio, Sortino ratio, max drawdown, VaR
- **Trade Statistics**: Win rate, profit factor, expectancy
- **Duration Analysis**: Average trade duration, consecutive wins/losses

## 🔌 API Integration

### Solana Tracker API
- Real-time and historical price data
- Trading volume and market data
- Token information and metadata

### Birdeye API
- Alternative data source
- Price history and market data
- Token analytics

## 🚨 Risk Disclaimer

**IMPORTANT**: This is educational software for learning purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Never trade with money you cannot afford to lose.

- This software is provided "as is" without warranty
- The authors are not responsible for any trading losses
- Always do your own research before making trading decisions
- Consider paper trading before using real funds

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter issues:

1. Check the logs in `trading_bot.log`
2. Ensure all dependencies are installed correctly
3. Verify your API keys are configured
4. Check network connectivity for data fetching

## 🚀 Future Enhancements

- [ ] Live trading integration with Jupiter/Raydium
- [ ] Machine learning signal generation
- [ ] Multi-token portfolio management
- [ ] Web dashboard interface
- [ ] Mobile notifications
- [ ] Advanced risk management
- [ ] Paper trading mode
- [ ] More data providers
- [ ] Social trading features

## 📚 Additional Resources

- [Solana Documentation](https://docs.solana.com/)
- [Technical Analysis Library](https://github.com/bukosabino/ta)
- [Solana Tracker API](https://solanatracker.io/api)
- [Jupiter Exchange](https://jup.ag/)

---

**Happy Trading! 🚀**

*Remember: Past performance is not indicative of future results. Trade responsibly.*