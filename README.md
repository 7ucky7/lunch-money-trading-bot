# Espero Trading Bot

A powerful, modular cryptocurrency trading bot with support for traditional technical analysis strategies and machine learning models.

## Features

- **Real-Time Data Streaming**: WebSocket connections for live market data
- **Multiple Strategy Support**: Combine technical and machine learning approaches
- **Position Management**: Automatic tracking of open positions and P&L
- **Risk Management**: Configurable risk limits and position sizing
- **Backtesting Engine**: Test strategies on historical data
- **Paper Trading Mode**: Practice without risking real funds
- **Customizable Configuration**: Tailor the bot to your trading style

## Architecture

The bot is built with a modular architecture that separates concerns and allows for easy extension:

- **Data Manager**: Handles data collection, processing, and storage
- **Strategy Framework**: Provides a common interface for implementing trading strategies
- **Position Manager**: Tracks and manages trading positions
- **Order Validator**: Ensures trades comply with exchange and risk rules
- **Trading Bot**: Orchestrates all components and manages the trading cycle

## Included Strategies

The bot comes with several built-in strategies:

1. **Moving Average Crossover**: Generates signals based on MA crossovers
2. **RSI**: Identifies overbought and oversold conditions
3. **Bollinger Bands**: Trades mean reversion and breakouts
4. **MACD**: Uses MACD histogram reversals and zero-line crossovers
5. **ML Strategy** (optional): Combines LSTM, Random Forest, and Gradient Boosting models

## Installation

### Prerequisites

- Python 3.8+
- Coinbase API credentials

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/espero-trading-bot.git
   cd espero-trading-bot
   ```

2. Run the setup script to create a virtual environment and install dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

4. Create a `config.json` file with your API credentials and trading parameters (see Configuration section below).

## Configuration

The bot is configured using a JSON file. Here's an example configuration:

```json
{
  "api_key": "YOUR_API_KEY",
  "api_secret": "YOUR_API_SECRET",
  "environment": "sandbox",
  "products": ["BTC-USD", "ETH-USD"],
  "cycle_interval": 60,
  "min_trade_confidence": 0.65,
  "allow_multiple_positions": false,
  "risk": {
    "risk_per_trade": 0.02,
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.1,
    "max_open_positions": 5,
    "max_daily_drawdown": 0.05
  },
  "signal_weights": {
    "MACrossoverStrategy": 1.0,
    "RSIStrategy": 1.0,
    "BollingerBandsStrategy": 1.0,
    "MACDStrategy": 1.0,
    "MLStrategy": 2.0
  },
  "strategies": {
    "ma_crossover": {
      "short_period": 20,
      "long_period": 50,
      "signal_threshold": 0.5,
      "timeframes": ["1h", "4h", "1d"]
    },
    "rsi": {
      "rsi_period": 14,
      "overbought_threshold": 70,
      "oversold_threshold": 30,
      "signal_threshold": 0.5,
      "timeframes": ["1h", "4h"]
    },
    "bollinger_bands": {
      "period": 20,
      "std_dev": 2.0,
      "signal_threshold": 0.5,
      "timeframes": ["1h", "4h", "1d"]
    },
    "macd": {
      "fast_period": 12,
      "slow_period": 26,
      "signal_period": 9,
      "signal_threshold": 0.5,
      "timeframes": ["1h", "4h", "1d"]
    },
    "ml_strategy": {
      "lstm_lookback": 30,
      "forest_max_depth": 10,
      "gradient_boosting_n_estimators": 100,
      "use_ensemble": true,
      "timeframes": ["1h", "4h", "1d"]
    }
  },
  "backtest": {
    "initial_balance": 10000,
    "fee_rate": 0.001,
    "slippage": 0.0005
  }
}
```

## Usage

### Paper Trading Mode

Run the bot in paper trading mode (no real trades):

```
python trading_bot.py --config config.json --paper
```

### Live Trading Mode

Run the bot with real trading (use with caution):

```
python trading_bot.py --config config.json --live
```

The bot will prompt for confirmation before starting live trading.

### Backtesting

Test strategies using historical data:

```
python backtesting.py --config config.json --product BTC-USD --start 2023-01-01 --end 2023-12-31 --plot
```

Additional options:
- `--plot`: Generate a performance chart
- `--save results.json`: Save detailed results to a file

## Testing Components

### Test Data Manager

```
python test_data_manager.py
```

### Test Position Manager

```
python test_position_manager.py
```

### Test Order Validator

```
python test_order_validator.py
```

## Creating Custom Strategies

You can create custom strategies by extending the `Strategy` class:

```python
from strategy_framework import Strategy, Signal, SignalType, TimeFrame

class MyCustomStrategy(Strategy):
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize your strategy parameters
        
    async def generate_signals(self, data, product_id):
        signals = []
        # Implement your signal generation logic
        return signals
```

Register your strategy in the trading bot:

```python
from my_custom_strategy import MyCustomStrategy

# In the _register_strategies method:
self.strategy_registry.register(
    MyCustomStrategy(strategy_configs.get('my_custom_strategy', {}))
)
```

## Important Notes

- **API Rate Limits**: Be aware of exchange rate limits
- **Risk Management**: Always use proper risk management settings
- **Testing**: Thoroughly test in paper trading mode before using real funds
- **Security**: Keep your API credentials secure

## Disclaimer

Trading cryptocurrencies involves significant risk. This software is provided for educational and research purposes only. Use at your own risk.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Coinbase for providing the API
- All the open-source libraries that made this project possible 