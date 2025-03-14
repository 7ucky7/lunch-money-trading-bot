# Lunch Money Trading Bot

A simple cryptocurrency trading bot designed to generate small daily profits using a grid trading strategy on Coinbase Pro (now Coinbase Advanced Trade API).

## DISCLAIMER

**USE AT YOUR OWN RISK**

This trading bot:
- Is provided for educational purposes only
- Involves substantial risk of loss
- Is not guaranteed to make profits
- Should never be used with money you cannot afford to lose
- Requires proper understanding of cryptocurrency markets and trading

## Features

- Implements a grid trading strategy that places multiple buy and sell orders at predetermined intervals
- Aims to profit from natural market volatility
- Tracks daily and total profits
- Logs all transactions and activity
- Targets a customizable daily profit goal

## Requirements

- Python 3.7+
- Coinbase Pro API credentials with trading privileges
- Sufficient funds in your Coinbase Pro account

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your Coinbase Pro API credentials:
   ```
   cp .env.example .env
   ```
4. Edit `.env` with your actual API credentials

## Configuration

You can adjust the trading parameters in `lunch_money_bot.py`:

```python
TRADING_PAIR = "BTC-USD"  # The cryptocurrency pair to trade
TARGET_PROFIT_USD = 20.00  # Target profit in USD
GRID_LEVELS = 5  # Number of buy/sell grid levels
GRID_SPREAD_PERCENTAGE = 0.5  # Percentage between grid levels
POSITION_SIZE_USD = 100.00  # Size of each position in USD
```

## Usage

Run the bot:

```
python lunch_money_bot.py
```

The bot will:
1. Set up a grid of buy and sell orders around the current market price
2. Check for filled orders every 5 minutes
3. Automatically reset the grid when necessary
4. Track profits and reset the daily counter at midnight

## Logs

All activity is logged to `trading_bot.log` with timestamps and details.

## Security Considerations

- Never share your API credentials
- Use IP whitelisting in your Coinbase Pro settings
- Consider using a dedicated Coinbase Pro account with limited funds
- Keep your API secret keys secure
- Regularly monitor the bot's activity

## License

This project is licensed under the MIT License - see the LICENSE file for details. 