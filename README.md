# 📈 Telegram Signal Bot

A fully automated **Telegram trading signal bot** that scans markets 24/7 and sends entry/exit alerts when multiple **Smart Money Concepts (SMC)** confirmations align.

## ✨ Features
- Real-time signals for:
  - **BTC/USDT** (Binance)
  - **XAUUSD** (Gold via Bybit/Alpha Vantage)
  - **GBPUSD** (via Binance proxy or external API)
- Uses **SMC strategies**: FVG, OB, Liquidity, Fibonacci zones, Breaker/Mitigation blocks, etc.
- Sends signals with **Entry, TP1, TP2, and SL** directly to Telegram.
- Multi-timeframe analysis (5m, 15m, 1h, …).
- Modular strategy support (SMC, EMA+RSI).

## ⚙️ Tech Stack
- Python 3.10+
- [CCXT](https://github.com/ccxt/ccxt) – exchange data
- [yFinance](https://github.com/ranaroussi/yfinance) / [Alpha Vantage](https://www.alphavantage.co/) – FX & Gold
- [APScheduler](https://apscheduler.readthedocs.io/) – scheduling scans
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) – Telegram integration
- Pandas – data processing

## 🚀 Installation
1. Clone repo:
   ```bash
   git clone https://github.com/yourusername/telegram-signal-bot.git
   cd telegram-signal-bot
   python -m venv .venv
   .venv\Scripts\activate   # on Windows
   pip install -r requirements.txt
   BOT_TOKEN=your_telegram_bot_token
   ALPHAVANTAGE_API_KEY=your_alpha_vantage_key
   python bot.py

And that's all.

