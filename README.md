# Kraken Sentinel V15.0

**Build Once, Evolve Forever** – A production-ready, autonomous cryptocurrency trading system for Kraken exchange.

## Overview

Kraken Sentinel is a fully autonomous trading bot that executes long and short positions on spot markets using two proprietary strategies:
- **Approach B (Momentum)**: High-frequency directional trades (2-6h holds)
- **Approach C (Swing)**: Swing trading on technical patterns (4-24h holds)

The system validates all strategies using **combined ICIR + Walk-Forward validation**, sizes positions via **Kelly Criterion**, manages risk with **ATR-based stops**, and evolves parameters autonomously via **genetic algorithm**.

## Key Features

✅ **Spot & Margin Ready**: Phase 1 launches spot-only; Phase 2 toggles margin/futures  
✅ **Game Theory Embedded**: Liquidity filters, funding rate validation, manipulation detection  
✅ **Self-Evolving**: ICIR-driven parameter mutation; no manual tuning required  
✅ **Multi-Factor Asset Selection**: 12-factor scoring model for asset ranking  
✅ **LLM Strategist**: Groq + OpenRouter for failure analysis and market context  
✅ **Production Dashboard**: Motion graphics, real-time P&L, equity curve, ICIR tracking  
✅ **Firebase Native**: Cloud Functions, Firestore, Scheduler, Hosting—100% serverless  
✅ **Zero Cost**: $0/month (no billing card required)  

## Tech Stack

- **Runtime**: Python 3.11 (Cloud Functions)
- **Database**: Firestore (NoSQL)
- **Scheduler**: Google Cloud Scheduler
- **Hosting**: Firebase Hosting
- **Exchanges**: Kraken REST API (free, no key required for public data)
- **LLM**: Groq API (free tier) + OpenRouter (failover)
- **Data**: CoinPaprika, CoinMarketCap, alternative.me, Iturri API

## Architecture
Cloud Scheduler (triggers)
↓
Cloud Functions (Python)
↓
Firestore (state + trades + ICIR)
↓
Kraken REST API (execution)
↓
Firebase Hosting (dashboard)
## Strategies

### Approach B (Momentum)
- **Long Entry**: Breakout + Volume Spike + Momentum
- **Short Entry**: Exhaustion + Structure Break + Funding Rate
- **Hold Time**: Long 2-6 hours / Short 1-4 hours
- **Exit**: Trailing Stop (2x ATR)

### Approach C (Swing)
- **Long Entry**: MA50 + RSI > 60 + Volume > 1.5x
- **Short Entry**: MA50 + RSI < 40 + Volume > 1.2x
- **Hold Time**: Long 4-24 hours / Short 2-12 hours
- **Exit**: MA20 Cross + RSI confirmation

## Validation

All strategies pass **combined ICIR + Walk-Forward validation**:
- ICIR > 0.2 on in-sample data ✅
- ICIR > 0.2 on out-of-sample data ✅
- ICIR consistency across 5+ rolling windows ✅
- Profit factor > 1.2 ✅

## Risk Management

- **Stop-Loss**: ATR-based (2x ATR), dynamic with volatility
- **Position Sizing**: Kelly Criterion (max 10% of pool)
- **Daily Loss Limit**: -5% → freezes strategy for 24h
- **Max Drawdown**: -10% → halves position sizes
- **Circuit Breakers**: ICIR < 0.2 for 3 days → freeze
- **Max Open Positions**: 2 per strategy

## Asset Selection

Multi-factor scoring model with 12 factors:
- **Momentum** (35%): Price change, RSI, MA50 distance
- **Volume** (25%): Volume spike, 24h USD volume
- **Sentiment** (15%): Social velocity, polarity
- **Technical** (15%): ADX, Bollinger Bands
- **Microstructure** (10%): Order Book Imbalance

## Capital Deployment

Phase 1 launches with $300 ($150 per strategy). Scales to $1,390 via 7-phase deployment:
- Phase 1: $300
- Phase 2-6: +$200 each
- Phase 7: +$90

Scaling gates: ICIR > 0.2, profitability > 5% per month, no loss limit breaches.

## Game Theory

Every decision is game-theoretically sound:
- **Liquidity validation**: $1M+ 24h volume required
- **Volume confirmation**: 2x average volume
- **Sentiment filters**: 50%+ positive mention ratio
- **Funding rate checks**: Detect over-leverage
- **HMM regime classification**: Avoid manipulated markets
- **Manipulation detection**: LLM Strategist analysis

## Data Sources (All Free)

| Data Type | Source | Cost |
|-----------|--------|------|
| OHLCV, Trades, Order Book | Kraken Public API | Free, no key |
| Market data | CoinPaprika / CoinMarketCap | Free, no key |
| Funding rates | agent-data-api-mcp | Free tool |
| Market regime | jarvis-market-signals | Free tier |
| Fear & Greed | alternative.me | Free, no key |
| Social sentiment | cryptocurrency.cv | Free, no key |
| Macro (VIX, DXY) | Iturri API | Free |

## Deployment

### Firebase Setup (No Billing Card)
1. Create Firebase project
2. Enable Firestore (Native mode)
3. Enable Cloud Functions (Python 3.11)
4. Enable Cloud Scheduler
5. Enable Firebase Hosting
6. Deploy via `firebase deploy`

### CI/CD
GitHub Actions auto-deploys on push to `main` branch after passing tests.

## Monitoring

- **Telegram Commands**: `/status`, `/kill`, `/resume`, `/report`
- **Real-time Notifications**: FCM on trades
- **Daily Summaries**: ICIR + P&L via LLM Strategist
- **Dashboard**: Live equity curve, ICIR tracking, open positions

## Project Status

| Aspect | Status |
|--------|--------|
| Version | V15.0 (Ultimate Final Locked) |
| Build Philosophy | Complete feature set built upfront |
| Cost | $0/month |
| Build Time | 4-6 weeks |
| Trading Mode | Spot (Phase 1), Margin toggle (Phase 2) |

## License

MIT License – See [LICENSE](LICENSE) file for details.

## Author

Belinda

---

**Build Once, Evolve Forever** 🚀
