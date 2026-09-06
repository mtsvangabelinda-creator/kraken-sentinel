`docs/architecture.md`** — Complete system architecture documentation.

Copy this directly into GitHub web UI as `docs/architecture.md`:

```markdown
# Kraken Sentinel V15.0 - System Architecture

**Version:** 15.0  
**Last Updated:** September 2026  
**Author:** Belinda  
**License:** MIT  

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Layers](#architecture-layers)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Firestore Schema](#firestore-schema)
6. [Strategy Mechanics](#strategy-mechanics)
7. [Risk Management](#risk-management)
8. [Evolution Engine](#evolution-engine)
9. [LLM Integration](#llm-integration)
10. [Dashboard Architecture](#dashboard-architecture)
11. [Deployment Model](#deployment-model)

---

## System Overview

Kraken Sentinel V15.0 is a **fully autonomous cryptocurrency trading bot** running on Firebase Cloud Functions with Firestore for real-time state management. It executes two parallel strategies (Momentum & Swing) on Kraken spot markets with zero-cost LLM decision support.

### Key Attributes

| Aspect | Detail |
|--------|--------|
| **Platform** | Firebase Cloud Functions (Python 3.11) |
| **Database** | Firestore (real-time, event-driven) |
| **Exchange** | Kraken REST API (spot trading, Phase 1) |
| **Strategies** | Momentum (2-6h holds) + Swing (4-24h holds) |
| **Capital** | $300 Phase 1 ($150/strategy), scales to $1,390 over 7 phases |
| **LLM** | Groq (Llama 3.3 70B) + OpenRouter failover |
| **Risk Model** | ATR stops (2x), Kelly Criterion (25% conservative), daily loss -5%, max drawdown -10% |
| **Validation** | ICIR (>0.2) + Walk-Forward (Spearman correlation) |
| **Dashboard** | Real-time Firestore updates, Chart.js, cyberpunk UI |
| **Cost** | $0/month (free tiers: Kraken, CoinPaprika, alternative.me, Groq) |

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                  PRESENTATION LAYER                      │
│  Dashboard (HTML/CSS/JS) - Real-time Firestore updates  │
└────────────────────┬────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────┐
│                 DATA AGGREGATION LAYER                    │
│  Asset Scoring | Regime Detection | Circuit Breakers    │
└────────────────────┬────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────┐
│              DECISION ENGINE LAYER                        │
│  Momentum Scan | Swing Scan | LLM Strategist             │
│  Signal Generation | Confidence Scoring                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────┐
│            EXECUTION & RISK LAYER                        │
│  Entry/Exit Logic | ATR Stops | Kelly Sizing             │
│  Position Lifecycle | Margin Toggle                      │
└────────────────────┬────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────┐
│              EVOLUTION LAYER                              │
│  Genetic Algorithm | Parameter Optimization              │
│  Validation Reports | Fitness Tracking                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌─────────────────────▼────────────────────────────────────┐
│           PERSISTENCE & API LAYER                         │
│  Firestore Collections | Kraken API | Telegram Bot      │
└─────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. **Data Ingestion** (`fetch_ohlcv.py`)

Fetches historical and real-time OHLCV from Kraken REST API.

```
Kraken REST → OHLCV (1h bars) → Firestore (ohlcv_data)
└─ Pairs: BTC/USD, ETH/USD, SOL/USD, etc.
└─ History: 500 bars (500h ≈ 20 days)
└─ Update: On Cloud Scheduler trigger (hourly)
```

**Key Functions:**
- `fetch_kraken_ohlcv(pair, timeframe, start, end)` → DataFrame
- Caches in Firestore for efficiency
- Handles API rate limits gracefully

### 2. **Asset Scoring** (`asset_scoring.py`)

Multi-factor scoring system for pair selection.

```
Inputs:
├─ Volatility (14-day ATR ÷ close)
├─ Volume (24h / 90-day avg)
├─ Momentum (z-score of returns)
├─ Trend Strength (price vs MA50)
└─ LLM Sentiment (from strategist)

Output: composite_score (0-1) for each pair
Storage: Firestore (asset_scores collection)
```

**Calculation:**
```python
composite_score = (
    0.25 * norm_volatility +
    0.20 * norm_volume +
    0.25 * norm_momentum +
    0.15 * norm_trend +
    0.15 * llm_sentiment
)
```

### 3. **Regime Detection** (`regime_detection.py`)

Hidden Markov Model (5 states) for market context.

```
States:
├─ Trend (strong directional move, low volatility)
├─ Choppy (bidirectional chop, medium volatility)
├─ Ranging (tight bounds, low volatility)
├─ Vol-Low (calm, ready to break)
└─ Vol-High (chaotic, high risk)

Inputs: Close prices, ATR, returns volatility
Output: current_state, confidence (0-1), state_probabilities
Storage: Firestore (regime_history)
Update: Hourly with new bar
```

**Viterbi Algorithm** for state inference.

### 4. **Momentum Strategy** (`scan_momentum.py`)

Z-score breakout + RSI + volume spike.

```
Entry Signal:
├─ Z-Score > 2.0 (price 2σ above MA)
├─ RSI 40-70 (not overbought, room to run)
├─ Volume spike > 1.5x (confirmation)
└─ Market regime: NOT Ranging (needs breakout room)

Confidence: min(abs(z_score) / 3.0, 1.0)
Hold Time: 2-6 hours (timeout exit)
```

**Exit Logic:**
- Stop Loss: Entry - 2×ATR
- Take Profit: Entry + 2×(Entry - Stop)
- Timeout: 6 hours elapsed

### 5. **Swing Strategy** (`generate_swing_signals.py`)

Mean reversion + moving average + RSI.

```
Entry Signal:
├─ Price > MA50 (above trend, long setup)
├─ RSI 40-70 (pullback but not overbought)
└─ OR Price < MA50 (short setup)
└─ RSI 30-60 (pullback but not oversold)

Confidence: min((price - ma50) / (ma50 * 0.05), 1.0)
Hold Time: 4-24 hours long, 2-12 hours short
```

**Exit Logic:**
- Stop Loss: Entry ± 2×ATR (directional)
- Take Profit: 2× risk (1:2 risk/reward)
- Timeout: 24h long, 12h short

### 6. **Execution Engine** (`execute_trade.py`)

Order placement + pre-flight checks.

```
Pre-Flight Checks:
├─ Circuit breaker status (daily loss, max DD)
├─ Capital available (Kelly sized)
├─ Margin requirements met
└─ Signal confidence > threshold (0.5)

Order Placement:
├─ Type: Limit order (1% above/below market)
├─ Timeout: 5 minutes (cancel if unfilled)
├─ Slippage budget: 0.1% (price safety)
└─ Dry-run mode: Simulate without executing

Position Tracking:
├─ Store in Firestore (open_positions)
├─ Calculate unrealized P&L
├─ Monitor stops and targets
└─ Update equity in real-time
```

### 7. **Position Management** (`close_trade.py`)

Lifecycle management: entry → monitoring → exit.

```
Monitoring Loop (runs every 5 min):
1. Fetch current price
2. Check if stop loss hit → Execute at SL
3. Check if take profit hit → Execute at TP
4. Check if timeout reached → Exit at market
5. Update unrealized P&L
6. Log to trade_history

On Close:
├─ Realized P&L = exit_price - entry_price
├─ Return % = (exit_price - entry_price) / entry_price * 100
├─ Store in trade_history
├─ Update equity curve
└─ Notify via Telegram
```

### 8. **Risk Manager** (`risk_manager.py`)

Position sizing + circuit breakers.

```
Position Sizing (Kelly Criterion):
1. Calculate fair Kelly: (win_rate × avg_win - loss_rate × avg_loss) / avg_win
2. Apply conservative factor: 0.25 (25% of Kelly)
3. Size = capital × kelly_fraction × risk_per_trade

Circuit Breakers:
├─ Daily Loss Limit: -5% (stop all trading if hit)
├─ Max Drawdown: -10% (pause, wait for recovery)
├─ Max Positions: 5 concurrent trades
├─ Margin Utilization: 50% max
└─ Risk Per Trade: 2% of capital max

ATR Stop Calculation:
├─ ATR = Average True Range (14-period)
├─ Stop Loss = Entry ± 2 × ATR
├─ Ensures stops aren't too tight
└─ Adapts to volatility
```

### 9. **Validation Framework** (`validation.py`)

ICIR + Walk-Forward validation.

```
ICIR (Information Coefficient Information Ratio):
├─ Measures autocorrelation in P&L
├─ ICIR = correlation(trade_sequence, returns) × √N / σ(returns)
├─ Threshold: > 0.2 (indicates real edge)
└─ Updated daily with new trades

Walk-Forward Analysis:
├─ Divide historical data into rolling windows
├─ Optimize on first window, test on next
├─ Spearman rank correlation between in-sample & out-of-sample
├─ Indicates parameter stability
└─ Stored in validation_reports
```

### 10. **Evolution Engine** (`evolution_engine.py`)

Genetic algorithm for parameter optimization.

```
Parameters to Evolve:
├─ Momentum: z_score_threshold, rsi_range, volume_multiplier
├─ Swing: ma_length, rsi_range, hold_time_max
├─ Risk: atr_multiplier, kelly_fraction
└─ General: entry_confidence_min, daily_loss_limit

Population: 10 individuals
Generations: 50 (runs weekly)
Fitness: ICIR × win_rate × recovery_factor

Selection: Tournament (top 2)
Crossover: Single-point
Mutation: 15% (Gaussian random walk)

Storage:
├─ evolution_history: Fitness per generation
├─ genetic_history: Variant performance
├─ config: Current best parameters
└─ All validated before deployment
```

### 11. **LLM Strategist** (`llm_strategist.py`)

Groq API for real-time strategic insights.

```
Groq Configuration:
├─ Model: Llama 3.3 70B
├─ Max tokens: 500
├─ Temperature: 0.3 (deterministic)
└─ Free tier: 14,400 RPM

Fallback: OpenRouter (Llama 2 if Groq fails)

Queries:
1. "Should we trade BTC/USD now? (regime: {regime}, signals: {signals})"
   → Returns: enter/wait/exit with confidence

2. "What's the market sentiment? (news: {headlines})"
   → Returns: bullish/bearish/neutral with score

3. "Analyze this trade: entry={entry}, exit={exit}, pnl={pnl}"
   → Returns: root cause analysis, lessons learned

Input Context:
├─ Current regime (HMM state)
├─ Pending signals
├─ Open positions
├─ Recent P&L
└─ News headlines (alternative.me)

Output:
├─ Recommendation: enter/wait/exit
├─ Confidence: 0-1
├─ Reasoning: 1-2 sentences
└─ Stored in signals for audit
```

### 12. **Notifications** (`send_notification.py`)

Telegram alerts for human oversight.

```
Telegram Bot Commands:
├─ /status → Current positions, equity, P&L
├─ /kill → Stop all trading immediately
├─ /resume → Resume trading after kill
├─ /report → Daily/weekly summary
├─ /help → Command list

Automatic Alerts:
├─ Entry: "{pair} LONG entry at ${price} (z={z:.2f})"
├─ Exit: "{pair} LONG exit at ${price} PnL=${pnl:.2f} ({ret:.1f}%)"
├─ Circuit Breaker: "⚠️ CIRCUIT BREAKER: {reason}"
├─ Error: "❌ Error in {component}: {message}"
└─ Daily Report: Summary + metrics
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION (Hourly)                                  │
│    Kraken API → fetch_kraken_ohlcv() → Firestore (ohlcv_data)
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 2. INDICATORS & SCORING (Hourly)                            │
│    RSI, ATR, MA50, Z-Score → Asset Scoring → Firestore     │
│    HMM Regime Detection → Firestore (regime_history)       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 3. SIGNAL GENERATION (Hourly)                               │
│    Momentum Scan + Swing Scan → Confidence Scoring          │
│    → Firestore (signals)                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 4. LLM VALIDATION (Per Signal)                              │
│    Groq API → Strategic confirmation                        │
│    → Update signal confidence → Firestore                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 5. EXECUTION (Per Signal)                                   │
│    Pre-flight checks → Size (Kelly) → Place order           │
│    → Firestore (open_positions)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 6. POSITION MONITORING (Every 5 min)                        │
│    Check stops/targets → Update P&L → Close if triggered   │
│    → Firestore (trade_history)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 7. EQUITY & METRICS (Hourly)                                │
│    Calculate equity → ICIR → Drawdown → Firestore          │
│    Validation reports → Evolution trigger                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 8. EVOLUTION (Weekly)                                       │
│    Genetic algorithm optimization → Test parameters         │
│    → Update config if ICIR > 0.2 + Walk-Forward pass       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ 9. DASHBOARD & ALERTS                                       │
│    Firestore listeners → Dashboard updates (real-time)      │
│    Telegram notifications → Human in loop                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Firestore Schema

### Collections Overview

| Collection | Purpose | Access | Documents |
|-----------|---------|--------|-----------|
| `signals` | Pending trade signals | Public read | timestamp DESC, limit 100 |
| `open_positions` | Active trades | Public read | pair, strategy, updated |
| `trade_history` | Closed trades + P&L | Public read | exit_timestamp DESC, limit 1000 |
| `equity_history` | Balance snapshots | Public read | timestamp DESC, limit 5000 |
| `regime_history` | HMM regime states | Public read | timestamp DESC, limit 100 |
| `asset_scores` | Multi-factor scoring | Public read | composite_score DESC |
| `ohlcv_data` | Current OHLCV bars | Public read | pair, latest 500 bars |
| `config` | Strategy parameters | Admin only | Global settings |
| `circuit_breakers` | Risk limits + status | Admin only | daily_loss, max_dd, paused |
| `evolution_history` | GA evolution records | Admin only | timestamp DESC, generations |
| `genetic_history` | Variant performance | Admin only | fitness ranking |
| `validation_reports` | ICIR + walk-forward | Admin only | timestamp DESC |
| `margin_toggle` | Margin status | Admin only | enabled bool |
| `ohlcv_historical` | Archive (backup) | Admin only | timestamp DESC |

### Document Schemas

#### `signals` Collection
```json
{
  "pair": "BTC/USD",
  "strategy": "momentum",
  "timestamp": "2026-09-06T15:00:00Z",
  "z_score": 2.3,
  "rsi": 62.5,
  "confidence": 0.85,
  "direction": "long",
  "suggested_entry": 43250.50,
  "suggested_stop": 42000.00,
  "llm_confirmation": true,
  "llm_reasoning": "Strong breakout with volume confirmation"
}
```

#### `open_positions` Collection
```json
{
  "pair": "BTC/USD",
  "strategy": "momentum",
  "entry_time": "2026-09-06T15:30:00Z",
  "entry_price": 43250.50,
  "size": 0.005,
  "current_price": 43500.00,
  "unrealized_pnl": 1.25,
  "stop_loss": 42000.00,
  "take_profit": 44500.00,
  "duration_hours": 2.5,
  "margin_used": false
}
```

#### `trade_history` Collection
```json
{
  "pair": "BTC/USD",
  "strategy": "swing",
  "entry_time": "2026-09-05T10:00:00Z",
  "entry_price": 42000.00,
  "exit_time": "2026-09-05T18:00:00Z",
  "exit_price": 42500.00,
  "size": 0.01,
  "pnl": 5.00,
  "return_pct": 1.19,
  "duration_hours": 8.0,
  "reason": "tp",
  "icir_contribution": 0.15
}
```

#### `regime_history` Collection
```json
{
  "timestamp": "2026-09-06T15:00:00Z",
  "current_state": 0,
  "state_names": ["Trend", "Choppy", "Ranging", "Vol-Low", "Vol-High"],
  "confidence": 0.92,
  "state_probabilities": [0.92, 0.05, 0.02, 0.01, 0.00],
  "atr": 125.50,
  "volatility": 0.018
}
```

---

## Strategy Mechanics

### Momentum (2-6h Holds)

**Principle:** Ride breakouts with trend confirmation.

```
Entry Conditions:
├─ Z-Score > 2.0 (price breakout)
├─ RSI 40-70 (not overbought)
├─ Volume spike > 1.5x (confirmation)
└─ Regime != Ranging (needs room)

Exit Conditions:
├─ Stop: Entry - 2×ATR
├─ Profit: Entry + 2×(Entry - Stop)
├─ Timeout: 6 hours
└─ Whipsaw: Reverse signal

Typical Trade:
Entry:    BTC/USD at 43,250 (z=+2.3)
Stop:     42,000 (2×ATR)
Target:   44,500 (2:1)
Outcome:  +$1,250 (2.9%) in 3 hours
```

### Swing (4-24h Holds)

**Principle:** Trade price action around moving averages.

```
Entry Conditions (Long):
├─ Price > MA50
├─ RSI 40-70 (pullback, not overbought)
└─ Higher than yesterday

Entry Conditions (Short):
├─ Price < MA50
├─ RSI 30-60 (bounce, not oversold)
└─ Lower than yesterday

Exit Conditions:
├─ Stop: Entry ± 2×ATR
├─ Profit: 2× risk
├─ Timeout: 24h long, 12h short
└─ Regime change: Exit early if Ranging

Typical Trade:
Entry:    ETH/USD at 2,300 (MA50: 2,280)
Stop:     2,200 (2×ATR)
Target:   2,400 (2:1)
Outcome:  +$100 (4.3%) in 12 hours
```

---

## Risk Management

### ATR-Based Stops

Adapts to volatility. In high-volatility periods, stops widen. In calm periods, stops tighten.

```
ATR = Average True Range (14-period)
Stop Loss = Entry ± 2 × ATR

Example:
Entry:    100
ATR:      2.50
Stop:     Entry - 2×2.50 = 95 (5% risk)

High Volatility:
Entry:    100
ATR:      5.00
Stop:     Entry - 2×5.00 = 90 (10% risk)
```

### Kelly Criterion Position Sizing

Conservative (25% of mathematical Kelly) to reduce drawdown.

```
Fair Kelly = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
Conservative Kelly = Fair Kelly × 0.25

Example:
Win rate:   60% (0.6)
Avg win:    +2% (0.02)
Avg loss:   -1% (0.01)

Fair Kelly = (0.6 × 0.02 - 0.4 × 0.01) / 0.02 = 0.4 (40%)
Conservative = 0.4 × 0.25 = 0.1 (10%)

Size = Capital × 0.1 = $300 × 0.1 = $30 per trade
```

### Circuit Breakers

Automatic pause mechanisms.

```
Daily Loss Limit: -5%
├─ If loss > $15 → stop all trading
├─ Reset at UTC midnight
└─ Alert via Telegram

Max Drawdown: -10%
├─ If drawdown > $30 → pause entry signals
├─ Allow exits only
├─ Resume when drawdown recovers to -5%
└─ Freeze positions at -15% (emergency stop)

Max Positions: 5
├─ Never more than 5 concurrent trades
└─ Ensures diversification

Margin Utilization: 50% max
├─ Don't use more than 50% of available margin
└─ Safety buffer for volatility
```

---

## Evolution Engine

### Genetic Algorithm

Runs weekly, optimizes parameters for new market regime.

```
Parameters (30 total):
├─ Momentum: z_threshold (1.5-3.0), rsi_min (30-50), rsi_max (60-80), vol_mult (1.0-2.0)
├─ Swing: ma_length (20-100), rsi_min (20-40), rsi_max (60-80), hold_max (4-48h)
├─ Risk: atr_mult (1.5-3.0), kelly_fraction (0.1-0.5)
└─ General: confidence_min (0.3-0.7), daily_loss (-3% to -10%)

Fitness Function:
fitness = ICIR × win_rate × recovery_factor × profitability_index

Population: 10 individuals
Generations: 50 per week

Selection: Tournament (pick best 2 parents)
Crossover: Single-point
Mutation: 15% chance, Gaussian ±10%

Validation:
├─ All candidates tested on walk-forward window
├─ ICIR must be > 0.2
├─ Sharpe > 0.5
└─ Max drawdown < 15%

Deployment:
├─ Best variant replaces config if ICIR > previous
├─ Rolled back if real-time performance drops
└─ Logged in genetic_history for audit
```

---

## LLM Integration

### Groq API Strategy

**Model:** Llama 3.3 70B (free tier)  
**Fallback:** OpenRouter (Llama 2)

```
Use Cases:

1. Signal Validation (per signal)
   Query: "Should we trade {pair} {direction} at {price}? 
           Regime: {regime}, Confidence: {conf}"
   Response: "Yes/No/Wait, confidence: 0.85, reasoning: ..."
   Cost: ~100 tokens/signal

2. Sentiment Analysis (daily)
   Query: "Market sentiment from headlines: {headlines}"
   Response: "Bullish 0.7" (weighted -1 to +1)
   Cost: ~150 tokens/day

3. Trade Analysis (post-close)
   Query: "Analyze: entry={entry}, exit={exit}, pnl={pnl}. Why?"
   Response: "Took profit on breakout confirmation..."
   Cost: ~200 tokens/trade

Rate Limits:
├─ 14,400 RPM (240 per minute)
├─ ≈ 1 signal validation per minute OK
├─ Batching for daily reports
└─ Graceful degration if limit hit
```

---

## Dashboard Architecture

### Frontend Stack

| Component | Purpose |
|-----------|---------|
| HTML | Document structure (7 cards, 14 metrics) |
| CSS | Cyberpunk styling (neon green/cyan/magenta) |
| JavaScript | Real-time Firestore listeners |
| Chart.js | Equity & evolution charts |
| Firebase JS SDK | Firestore real-time listeners |

### Real-Time Updates

```javascript
// Every collection has a listener
onSnapshot(collection(db, 'open_positions'), (snapshot) => {
  // Update positions table
  // Recalculate metrics
  // Refresh charts
  // Update UI instantly
});

// All driven by Firestore document changes
// No polling, pure event-driven
```

### Metrics Calculated

**Key Metrics:**
- Active Positions: count of open_positions
- Total P&L: sum of trade_history.pnl
- Win Rate: (wins / total) × 100
- Current Equity: latest equity_history.equity
- Max Drawdown: (peak - trough) / peak × 100
- Unrealized P&L: sum of open_positions.unrealized_pnl

**Charts:**
- Equity Curve: 7-day history with sparkline
- Regime History: Current state with confidence
- Asset Scores: Top 5 pairs with composite score
- Evolution: Best fitness per generation

---

## Deployment Model

### Firebase Cloud Functions

```
Event: Cloud Scheduler trigger (hourly)
Function: main.py (Python 3.11)
Handler: def main(request)
Timeout: 540s (9 minutes)
Memory: 512MB
VPC: Standard
Logging: Cloud Logging
```

### Execution Flow

```
Hour 0:00 → Cloud Scheduler trigger
   ├─ fetch_ohlcv() → Download new bars
   ├─ calculate_indicators() → RSI, ATR, MA50, Z-Score
   ├─ regime_detection() → HMM inference
   ├─ asset_scoring() → Multi-factor scores
   ├─ momentum_scan() → Generate momentum signals
   ├─ swing_scan() → Generate swing signals
   ├─ llm_strategist() → Validate with Groq
   ├─ execute_trade() → Place orders for approved signals
   ├─ check_positions() → Monitor all open trades
   ├─ close_trade() → Execute stops/targets
   ├─ calculate_metrics() → ICIR, drawdown, equity
   ├─ evolution_engine() (if weekly) → GA optimization
   └─ send_notification() → Telegram alerts

Total time: ~30 seconds (within 9-minute timeout)
```

### Environment Variables

```bash
# Kraken API
KRAKEN_API_KEY=your_key
KRAKEN_API_SECRET=your_secret

# LLM
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key (fallback)

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Firebase (auto-injected in Cloud Functions)
FIREBASE_PROJECT_ID=your_project
FIREBASE_SERVICE_ACCOUNT=/path/to/service-account.json

# Trading
CAPITAL_POOL_USD=300
DRY_RUN=false (set true for testing)

# Optional
FCM_SERVER_KEY=for push notifications
FIREBASE_HOSTING_URL=for dashboard link in alerts
```

### Dry-Run Mode

Set `DRY_RUN=true` in environment:

```python
if os.getenv('DRY_RUN') == 'true':
    # Log order instead of placing
    # Simulate fills at market
    # Calculate P&L without real money
    # Update Firestore for testing
```

---

## Security Model

### Firestore Rules

```
// Public read (dashboard)
match /signals/{document=**} {
  allow read: if true;
  allow write: if false;
}

// Admin only (configuration)
match /config/{document=**} {
  allow read, write: if request.auth.uid in ['admin_uid'];
}

// Service account only (trading)
match /open_positions/{document=**} {
  allow read: if true;  // Dashboard read
  allow write: if request.auth.uid == service_account_uid;
}
```

### API Security

- Kraken API keys stored in Cloud Secret Manager
- Telegram token stored in Cloud Secret Manager
- Groq/OpenRouter keys in Secret Manager
- No keys in code or Git

---

## Monitoring & Logging

### Cloud Logging Integration

```
Log entries tagged by function:
├─ INFO: Trade execution, signals, equity updates
├─ WARNING: Circuit breaker triggers, low confidence
└─ ERROR: API failures, validation failures

Example:
[2026-09-06 15:30:00] INFO execute_trade: 
  BTC/USD LONG entry at 43250.50, size 0.005, SL 42000

[2026-09-06 15:35:00] INFO check_positions: 
  BTC/USD TP hit at 44500, PnL +1250
```

### Metrics & Observability

- Cloud Monitoring: Function execution time, errors
- Firestore: Document write counts, read counts
- Custom: ICIR, win rate, max drawdown (stored in Firestore)

---

## Roadmap

### Phase 1 (Current)
- ✅ Momentum + Swing on Kraken spot
- ✅ $300 capital, 2 strategies
- ✅ Firebase Cloud Functions
- ✅ Dashboard + Telegram alerts

### Phase 2
- Margin trading (2x leverage)
- Additional exchange: Binance spot
- More pairs: 10+ altcoins
- Capital: $600

### Phase 3
- Perpetual futures (Kraken/Binance)
- Advanced techniques: Spread trading
- Capital: $1,000

### Phase 4-7
- Full scale automation
- Multi-exchange arbitrage
- ML/RL optimization
- Capital: $1,390

---

## References

- **Firestore Documentation:** https://firebase.google.com/docs/firestore
- **Cloud Functions:** https://firebase.google.com/docs/functions
- **Kraken API:** https://docs.kraken.com/rest/
- **Groq API:** https://console.groq.com
- **Chart.js:** https://www.chartjs.org

---

**End of Architecture Documentation**
```

**Notes for File 33:**
- Comprehensive system architecture (15 major sections)
- All components mapped with data flow
- Firestore schema with actual document examples
- Strategy mechanics with concrete examples
- Risk management with calculations
- Evolution engine algorithm details
- LLM integration strategy
- Dashboard architecture
- Deployment model with execution flow
- Security model with Firestore rules
- Monitoring & logging integration
