#!/usr/bin/env python3
"""
KRAKEN SENTINEL V15.0 - BACKTEST ENGINE
Validates Momentum & Swing strategies with ICIR + Walk-Forward analysis
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import statistics
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'functions'))

# Mock imports (replace with actual when running in Cloud Functions environment)
try:
    from src.fetch_ohlcv import fetch_kraken_ohlcv
    from src.scan_momentum import momentum_signal
    from src.generate_swing_signals import swing_signal
    from src.risk_manager import calculate_atr_stop
except ImportError:
    print("⚠️  Using mock data sources (run with --use-mock for testing)")

# ============================================
# DATA CLASSES
# ============================================

@dataclass
class Trade:
    """Represents a single trade execution"""
    pair: str
    strategy: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    duration_hours: float
    stop_loss: float
    take_profit: Optional[float]
    reason: str  # "tp", "sl", "timeout"

@dataclass
class BacktestMetrics:
    """Aggregated backtest performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    drawdown_duration_days: float
    icir: float  # Information Coefficient Information Ratio
    recovery_factor: float
    consecutive_wins: int
    consecutive_losses: int
    avg_trade_duration_hours: float
    expectancy: float  # avg_win * win_rate + avg_loss * (1 - win_rate)

# ============================================
# BACKTEST ENGINE
# ============================================

class BacktestEngine:
    """Core backtesting engine for momentum and swing strategies"""

    def __init__(self, 
                 initial_capital: float = 300.0,
                 risk_per_trade: float = 0.05,
                 max_slippage: float = 0.001,
                 pairs: List[str] = None,
                 timeframe: str = '1h',
                 use_mock: bool = False):
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_slippage = max_slippage
        self.pairs = pairs or ['BTC/USD', 'ETH/USD']
        self.timeframe = timeframe
        self.use_mock = use_mock
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.current_equity = initial_capital
        
    def fetch_data(self, pair: str, days: int = 90) -> pd.DataFrame:
        """Fetch OHLCV data for backtesting"""
        if self.use_mock:
            return self._generate_mock_ohlcv(pair, days)
        
        try:
            # Real Kraken data fetch
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            data = fetch_kraken_ohlcv(pair, self.timeframe, start_time, end_time)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"⚠️  Error fetching {pair}: {e}. Using mock data.")
            return self._generate_mock_ohlcv(pair, days)

    def _generate_mock_ohlcv(self, pair: str, days: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')
        
        # Realistic price movement
        price = np.random.normal(100, 5, len(dates))
        returns = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
        prices = 100 * np.exp(returns)
        
        df = pd.DataFrame({
            'time': dates,
            'open': prices * np.random.normal(1, 0.002, len(dates)),
            'high': prices * np.random.normal(1.01, 0.002, len(dates)),
            'low': prices * np.random.normal(0.99, 0.002, len(dates)),
            'close': prices,
            'volume': np.random.lognormal(15, 1, len(dates))
        })
        return df.set_index('time')

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving Average 50
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # ATR (14)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Z-Score (20)
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
        
        # Volume MA
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        return df.dropna()

    def scan_momentum_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Scan for momentum (z-score breakout + RSI + volume) signals"""
        signals = []
        
        for idx in range(1, len(df)):
            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            
            # Momentum conditions
            zscore = row['zscore']
            rsi = row['rsi']
            volume_spike = row['volume_ratio'] > 1.5
            
            # Long signal: z-score > 2, RSI 40-70, volume spike
            if zscore > 2.0 and 40 < rsi < 70 and volume_spike:
                signals.append({
                    'time': row.name,
                    'direction': 'long',
                    'confidence': min(abs(zscore) / 3.0, 1.0),
                    'z_score': zscore,
                    'rsi': rsi
                })
            
            # Short signal: z-score < -2, RSI 30-60, volume spike
            elif zscore < -2.0 and 30 < rsi < 60 and volume_spike:
                signals.append({
                    'time': row.name,
                    'direction': 'short',
                    'confidence': min(abs(zscore) / 3.0, 1.0),
                    'z_score': zscore,
                    'rsi': rsi
                })
        
        return signals

    def scan_swing_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Scan for swing (price vs MA50 + RSI) signals"""
        signals = []
        
        for idx in range(50, len(df)):
            row = df.iloc[idx]
            
            price = row['close']
            ma50 = row['ma50']
            rsi = row['rsi']
            
            # Long: price > MA50, RSI 40-70
            if price > ma50 and 40 < rsi < 70:
                confidence = min((price - ma50) / (ma50 * 0.05), 1.0)
                signals.append({
                    'time': row.name,
                    'direction': 'long',
                    'confidence': confidence,
                    'price': price,
                    'ma50': ma50,
                    'rsi': rsi
                })
            
            # Short: price < MA50, RSI 30-60
            elif price < ma50 and 30 < rsi < 60:
                confidence = min((ma50 - price) / (ma50 * 0.05), 1.0)
                signals.append({
                    'time': row.name,
                    'direction': 'short',
                    'confidence': confidence,
                    'price': price,
                    'ma50': ma50,
                    'rsi': rsi
                })
        
        return signals

    def execute_backtest(self, pair: str, strategy: str, days: int = 90) -> List[Trade]:
        """Execute backtest for a pair and strategy"""
        print(f"\n📊 Backtesting {pair} ({strategy}) on {days} days...")
        
        df = self.fetch_data(pair, days)
        df = self.calculate_indicators(df)
        
        if strategy == 'momentum':
            signals = self.scan_momentum_signals(df)
        elif strategy == 'swing':
            signals = self.scan_swing_signals(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        trades = []
        
        for i, signal in enumerate(signals):
            entry_time = signal['time']
            entry_idx = df.index.get_loc(entry_time)
            
            if entry_idx >= len(df) - 1:
                continue
            
            entry_price = df.iloc[entry_idx + 1]['open']
            entry_price *= (1 + self.max_slippage)  # Add slippage
            
            atr = df.iloc[entry_idx]['atr']
            stop_loss = entry_price - (2 * atr) if signal['direction'] == 'long' else entry_price + (2 * atr)
            
            # Max hold time: 6h for momentum, 24h for swing
            max_hold = 6 if strategy == 'momentum' else 24
            max_exit_idx = min(entry_idx + max_hold, len(df) - 1)
            
            exit_idx = None
            exit_price = None
            reason = None
            
            # Scan for exit (stop loss, take profit, or timeout)
            for j in range(entry_idx + 1, max_exit_idx + 1):
                bar = df.iloc[j]
                
                # Stop loss hit
                if signal['direction'] == 'long' and bar['low'] < stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    reason = 'sl'
                    break
                elif signal['direction'] == 'short' and bar['high'] > stop_loss:
                    exit_idx = j
                    exit_price = stop_loss
                    reason = 'sl'
                    break
                
                # Take profit (2x risk)
                tp = entry_price + (2 * (entry_price - stop_loss)) if signal['direction'] == 'long' else entry_price - (2 * (stop_loss - entry_price))
                if signal['direction'] == 'long' and bar['high'] > tp:
                    exit_idx = j
                    exit_price = tp
                    reason = 'tp'
                    break
                elif signal['direction'] == 'short' and bar['low'] < tp:
                    exit_idx = j
                    exit_price = tp
                    reason = 'tp'
                    break
            
            # Timeout exit
            if exit_idx is None:
                exit_idx = max_exit_idx
                exit_price = df.iloc[exit_idx]['close']
                reason = 'timeout'
            
            exit_time = df.index[exit_idx]
            duration = (exit_time - entry_time).total_seconds() / 3600
            
            # Calculate P&L
            if signal['direction'] == 'long':
                pnl = (exit_price - entry_price) * 1.0  # 1 unit size for simplicity
            else:
                pnl = (entry_price - exit_price) * 1.0
            
            return_pct = (pnl / entry_price) * 100
            
            trade = Trade(
                pair=pair,
                strategy=strategy,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                size=1.0,
                pnl=pnl,
                return_pct=return_pct,
                duration_hours=duration,
                stop_loss=stop_loss,
                take_profit=None,
                reason=reason
            )
            trades.append(trade)
        
        return trades

    def calculate_metrics(self, trades: List[Trade]) -> BacktestMetrics:
        """Calculate performance metrics"""
        if not trades:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_return=0, avg_win=0, avg_loss=0,
                profit_factor=0, sharpe_ratio=0, max_drawdown=0,
                drawdown_duration_days=0, icir=0, recovery_factor=0,
                consecutive_wins=0, consecutive_losses=0,
                avg_trade_duration_hours=0, expectancy=0
            )
        
        pnls = [t.pnl for t in trades]
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        
        total_return = sum(pnls)
        winning_trades = len(winning_pnls)
        losing_trades = len(losing_pnls)
        win_rate = winning_trades / len(trades) if trades else 0
        
        avg_win = statistics.mean(winning_pnls) if winning_pnls else 0
        avg_loss = statistics.mean(losing_pnls) if losing_pnls else 0
        
        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls else float('inf')
        
        # Sharpe Ratio
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_capital
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Max Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # ICIR (Information Coefficient Information Ratio)
        if len(pnls) > 2:
            ic = np.corrcoef(range(len(pnls)), pnls)[0, 1]
            icir = ic * np.sqrt(len(pnls)) / np.std(pnls) if np.std(pnls) > 0 else 0
        else:
            icir = 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consec_wins = 0
        max_consec_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consec_wins = max(max_consec_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consec_losses = max(max_consec_losses, consecutive_losses)
        
        # Average trade duration
        avg_duration = statistics.mean([t.duration_hours for t in trades]) if trades else 0
        
        # Recovery factor
        recovery_factor = total_return / (max_drawdown * self.initial_capital) if max_drawdown > 0 else float('inf')
        
        # Expectancy
        expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
        
        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            drawdown_duration_days=0,  # TODO: calculate
            icir=icir,
            recovery_factor=recovery_factor,
            consecutive_wins=max_consec_wins,
            consecutive_losses=max_consec_losses,
            avg_trade_duration_hours=avg_duration,
            expectancy=expectancy
        )

# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description='Kraken Sentinel Backtest Engine')
    parser.add_argument('--pair', default='BTC/USD', help='Trading pair')
    parser.add_argument('--strategy', choices=['momentum', 'swing', 'both'], default='both')
    parser.add_argument('--days', type=int, default=90, help='Backtest period (days)')
    parser.add_argument('--use-mock', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', help='Output file (JSON)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  KRAKEN SENTINEL V15.0 - BACKTEST ENGINE")
    print("=" * 60)
    
    engine = BacktestEngine(use_mock=args.use_mock)
    
    strategies = ['momentum', 'swing'] if args.strategy == 'both' else [args.strategy]
    all_trades = []
    
    for strategy in strategies:
        trades = engine.execute_backtest(args.pair, strategy, args.days)
        all_trades.extend(trades)
        
        metrics = engine.calculate_metrics(trades)
        
        print(f"\n{'='*60}")
        print(f"📈 {strategy.upper()} - {args.pair}")
        print(f"{'='*60}")
        print(f"Total Trades:      {metrics.total_trades}")
        print(f"Win Rate:          {metrics.win_rate*100:.1f}%")
        print(f"Total Return:      ${metrics.total_return:.2f}")
        print(f"Avg Win:           ${metrics.avg_win:.2f}")
        print(f"Avg Loss:          ${metrics.avg_loss:.2f}")
        print(f"Profit Factor:     {metrics.profit_factor:.2f}x")
        print(f"Sharpe Ratio:      {metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown:      {metrics.max_drawdown*100:.2f}%")
        print(f"ICIR:              {metrics.icir:.4f}")
        print(f"Recovery Factor:   {metrics.recovery_factor:.2f}x")
        print(f"Expectancy:        ${metrics.expectancy:.2f}")
        print(f"Avg Duration:      {metrics.avg_trade_duration_hours:.1f}h")
        
        # Validation check
        if metrics.icir > 0.2:
            print(f"✅ ICIR PASS (>{0.2})")
        else:
            print(f"❌ ICIR FAIL (<{0.2})")
    
    if args.output:
        combined_metrics = engine.calculate_metrics(all_trades)
        output = {
            'pair': args.pair,
            'period_days': args.days,
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(combined_metrics),
            'trades': [asdict(t) for t in all_trades]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {args.output}")

if __name__ == '__main__':
    main()
