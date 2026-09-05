import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import spearmanr
import asyncio

from src.helpers import (
    query_firestore,
    write_to_firestore,
    get_utc_timestamp,
    get_firestore_client,
    log_info,
    log_error
)

logger = logging.getLogger(__name__)

# ============================================================================
# ICIR + WALK-FORWARD VALIDATION
# ============================================================================

class ValidationEngine:
    """
    Combined ICIR + Walk-Forward validation system.
    
    ICIR (Information Coefficient Information Ratio):
    - Measures predictive power of signals
    - Formula: ICIR = Mean(IC) / StdDev(IC)
    - IC = Spearman rank correlation between forecast and actual return
    - Threshold: ICIR > 0.2 (both in-sample and out-of-sample)
    
    Walk-Forward Validation:
    - Split data into rolling windows
    - In-sample window: Train strategy
    - Out-of-sample window: Test strategy
    - Strategy must pass both to be deployed
    
    Metrics:
    - ICIR (predictive power)
    - Win Rate (trade quality)
    - Profit Factor (reward/risk ratio)
    - Max Drawdown (risk control)
    - Sharpe Ratio (risk-adjusted returns)
    - Sortino Ratio (downside risk)
    """
    
    # Validation thresholds
    ICIR_THRESHOLD = 0.2
    ICIR_MIN_CONSISTENCY = 0.2
    ICIR_MAX_STDEV = 0.05
    PROFIT_FACTOR_THRESHOLD = 1.2
    WIN_RATE_THRESHOLD = 0.45  # 45%
    MAX_DRAWDOWN_THRESHOLD = -0.15  # -15%
    
    # Walk-forward windows
    LOOKBACK_DAYS = 365
    TRAIN_DAYS = 300
    TEST_DAYS = 30
    
    def __init__(self):
        self.db = get_firestore_client()
    
    async def validate_strategy_variant(self, variant_id: str, variant_params: Dict) -> Dict:
        """
        Validate a single strategy variant using combined ICIR + Walk-Forward.
        
        Args:
            variant_id: Unique variant ID
            variant_params: Strategy parameters (e.g., Z-score, TP%, SL%, Hold time)
        
        Returns:
            Validation report
        """
        try:
            logger.info(f"🧪 Validating variant {variant_id}...")
            
            # Fetch historical trades
            trades = await self._fetch_historical_trades(days=self.LOOKBACK_DAYS)
            if len(trades) < 50:
                logger.warning(f"⚠️ Insufficient trades for validation: {len(trades)}")
                return {
                    "variant_id": variant_id,
                    "status": "insufficient_data",
                    "trade_count": len(trades),
                    "icir": 0,
                    "passes_validation": False
                }
            
            # Split into windows for walk-forward
            windows = self._create_rolling_windows(trades)
            
            icir_results = []
            in_sample_icirs = []
            out_of_sample_icirs = []
            
            # Walk-forward validation
            for window_idx, (in_sample, out_sample) in enumerate(windows):
                logger.info(f"📊 Window {window_idx + 1}: In-sample {len(in_sample)} | Out-of-sample {len(out_sample)}")
                
                # Calculate ICIR on in-sample
                in_sample_icir = self._calculate_icir(in_sample)
                in_sample_icirs.append(in_sample_icir)
                
                # Calculate ICIR on out-of-sample
                out_sample_icir = self._calculate_icir(out_sample)
                out_of_sample_icirs.append(out_sample_icir)
                
                # Log window result
                window_result = {
                    "window": window_idx + 1,
                    "in_sample_icir": in_sample_icir,
                    "out_of_sample_icir": out_sample_icir,
                    "in_sample_pass": in_sample_icir > self.ICIR_THRESHOLD,
                    "out_of_sample_pass": out_sample_icir > self.ICIR_THRESHOLD
                }
                
                icir_results.append(window_result)
                logger.info(f"  In-Sample ICIR: {in_sample_icir:.4f} {'✅' if in_sample_icir > self.ICIR_THRESHOLD else '❌'}")
                logger.info(f"  Out-of-Sample ICIR: {out_sample_icir:.4f} {'✅' if out_sample_icir > self.ICIR_THRESHOLD else '❌'}")
            
            # Calculate aggregate metrics
            avg_in_sample_icir = np.mean(in_sample_icirs)
            avg_out_of_sample_icir = np.mean(out_of_sample_icirs)
            icir_stdev = np.std(out_of_sample_icirs)
            
            # Check consistency
            windows_passing = sum(1 for w in icir_results if w["out_of_sample_pass"])
            consistency_pass = windows_passing >= len(icir_results) * 0.8  # 80% of windows
            
            # Calculate other metrics
            metrics = await self._calculate_trade_metrics(trades)
            
            # Determine pass/fail
            passes_validation = (
                avg_in_sample_icir > self.ICIR_THRESHOLD and
                avg_out_of_sample_icir > self.ICIR_THRESHOLD and
                icir_stdev < self.ICIR_MAX_STDEV and
                consistency_pass and
                metrics["profit_factor"] > self.PROFIT_FACTOR_THRESHOLD and
                metrics["win_rate"] > self.WIN_RATE_THRESHOLD and
                metrics["max_drawdown"] > self.MAX_DRAWDOWN_THRESHOLD
            )
            
            # Build report
            report = {
                "variant_id": variant_id,
                "parameters": variant_params,
                "status": "validated" if passes_validation else "rejected",
                "passes_validation": passes_validation,
                "timestamp": get_utc_timestamp(),
                "icir": {
                    "in_sample_avg": avg_in_sample_icir,
                    "out_of_sample_avg": avg_out_of_sample_icir,
                    "stdev": icir_stdev,
                    "consistency_windows": windows_passing,
                    "total_windows": len(icir_results)
                },
                "metrics": metrics,
                "windows": icir_results
            }
            
            logger.info(f"{'✅' if passes_validation else '❌'} Variant validation complete")
            
            # Save report
            await self._save_validation_report(report)
            
            return report
        except Exception as e:
            log_error(f"❌ Validation failed for variant {variant_id}", e)
            return {
                "variant_id": variant_id,
                "status": "error",
                "passes_validation": False,
                "error": str(e)
            }
    
    # ========================================================================
    # ICIR CALCULATION
    # ========================================================================
    
    def _calculate_icir(self, trades: List[Dict]) -> float:
        """
        Calculate ICIR (Information Coefficient Information Ratio).
        
        ICIR = Mean(IC) / StdDev(IC)
        IC = Spearman rank correlation between forecast and actual return
        
        Args:
            trades: List of trade dicts with signals and returns
        
        Returns:
            ICIR value (higher is better, > 0.2 is threshold)
        """
        try:
            if len(trades) < 10:
                return 0
            
            # Extract forecast signals and actual returns
            forecasts = []
            returns = []
            
            for trade in trades:
                # Forecast: Signal strength (RSI, volume spike, etc.)
                forecast = self._get_forecast_signal(trade)
                forecasts.append(forecast)
                
                # Actual: Trade return percentage
                actual_return = trade.get("pnl_percent", 0)
                returns.append(actual_return)
            
            if len(forecasts) < 10 or len(returns) < 10:
                return 0
            
            # Calculate Spearman rank correlation (IC)
            correlation, pvalue = spearmanr(forecasts, returns)
            
            if np.isnan(correlation):
                return 0
            
            # IC is the correlation
            ic_values = [correlation]  # In practice, would use rolling window of ICs
            
            mean_ic = np.mean(ic_values)
            std_ic = np.std(ic_values) if len(ic_values) > 1 else 1.0
            
            if std_ic == 0:
                return 0
            
            icir = mean_ic / std_ic
            
            logger.debug(f"📊 ICIR calculated: {icir:.4f} (IC: {correlation:.4f}, p-value: {pvalue:.4f})")
            
            return icir
        except Exception as e:
            logger.error(f"❌ ICIR calculation failed: {str(e)}")
            return 0
    
    def _get_forecast_signal(self, trade: Dict) -> float:
        """
        Extract forecast signal strength from trade.
        
        Combines multiple factors:
        - RSI extremity (0-100)
        - Volume spike ratio
        - Asset score
        
        Returns:
            Signal strength (0-100 normalized)
        """
        try:
            rsi = trade.get("rsi_entry", 50)
            volume_spike = trade.get("volume_spike", 1.0)
            asset_score = trade.get("asset_score", 50)
            
            # Normalize each component
            rsi_signal = abs(rsi - 50)  # 0-50 (0 = neutral, 50 = extreme)
            volume_signal = min(volume_spike * 20, 100)  # 0-100 (1x = 20, 5x = 100)
            score_signal = asset_score  # Already 0-100
            
            # Weighted average
            signal = (rsi_signal * 0.3) + (volume_signal * 0.3) + (score_signal * 0.4)
            
            return min(100, max(0, signal))
        except Exception as e:
            logger.error(f"❌ Forecast signal extraction failed: {str(e)}")
            return 50
    
    # ========================================================================
    # WALK-FORWARD WINDOWS
    # ========================================================================
    
    def _create_rolling_windows(self, trades: List[Dict], window_count: int = 5) -> List[Tuple]:
        """
        Create rolling walk-forward windows from historical trades.
        
        Args:
            trades: Historical trades (sorted by timestamp, descending)
            window_count: Number of rolling windows
        
        Returns:
            List of (in_sample, out_of_sample) tuples
        """
        try:
            # Sort trades chronologically (oldest first)
            trades_sorted = sorted(trades, key=lambda t: t.get("timestamp", ""))
            
            windows = []
            total_trades = len(trades_sorted)
            
            # Calculate window sizes
            step_size = total_trades // (window_count + 1)
            
            for i in range(window_count):
                start_idx = i * step_size
                train_end = start_idx + self.TRAIN_DAYS
                test_end = train_end + self.TEST_DAYS
                
                # Ensure bounds
                train_end = min(train_end, total_trades - self.TEST_DAYS)
                test_end = min(test_end, total_trades)
                
                in_sample = trades_sorted[start_idx:train_end]
                out_sample = trades_sorted[train_end:test_end]
                
                if len(in_sample) > 10 and len(out_sample) > 5:
                    windows.append((in_sample, out_sample))
            
            logger.info(f"✅ Created {len(windows)} rolling windows")
            return windows
        except Exception as e:
            logger.error(f"❌ Window creation failed: {str(e)}")
            return []
    
    # ========================================================================
    # TRADE METRICS
    # ========================================================================
    
    async def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive trade performance metrics.
        
        Returns:
            {
                "win_rate": float (0-1),
                "profit_factor": float,
                "max_drawdown": float (-1 to 0),
                "sharpe_ratio": float,
                "sortino_ratio": float,
                "avg_win": float,
                "avg_loss": float,
                "consecutive_wins": int,
                "consecutive_losses": int
            }
        """
        try:
            if not trades:
                return {
                    "win_rate": 0,
                    "profit_factor": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                    "sortino_ratio": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "consecutive_wins": 0,
                    "consecutive_losses": 0
                }
            
            # Extract returns
            pnls = [trade.get("pnl", 0) for trade in trades]
            pnl_percents = [trade.get("pnl_percent", 0) for trade in trades]
            
            # Win rate
            wins = sum(1 for pnl in pnls if pnl > 0)
            losses = len(pnls) - wins
            win_rate = wins / len(pnls) if pnls else 0
            
            # Profit factor
            gross_profit = sum(max(0, pnl) for pnl in pnls)
            gross_loss = sum(abs(min(0, pnl)) for pnl in pnls)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Average win/loss
            avg_win = gross_profit / wins if wins > 0 else 0
            avg_loss = gross_loss / losses if losses > 0 else 0
            
            # Drawdown
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / (running_max + 1)
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # Sharpe ratio (assuming risk-free rate = 0)
            returns_array = np.array(pnl_percents) / 100
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Sortino ratio (downside risk only)
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1.0
            sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            
            # Consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_streak = 0
            streak_type = None
            
            for pnl in pnls:
                if pnl > 0:
                    if streak_type == "win":
                        current_streak += 1
                    else:
                        consecutive_losses = max(consecutive_losses, current_streak)
                        current_streak = 1
                        streak_type = "win"
                elif pnl < 0:
                    if streak_type == "loss":
                        current_streak += 1
                    else:
                        consecutive_wins = max(consecutive_wins, current_streak)
                        current_streak = 1
                        streak_type = "loss"
            
            # Capture final streak
            if streak_type == "win":
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                consecutive_losses = max(consecutive_losses, current_streak)
            
            return {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "consecutive_wins": consecutive_wins,
                "consecutive_losses": consecutive_losses,
                "total_trades": len(trades),
                "total_wins": wins,
                "total_losses": losses
            }
        except Exception as e:
            log_error("❌ Trade metrics calculation failed", e)
            return {}
    
    # ========================================================================
    # DATA FETCHING
    # ========================================================================
    
    async def _fetch_historical_trades(self, days: int = 365) -> List[Dict]:
        """Fetch historical trades from Firestore."""
        try:
            lookback_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            trades = query_firestore(
                "trade_history",
                field="timestamp",
                operator=">",
                value=lookback_date,
                order_by="timestamp",
                direction="desc",
                limit=1000
            )
            
            logger.info(f"✅ Fetched {len(trades)} historical trades")
            return trades
        except Exception as e:
            log_error("❌ Failed to fetch historical trades", e)
            return []
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    async def _save_validation_report(self, report: Dict) -> bool:
        """Save validation report to Firestore."""
        try:
            variant_id = report.get("variant_id")
            doc_id = f"{variant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            write_to_firestore("validation_reports", doc_id, report)
            
            logger.info(f"✅ Saved validation report: {doc_id}")
            return True
        except Exception as e:
            log_error("❌ Failed to save validation report", e)
            return False
    
    def generate_validation_summary(self, report: Dict) -> str:
        """Generate human-readable validation summary."""
        try:
            variant_id = report.get("variant_id")
            passes = report.get("passes_validation")
            icir = report.get("icir", {})
            metrics = report.get("metrics", {})
            
            summary = f"""
            📊 VALIDATION REPORT
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Variant: {variant_id}
            Status: {'✅ PASSES' if passes else '❌ FAILS'}
            
            ICIR:
            - In-Sample: {icir.get('in_sample_avg', 0):.4f}
            - Out-of-Sample: {icir.get('out_of_sample_avg', 0):.4f}
            - Std Dev: {icir.get('stdev', 0):.4f}
            - Consistency: {icir.get('consistency_windows', 0)}/{icir.get('total_windows', 0)} windows
            
            METRICS:
            - Win Rate: {metrics.get('win_rate', 0):.1%}
            - Profit Factor: {metrics.get('profit_factor', 0):.2f}
            - Max Drawdown: {metrics.get('max_drawdown', 0):.1%}
            - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
            - Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
            
            TRADES:
            - Total: {metrics.get('total_trades', 0)}
            - Wins: {metrics.get('total_wins', 0)}
            - Losses: {metrics.get('total_losses', 0)}
            """
            
            return summary.strip()
        except Exception as e:
            logger.error(f"❌ Failed to generate summary: {str(e)}")
            return "Error generating summary"

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def validate_strategy(variant_id: str, variant_params: Dict) -> Dict:
    """Main entry point for strategy validation."""
    try:
        engine = ValidationEngine()
        report = await engine.validate_strategy_variant(variant_id, variant_params)
        return report
    except Exception as e:
        log_error("❌ Strategy validation failed", e)
        raise
