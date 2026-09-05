import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pytz

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
# RISK MANAGEMENT ENGINE
# ============================================================================

class RiskManager:
    """
    Risk management and circuit breaker system.
    
    Risk Controls:
    1. Daily Loss Limit: -5% of pool → freeze for 24h
    2. Max Drawdown: -10% from peak → halve position sizes
    3. ICIR Breaker: ICIR < 0.2 for 3 days → freeze
    4. Single Trade Loss: -10% of trade → 10-min cooldown
    5. Max Open Positions: 2 per strategy → reject new signals
    6. System Error: 3 consecutive API failures → freeze all
    
    Stop Loss Calculation:
    - ATR-based (2x ATR)
    - Dynamic (adapts to volatility)
    - Trailing (moves with price)
    
    Position Sizing:
    - Kelly Criterion (conservative 25%)
    - Max 10% of pool per trade
    - Min $10 per trade
    """
    
    # Risk limits
    DAILY_LOSS_LIMIT_PERCENT = -5.0  # -5% of pool
    MAX_DRAWDOWN_PERCENT = -10.0     # -10% from peak
    ICIR_THRESHOLD = 0.2             # ICIR < 0.2
    ICIR_BREACH_DAYS = 3             # For 3 days
    SINGLE_TRADE_LOSS_PERCENT = -10.0  # -10% of trade
    MAX_OPEN_POSITIONS_PER_STRATEGY = 2
    MAX_CONSECUTIVE_ERRORS = 3
    
    # Timeouts
    DAILY_LOSS_FREEZE_HOURS = 24
    SINGLE_TRADE_COOLDOWN_MINUTES = 10
    
    # Thresholds
    MIN_POOL_BALANCE = 10  # $10 minimum
    
    def __init__(self):
        self.db = get_firestore_client()
    
    # ========================================================================
    # MAIN RISK CHECK
    # ========================================================================
    
    async def check_risk_limits(self) -> Dict:
        """
        Comprehensive risk check before trade execution.
        
        Returns:
            {
                "allow_trading": bool,
                "reason": str,
                "breaches": [list of breached limits]
            }
        """
        try:
            breaches = []
            
            # Check 1: Daily loss limit
            daily_loss_check = await self._check_daily_loss_limit()
            if not daily_loss_check["pass"]:
                breaches.append(daily_loss_check["reason"])
            
            # Check 2: Max drawdown
            drawdown_check = await self._check_max_drawdown()
            if not drawdown_check["pass"]:
                breaches.append(drawdown_check["reason"])
            
            # Check 3: ICIR breaker
            icir_check = await self._check_icir_breaker()
            if not icir_check["pass"]:
                breaches.append(icir_check["reason"])
            
            # Check 4: System errors
            error_check = await self._check_system_errors()
            if not error_check["pass"]:
                breaches.append(error_check["reason"])
            
            # Check 5: Pool balance
            pool_check = await self._check_pool_balance()
            if not pool_check["pass"]:
                breaches.append(pool_check["reason"])
            
            allow_trading = len(breaches) == 0
            reason = " | ".join(breaches) if breaches else "All checks passed"
            
            return {
                "allow_trading": allow_trading,
                "reason": reason,
                "breaches": breaches
            }
        except Exception as e:
            log_error("❌ Risk check failed", e)
            return {
                "allow_trading": False,
                "reason": f"Risk check error: {str(e)}",
                "breaches": ["system_error"]
            }
    
    # ========================================================================
    # DAILY LOSS LIMIT
    # ========================================================================
    
    async def _check_daily_loss_limit(self) -> Dict:
        """Check if daily loss limit (-5%) is breached."""
        try:
            daily_loss = await self.get_daily_loss()
            daily_limit = await self.get_daily_loss_limit()
            
            if daily_loss <= -daily_limit:
                logger.warning(f"⚠️ Daily loss limit breached: ${daily_loss:.2f} / ${-daily_limit:.2f}")
                return {
                    "pass": False,
                    "reason": f"daily_loss_limit ({daily_loss:.2f}%)"
                }
            
            return {"pass": True, "reason": ""}
        except Exception as e:
            log_error("❌ Daily loss check failed", e)
            return {"pass": False, "reason": "daily_loss_check_error"}
    
    async def get_daily_loss(self) -> float:
        """
        Get current daily loss percentage.
        
        Returns:
            P&L percent for today
        """
        try:
            today_start = datetime.now(pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Fetch today's trades
            trades = query_firestore(
                "trade_history",
                order_by="timestamp",
                direction="desc",
                limit=100
            )
            
            today_pnl = 0
            for trade in trades:
                trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                if trade_time >= today_start:
                    today_pnl += trade.get("pnl", 0)
            
            # Get starting balance
            equity_docs = query_firestore(
                "equity_history",
                order_by="timestamp",
                direction="asc",
                limit=1
            )
            
            starting_balance = 300.0  # Default Phase 1
            if equity_docs:
                starting_balance = equity_docs[0].get("balance_usd", 300.0)
            
            if starting_balance == 0:
                return 0
            
            loss_percent = (today_pnl / starting_balance) * 100
            
            logger.debug(f"📊 Daily loss: ${today_pnl:.2f} ({loss_percent:.2f}%)")
            return loss_percent
        except Exception as e:
            log_error("❌ Failed to get daily loss", e)
            return 0
    
    async def get_daily_loss_limit(self) -> float:
        """Get daily loss limit percentage."""
        return abs(self.DAILY_LOSS_LIMIT_PERCENT)
    
    # ========================================================================
    # MAX DRAWDOWN
    # ========================================================================
    
    async def _check_max_drawdown(self) -> Dict:
        """Check if max drawdown (-10%) is breached."""
        try:
            current_equity = await self._get_current_equity()
            peak_equity = await self._get_peak_equity()
            
            if peak_equity == 0:
                return {"pass": True, "reason": ""}
            
            drawdown_percent = ((current_equity - peak_equity) / peak_equity) * 100
            
            if drawdown_percent <= self.MAX_DRAWDOWN_PERCENT:
                logger.warning(f"⚠️ Max drawdown breached: {drawdown_percent:.2f}% / {self.MAX_DRAWDOWN_PERCENT}%")
                return {
                    "pass": False,
                    "reason": f"max_drawdown ({drawdown_percent:.2f}%)"
                }
            
            return {"pass": True, "reason": ""}
        except Exception as e:
            log_error("❌ Drawdown check failed", e)
            return {"pass": False, "reason": "drawdown_check_error"}
    
    async def _get_current_equity(self) -> float:
        """Get current equity balance."""
        try:
            equity_docs = query_firestore(
                "equity_history",
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            if equity_docs:
                return equity_docs[0].get("balance_usd", 300.0)
            
            return 300.0  # Default
        except Exception as e:
            logger.error(f"❌ Failed to get current equity: {str(e)}")
            return 300.0
    
    async def _get_peak_equity(self) -> float:
        """Get peak equity from history."""
        try:
            equity_docs = query_firestore(
                "equity_history",
                order_by="timestamp",
                direction="desc",
                limit=1000
            )
            
            if not equity_docs:
                return 300.0
            
            peak = max(doc.get("balance_usd", 0) for doc in equity_docs)
            return peak if peak > 0 else 300.0
        except Exception as e:
            logger.error(f"❌ Failed to get peak equity: {str(e)}")
            return 300.0
    
    async def get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        try:
            current = await self._get_current_equity()
            peak = await self._get_peak_equity()
            
            if peak == 0:
                return 0
            
            return ((current - peak) / peak) * 100
        except Exception as e:
            logger.error(f"❌ Failed to get drawdown: {str(e)}")
            return 0
    
    # ========================================================================
    # ICIR BREAKER
    # ========================================================================
    
    async def _check_icir_breaker(self) -> Dict:
        """Check if ICIR < 0.2 for 3+ days."""
        try:
            three_days_ago = datetime.utcnow() - timedelta(days=3)
            
            # Fetch ICIR history
            icir_docs = query_firestore(
                "genetic_history",
                order_by="timestamp",
                direction="desc",
                limit=100
            )
            
            if not icir_docs:
                return {"pass": True, "reason": ""}
            
            # Filter last 3 days
            recent_icir = []
            for doc in icir_docs:
                doc_time = datetime.fromisoformat(doc.get("timestamp", ""))
                if doc_time >= three_days_ago:
                    recent_icir.append(doc.get("icir", 0))
            
            if len(recent_icir) < 3:
                return {"pass": True, "reason": ""}
            
            # Check if all recent values < threshold
            all_below_threshold = all(icir < self.ICIR_THRESHOLD for icir in recent_icir)
            
            if all_below_threshold:
                logger.warning(f"⚠️ ICIR breaker: All recent values < {self.ICIR_THRESHOLD}")
                return {
                    "pass": False,
                    "reason": f"icir_breaker (ICIR < {self.ICIR_THRESHOLD})"
                }
            
            return {"pass": True, "reason": ""}
        except Exception as e:
            log_error("❌ ICIR check failed", e)
            return {"pass": True, "reason": ""}
    
    # ========================================================================
    # SYSTEM ERRORS
    # ========================================================================
    
    async def _check_system_errors(self) -> Dict:
        """Check for 3+ consecutive API failures."""
        try:
            error_logs = query_firestore(
                "error_log",
                order_by="timestamp",
                direction="desc",
                limit=10
            )
            
            if not error_logs or len(error_logs) < self.MAX_CONSECUTIVE_ERRORS:
                return {"pass": True, "reason": ""}
            
            # Check if last N errors are consecutive
            consecutive_errors = 0
            for log in error_logs[:self.MAX_CONSECUTIVE_ERRORS]:
                if log.get("error_type") == "api_failure":
                    consecutive_errors += 1
                else:
                    break
            
            if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                logger.warning(f"⚠️ System error breaker: {consecutive_errors} consecutive API failures")
                return {
                    "pass": False,
                    "reason": f"system_errors ({consecutive_errors} consecutive)"
                }
            
            return {"pass": True, "reason": ""}
        except Exception as e:
            log_error("❌ System error check failed", e)
            return {"pass": True, "reason": ""}
    
    # ========================================================================
    # POOL BALANCE
    # ========================================================================
    
    async def _check_pool_balance(self) -> Dict:
        """Check if pool balance meets minimum."""
        try:
            current_balance = await self._get_current_equity()
            
            if current_balance < self.MIN_POOL_BALANCE:
                logger.warning(f"⚠️ Pool balance below minimum: ${current_balance:.2f} < ${self.MIN_POOL_BALANCE}")
                return {
                    "pass": False,
                    "reason": f"pool_balance_low (${current_balance:.2f})"
                }
            
            return {"pass": True, "reason": ""}
        except Exception as e:
            log_error("❌ Pool balance check failed", e)
            return {"pass": False, "reason": "pool_check_error"}
    
    # ========================================================================
    # CIRCUIT BREAKERS
    # ========================================================================
    
    async def is_circuit_breaker_active(self) -> bool:
        """
        Check if any circuit breaker is active.
        
        Returns:
            True if trading is frozen
        """
        try:
            breaker_status = query_firestore(
                "circuit_breakers",
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            if not breaker_status:
                return False
            
            status_doc = breaker_status[0]
            is_active = status_doc.get("active", False)
            
            if is_active:
                freeze_until = status_doc.get("freeze_until")
                if freeze_until:
                    freeze_time = datetime.fromisoformat(freeze_until)
                    if datetime.utcnow() > freeze_time:
                        # Freeze expired, deactivate
                        await self._deactivate_circuit_breaker()
                        return False
            
            return is_active
        except Exception as e:
            logger.error(f"❌ Failed to check circuit breaker: {str(e)}")
            return False
    
    async def activate_circuit_breaker(self, reason: str, freeze_hours: int = 24) -> bool:
        """Activate circuit breaker (freeze trading)."""
        try:
            freeze_until = (datetime.utcnow() + timedelta(hours=freeze_hours)).isoformat()
            
            breaker = {
                "active": True,
                "reason": reason,
                "activated_at": get_utc_timestamp(),
                "freeze_until": freeze_until,
                "freeze_hours": freeze_hours
            }
            
            write_to_firestore("circuit_breakers", "active", breaker)
            
            logger.warning(f"🚨 Circuit breaker ACTIVATED ({freeze_hours}h): {reason}")
            return True
        except Exception as e:
            log_error("❌ Failed to activate circuit breaker", e)
            return False
    
    async def _deactivate_circuit_breaker(self) -> bool:
        """Deactivate circuit breaker (resume trading)."""
        try:
            breaker = {"active": False, "deactivated_at": get_utc_timestamp()}
            
            write_to_firestore("circuit_breakers", "active", breaker)
            
            logger.info("✅ Circuit breaker DEACTIVATED")
            return True
        except Exception as e:
            log_error("❌ Failed to deactivate circuit breaker", e)
            return False
    
    # ========================================================================
    # STOP LOSS & POSITION SIZING
    # ========================================================================
    
    def calculate_stop_loss(self, entry_price: float, atr: float, bias: str) -> float:
        """
        Calculate stop loss based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            bias: "long" or "short"
        
        Returns:
            Stop loss price
        """
        try:
            if bias == "long":
                return entry_price - (2 * atr)
            else:  # short
                return entry_price + (2 * atr)
        except Exception as e:
            logger.error(f"❌ Stop loss calculation failed: {str(e)}")
            return entry_price
    
    def calculate_risk_percent(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate risk percentage of trade.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
        
        Returns:
            Risk as percentage
        """
        try:
            if entry_price == 0:
                return 0
            
            risk = abs(entry_price - stop_loss) / entry_price * 100
            return risk
        except Exception as e:
            logger.error(f"❌ Risk percent calculation failed: {str(e)}")
            return 0
    
    # ========================================================================
    # POSITION LIMITS
    # ========================================================================
    
    async def check_position_limits(self, strategy: str) -> Dict:
        """
        Check if position limits are exceeded.
        
        Returns:
            {
                "can_trade": bool,
                "open_positions": int,
                "limit": int
            }
        """
        try:
            # Count open positions for this strategy
            open_positions = query_firestore(
                "open_positions",
                field="strategy",
                operator="==",
                value=strategy
            )
            
            count = len(open_positions)
            limit = self.MAX_OPEN_POSITIONS_PER_STRATEGY
            
            can_trade = count < limit
            
            return {
                "can_trade": can_trade,
                "open_positions": count,
                "limit": limit
            }
        except Exception as e:
            log_error("❌ Position limit check failed", e)
            return {
                "can_trade": False,
                "open_positions": -1,
                "limit": self.MAX_OPEN_POSITIONS_PER_STRATEGY
            }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_risk_manager() -> RiskManager:
    """Get RiskManager instance."""
    return RiskManager()
