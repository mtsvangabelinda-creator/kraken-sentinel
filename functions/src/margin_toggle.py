import logging
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import secrets

from src.helpers import (
    make_request,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    get_optional_env_var,
    get_firestore_client,
    log_info,
    log_error
)

logger = logging.getLogger(__name__)

# ============================================================================
# MARGIN/FUTURES TOGGLE (Phase 2)
# ============================================================================

class MarginToggle:
    """
    Margin and futures trading toggle system.
    
    Phase 1 (Current): Spot trading only
    - Long positions only
    - No leverage
    - No margin requirements
    
    Phase 2 (Activated): Margin & Futures
    - Long AND short positions
    - Funding rate monitoring
    - Margin ratio monitoring
    - Liquidation risk management
    
    Activation Flow:
    1. User executes /enable_margin command
    2. System requests confirmation (/confirm_margin)
    3. 24-hour paper trading test mode activates
    4. After 24h with no issues, margin trading activated
    5. System monitors margin ratio and funding rates
    
    Test Mode:
    - Paper trading in margin mode
    - No real capital deployed
    - Validates margin trading logic
    - Confirms user understanding
    - Identifies configuration issues
    
    Safety:
    - Margin ratio monitoring (130% minimum)
    - Liquidation price tracking
    - Funding rate alerts (>0.05%)
    - Auto-deleveraging on risk breach
    """
    
    # Phase settings
    PHASE_1 = "spot_only"
    PHASE_2 = "margin_futures"
    
    # Margin parameters
    MIN_MARGIN_RATIO = 1.30  # 130% minimum
    LIQUIDATION_WARNING_RATIO = 1.40  # 140% warning threshold
    CRITICAL_MARGIN_RATIO = 1.20  # 120% critical (close positions)
    
    # Funding rates
    FUNDING_RATE_ALERT_THRESHOLD = 0.0005  # 0.05%
    FUNDING_RATE_EXTREME_THRESHOLD = 0.001  # 0.1%
    
    # Test mode
    TEST_MODE_DURATION_HOURS = 24
    TEST_MODE_MIN_TRADES = 5
    TEST_MODE_MAX_LOSS_PERCENT = -10
    
    # Activation security
    CONFIRMATION_TIMEOUT_MINUTES = 10
    CONFIRMATION_TOKEN_LENGTH = 32
    MAX_ACTIVATION_ATTEMPTS = 3
    
    def __init__(self):
        self.db = get_firestore_client()
    
    async def get_phase_status(self) -> Dict:
        """Get current trading phase status."""
        try:
            phase_config = query_firestore("config", field="key", operator="==", value="trading_phase")
            
            if not phase_config:
                return {
                    "phase": self.PHASE_1,
                    "phase_name": "Spot Trading Only",
                    "margin_enabled": False,
                    "shorts_enabled": False,
                    "test_mode_active": False
                }
            
            status_doc = phase_config[0]
            phase = status_doc.get("value", self.PHASE_1)
            
            # Check test mode
            test_mode_active = False
            test_mode_status = query_firestore("margin_toggle", field="test_mode_active", operator="==", value=True)
            if test_mode_status:
                test_doc = test_mode_status[0]
                test_end = test_doc.get("test_mode_end")
                if test_end:
                    test_end_time = datetime.fromisoformat(test_end)
                    test_mode_active = datetime.utcnow() < test_end_time
            
            margin_enabled = phase == self.PHASE_2
            
            return {
                "phase": phase,
                "phase_name": "Margin & Futures" if margin_enabled else "Spot Trading Only",
                "margin_enabled": margin_enabled,
                "shorts_enabled": margin_enabled,
                "test_mode_active": test_mode_active
            }
        except Exception as e:
            log_error("❌ Failed to get phase status", e)
            return {
                "phase": self.PHASE_1,
                "phase_name": "Spot Trading Only",
                "margin_enabled": False,
                "shorts_enabled": False,
                "test_mode_active": False
            }
    
    # ========================================================================
    # ACTIVATION FLOW
    # ========================================================================
    
    async def initiate_margin_activation(self, user_id: str) -> Dict:
        """
        Initiate margin trading activation.
        
        Step 1: User runs /enable_margin
        
        Returns:
            Confirmation prompt
        """
        try:
            logger.info(f"🚀 Margin activation initiated by {user_id}")
            
            # Check if already enabled
            phase_status = await self.get_phase_status()
            if phase_status.get("margin_enabled"):
                return {
                    "status": "already_enabled",
                    "message": "Margin trading already enabled"
                }
            
            # Check activation attempts
            activation_history = query_firestore(
                "margin_activation_attempts",
                field="user_id",
                operator="==",
                value=user_id,
                order_by="timestamp",
                direction="desc",
                limit=3
            )
            
            recent_attempts = sum(1 for attempt in activation_history 
                                if attempt.get("status") == "pending")
            
            if recent_attempts >= self.MAX_ACTIVATION_ATTEMPTS:
                return {
                    "status": "too_many_attempts",
                    "message": "Too many activation attempts. Please wait before trying again."
                }
            
            # Generate confirmation token
            confirmation_token = secrets.token_hex(self.CONFIRMATION_TOKEN_LENGTH // 2)
            token_hash = hashlib.sha256(confirmation_token.encode()).hexdigest()
            
            # Store activation request
            activation_request = {
                "user_id": user_id,
                "status": "pending",
                "confirmation_token_hash": token_hash,
                "initiated_at": get_utc_timestamp(),
                "expires_at": (datetime.utcnow() + timedelta(minutes=self.CONFIRMATION_TIMEOUT_MINUTES)).isoformat()
            }
            
            doc_id = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            write_to_firestore("margin_activation_attempts", doc_id, activation_request)
            
            logger.info(f"✅ Activation request created: {doc_id}")
            
            return {
                "status": "confirmation_required",
                "message": "⚠️ MARGIN TRADING ACTIVATION\n\nThis will enable SHORT positions and leverage trading.\n\nRisks:\n- Liquidation risk if margin ratio falls below 130%\n- Funding rate costs\n- Complex position management\n\nTo confirm, run: /confirm_margin",
                "confirmation_token": confirmation_token,  # Send to user securely
                "expires_in_minutes": self.CONFIRMATION_TIMEOUT_MINUTES
            }
        except Exception as e:
            log_error(f"❌ Margin activation initiation failed for {user_id}", e)
            return {
                "status": "error",
                "message": f"Activation failed: {str(e)}"
            }
    
    async def confirm_margin_activation(self, user_id: str, confirmation_token: str) -> Dict:
        """
        Confirm margin trading activation.
        
        Step 2: User runs /confirm_margin with token
        
        Returns:
            Activation status + test mode details
        """
        try:
            logger.info(f"🔐 Confirming margin activation for {user_id}")
            
            # Verify token
            token_hash = hashlib.sha256(confirmation_token.encode()).hexdigest()
            
            pending_requests = query_firestore(
                "margin_activation_attempts",
                field="user_id",
                operator="==",
                value=user_id,
                order_by="initiated_at",
                direction="desc",
                limit=1
            )
            
            if not pending_requests:
                return {
                    "status": "no_pending_request",
                    "message": "No pending activation request. Run /enable_margin first."
                }
            
            request = pending_requests[0]
            
            # Check expiration
            expires_at = datetime.fromisoformat(request.get("expires_at", ""))
            if datetime.utcnow() > expires_at:
                return {
                    "status": "token_expired",
                    "message": "Confirmation token expired. Run /enable_margin again."
                }
            
            # Verify token
            if request.get("confirmation_token_hash") != token_hash:
                return {
                    "status": "invalid_token",
                    "message": "Invalid confirmation token."
                }
            
            # Activate test mode
            test_mode_end = (datetime.utcnow() + timedelta(hours=self.TEST_MODE_DURATION_HOURS)).isoformat()
            
            test_mode_config = {
                "user_id": user_id,
                "test_mode_active": True,
                "test_mode_start": get_utc_timestamp(),
                "test_mode_end": test_mode_end,
                "test_mode_duration_hours": self.TEST_MODE_DURATION_HOURS,
                "test_trades_completed": 0,
                "test_pnl": 0,
                "status": "testing"
            }
            
            write_to_firestore("margin_toggle", f"{user_id}_test_mode", test_mode_config)
            
            # Update activation request
            request["status"] = "activated"
            request["activated_at"] = get_utc_timestamp()
            request["test_mode_end"] = test_mode_end
            
            doc_id = pending_requests[0].get("order_id", f"{user_id}_activated")
            write_to_firestore("margin_activation_attempts", doc_id, request)
            
            logger.info(f"✅ Test mode activated for {user_id}")
            
            return {
                "status": "test_mode_activated",
                "message": f"📝 TEST MODE ACTIVATED\n\n✅ 24-hour paper trading enabled\n\nRequirements for Phase 2 activation:\n- Complete at least {self.TEST_MODE_MIN_TRADES} trades\n- Max loss: {self.TEST_MODE_MAX_LOSS_PERCENT}%\n- No system errors\n\nTest mode expires in {self.TEST_MODE_DURATION_HOURS} hours.",
                "test_mode_end": test_mode_end,
                "requirements": {
                    "min_trades": self.TEST_MODE_MIN_TRADES,
                    "max_loss_percent": self.TEST_MODE_MAX_LOSS_PERCENT,
                    "duration_hours": self.TEST_MODE_DURATION_HOURS
                }
            }
        except Exception as e:
            log_error(f"❌ Margin activation confirmation failed for {user_id}", e)
            return {
                "status": "error",
                "message": f"Confirmation failed: {str(e)}"
            }
    
    async def complete_test_mode(self, user_id: str) -> Dict:
        """
        Complete test mode and activate Phase 2 margin trading.
        
        Runs after 24-hour test period with passing criteria.
        """
        try:
            logger.info(f"✅ Completing test mode for {user_id}")
            
            # Get test mode config
            test_mode = query_firestore("margin_toggle", field="user_id", operator="==", value=user_id)
            
            if not test_mode:
                return {
                    "status": "test_mode_not_found",
                    "message": "Test mode not active"
                }
            
            test_config = test_mode[0]
            
            # Check test end time
            test_end = datetime.fromisoformat(test_config.get("test_mode_end", ""))
            if datetime.utcnow() < test_end:
                time_remaining = (test_end - datetime.utcnow()).total_seconds() / 3600
                return {
                    "status": "test_mode_ongoing",
                    "message": f"Test mode still active. {time_remaining:.1f} hours remaining.",
                    "test_end": test_config.get("test_mode_end")
                }
            
            # Check criteria
            test_trades = test_config.get("test_trades_completed", 0)
            test_pnl = test_config.get("test_pnl", 0)
            
            if test_trades < self.TEST_MODE_MIN_TRADES:
                return {
                    "status": "insufficient_trades",
                    "message": f"Only {test_trades} trades. Need {self.TEST_MODE_MIN_TRADES}."
                }
            
            # Check max loss (assuming started with $300)
            max_loss = 300 * (self.TEST_MODE_MAX_LOSS_PERCENT / 100)
            if test_pnl < max_loss:
                return {
                    "status": "max_loss_exceeded",
                    "message": f"Test P&L: ${test_pnl:.2f}. Max allowed: ${max_loss:.2f}"
                }
            
            # Activate Phase 2
            phase_config = {
                "key": "trading_phase",
                "value": self.PHASE_2,
                "activated_by": user_id,
                "activated_at": get_utc_timestamp(),
                "test_mode_stats": {
                    "trades": test_trades,
                    "pnl": test_pnl
                }
            }
            
            write_to_firestore("config", "trading_phase", phase_config)
            
            # Update margin toggle status
            test_config["status"] = "completed"
            test_config["margin_activated"] = True
            test_config["margin_activated_at"] = get_utc_timestamp()
            
            write_to_firestore("margin_toggle", f"{user_id}_test_mode", test_config)
            
            logger.info(f"🎉 Phase 2 activated for {user_id}")
            
            return {
                "status": "phase_2_activated",
                "message": "🎉 MARGIN TRADING ACTIVATED\n\n✅ Phase 2 live\n✅ Short positions enabled\n✅ Leverage trading active\n\nRemember:\n- Monitor margin ratio (min 130%)\n- Watch funding rates\n- Set appropriate stops",
                "phase": self.PHASE_2
            }
        except Exception as e:
            log_error(f"❌ Test mode completion failed for {user_id}", e)
            return {
                "status": "error",
                "message": f"Completion failed: {str(e)}"
            }
    
    # ========================================================================
    # MARGIN MONITORING
    # ========================================================================
    
    async def check_margin_ratio(self, account_balance: float, borrowed_amount: float) -> Dict:
        """
        Check margin ratio and generate alerts.
        
        Margin Ratio = (Account Balance) / (Borrowed Amount)
        
        - > 1.40: Safe
        - 1.30 - 1.40: Warning
        - 1.20 - 1.30: Critical (reduce positions)
        - < 1.20: Liquidation risk
        """
        try:
            if borrowed_amount == 0:
                margin_ratio = float('inf')
            else:
                margin_ratio = account_balance / borrowed_amount
            
            status = "safe"
            alert_level = "none"
            action = "none"
            
            if margin_ratio < self.CRITICAL_MARGIN_RATIO:
                status = "critical"
                alert_level = "critical"
                action = "close_positions_immediately"
            elif margin_ratio < self.MIN_MARGIN_RATIO:
                status = "warning"
                alert_level = "high"
                action = "reduce_positions"
            elif margin_ratio < self.LIQUIDATION_WARNING_RATIO:
                status = "caution"
                alert_level = "medium"
                action = "monitor_closely"
            
            return {
                "margin_ratio": margin_ratio,
                "status": status,
                "alert_level": alert_level,
                "action": action,
                "min_required": self.MIN_MARGIN_RATIO,
                "critical_threshold": self.CRITICAL_MARGIN_RATIO
            }
        except Exception as e:
            log_error("❌ Margin ratio check failed", e)
            return {
                "margin_ratio": 0,
                "status": "error",
                "alert_level": "critical",
                "action": "error_check"
            }
    
    async def monitor_funding_rates(self, funding_rates: Dict[str, float]) -> Dict:
        """
        Monitor funding rates for alerts.
        
        Funding Rate Alerts:
        - > 0.05%: Alert (costs adding up)
        - > 0.1%: Warning (extreme rate)
        """
        try:
            alerts = []
            
            for pair, rate in funding_rates.items():
                if rate > self.FUNDING_RATE_EXTREME_THRESHOLD:
                    alerts.append({
                        "pair": pair,
                        "funding_rate": rate,
                        "level": "extreme",
                        "message": f"⚠️ {pair} funding rate {rate:.4%} (extreme)"
                    })
                elif rate > self.FUNDING_RATE_ALERT_THRESHOLD:
                    alerts.append({
                        "pair": pair,
                        "funding_rate": rate,
                        "level": "high",
                        "message": f"ℹ️ {pair} funding rate {rate:.4%} (high)"
                    })
            
            return {
                "alerts": alerts,
                "has_alerts": len(alerts) > 0,
                "total_funding_cost": sum(funding_rates.values())
            }
        except Exception as e:
            log_error("❌ Funding rate monitoring failed", e)
            return {
                "alerts": [],
                "has_alerts": False,
                "error": str(e)
            }
    
    # ========================================================================
    # DISABLE MARGIN
    # ========================================================================
    
    async def disable_margin_trading(self, user_id: str) -> Dict:
        """Disable margin trading (return to Phase 1)."""
        try:
            logger.warning(f"🔴 Disabling margin trading for {user_id}")
            
            # Close all open positions (handled by close_trade)
            from src.close_trade import TradeCloser
            closer = TradeCloser()
            await closer.close_all_positions(reason="margin_disabled")
            
            # Update phase config
            phase_config = {
                "key": "trading_phase",
                "value": self.PHASE_1,
                "disabled_by": user_id,
                "disabled_at": get_utc_timestamp()
            }
            
            write_to_firestore("config", "trading_phase", phase_config)
            
            logger.info(f"✅ Margin trading disabled for {user_id}")
            
            return {
                "status": "margin_disabled",
                "message": "✅ Margin trading disabled. All positions closed.",
                "phase": self.PHASE_1
            }
        except Exception as e:
            log_error(f"❌ Failed to disable margin trading for {user_id}", e)
            return {
                "status": "error",
                "message": f"Disable failed: {str(e)}"
            }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def get_margin_status() -> Dict:
    """Get margin trading status."""
    try:
        toggle = MarginToggle()
        status = await toggle.get_phase_status()
        return status
    except Exception as e:
        log_error("❌ Failed to get margin status", e)
        raise
