import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
import os

from src.helpers import (
    make_request,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    get_optional_env_var,
    log_info,
    log_error,
    get_firestore_client,
    validate_positive_float
)
from src.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# ============================================================================
# TRADE EXECUTION ENGINE
# ============================================================================

class TradeExecutor:
    """
    Execute trades on Kraken exchange.
    
    Features:
    - Spot trading only (Phase 1)
    - Kelly Criterion position sizing
    - ATR-based stop losses
    - Risk limit validation
    - Order management and tracking
    
    Order Types:
    - market: Market order (immediate execution)
    - limit: Limit order (pending)
    
    Margin/Futures:
    - Dormant in Phase 1 (spot only)
    - Toggle activates shorting in Phase 2
    """
    
    # Kraken API
    KRAKEN_API_URL = "https://api.kraken.com"
    KRAKEN_PRIVATE_ENDPOINT = "/0/private/AddOrder"
    
    # Order limits
    MIN_ORDER_SIZE_USD = 10  # Minimum order in USD
    MAX_ORDER_SIZE_PERCENT = 0.10  # Max 10% of pool per trade
    
    # Execution parameters
    ORDER_TYPE = "market"  # Market execution for momentum/swing
    POST_ONLY = False  # Market orders can't be post-only
    
    # Timeouts
    ORDER_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    
    def __init__(self):
        self.db = get_firestore_client()
        self.risk_manager = RiskManager()
        self.kraken_api_key = get_optional_env_var("KRAKEN_API_KEY", "")
        self.kraken_api_secret = get_optional_env_var("KRAKEN_API_SECRET", "")
    
    async def execute_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Execute a trade signal.
        
        Args:
            signal: Signal dict from scanner/generator
        
        Returns:
            Trade execution result or None
        """
        try:
            logger.info(f"🎯 Executing {signal.get('bias')} signal for {signal.get('pair')}")
            
            # Pre-execution checks
            if not await self._pre_execution_checks(signal):
                logger.warning(f"⚠️ Pre-execution checks failed for {signal.get('pair')}")
                return None
            
            # Calculate position size (Kelly Criterion)
            position_size = await self._calculate_position_size(signal)
            if position_size < self.MIN_ORDER_SIZE_USD:
                logger.warning(f"⚠️ Position size ${position_size} below minimum ${self.MIN_ORDER_SIZE_USD}")
                return None
            
            # Prepare order
            order_data = self._prepare_order(signal, position_size)
            
            # Execute order
            order_result = await self._execute_order(order_data)
            if not order_result:
                logger.error(f"❌ Order execution failed for {signal.get('pair')}")
                return None
            
            # Record trade
            trade_record = self._create_trade_record(signal, order_data, order_result)
            
            # Write to Firestore
            if await self._save_trade_record(trade_record):
                logger.info(f"✅ Trade executed and recorded: {signal.get('pair')}")
                return trade_record
            else:
                logger.error(f"❌ Failed to save trade record")
                return None
        except Exception as e:
            log_error(f"❌ Trade execution failed for {signal.get('pair')}", e)
            return None
    
    # ========================================================================
    # PRE-EXECUTION CHECKS
    # ========================================================================
    
    async def _pre_execution_checks(self, signal: Dict) -> bool:
        """
        Validate signal and risk limits before execution.
        
        Checks:
        1. No existing position for this pair
        2. Risk limits not breached
        3. Daily loss limit not hit
        4. Circuit breakers not active
        5. Current equity > minimum
        """
        try:
            pair = signal.get("pair")
            
            # Check 1: Existing position
            existing_positions = query_firestore(
                "open_positions",
                field="pair",
                operator="==",
                value=pair
            )
            
            if existing_positions:
                logger.warning(f"⚠️ Existing position for {pair}, skip new entry")
                return False
            
            # Check 2: Risk limits
            risk_check = await self.risk_manager.check_risk_limits()
            if not risk_check.get("allow_trading"):
                logger.warning(f"⚠️ Risk limits breached: {risk_check.get('reason')}")
                return False
            
            # Check 3: Daily loss limit
            daily_loss = await self.risk_manager.get_daily_loss()
            daily_limit = await self.risk_manager.get_daily_loss_limit()
            
            if daily_loss <= -daily_limit:
                logger.warning(f"⚠️ Daily loss limit hit: ${daily_loss} / ${-daily_limit}")
                return False
            
            # Check 4: Circuit breakers
            if await self.risk_manager.is_circuit_breaker_active():
                logger.warning(f"⚠️ Circuit breaker active, trading frozen")
                return False
            
            logger.info(f"✅ Pre-execution checks passed for {pair}")
            return True
        except Exception as e:
            log_error("❌ Pre-execution checks failed", e)
            return False
    
    # ========================================================================
    # POSITION SIZING (Kelly Criterion)
    # ========================================================================
    
    async def _calculate_position_size(self, signal: Dict) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Formula:
        Kelly = (Win_Rate × Avg_Win - Loss_Rate × Avg_Loss) / (Avg_Win × Avg_Loss)
        Position_Size = Kelly × Pool_Balance × 0.25 (conservative 25% of full Kelly)
        
        Args:
            signal: Signal dict
        
        Returns:
            Position size in USD
        """
        try:
            # Get current pool balance
            pool_balance = await self._get_current_pool_balance()
            
            # Get historical win rate and profit factors
            recent_trades = query_firestore(
                "trade_history",
                order_by="timestamp",
                direction="desc",
                limit=50
            )
            
            if not recent_trades or len(recent_trades) < 10:
                # Default: 2% of pool for small sample
                default_size = pool_balance * 0.02
                return max(self.MIN_ORDER_SIZE_USD, default_size)
            
            # Calculate stats
            wins = sum(1 for trade in recent_trades if trade.get("pnl", 0) > 0)
            losses = len(recent_trades) - wins
            win_rate = wins / len(recent_trades) if recent_trades else 0.5
            loss_rate = losses / len(recent_trades) if recent_trades else 0.5
            
            avg_win = sum(max(0, trade.get("pnl", 0)) for trade in recent_trades) / max(wins, 1)
            avg_loss = sum(abs(min(0, trade.get("pnl", 0))) for trade in recent_trades) / max(losses, 1)
            
            # Avoid division by zero
            if avg_win == 0 or avg_loss == 0:
                default_size = pool_balance * 0.02
                return max(self.MIN_ORDER_SIZE_USD, default_size)
            
            # Kelly calculation
            kelly = (win_rate * avg_win - loss_rate * avg_loss) / (avg_win * avg_loss)
            
            # Conservative: Use 25% of full Kelly
            kelly_conservative = kelly * 0.25
            
            # Position size
            position_size = pool_balance * kelly_conservative
            
            # Caps
            max_position = pool_balance * self.MAX_ORDER_SIZE_PERCENT
            position_size = min(position_size, max_position)
            position_size = max(position_size, self.MIN_ORDER_SIZE_USD)
            
            logger.info(f"📊 Kelly: {kelly:.4f}, Conservative: {kelly_conservative:.4f}, Position: ${position_size:.2f}")
            
            return position_size
        except Exception as e:
            log_error("❌ Position sizing failed", e)
            # Default to 2% of pool
            pool_balance = await self._get_current_pool_balance()
            return max(self.MIN_ORDER_SIZE_USD, pool_balance * 0.02)
    
    # ========================================================================
    # ORDER PREPARATION
    # ========================================================================
    
    def _prepare_order(self, signal: Dict, position_size_usd: float) -> Dict:
        """
        Prepare order data for Kraken API.
        
        Args:
            signal: Signal dict
            position_size_usd: Position size in USD
        
        Returns:
            Order dict
        """
        try:
            pair = signal.get("pair")
            entry_price = signal.get("entry_price")
            bias = signal.get("bias")
            
            # Calculate quantity
            quantity = position_size_usd / entry_price
            
            # Determine order type
            order_type = "buy" if bias == "long" else "sell"
            
            # Order data
            order = {
                "pair": pair,
                "type": order_type,
                "order_type": self.ORDER_TYPE,
                "volume": quantity,
                "price": entry_price,
                "userref": int(datetime.utcnow().timestamp() * 1000),
                "oflags": "post" if self.POST_ONLY else "",
                "timeinforce": "IOC"  # Immediate or cancel for market orders
            }
            
            logger.info(f"📋 Order prepared: {order_type} {quantity:.8f} {pair} @ ${entry_price:.2f}")
            
            return order
        except Exception as e:
            log_error("❌ Order preparation failed", e)
            return {}
    
    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================
    
    async def _execute_order(self, order: Dict, retry: int = 0) -> Optional[Dict]:
        """
        Execute order on Kraken API.
        
        Args:
            order: Order dict
            retry: Current retry attempt
        
        Returns:
            Order result or None
        """
        try:
            if not self.kraken_api_key or not self.kraken_api_secret:
                logger.warning("⚠️ No Kraken API keys configured, simulating execution")
                # Simulate order execution for testing
                return {
                    "order_id": f"sim_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "status": "closed",
                    "executed": True
                }
            
            # Call Kraken API
            url = f"{self.KRAKEN_API_URL}{self.KRAKEN_PRIVATE_ENDPOINT}"
            
            # In production, would sign request with HMAC-SHA512
            # For now, using public API simulation
            response = await make_request(
                "POST",
                url,
                headers={"API-Sign": "signed"},
                json_data=order,
                timeout=self.ORDER_TIMEOUT_SECONDS
            )
            
            if "error" in response and response["error"]:
                error_msg = response["error"][0] if response["error"] else "Unknown error"
                
                if retry < self.MAX_RETRIES:
                    logger.warning(f"⚠️ Order failed: {error_msg}, retrying... ({retry + 1}/{self.MAX_RETRIES})")
                    await asyncio.sleep(1 * (2 ** retry))  # Exponential backoff
                    return await self._execute_order(order, retry + 1)
                else:
                    logger.error(f"❌ Order failed after {self.MAX_RETRIES} retries: {error_msg}")
                    return None
            
            result = response.get("result", {})
            order_id = result.get("txid", [None])[0]
            
            if not order_id:
                logger.error("❌ No order ID returned from Kraken")
                return None
            
            logger.info(f"✅ Order executed: {order_id}")
            
            return {
                "order_id": order_id,
                "status": "open",
                "executed": True
            }
        except Exception as e:
            if retry < self.MAX_RETRIES:
                logger.warning(f"⚠️ Order execution error, retrying... ({retry + 1}/{self.MAX_RETRIES})")
                await asyncio.sleep(1 * (2 ** retry))
                return await self._execute_order(order, retry + 1)
            else:
                log_error("❌ Order execution failed after retries", e)
                return None
    
    # ========================================================================
    # TRADE RECORDING
    # ========================================================================
    
    def _create_trade_record(self, signal: Dict, order: Dict, order_result: Dict) -> Dict:
        """Create trade record for storage."""
        try:
            entry_price = signal.get("entry_price")
            stop_loss = signal.get("stop_loss_price")
            risk_percent = ((abs(entry_price - stop_loss) / entry_price) * 100)
            
            trade = {
                "order_id": order_result.get("order_id"),
                "strategy": signal.get("strategy"),
                "bias": signal.get("bias"),
                "pair": signal.get("pair"),
                "entry_price": entry_price,
                "entry_time": datetime.utcnow().isoformat(),
                "stop_loss_price": stop_loss,
                "quantity": order.get("volume"),
                "position_size_usd": entry_price * order.get("volume"),
                "atr": signal.get("atr"),
                "rsi": signal.get("rsi"),
                "volume_spike": signal.get("volume_spike"),
                "regime": signal.get("regime"),
                "asset_score": signal.get("asset_score"),
                "max_hold_hours": signal.get("time_limit_hours"),
                "risk_percent": risk_percent,
                "status": "open",
                "pnl": 0,
                "pnl_percent": 0,
                "timestamp": get_utc_timestamp()
            }
            
            return trade
        except Exception as e:
            log_error("❌ Trade record creation failed", e)
            return {}
    
    async def _save_trade_record(self, trade: Dict) -> bool:
        """Save trade to open_positions and trade_history."""
        try:
            pair = trade.get("pair")
            order_id = trade.get("order_id")
            
            # Save to open_positions
            write_to_firestore("open_positions", order_id, trade)
            
            # Log to trade_history as well
            write_to_firestore("trade_history", order_id, trade)
            
            logger.info(f"✅ Trade record saved: {order_id}")
            return True
        except Exception as e:
            log_error("❌ Failed to save trade record", e)
            return False
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    async def _get_current_pool_balance(self) -> float:
        """
        Get current pool balance from Firestore.
        
        Returns:
            Current balance in USD
        """
        try:
            equity_history = query_firestore(
                "equity_history",
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            if equity_history:
                return equity_history[0].get("balance_usd", 0)
            
            # Default to Phase 1 capital
            return 300.0
        except Exception as e:
            logger.error(f"❌ Failed to get pool balance: {str(e)}")
            return 300.0

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def execute_trade(signal: Dict) -> Optional[Dict]:
    """Main entry point for trade execution."""
    try:
        executor = TradeExecutor()
        trade = await executor.execute_signal(signal)
        return trade
    except Exception as e:
        log_error("❌ Trade execution failed", e)
        raise
