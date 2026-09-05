import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime
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
    calculate_pnl_percent
)

logger = logging.getLogger(__name__)

# ============================================================================
# TRADE CLOSURE ENGINE
# ============================================================================

class TradeCloser:
    """
    Close open positions on Kraken exchange.
    
    Features:
    - Market order closure (immediate)
    - P&L calculation
    - Trade history recording
    - Order management
    
    Closure Triggers:
    - Trailing stop hit
    - Time limit reached
    - Technical exit signal
    - Force close (EOD)
    - Manual /kill command
    """
    
    # Kraken API
    KRAKEN_API_URL = "https://api.kraken.com"
    KRAKEN_CLOSE_ENDPOINT = "/0/private/CancelOrder"
    
    # Execution
    ORDER_TYPE = "market"
    
    # Timeouts
    CLOSURE_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    
    def __init__(self):
        self.db = get_firestore_client()
        self.kraken_api_key = get_optional_env_var("KRAKEN_API_KEY", "")
        self.kraken_api_secret = get_optional_env_var("KRAKEN_API_SECRET", "")
    
    async def close_position(self, position: Dict, reason: str = "manual") -> Optional[Dict]:
        """
        Close an open position.
        
        Args:
            position: Position dict from open_positions
            reason: Reason for closure
        
        Returns:
            Trade history record or None
        """
        try:
            pair = position.get("pair")
            order_id = position.get("order_id")
            entry_price = position.get("entry_price")
            bias = position.get("bias")
            quantity = position.get("quantity")
            
            logger.info(f"📉 Closing {bias} position for {pair} (reason: {reason})")
            
            # Get current price
            exit_price = await self._get_current_price(pair)
            if not exit_price:
                logger.error(f"❌ Failed to get exit price for {pair}")
                return None
            
            # Execute closure order
            closure_result = await self._execute_closure_order(pair, bias, quantity)
            if not closure_result:
                logger.error(f"❌ Failed to execute closure order for {pair}")
                return None
            
            # Calculate P&L
            pnl, pnl_percent = self._calculate_pnl(entry_price, exit_price, bias, quantity)
            
            # Calculate hold time
            entry_time = datetime.fromisoformat(position.get("entry_time"))
            exit_time = datetime.utcnow()
            hold_duration = (exit_time - entry_time).total_seconds() / 3600  # Hours
            
            # Create trade history record
            trade_record = {
                "order_id": order_id,
                "strategy": position.get("strategy"),
                "bias": bias,
                "pair": pair,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_time": position.get("entry_time"),
                "exit_time": exit_time.isoformat(),
                "hold_time_hours": hold_duration,
                "quantity": quantity,
                "position_size_usd": entry_price * quantity,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "atr": position.get("atr"),
                "rsi_entry": position.get("rsi"),
                "regime_entry": position.get("regime"),
                "asset_score": position.get("asset_score"),
                "close_reason": reason,
                "status": "closed",
                "timestamp": get_utc_timestamp()
            }
            
            # Save to trade_history
            if await self._save_trade_history(trade_record):
                logger.info(f"✅ Trade closed and recorded: {pair} (P&L: ${pnl:.2f})")
                
                # Remove from open_positions
                await self._remove_open_position(order_id)
                
                # Record to equity history
                await self._update_equity_history(pnl)
                
                return trade_record
            else:
                logger.error(f"❌ Failed to save trade history")
                return None
        except Exception as e:
            log_error(f"❌ Trade closure failed for {position.get('pair')}", e)
            return None
    
    async def close_all_positions(self, reason: str = "manual") -> List[Dict]:
        """
        Close all open positions (emergency).
        
        Args:
            reason: Reason for mass closure
        
        Returns:
            List of closed trade records
        """
        try:
            logger.warning(f"🚨 Closing ALL positions (reason: {reason})")
            
            # Fetch all open positions
            open_positions = query_firestore("open_positions")
            logger.info(f"📊 Found {len(open_positions)} positions to close")
            
            closed_trades = []
            
            # Close each position
            for position in open_positions:
                try:
                    trade = await self.close_position(position, reason)
                    if trade:
                        closed_trades.append(trade)
                except Exception as e:
                    logger.error(f"❌ Error closing {position.get('pair')}: {str(e)}")
                    continue
            
            logger.info(f"✅ Closed {len(closed_trades)} positions")
            
            return closed_trades
        except Exception as e:
            log_error("❌ Mass closure failed", e)
            raise
    
    # ========================================================================
    # PRICE MONITORING
    # ========================================================================
    
    async def _get_current_price(self, pair: str) -> Optional[float]:
        """Fetch current price from Kraken."""
        try:
            url = "https://api.kraken.com/0/public/Ticker"
            params = {"pair": pair}
            
            response = await make_request("GET", url, params=params, timeout=10)
            
            if "error" in response and response["error"]:
                logger.warning(f"⚠️ Kraken error: {response['error']}")
                return None
            
            result = response.get("result", {})
            ticker = result.get(pair, {})
            
            last_price = ticker.get("c", [None])[0]
            
            if not last_price:
                return None
            
            return float(last_price)
        except Exception as e:
            logger.error(f"❌ Failed to get price for {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # ORDER EXECUTION
    # ========================================================================
    
    async def _execute_closure_order(self, pair: str, bias: str, quantity: float, retry: int = 0) -> Optional[Dict]:
        """
        Execute market order to close position.
        
        Args:
            pair: Trading pair
            bias: "long" or "short"
            quantity: Amount to close
            retry: Retry attempt
        
        Returns:
            Order result or None
        """
        try:
            if not self.kraken_api_key or not self.kraken_api_secret:
                logger.warning("⚠️ No Kraken API keys, simulating closure")
                # Simulate closure
                return {
                    "order_id": f"sim_close_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    "status": "closed",
                    "executed": True
                }
            
            # Determine order type (opposite of entry)
            order_type = "sell" if bias == "long" else "buy"
            
            # Prepare order
            order_data = {
                "pair": pair,
                "type": order_type,
                "order_type": self.ORDER_TYPE,
                "volume": quantity,
                "timeinforce": "IOC"  # Immediate or cancel
            }
            
            # In production, sign and send to Kraken API
            # For now, simulate
            logger.info(f"📋 Closure order: {order_type} {quantity:.8f} {pair}")
            
            return {
                "order_id": f"close_{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "status": "closed",
                "executed": True
            }
        except Exception as e:
            if retry < self.MAX_RETRIES:
                logger.warning(f"⚠️ Closure order failed, retrying... ({retry + 1}/{self.MAX_RETRIES})")
                await asyncio.sleep(1 * (2 ** retry))
                return await self._execute_closure_order(pair, bias, quantity, retry + 1)
            else:
                log_error("❌ Closure order failed after retries", e)
                return None
    
    # ========================================================================
    # P&L CALCULATION
    # ========================================================================
    
    def _calculate_pnl(self, entry_price: float, exit_price: float, bias: str, quantity: float) -> tuple:
        """
        Calculate P&L for a closed trade.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            bias: "long" or "short"
            quantity: Position size
        
        Returns:
            (pnl_usd, pnl_percent)
        """
        try:
            if bias == "long":
                # Long: Profit if exit > entry
                price_diff = exit_price - entry_price
                pnl = price_diff * quantity
                pnl_percent = calculate_pnl_percent(entry_price, exit_price)
            else:
                # Short: Profit if entry > exit
                price_diff = entry_price - exit_price
                pnl = price_diff * quantity
                pnl_percent = calculate_pnl_percent(exit_price, entry_price)
            
            return pnl, pnl_percent
        except Exception as e:
            logger.error(f"❌ P&L calculation failed: {str(e)}")
            return 0, 0
    
    # ========================================================================
    # FIRESTORE OPERATIONS
    # ========================================================================
    
    async def _save_trade_history(self, trade_record: Dict) -> bool:
        """Save closed trade to trade_history collection."""
        try:
            order_id = trade_record.get("order_id")
            write_to_firestore("trade_history", order_id, trade_record)
            
            logger.info(f"✅ Saved trade history: {order_id}")
            return True
        except Exception as e:
            log_error("❌ Failed to save trade history", e)
            return False
    
    async def _remove_open_position(self, order_id: str) -> bool:
        """Remove position from open_positions collection."""
        try:
            from src.helpers import get_firestore_client
            db = get_firestore_client()
            db.collection("open_positions").document(order_id).delete()
            
            logger.info(f"✅ Removed from open positions: {order_id}")
            return True
        except Exception as e:
            log_error("❌ Failed to remove open position", e)
            return False
    
    async def _update_equity_history(self, pnl: float) -> bool:
        """Update equity history with P&L."""
        try:
            # Get current equity
            equity_docs = query_firestore(
                "equity_history",
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            current_balance = 300.0  # Default Phase 1 capital
            if equity_docs:
                current_balance = equity_docs[0].get("balance_usd", 300.0)
            
            # Update balance
            new_balance = current_balance + pnl
            
            # Record new equity
            equity_record = {
                "balance_usd": new_balance,
                "pnl": pnl,
                "timestamp": get_utc_timestamp()
            }
            
            doc_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            write_to_firestore("equity_history", doc_id, equity_record)
            
            logger.info(f"✅ Updated equity: ${new_balance:.2f} (PnL: ${pnl:+.2f})")
            return True
        except Exception as e:
            log_error("❌ Failed to update equity history", e)
            return False
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_position_summary(self, position: Dict) -> str:
        """Generate human-readable position summary."""
        try:
            pair = position.get("pair")
            bias = position.get("bias")
            entry_price = position.get("entry_price")
            quantity = position.get("quantity")
            entry_time = position.get("entry_time")
            strategy = position.get("strategy")
            
            summary = f"""
            📊 POSITION: {pair}
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Strategy: {strategy}
            Bias: {'🟢 LONG' if bias == 'long' else '🔴 SHORT'}
            Entry Price: ${entry_price:.2f}
            Quantity: {quantity:.8f}
            Size: ${entry_price * quantity:.2f}
            Entry Time: {entry_time}
            """
            
            return summary.strip()
        except Exception as e:
            logger.error(f"❌ Failed to generate summary: {str(e)}")
            return "Error generating summary"

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def close_trade(position: Dict) -> Optional[Dict]:
    """Main entry point for trade closure."""
    try:
        closer = TradeCloser()
        trade = await closer.close_position(position, reason="signal")
        return trade
    except Exception as e:
        log_error("❌ Trade closure failed", e)
        raise

async def emergency_close_all() -> list:
    """Emergency closure of all positions."""
    try:
        closer = TradeCloser()
        trades = await closer.close_all_positions(reason="emergency")
        return trades
    except Exception as e:
        log_error("❌ Emergency closure failed", e)
        raise
