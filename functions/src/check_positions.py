import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz

from src.helpers import (
    make_request,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    get_firestore_client,
    log_info,
    log_error
)

logger = logging.getLogger(__name__)

# ============================================================================
# POSITION MONITOR
# ============================================================================

class PositionMonitor:
    """
    Monitor open positions and check exit conditions.
    
    Exit Triggers:
    1. Trailing Stop: Price drops -2x ATR from peak (HIGH)
    2. Time Limit: Max hold time reached (HIGH)
    3. Momentum Death: Price < MA50 for long (MEDIUM)
    4. Momentum Reversal: Price > MA50 for short (MEDIUM)
    5. Daily Cutoff: 22:00 UTC for shorts (LOW)
    6. Force Close: 23:00 UTC all positions (CRITICAL)
    
    Features:
    - Real-time price monitoring
    - Peak tracking (for trailing stops)
    - Time limit validation
    - Technical indicator confirmation
    - Forced closure at EOD
    """
    
    # Kraken API
    KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker"
    
    # Exit parameters
    TRAILING_STOP_DISTANCE = 2.0  # 2x ATR
    TIME_LIMIT_BUFFER_MINUTES = 5  # Close 5 min before limit
    
    # Cutoff times (UTC)
    DAILY_CUTOFF_HOUR = 22  # 22:00 UTC (EOD prep)
    FORCE_CLOSE_HOUR = 23   # 23:00 UTC (critical)
    
    def __init__(self):
        self.db = get_firestore_client()
    
    async def check_all_positions(self) -> List[Dict]:
        """
        Check all open positions for exit conditions.
        
        Returns:
            List of positions requiring closure
        """
        try:
            logger.info("🔎 Checking all open positions...")
            
            # Fetch all open positions
            open_positions = query_firestore("open_positions")
            logger.info(f"📊 Found {len(open_positions)} open positions")
            
            positions_to_close = []
            
            # Check each position
            for position in open_positions:
                try:
                    should_close, reason = await self._check_position_exit(position)
                    
                    if should_close:
                        positions_to_close.append({
                            "position": position,
                            "reason": reason,
                            "timestamp": get_utc_timestamp()
                        })
                except Exception as e:
                    logger.error(f"❌ Error checking position {position.get('pair')}: {str(e)}")
                    continue
            
            logger.info(f"✅ Position check complete: {len(positions_to_close)} to close")
            
            return positions_to_close
        except Exception as e:
            log_error("❌ Position check failed", e)
            raise
    
    async def _check_position_exit(self, position: Dict) -> tuple:
        """
        Check if a position should be exited.
        
        Returns:
            (should_close: bool, reason: str)
        """
        try:
            pair = position.get("pair")
            entry_price = position.get("entry_price")
            entry_time = position.get("entry_time")
            stop_loss = position.get("stop_loss_price")
            max_hold_hours = position.get("max_hold_hours")
            atr = position.get("atr")
            bias = position.get("bias")
            
            # Get current price
            current_price = await self._get_current_price(pair)
            if not current_price:
                return False, "No price data"
            
            # Check 1: Force close (23:00 UTC - CRITICAL)
            if self._is_force_close_time():
                logger.warning(f"⚠️ Force close time reached for {pair}")
                return True, "force_close_23utc"
            
            # Check 2: Trailing stop (HIGH)
            peak_price = position.get("peak_price", entry_price)
            trailing_stop_price = peak_price - (self.TRAILING_STOP_DISTANCE * atr)
            
            if bias == "long" and current_price < trailing_stop_price:
                logger.info(f"🛑 Trailing stop hit for {pair} LONG: ${current_price:.2f} < ${trailing_stop_price:.2f}")
                return True, "trailing_stop_long"
            
            if bias == "short" and current_price > trailing_stop_price:
                logger.info(f"🛑 Trailing stop hit for {pair} SHORT: ${current_price:.2f} > ${trailing_stop_price:.2f}")
                return True, "trailing_stop_short"
            
            # Check 3: Time limit (HIGH)
            entry_datetime = datetime.fromisoformat(entry_time)
            hold_duration = datetime.utcnow() - entry_datetime
            max_hold_duration = timedelta(hours=max_hold_hours - (self.TIME_LIMIT_BUFFER_MINUTES / 60))
            
            if hold_duration > max_hold_duration:
                logger.info(f"⏱️ Time limit reached for {pair}: {hold_duration.total_seconds()/3600:.1f}h > {max_hold_hours}h")
                return True, "time_limit"
            
            # Check 4: Momentum death (MEDIUM - long only)
            if bias == "long":
                ma50 = position.get("ma50", entry_price)
                if current_price < ma50:
                    logger.info(f"📉 Momentum death for {pair} LONG: ${current_price:.2f} < MA50 ${ma50:.2f}")
                    return True, "momentum_death_long"
            
            # Check 5: Momentum reversal (MEDIUM - short only)
            if bias == "short":
                ma50 = position.get("ma50", entry_price)
                if current_price > ma50:
                    logger.info(f"📈 Momentum reversal for {pair} SHORT: ${current_price:.2f} > MA50 ${ma50:.2f}")
                    return True, "momentum_reversal_short"
            
            # Check 6: Daily cutoff (LOW - shorts only before 22:00 UTC)
            if bias == "short" and self._is_daily_cutoff_time():
                logger.info(f"🌙 Daily cutoff approaching for {pair} SHORT (22:00 UTC)")
                return True, "daily_cutoff_short"
            
            # Check 7: Stop loss breach (CRITICAL - safety)
            if bias == "long" and current_price <= stop_loss:
                logger.warning(f"🚨 Stop loss hit for {pair} LONG: ${current_price:.2f} <= ${stop_loss:.2f}")
                return True, "stop_loss_long"
            
            if bias == "short" and current_price >= stop_loss:
                logger.warning(f"🚨 Stop loss hit for {pair} SHORT: ${current_price:.2f} >= ${stop_loss:.2f}")
                return True, "stop_loss_short"
            
            return False, "no_exit"
        except Exception as e:
            logger.error(f"❌ Position exit check failed: {str(e)}")
            return False, f"error: {str(e)}"
    
    # ========================================================================
    # PRICE MONITORING
    # ========================================================================
    
    async def _get_current_price(self, pair: str) -> Optional[float]:
        """Fetch current price from Kraken."""
        try:
            params = {"pair": pair}
            response = await make_request("GET", self.KRAKEN_TICKER_URL, params=params)
            
            if "error" in response and response["error"]:
                logger.warning(f"⚠️ Kraken error for {pair}: {response['error']}")
                return None
            
            result = response.get("result", {})
            ticker = result.get(pair, {})
            
            # Get last price
            last_price = ticker.get("c", [None])[0]
            
            if not last_price:
                logger.warning(f"⚠️ No price data for {pair}")
                return None
            
            return float(last_price)
        except Exception as e:
            logger.error(f"❌ Failed to get price for {pair}: {str(e)}")
            return None
    
    async def update_peak_price(self, position: Dict, current_price: float) -> bool:
        """
        Update peak price for trailing stop calculation.
        
        Peak price only moves up (for longs) or down (for shorts).
        
        Args:
            position: Position dict
            current_price: Current market price
        
        Returns:
            True if peak updated
        """
        try:
            peak_price = position.get("peak_price", position.get("entry_price"))
            bias = position.get("bias")
            
            # LONG: Peak only increases
            if bias == "long" and current_price > peak_price:
                position["peak_price"] = current_price
                position["peak_price_updated"] = get_utc_timestamp()
                logger.debug(f"📈 Updated peak for {position.get('pair')}: ${current_price:.2f}")
                return True
            
            # SHORT: Peak only decreases
            if bias == "short" and current_price < peak_price:
                position["peak_price"] = current_price
                position["peak_price_updated"] = get_utc_timestamp()
                logger.debug(f"📉 Updated peak for {position.get('pair')}: ${current_price:.2f}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Failed to update peak price: {str(e)}")
            return False
    
    # ========================================================================
    # TIME UTILITIES
    # ========================================================================
    
    def _is_force_close_time(self) -> bool:
        """
        Check if current time is force close time (23:00 UTC).
        
        Returns:
            True if >= 23:00 UTC
        """
        try:
            utc_now = datetime.now(pytz.UTC)
            hour = utc_now.hour
            
            return hour >= self.FORCE_CLOSE_HOUR
        except Exception as e:
            logger.error(f"❌ Force close time check failed: {str(e)}")
            return False
    
    def _is_daily_cutoff_time(self) -> bool:
        """
        Check if current time is daily cutoff (22:00 UTC).
        
        Returns:
            True if >= 22:00 UTC
        """
        try:
            utc_now = datetime.now(pytz.UTC)
            hour = utc_now.hour
            
            return hour >= self.DAILY_CUTOFF_HOUR
        except Exception as e:
            logger.error(f"❌ Daily cutoff time check failed: {str(e)}")
            return False
    
    # ========================================================================
    # BATCH MONITORING
    # ========================================================================
    
    async def monitor_positions_continuous(self, check_interval_seconds: int = 300):
        """
        Continuously monitor positions at regular intervals.
        
        Args:
            check_interval_seconds: Seconds between checks (default 5 min)
        """
        try:
            logger.info(f"🔄 Starting continuous position monitoring (interval: {check_interval_seconds}s)")
            
            while True:
                try:
                    positions_to_close = await self.check_all_positions()
                    
                    for close_request in positions_to_close:
                        try:
                            # Trigger close (will be handled by close_trade function)
                            position = close_request.get("position")
                            reason = close_request.get("reason")
                            
                            # Write close signal to Firestore
                            close_signal = {
                                "position_id": position.get("order_id"),
                                "pair": position.get("pair"),
                                "reason": reason,
                                "timestamp": get_utc_timestamp(),
                                "current_price": await self._get_current_price(position.get("pair"))
                            }
                            
                            write_to_firestore("close_signals", f"{position.get('order_id')}_close", close_signal)
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing close signal: {str(e)}")
                    
                    # Wait for next check
                    await asyncio.sleep(check_interval_seconds)
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {str(e)}")
                    await asyncio.sleep(check_interval_seconds)
        except Exception as e:
            log_error("❌ Continuous monitoring failed", e)
            raise

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def check_positions():
    """Main entry point for position checking."""
    try:
        monitor = PositionMonitor()
        positions_to_close = await monitor.check_all_positions()
        return positions_to_close
    except Exception as e:
        log_error("❌ Position check failed", e)
        raise
