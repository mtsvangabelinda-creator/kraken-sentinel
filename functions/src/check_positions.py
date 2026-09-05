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
            
            # Check 4: Momentum death (MEDIUM - long only
