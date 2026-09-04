import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.helpers import (
    make_request,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    log_info,
    log_error,
    get_firestore_client
)
from src.regime_detection import RegimeDetector
from src.asset_scoring import AssetScorer

logger = logging.getLogger(__name__)

# ============================================================================
# MOMENTUM SCANNER (Approach B)
# ============================================================================

class MomentumScanner:
    """
    Approach B: Momentum strategy scanner.
    
    Entry Signals:
    - LONG: Breakout + Volume Spike + Momentum (RSI > 50)
    - SHORT: Exhaustion + Structure Break + Funding Rate (RSI < 50)
    
    Hold Times:
    - LONG: 2-6 hours
    - SHORT: 1-4 hours
    
    Exit:
    - LONG: Trailing Stop (2x ATR) + Momentum Death (Price < MA50)
    - SHORT: Trailing Stop (2x ATR) + Momentum Reversal (Price > MA50)
    """
    
    # Entry thresholds
    BREAKOUT_THRESHOLD = 0.02  # 2% above recent high
    VOLUME_SPIKE_THRESHOLD = 2.0  # 2x average volume
    RSI_LONG_THRESHOLD = 50
    RSI_SHORT_THRESHOLD = 50
    FUNDING_RATE_THRESHOLD = 0.0005  # 0.05% for shorts
    
    # Asset selection
    MIN_VOLUME_USD = 1_000_000
    MIN_SCORE = 65  # Momentum entry candidate
    
    # Hold times
    LONG_HOLD_MIN_HOURS = 2
    LONG_HOLD_MAX_HOURS = 6
    SHORT_HOLD_MIN_HOURS = 1
    SHORT_HOLD_MAX_HOURS = 4
    
    def __init__(self):
        self.db = get_firestore_client()
        self.regime_detector = RegimeDetector()
        self.asset_scorer = AssetScorer()
    
    async def scan_all_assets(self) -> List[Dict]:
        """
        Scan all Kraken assets for momentum signals.
        
        Returns:
            List of signal dicts
        """
        try:
            logger.info("🔍 Starting momentum scan (Approach B)...")
            
            # Fetch all Kraken pairs
            kraken_pairs = await self._fetch_kraken_pairs()
            logger.info(f"📊 Scanning {len(kraken_pairs)} pairs")
            
            signals = []
            
            # Scan in parallel (batch of 10)
            for i in range(0, len(kraken_pairs), 10):
                batch = kraken_pairs[i:i+10]
                batch_signals = await asyncio.gather(*[
                    self._scan_pair(pair) for pair in batch
                ], return_exceptions=True)
                
                for signal in batch_signals:
                    if isinstance(signal, dict) and signal:
                        signals.append(signal)
            
            logger.info(f"✅ Momentum scan complete: {len(signals)} signals")
            
            return signals
        except Exception as e:
            log_error("❌ Momentum scan failed", e)
            raise
    
    async def _scan_pair(self, pair: str) -> Optional[Dict]:
        """
        Scan a single pair for momentum signals.
        
        Returns:
            Signal dict or None
        """
        try:
            # Fetch OHLCV data
            ohlcv = await self._fetch_ohlcv(pair)
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            # Get current prices
            current_price = ohlcv[-1]["close"]
            recent_high = max(candle["high"] for candle in ohlcv[-10:])
            recent_low = min(candle["low"] for candle in ohlcv[-10:])
            
            # Check regime
            regime = await self.regime_detector.detect_regime(pair, ohlcv)
            if not regime.get("game_theory_pass"):
                logger.debug(f"⚠️ {pair} failed game theory check")
                return None
            
            # Check asset score
            score = await self._get_asset_score(pair)
            if score < self.MIN_SCORE:
                return None
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(ohlcv)
            ma50 = self._calculate_ma50(ohlcv)
            volume_spike = await self._calculate_volume_spike(pair)
            atr = self._calculate_atr(ohlcv)
            
            # Check LONG entry signals
            long_signal = await self._check_long_entry(
                pair, current_price, recent_high, recent_low, ma50,
                rsi, volume_spike, atr, score, regime
            )
            
            if long_signal:
                return long_signal
            
            # Check SHORT entry signals
            short_signal = await self._check_short_entry(
                pair, current_price, recent_high, recent_low, ma50,
                rsi, volume_spike, atr, score, regime
            )
            
            if short_signal:
                return short_signal
            
            return None
        except Exception as e:
            logger.debug(f"⚠️ Failed to scan {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # LONG ENTRY SIGNALS
    # ========================================================================
    
    async def _check_long_entry(
        self,
        pair: str,
        price: float,
        recent_high: float,
        recent_low: float,
        ma50: float,
        rsi: float,
        volume_spike: float,
        atr: float,
        score: float,
        regime: Dict
    ) -> Optional[Dict]:
        """Check LONG entry conditions."""
        try:
            # Signal requirements:
            # 1. Breakout: Price > recent_high + 2% (dynamic resistance)
            # 2. Volume Spike: Current volume > 2x average
            # 3. Momentum: RSI > 50 (bullish)
            # 4. Regime: Not trending down
            
            breakout_threshold = recent_high * (1 + self.BREAKOUT_THRESHOLD)
            
            if price < breakout_threshold:
                return None
            
            if volume_spike < self.VOLUME_SPIKE_THRESHOLD:
                return None
            
            if rsi < self.RSI_LONG_THRESHOLD:
                return None
            
            # Regime check
            regime_name = regime.get("regime")
            if regime_name == "trending_down":
                logger.debug(f"⚠️ {pair} trending down, skip LONG")
                return None
            
            # Calculate position size (Kelly Criterion handled in execute_trade)
            stop_loss = price - (2 * atr)
            risk_percent = ((price - stop_loss) / price) * 100
            
            # Signal valid!
            signal = {
                "strategy": "approach_b",
                "bias": "long",
                "pair": pair,
                "entry_price": price,
                "entry_signal": "breakout_volume_momentum",
                "secondary_signal": "price_above_ma20",
                "confirmation": "rsi_bullish",
                "time_limit_hours": self.LONG_HOLD_MAX_HOURS,
                "stop_loss_price": stop_loss,
                "atr": atr,
                "rsi": rsi,
                "ma50": ma50,
                "volume_spike": volume_spike,
                "asset_score": score,
                "regime": regime_name,
                "regime_confidence": regime.get("confidence"),
                "timestamp": get_utc_timestamp()
            }
            
            logger.info(f"🟢 LONG signal: {pair} @ ${price:.2f} (RSI: {rsi:.1f}, Spike: {volume_spike:.2f}x)")
            
            # Write to Firestore
            doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_long"
            write_to_firestore("signals", doc_id, signal)
            
            return signal
        except Exception as e:
            logger.error(f"❌ LONG entry check failed for {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # SHORT ENTRY SIGNALS
    # ========================================================================
    
    async def _check_short_entry(
        self,
        pair: str,
        price: float,
        recent_high: float,
        recent_low: float,
        ma50: float,
        rsi: float,
        volume_spike: float,
        atr: float,
        score: float,
        regime: Dict
    ) -> Optional[Dict]:
        """Check SHORT entry conditions."""
        try:
            # Signal requirements:
            # 1. Exhaustion: Price failed to hold above recent high (decline after spike)
            # 2. Structure Break: Price breaks below recent low
            # 3. Momentum: RSI < 50 (bearish)
            # 4. Funding Rate: High (indicates over-leverage)
            # 5. Regime: Not trending up
            
            breakdown_threshold = recent_low * (1 - self.BREAKOUT_THRESHOLD)
            
            if price > breakdown_threshold:
                return None
            
            if volume_spike < self.VOLUME_SPIKE_THRESHOLD:
                return None
            
            if rsi > self.RSI_SHORT_THRESHOLD:
                return None
            
            # Regime check
            regime_name = regime.get("regime")
            if regime_name == "trending_up":
                logger.debug(f"⚠️ {pair} trending up, skip SHORT")
                return None
            
            # Check funding rate (if available)
            funding_rate = await self._get_funding_rate(pair)
            if funding_rate < self.FUNDING_RATE_THRESHOLD:
                logger.debug(f"⚠️ {pair} funding rate low ({funding_rate:.6f}), skip SHORT")
                return None
            
            # Calculate stop loss
            stop_loss = price + (2 * atr)
            risk_percent = ((stop_loss - price) / price) * 100
            
            # Signal valid!
            signal = {
                "strategy": "approach_b",
                "bias": "short",
                "pair": pair,
                "entry_price": price,
                "entry_signal": "exhaustion_structure_break",
                "secondary_signal": "funding_rate_high",
                "confirmation": "rsi_bearish",
                "time_limit_hours": self.SHORT_HOLD_MAX_HOURS,
                "stop_loss_price": stop_loss,
                "atr": atr,
                "rsi": rsi,
                "ma50": ma50,
                "volume_spike": volume_spike,
                "funding_rate": funding_rate,
                "asset_score": score,
                "regime": regime_name,
                "regime_confidence": regime.get("confidence"),
                "timestamp": get_utc_timestamp()
            }
            
            logger.info(f"🔴 SHORT signal: {pair} @ ${price:.2f} (RSI: {rsi:.1f}, Spike: {volume_spike:.2f}x)")
            
            # Write to Firestore
            doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_short"
            write_to_firestore("signals", doc_id, signal)
            
            return signal
        except Exception as e:
            logger.error(f"❌ SHORT entry check failed for {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # TECHNICAL INDICATORS
    # ========================================================================
    
    def _calculate_rsi(self, ohlcv: List[Dict], period: int = 14) -> float:
        """Calculate RSI (14)."""
        try:
            if len(ohlcv) < period + 1:
                return 50  # Neutral
            
            closes = [candle["close"] for candle in ohlcv]
            
            gains = []
            losses = []
            for i in range(1, len(closes)):
                change = closes[i] - closes[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100 if avg_gain > 0 else 50
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {str(e)}")
            return 50
    
    def _calculate_ma50(self, ohlcv: List[Dict]) -> float:
        """Calculate 50-period moving average."""
        try:
            if len(ohlcv) < 50:
                return ohlcv[-1]["close"]
            
            closes = [candle["close"] for candle in ohlcv[-50:]]
            return sum(closes) / 50
        except Exception as e:
            logger.error(f"❌ MA50 calculation failed: {str(e)}")
            return ohlcv[-1]["close"]
    
    def _calculate_atr(self, ohlcv: List[Dict], period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(ohlcv) < period + 1:
                return 0
            
            trs = []
            for i in range(1, len(ohlcv)):
                high = ohlcv[i]["high"]
                low = ohlcv[i]["low"]
                close_prev = ohlcv[i-1]["close"]
                
                tr = max(
                    high - low,
                    abs(high - close_prev),
                    abs(low - close_prev)
                )
                trs.append(tr)
            
            return sum(trs[-period:]) / period
        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {str(e)}")
            return 0
    
    # ========================================================================
    # DATA FETCHING
    # ========================================================================
    
    async def _fetch_kraken_pairs(self) -> List[str]:
        """Fetch all Kraken trading pairs."""
        try:
            url = "https://api.kraken.com/0/public/AssetPairs"
            response = await make_request("GET", url)
            
            pairs = [key for key in response.get("result", {}).keys() if not key.startswith("X")]
            return pairs
        except Exception as e:
            logger.error(f"❌ Failed to fetch Kraken pairs: {str(e)}")
            return []
    
    async def _fetch_ohlcv(self, pair: str, interval: str = "5m", limit: int = 100) -> List[Dict]:
        """Fetch OHLCV data from Kraken (5-minute candles for momentum)."""
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            params = {
                "pair": pair,
                "interval": interval,
                "limit": limit
            }
            
            response = await make_request("GET", url, params=params)
            
            if "error" in response and response["error"]:
                logger.debug(f"⚠️ Kraken error for {pair}: {response['error']}")
                return []
            
            result = response.get("result", {})
            ohlcv_data = result.get(pair, [])
            
            return [
                {
                    "time": candle[0],
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[6])
                }
                for candle in ohlcv_data
            ]
        except Exception as e:
            logger.error(f"❌ Failed to fetch OHLCV for {pair}: {str(e)}")
            return []
    
    async def _calculate_volume_spike(self, pair: str) -> float:
        """Calculate volume spike ratio."""
        try:
            # Fetch hourly data for average
            url = "https://api.kraken.com/0/public/OHLC"
            params = {"pair": pair, "interval": "60", "limit": 24}
            
            response = await make_request("GET", url, params=params)
            
            if "error" in response and response["error"]:
                return 1.0
            
            result = response.get("result", {})
            ohlcv_data = result.get(pair, [])
            
            if len(ohlcv_data) < 24:
                return 1.0
            
            volumes = [float(candle[6]) for candle in ohlcv_data[-24:]]
            current_volume = volumes[-1]
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        except Exception as e:
            logger.error(f"❌ Volume spike calculation failed: {str(e)}")
            return 1.0
    
    async def _get_asset_score(self, pair: str) -> float:
        """Get current asset score."""
        try:
            scores = query_firestore(
                "asset_scores",
                order_by="timestamp",
                direction="desc",
                limit=1
            )
            
            if not scores:
                return 50  # Default neutral score
            
            all_scores = scores[0].get("scores", {})
            return all_scores.get(pair, 50)
        except Exception as e:
            logger.error(f"❌ Failed to get asset score: {str(e)}")
            return 50
    
    async def _get_funding_rate(self, pair: str) -> float:
        """Get funding rate from data source."""
        try:
            # Fetch from agent-data-api-mcp or alternative source
            # For now, return placeholder
            url = "https://api.coingecko.com/api/v3/simple/price"
            # This is a placeholder - real integration needed
            return 0.0001  # Default 0.01%
        except Exception as e:
            logger.debug(f"⚠️ Failed to fetch funding rate: {str(e)}")
            return 0

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def scan_momentum():
    """Main entry point for momentum scanning."""
    try:
        scanner = MomentumScanner()
        signals = await scanner.scan_all_assets()
        return signals
    except Exception as e:
        log_error("❌ Momentum scan failed", e)
        raise
