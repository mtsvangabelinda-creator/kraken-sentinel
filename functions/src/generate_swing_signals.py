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

logger = logging.getLogger(__name__)

# ============================================================================
# SWING SIGNAL GENERATOR (Approach C)
# ============================================================================

class SwingSignalGenerator:
    """
    Approach C: Swing trading strategy signal generator.
    
    Entry Signals:
    - LONG: MA50 + RSI > 60 + Volume > 1.5x
    - SHORT: MA50 + RSI < 40 + Volume > 1.2x
    
    Exit Signals:
    - LONG: MA20 Cross (price < MA20) + RSI < 40
    - SHORT: MA20 Cross (price > MA20) + RSI > 60
    
    Hold Times:
    - LONG: 4-24 hours
    - SHORT: 2-12 hours
    
    Exit:
    - Trailing Stop (2x ATR)
    - MA20 Cross confirmation
    - RSI extreme reversal
    """
    
    # Entry thresholds
    RSI_LONG_THRESHOLD = 60
    RSI_SHORT_THRESHOLD = 40
    VOLUME_SPIKE_LONG = 1.5  # 1.5x average
    VOLUME_SPIKE_SHORT = 1.2  # 1.2x average
    
    # Asset selection
    MIN_VOLUME_USD = 1_000_000
    MIN_SCORE = 50  # Swing entry candidate
    
    # Hold times
    LONG_HOLD_MIN_HOURS = 4
    LONG_HOLD_MAX_HOURS = 24
    SHORT_HOLD_MIN_HOURS = 2
    SHORT_HOLD_MAX_HOURS = 12
    
    # Technical parameters
    MA50_PERIOD = 50
    MA20_PERIOD = 20
    RSI_PERIOD = 14
    ATR_PERIOD = 14
    
    def __init__(self):
        self.db = get_firestore_client()
        self.regime_detector = RegimeDetector()
    
    async def generate_all_signals(self) -> List[Dict]:
        """
        Generate swing signals for all Kraken assets.
        
        Returns:
            List of signal dicts
        """
        try:
            logger.info("📈 Starting swing signal generation (Approach C)...")
            
            # Fetch all pairs
            pairs = await self._fetch_all_pairs()
            logger.info(f"🔍 Scanning {len(pairs)} pairs for swing setups")
            
            signals = []
            
            # Process in parallel (batch of 15)
            for i in range(0, len(pairs), 15):
                batch = pairs[i:i+15]
                batch_signals = await asyncio.gather(*[
                    self._generate_pair_signal(pair) for pair in batch
                ], return_exceptions=True)
                
                for signal in batch_signals:
                    if isinstance(signal, dict) and signal:
                        signals.append(signal)
            
            logger.info(f"✅ Signal generation complete: {len(signals)} signals")
            
            return signals
        except Exception as e:
            log_error("❌ Swing signal generation failed", e)
            raise
    
    async def _generate_pair_signal(self, pair: str) -> Optional[Dict]:
        """
        Generate swing signal for a single pair.
        
        Returns:
            Signal dict or None
        """
        try:
            # Fetch 1-hour OHLCV data (swing uses hourly candles)
            ohlcv = await self._fetch_ohlcv_hourly(pair, limit=100)
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            # Get current price
            current_price = ohlcv[-1]["close"]
            
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
            ma50 = self._calculate_ma(ohlcv, period=50)
            ma20 = self._calculate_ma(ohlcv, period=20)
            rsi = self._calculate_rsi(ohlcv, period=14)
            volume_spike = self._calculate_volume_spike(ohlcv)
            atr = self._calculate_atr(ohlcv, period=14)
            
            # Check LONG entry signals
            long_signal = await self._check_long_entry(
                pair, current_price, ma50, ma20, rsi, volume_spike, atr, score, regime
            )
            
            if long_signal:
                return long_signal
            
            # Check SHORT entry signals
            short_signal = await self._check_short_entry(
                pair, current_price, ma50, ma20, rsi, volume_spike, atr, score, regime
            )
            
            if short_signal:
                return short_signal
            
            return None
        except Exception as e:
            logger.debug(f"⚠️ Failed to generate signal for {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # LONG ENTRY SIGNALS
    # ========================================================================
    
    async def _check_long_entry(
        self,
        pair: str,
        price: float,
        ma50: float,
        ma20: float,
        rsi: float,
        volume_spike: float,
        atr: float,
        score: float,
        regime: Dict
    ) -> Optional[Dict]:
        """
        Check LONG entry conditions.
        
        Requirements:
        1. Price > MA50 (uptrend confirmation)
        2. RSI > 60 (bullish momentum)
        3. Volume > 1.5x average (volume confirmation)
        4. Regime allows (not trending down)
        """
        try:
            # Requirement 1: Price > MA50
            if price <= ma50:
                return None
            
            # Requirement 2: RSI > 60
            if rsi <= self.RSI_LONG_THRESHOLD:
                return None
            
            # Requirement 3: Volume > 1.5x average
            if volume_spike < self.VOLUME_SPIKE_LONG:
                return None
            
            # Requirement 4: Regime check
            regime_name = regime.get("regime")
            if regime_name == "trending_down":
                logger.debug(f"⚠️ {pair} trending down, skip LONG")
                return None
            
            # Calculate stop loss
            stop_loss = price - (2 * atr)
            risk_percent = ((price - stop_loss) / price) * 100
            
            # Signal valid!
            signal = {
                "strategy": "approach_c",
                "bias": "long",
                "pair": pair,
                "entry_price": price,
                "entry_signal": "ma50_rsi_volume",
                "primary_signal": "ma50_bullish",
                "secondary_signal": "rsi_above_60",
                "confirmation": "volume_spike",
                "time_limit_hours": self.LONG_HOLD_MAX_HOURS,
                "stop_loss_price": stop_loss,
                "atr": atr,
                "ma50": ma50,
                "ma20": ma20,
                "rsi": rsi,
                "volume_spike": volume_spike,
                "asset_score": score,
                "regime": regime_name,
                "regime_confidence": regime.get("confidence"),
                "timestamp": get_utc_timestamp()
            }
            
            logger.info(f"🟢 LONG signal: {pair} @ ${price:.2f} (RSI: {rsi:.1f}, Volume: {volume_spike:.2f}x, MA50: ${ma50:.2f})")
            
            # Write to Firestore
            doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_swing_long"
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
        ma50: float,
        ma20: float,
        rsi: float,
        volume_spike: float,
        atr: float,
        score: float,
        regime: Dict
    ) -> Optional[Dict]:
        """
        Check SHORT entry conditions.
        
        Requirements:
        1. Price < MA50 (downtrend confirmation)
        2. RSI < 40 (bearish momentum)
        3. Volume > 1.2x average (volume confirmation)
        4. Regime allows (not trending up)
        """
        try:
            # Requirement 1: Price < MA50
            if price >= ma50:
                return None
            
            # Requirement 2: RSI < 40
            if rsi >= self.RSI_SHORT_THRESHOLD:
                return None
            
            # Requirement 3: Volume > 1.2x average
            if volume_spike < self.VOLUME_SPIKE_SHORT:
                return None
            
            # Requirement 4: Regime check
            regime_name = regime.get("regime")
            if regime_name == "trending_up":
                logger.debug(f"⚠️ {pair} trending up, skip SHORT")
                return None
            
            # Calculate stop loss
            stop_loss = price + (2 * atr)
            risk_percent = ((stop_loss - price) / price) * 100
            
            # Signal valid!
            signal = {
                "strategy": "approach_c",
                "bias": "short",
                "pair": pair,
                "entry_price": price,
                "entry_signal": "ma50_rsi_volume",
                "primary_signal": "ma50_bearish",
                "secondary_signal": "rsi_below_40",
                "confirmation": "volume_spike",
                "time_limit_hours": self.SHORT_HOLD_MAX_HOURS,
                "stop_loss_price": stop_loss,
                "atr": atr,
                "ma50": ma50,
                "ma20": ma20,
                "rsi": rsi,
                "volume_spike": volume_spike,
                "asset_score": score,
                "regime": regime_name,
                "regime_confidence": regime.get("confidence"),
                "timestamp": get_utc_timestamp()
            }
            
            logger.info(f"🔴 SHORT signal: {pair} @ ${price:.2f} (RSI: {rsi:.1f}, Volume: {volume_spike:.2f}x, MA50: ${ma50:.2f})")
            
            # Write to Firestore
            doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_swing_short"
            write_to_firestore("signals", doc_id, signal)
            
            return signal
        except Exception as e:
            logger.error(f"❌ SHORT entry check failed for {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # TECHNICAL INDICATORS
    # ========================================================================
    
    def _calculate_ma(self, ohlcv: List[Dict], period: int) -> float:
        """Calculate moving average."""
        try:
            if len(ohlcv) < period:
                return ohlcv[-1]["close"]
            
            closes = [candle["close"] for candle in ohlcv[-period:]]
            return sum(closes) / period
        except Exception as e:
            logger.error(f"❌ MA calculation failed: {str(e)}")
            return ohlcv[-1]["close"]
    
    def _calculate_rsi(self, ohlcv: List[Dict], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
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
    
    def _calculate_volume_spike(self, ohlcv: List[Dict]) -> float:
        """
        Calculate volume spike ratio (current vs average).
        
        Returns:
            Ratio of current volume to average volume
        """
        try:
            if len(ohlcv) < 20:
                return 1.0
            
            volumes = [candle["volume"] for candle in ohlcv[-20:]]
            current_volume = volumes[-1]
            avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
            
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        except Exception as e:
            logger.error(f"❌ Volume spike calculation failed: {str(e)}")
            return 1.0
    
    # ========================================================================
    # DATA FETCHING
    # ========================================================================
    
    async def _fetch_all_pairs(self) -> List[str]:
        """Fetch all Kraken trading pairs."""
        try:
            url = "https://api.kraken.com/0/public/AssetPairs"
            response = await make_request("GET", url)
            
            pairs = [key for key in response.get("result", {}).keys() if not key.startswith("X")]
            return pairs
        except Exception as e:
            logger.error(f"❌ Failed to fetch pairs: {str(e)}")
            return []
    
    async def _fetch_ohlcv_hourly(self, pair: str, limit: int = 100) -> List[Dict]:
        """Fetch hourly OHLCV data from Kraken."""
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            params = {
                "pair": pair,
                "interval": "60",  # 1 hour
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
                return 50  # Default
            
            all_scores = scores[0].get("scores", {})
            return all_scores.get(pair, 50)
        except Exception as e:
            logger.error(f"❌ Failed to get asset score: {str(e)}")
            return 50

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def generate_swing_signals():
    """Main entry point for swing signal generation."""
    try:
        generator = SwingSignalGenerator()
        signals = await generator.generate_all_signals()
        return signals
    except Exception as e:
        log_error("❌ Swing signal generation failed", e)
        raise
