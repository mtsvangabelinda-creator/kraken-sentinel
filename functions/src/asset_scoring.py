import logging
import asyncio
from typing import Dict, List, Optional, Tuple
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce

from src.helpers import (
    get_firestore_client,
    make_request,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    log_info,
    log_error,
    validate_positive_float
)

logger = logging.getLogger(__name__)

# ============================================================================
# ASSET SCORING MODEL (Multi-Factor)
# ============================================================================

class AssetScorer:
    """
    Multi-factor asset scoring model.
    
    Scores all Kraken assets based on 12 factors:
    - Momentum (35%): Price change, RSI, MA50 distance
    - Volume (25%): Volume spike, 24h USD volume
    - Sentiment (15%): Social velocity, polarity
    - Technical (15%): ADX, Bollinger Bands
    - Microstructure (10%): Order Book Imbalance
    """
    
    # Minimum requirements
    MIN_VOLUME_USD = 1_000_000  # $1M 24h minimum
    MIN_SCORE_APPROACH_B = 65   # Momentum entry
    MIN_SCORE_APPROACH_C = 50   # Swing entry
    MAX_CORRELATION = 0.7       # Avoid overexposure
    MAX_OPEN_POSITIONS = 2      # Per strategy
    
    # Factor weights
    MOMENTUM_WEIGHT = 0.35
    VOLUME_WEIGHT = 0.25
    SENTIMENT_WEIGHT = 0.15
    TECHNICAL_WEIGHT = 0.15
    MICROSTRUCTURE_WEIGHT = 0.10
    
    def __init__(self):
        self.db = get_firestore_client()
    
    async def score_all_assets(self) -> Dict[str, float]:
        """
        Score all Kraken-listed assets.
        
        Returns:
            Dict of {asset_pair: total_score}
        """
        try:
            logger.info("🎯 Starting asset scoring...")
            
            # Fetch all Kraken pairs
            kraken_pairs = await self._fetch_kraken_pairs()
            logger.info(f"📊 Scoring {len(kraken_pairs)} Kraken pairs")
            
            scores = {}
            for pair in kraken_pairs:
                try:
                    score = await self._score_asset(pair)
                    if score is not None:
                        scores[pair] = score
                except Exception as e:
                    logger.warning(f"⚠️ Failed to score {pair}: {str(e)}")
                    continue
            
            logger.info(f"✅ Scored {len(scores)} assets")
            
            # Save to Firestore
            await self._save_asset_scores(scores)
            
            return scores
        except Exception as e:
            log_error("❌ Asset scoring failed", e)
            raise
    
    async def _score_asset(self, pair: str) -> Optional[float]:
        """
        Score a single asset using multi-factor model.
        
        Args:
            pair: Trading pair (e.g., "BTCUSD")
        
        Returns:
            Total score (0-100) or None if asset doesn't meet minimum
        """
        try:
            # Fetch data for all factors
            momentum_score = await self._score_momentum(pair)
            volume_score = await self._score_volume(pair)
            sentiment_score = await self._score_sentiment(pair)
            technical_score = await self._score_technical(pair)
            microstructure_score = await self._score_microstructure(pair)
            
            # Calculate total score
            total_score = (
                (momentum_score * self.MOMENTUM_WEIGHT) +
                (volume_score * self.VOLUME_WEIGHT) +
                (sentiment_score * self.SENTIMENT_WEIGHT) +
                (technical_score * self.TECHNICAL_WEIGHT) +
                (microstructure_score * self.MICROSTRUCTURE_WEIGHT)
            )
            
            # Validate minimum volume
            volume_24h = await self._get_24h_volume_usd(pair)
            if volume_24h < self.MIN_VOLUME_USD:
                logger.warning(f"⚠️ {pair} volume ${volume_24h} < ${self.MIN_VOLUME_USD}")
                return None
            
            logger.info(f"📈 {pair} score: {total_score:.2f}/100")
            
            return total_score
        except Exception as e:
            logger.error(f"❌ Failed to score {pair}: {str(e)}")
            return None
    
    # ========================================================================
    # MOMENTUM SCORING (35%)
    # ========================================================================
    
    async def _score_momentum(self, pair: str) -> float:
        """
        Score momentum factor (35% weight).
        Sub-factors:
        - 24h Price Change (15%)
        - RSI (10%)
        - MA50 vs Price (10%)
        """
        try:
            price_change_score = await self._score_24h_price_change(pair)
            rsi_score = await self._score_rsi(pair)
            ma50_score = await self._score_ma50_distance(pair)
            
            momentum_score = (
                (price_change_score * 0.43) +  # 15% / 35%
                (rsi_score * 0.29) +            # 10% / 35%
                (ma50_score * 0.29)             # 10% / 35%
            )
            
            return min(100, max(0, momentum_score))
        except Exception as e:
            logger.error(f"❌ Momentum scoring failed for {pair}: {str(e)}")
            return 0
    
    async def _score_24h_price_change(self, pair: str) -> float:
        """Score 24h price change. Positive change = higher score."""
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=24)
            if not ohlcv or len(ohlcv) < 24:
                return 0
            
            open_price = ohlcv[0]["open"]
            close_price = ohlcv[-1]["close"]
            change_percent = ((close_price - open_price) / open_price) * 100
            
            # Map change to 0-100 score
            # +10% = 100, -10% = 0, 0% = 50
            score = 50 + (change_percent * 5)
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ Price change scoring failed: {str(e)}")
            return 0
    
    async def _score_rsi(self, pair: str) -> float:
        """
        Score RSI (14).
        RSI > 60 = overbought (momentum)
        RSI < 40 = oversold (bottoming)
        RSI 40-60 = neutral
        """
        try:
            rsi = await self._calculate_rsi(pair, period=14)
            if rsi is None:
                return 0
            
            # Stronger momentum at extremes
            if rsi > 60:
                score = 50 + (rsi - 60) * 2.5  # 60-100 RSI -> 50-100 score
            elif rsi < 40:
                score = 50 - (40 - rsi) * 2.5  # 0-40 RSI -> 0-50 score
            else:
                score = 50  # Neutral
            
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ RSI scoring failed: {str(e)}")
            return 0
    
    async def _score_ma50_distance(self, pair: str) -> float:
        """
        Score distance from MA50.
        Price > MA50 = bullish (higher score)
        Price < MA50 = bearish (lower score)
        """
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=50)
            if not ohlcv or len(ohlcv) < 50:
                return 0
            
            closes = [candle["close"] for candle in ohlcv]
            ma50 = sum(closes) / 50
            current_price = closes[-1]
            
            # Distance from MA50 as percentage
            distance_percent = ((current_price - ma50) / ma50) * 100
            
            # Map to score: -5% = 0, +5% = 100
            score = 50 + (distance_percent * 10)
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ MA50 scoring failed: {str(e)}")
            return 0
    
    # ========================================================================
    # VOLUME SCORING (25%)
    # ========================================================================
    
    async def _score_volume(self, pair: str) -> float:
        """
        Score volume factor (25% weight).
        Sub-factors:
        - Volume Spike Ratio (15%)
        - 24h Volume USD (10%)
        """
        try:
            spike_score = await self._score_volume_spike(pair)
            usd_volume_score = await self._score_usd_volume(pair)
            
            volume_score = (
                (spike_score * 0.60) +      # 15% / 25%
                (usd_volume_score * 0.40)   # 10% / 25%
            )
            
            return min(100, max(0, volume_score))
        except Exception as e:
            logger.error(f"❌ Volume scoring failed for {pair}: {str(e)}")
            return 0
    
    async def _score_volume_spike(self, pair: str) -> float:
        """
        Score volume spike (current vs 30-day average).
        Volume > 2x average = high score
        """
        try:
            current_volume = await self._get_current_volume(pair)
            avg_volume = await self._get_30day_avg_volume(pair)
            
            if avg_volume == 0:
                return 0
            
            spike_ratio = current_volume / avg_volume
            
            # Spike ratio to score: 1x = 0, 2x = 50, 3x = 75, 4x+ = 100
            if spike_ratio < 1:
                score = spike_ratio * 50
            elif spike_ratio < 2:
                score = 50 + (spike_ratio - 1) * 50
            elif spike_ratio < 3:
                score = 75 + (spike_ratio - 2) * 25
            else:
                score = 100
            
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ Volume spike scoring failed: {str(e)}")
            return 0
    
    async def _score_usd_volume(self, pair: str) -> float:
        """Score 24h USD volume. Higher = better."""
        try:
            volume_usd = await self._get_24h_volume_usd(pair)
            
            # Volume to score: $1M = 0, $10M = 50, $50M = 75, $100M+ = 100
            if volume_usd < 1_000_000:
                score = 0
            elif volume_usd < 10_000_000:
                score = (volume_usd - 1_000_000) / (10_000_000 - 1_000_000) * 50
            elif volume_usd < 50_000_000:
                score = 50 + (volume_usd - 10_000_000) / (50_000_000 - 10_000_000) * 25
            else:
                score = 100
            
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ USD volume scoring failed: {str(e)}")
            return 0
    
    # ========================================================================
    # SENTIMENT SCORING (15%)
    # ========================================================================
    
    async def _score_sentiment(self, pair: str) -> float:
        """
        Score sentiment factor (15% weight).
        Sub-factors:
        - Social Velocity (10%)
        - Sentiment Polarity (5%)
        """
        try:
            social_score = await self._score_social_velocity(pair)
            polarity_score = await self._score_polarity(pair)
            
            sentiment_score = (
                (social_score * 0.67) +     # 10% / 15%
                (polarity_score * 0.33)     # 5% / 15%
            )
            
            return min(100, max(0, sentiment_score))
        except Exception as e:
            logger.error(f"❌ Sentiment scoring failed for {pair}: {str(e)}")
            return 50  # Neutral default
    
    async def _score_social_velocity(self, pair: str) -> float:
        """Score social media mentions velocity."""
        try:
            # Fetch from cryptocurrency.cv API
            coin_name = pair.replace("USD", "").replace("USDT", "").lower()
            
            url = f"https://api.cryptocurrency.cv/metrics/sentiment/{coin_name}"
            response = await make_request("GET", url)
            
            mentions = response.get("mentions", 0)
            avg_mentions = response.get("avg_mentions", 1)
            
            velocity_ratio = mentions / avg_mentions if avg_mentions > 0 else 1
            
            # Map to score: 1x = 50, 2x = 75, 3x+ = 100
            if velocity_ratio < 1:
                score = velocity_ratio * 50
            elif velocity_ratio < 2:
                score = 50 + (velocity_ratio - 1) * 25
            else:
                score = 100
            
            return min(100, max(0, score))
        except Exception as e:
            logger.warning(f"⚠️ Social velocity scoring failed: {str(e)}")
            return 50  # Neutral default
    
    async def _score_polarity(self, pair: str) -> float:
        """Score sentiment polarity (positive vs negative mentions)."""
        try:
            coin_name = pair.replace("USD", "").replace("USDT", "").lower()
            
            url = f"https://api.cryptocurrency.cv/metrics/sentiment/{coin_name}"
            response = await make_request("GET", url)
            
            positive = response.get("positive_mentions", 0)
            negative = response.get("negative_mentions", 0)
            total = positive + negative
            
            if total == 0:
                return 50
            
            positive_ratio = positive / total
            
            # Map to score: 50% positive = 50, 75% positive = 75, 100% = 100, 25% = 25
            score = positive_ratio * 100
            return min(100, max(0, score))
        except Exception as e:
            logger.warning(f"⚠️ Polarity scoring failed: {str(e)}")
            return 50  # Neutral default
    
    # ========================================================================
    # TECHNICAL SCORING (15%)
    # ========================================================================
    
    async def _score_technical(self, pair: str) -> float:
        """
        Score technical factor (15% weight).
        Sub-factors:
        - ADX (5%)
        - Bollinger Bands (5%)
        """
        try:
            adx_score = await self._score_adx(pair)
            bb_score = await self._score_bollinger_bands(pair)
            
            technical_score = (
                (adx_score * 0.50) +   # 5% / 15%
                (bb_score * 0.50)      # 5% / 15%
            )
            
            return min(100, max(0, technical_score))
        except Exception as e:
            logger.error(f"❌ Technical scoring failed: {str(e)}")
            return 50
    
    async def _score_adx(self, pair: str) -> float:
        """Score ADX (Average Directional Index). Higher ADX = stronger trend."""
        try:
            adx = await self._calculate_adx(pair, period=14)
            if adx is None:
                return 50
            
            # ADX to score: 0-20 = weak, 20-50 = strong, 50+ = very strong
            if adx < 20:
                score = adx
            elif adx < 50:
                score = 20 + (adx - 20) * 1.6
            else:
                score = 100
            
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ ADX scoring failed: {str(e)}")
            return 50
    
    async def _score_bollinger_bands(self, pair: str) -> float:
        """
        Score Bollinger Bands position.
        Price touching upper band = bullish (high score)
        Price at middle = neutral (50)
        Price at lower band = bearish (low score)
        """
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=20)
            if not ohlcv or len(ohlcv) < 20:
                return 50
            
            closes = [candle["close"] for candle in ohlcv]
            
            # Calculate SMA20
            sma20 = sum(closes) / 20
            
            # Calculate standard deviation
            variance = sum((c - sma20) ** 2 for c in closes) / 20
            std_dev = variance ** 0.5
            
            upper_band = sma20 + (2 * std_dev)
            lower_band = sma20 - (2 * std_dev)
            current_price = closes[-1]
            
            # Position within bands (0-1)
            if upper_band == lower_band:
                position = 0.5
            else:
                position = (current_price - lower_band) / (upper_band - lower_band)
            
            # Map to score: 0 (lower band) = 0, 0.5 (middle) = 50, 1 (upper band) = 100
            score = position * 100
            return min(100, max(0, score))
        except Exception as e:
            logger.error(f"❌ Bollinger Bands scoring failed: {str(e)}")
            return 50
    
    # ========================================================================
    # MICROSTRUCTURE SCORING (10%)
    # ========================================================================
    
    async def _score_microstructure(self, pair: str) -> float:
        """
        Score microstructure factor (10% weight).
        Sub-factors:
        - Order Book Imbalance (OBI) (5%)
        """
        try:
            obi_score = await self._score_obi(pair)
            return min(100, max(0, obi_score))
        except Exception as e:
            logger.error(f"❌ Microstructure scoring failed: {str(e)}")
            return 50
    
    async def _score_obi(self, pair: str) -> float:
        """
        Score Order Book Imbalance.
        More bids than asks = bullish (high score)
        Balanced = neutral (50)
        More asks than bids = bearish (low score)
        """
        try:
            order_book = await self._fetch_order_book(pair, depth=100)
            
            total_bids = sum(bid[1] for bid in order_book.get("bids", []))
            total_asks = sum(ask[1] for ask in order_book.get("asks", []))
            
            if total_bids + total_asks == 0:
                return 50
            
            bid_ratio = total_bids / (total_bids + total_asks)
            
            # Map to score: 0% bids = 0, 50% = 50, 100% bids = 100
            score = bid_ratio * 100
            return min(100, max(0, score))
        except Exception as e:
            logger.warning(f"⚠️ OBI scoring failed: {str(e)}")
            return 50
    
    # ========================================================================
    # DATA FETCHING HELPERS
    # ========================================================================
    
    async def _fetch_kraken_pairs(self) -> List[str]:
        """Fetch all tradable pairs from Kraken."""
        try:
            url = "https://api.kraken.com/0/public/AssetPairs"
            response = await make_request("GET", url)
            
            pairs = []
            for pair_name in response.get("result", {}).keys():
                if not pair_name.startswith("X"):  # Filter out indices
                    pairs.append(pair_name)
            
            logger.info(f"✅ Fetched {len(pairs)} pairs from Kraken")
            return pairs
        except Exception as e:
            logger.error(f"❌ Failed to fetch pairs: {str(e)}")
            return []
    
    async def _fetch_ohlcv(self, pair: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        """Fetch OHLCV data from Kraken."""
        try:
            url = "https://api.kraken.com/0/public/OHLC"
            params = {
                "pair": pair,
                "interval": interval,
                "limit": limit
            }
            
            response = await make_request("GET", url, params=params)
            
            if "error" in response and response["error"]:
                logger.warning(f"⚠️ Kraken error for {pair}: {response['error']}")
                return []
            
            # Parse response
            result = response.get("result", {})
            ohlcv_data = result.get(pair, [])
            
            # Convert to list of dicts
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
    
    async def _fetch_order_book(self, pair: str, depth: int = 100) -> Dict:
        """Fetch order book from Kraken."""
        try:
            url = "https://api.kraken.com/0/public/Depth"
            params = {"pair": pair, "count": depth}
            
            response = await make_request("GET", url, params=params)
            
            if "error" in response and response["error"]:
                logger.warning(f"⚠️ Kraken error for {pair}: {response['error']}")
                return {"bids": [], "asks": []}
            
            result = response.get("result", {})
            order_book = result.get(pair, {})
            
            return {
                "bids": [[float(bid[0]), float(bid[1])] for bid in order_book.get("bids", [])],
                "asks": [[float(ask[0]), float(ask[1])] for ask in order_book.get("asks", [])]
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch order book for {pair}: {str(e)}")
            return {"bids": [], "asks": []}
    
    async def _get_current_volume(self, pair: str) -> float:
        """Get current 24h volume."""
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=24)
            return sum(candle["volume"] for candle in ohlcv)
        except Exception as e:
            logger.error(f"❌ Failed to get current volume: {str(e)}")
            return 0
    
    async def _get_30day_avg_volume(self, pair: str) -> float:
        """Get 30-day average volume."""
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1d", limit=30)
            if not ohlcv:
                return 0
            return sum(candle["volume"] for candle in ohlcv) / len(ohlcv)
        except Exception as e:
            logger.error(f"❌ Failed to get 30-day avg volume: {str(e)}")
            return 0
    
    async def _get_24h_volume_usd(self, pair: str) -> float:
        """Get 24h volume in USD."""
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=24)
            if not ohlcv:
                return 0
            
            # Volume in USD = volume * close price
            volume_usd = sum(
                candle["volume"] * candle["close"]
                for candle in ohlcv
            )
            
            return volume_usd
        except Exception as e:
            logger.error(f"❌ Failed to get 24h volume USD: {str(e)}")
            return 0
    
    # ========================================================================
    # TECHNICAL INDICATOR CALCULATIONS
    # ========================================================================
    
    async def _calculate_rsi(self, pair: str, period: int = 14) -> Optional[float]:
        """Calculate RSI (Relative Strength Index)."""
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=period + 100)
            if not ohlcv or len(ohlcv) < period + 1:
                return None
            
            closes = [candle["close"] for candle in ohlcv]
            
            # Calculate gains and losses
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
            
            # Average gain and loss
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100 if avg_gain > 0 else 50
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {str(e)}")
            return None
    
    async def _calculate_adx(self, pair: str, period: int = 14) -> Optional[float]:
        """Calculate ADX (Average Directional Index)."""
        try:
            ohlcv = await self._fetch_ohlcv(pair, interval="1h", limit=period + 100)
            if not ohlcv or len(ohlcv) < period + 1:
                return None
            
            highs = [candle["high"] for candle in ohlcv]
            lows = [candle["low"] for candle in ohlcv]
            closes = [candle["close"] for candle in ohlcv]
            
            # Calculate directional movements
            up_moves = []
            down_moves = []
            for i in range(1, len(highs)):
                up = highs[i] - highs[i-1]
                down = lows[i-1] - lows[i]
                
                up_moves.append(max(up, 0) if up > down else 0)
                down_moves.append(max(down, 0) if down > up else 0)
            
            # True Range
            trs = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
                trs.append(tr)
            
            # Smoothed values
            plus_di = sum(up_moves[-period:]) / sum(trs[-period:]) * 100
            minus_di = sum(down_moves[-period:]) / sum(trs[-period:]) * 100
            
            di_sum = abs(plus_di - minus_di)
            di_total = plus_di + minus_di
            
            if di_total == 0:
                return 0
            
            adx = (di_sum / di_total) * 100
            return adx
        except Exception as e:
            logger.error(f"❌ ADX calculation failed: {str(e)}")
            return None
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    async def _save_asset_scores(self, scores: Dict[str, float]) -> bool:
        """Save asset scores to Firestore."""
        try:
            data = {
                "timestamp": get_utc_timestamp(),
                "scores": scores,
                "count": len(scores)
            }
            
            doc_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            write_to_firestore("asset_scores", doc_id, data)
            
            logger.info(f"✅ Saved {len(scores)} asset scores")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save asset scores: {str(e)}")
            return False

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def score_all_assets():
    """Main entry point for asset scoring."""
    try:
        scorer = AssetScorer()
        scores = await scorer.score_all_assets()
        return scores
    except Exception as e:
        log_error("❌ Asset scoring failed", e)
        raise
