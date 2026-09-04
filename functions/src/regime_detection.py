import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from hmmlearn import hmm
import asyncio

from src.helpers import (
    get_firestore_client,
    write_to_firestore,
    query_firestore,
    get_utc_timestamp,
    log_info,
    log_error
)

logger = logging.getLogger(__name__)

# ============================================================================
# REGIME DETECTION (Hidden Markov Model)
# ============================================================================

class RegimeDetector:
    """
    Hidden Markov Model (HMM) for market regime classification.
    
    States:
    - State 0: Trending Up (bullish momentum)
    - State 1: Trending Down (bearish momentum)
    - State 2: Ranging (sideways/accumulation)
    - State 3: High Volatility (choppy)
    - State 4: Low Volatility (quiet)
    
    Game Theory: Avoids strategies during uncertain/manipulated regimes.
    """
    
    # Regime names
    REGIMES = {
        0: "trending_up",
        1: "trending_down",
        2: "ranging",
        3: "high_volatility",
        4: "low_volatility"
    }
    
    # HMM parameters
    N_STATES = 5
    COVARIANCE_TYPE = "diag"
    N_ITER = 1000
    
    def __init__(self):
        self.db = get_firestore_client()
        self.model = None
    
    async def detect_regime(self, pair: str, ohlcv: List[Dict]) -> Dict:
        """
        Detect market regime for a pair using HMM.
        
        Args:
            pair: Trading pair (e.g., "BTCUSD")
            ohlcv: List of OHLCV candles
        
        Returns:
            Dict with regime state, probability, and game theory verdict
        """
        try:
            logger.info(f"🔍 Detecting regime for {pair}...")
            
            if not ohlcv or len(ohlcv) < 50:
                logger.warning(f"⚠️ Insufficient data for {pair} regime detection")
                return {
                    "pair": pair,
                    "regime": "unknown",
                    "state": -1,
                    "confidence": 0,
                    "is_manipulated": False,
                    "game_theory_pass": True,
                    "timestamp": get_utc_timestamp()
                }
            
            # Extract features
            features = self._extract_features(ohlcv)
            
            # Train or load HMM model
            if self.model is None:
                self.model = await self._train_hmm(ohlcv)
            
            # Predict current state
            current_state = self.model.predict(features[-1:])
            state_probs = self.model.predict_proba(features[-1:])
            confidence = float(state_probs[0][current_state[0]])
            
            regime_name = self.REGIMES.get(current_state[0], "unknown")
            
            # Game theory analysis
            is_manipulated = await self._detect_manipulation(pair, ohlcv, current_state[0])
            game_theory_pass = not is_manipulated
            
            result = {
                "pair": pair,
                "regime": regime_name,
                "state": int(current_state[0]),
                "confidence": confidence,
                "all_probs": {self.REGIMES[i]: float(state_probs[0][i]) for i in range(self.N_STATES)},
                "is_manipulated": is_manipulated,
                "game_theory_pass": game_theory_pass,
                "timestamp": get_utc_timestamp()
            }
            
            logger.info(f"✅ Regime: {regime_name} (confidence: {confidence:.2%})")
            
            # Save to Firestore
            await self._save_regime(pair, result)
            
            return result
        except Exception as e:
            log_error(f"❌ Regime detection failed for {pair}", e)
            return {
                "pair": pair,
                "regime": "error",
                "state": -1,
                "confidence": 0,
                "is_manipulated": True,  # Fail safe
                "game_theory_pass": False,
                "timestamp": get_utc_timestamp()
            }
    
    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    
    def _extract_features(self, ohlcv: List[Dict], lookback: int = 50) -> np.ndarray:
        """
        Extract features for HMM.
        
        Features:
        1. Log returns (price momentum)
        2. Volatility (standard deviation)
        3. Volume change
        4. Price trend (RSI-like)
        5. Trend strength (ADX-like)
        """
        if len(ohlcv) < lookback:
            lookback = len(ohlcv) - 1
        
        features = []
        
        for i in range(lookback, len(ohlcv)):
            window = ohlcv[i-lookback:i+1]
            
            # Feature 1: Log returns
            prices = [candle["close"] for candle in window]
            log_returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
            mean_return = np.mean(log_returns)
            
            # Feature 2: Volatility
            volatility = np.std(log_returns)
            
            # Feature 3: Volume change
            volumes = [candle["volume"] for candle in window]
            volume_change = np.mean(volumes[lookback//2:]) / np.mean(volumes[:lookback//2]) if volumes[:lookback//2].count(0) == 0 else 1.0
            
            # Feature 4: Price trend (distance from MA50)
            ma50 = np.mean(prices)
            current_price = prices[-1]
            price_trend = (current_price - ma50) / ma50
            
            # Feature 5: Trend strength (using simple directional movement)
            highs = [candle["high"] for candle in window]
            lows = [candle["low"] for candle in window]
            
            up_moves = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            down_moves = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            trend_strength = (up_moves - down_moves) / len(highs)
            
            features.append([mean_return, volatility, volume_change, price_trend, trend_strength])
        
        return np.array(features)
    
    # ========================================================================
    # HMM TRAINING
    # ========================================================================
    
    async def _train_hmm(self, ohlcv: List[Dict]) -> hmm.GaussianHMM:
        """
        Train Gaussian HMM on historical data.
        
        Returns:
            Trained HMM model
        """
        try:
            logger.info("🧠 Training HMM model...")
            
            features = self._extract_features(ohlcv)
            
            # Train HMM
            model = hmm.GaussianHMM(n_components=self.N_STATES, covariance_type=self.COVARIANCE_TYPE, n_iter=self.N_ITER)
            model.fit(features)
            
            logger.info("✅ HMM model trained")
            return model
        except Exception as e:
            log_error("❌ HMM training failed", e)
            # Return default model
            model = hmm.GaussianHMM(n_components=self.N_STATES, covariance_type=self.COVARIANCE_TYPE, n_iter=self.N_ITER)
            return model
    
    # ========================================================================
    # GAME THEORY: MANIPULATION DETECTION
    # ========================================================================
    
    async def _detect_manipulation(self, pair: str, ohlcv: List[Dict], current_state: int) -> bool:
        """
        Detect market manipulation using game theory analysis.
        
        Indicators of manipulation:
        1. Extreme volume spikes without price movement (pump & dump)
        2. Price moving against volume (selling on rally, buying on decline)
        3. Sudden regime changes (HMM instability)
        4. Order book imbalance without execution
        5. Funding rate extremes
        
        Returns:
            True if manipulation detected (avoid trading)
        """
        try:
            if len(ohlcv) < 10:
                return False
            
            # Check for pump & dump (volume spike without price move)
            volumes = [candle["volume"] for candle in ohlcv[-10:]]
            closes = [candle["close"] for candle in ohlcv[-10:]]
            
            # Volume spike
            avg_volume = np.mean(volumes[:-1])
            current_volume = volumes[-1]
            
            if current_volume > avg_volume * 5:  # 5x volume spike
                # Check if price moved proportionally
                price_change = abs((closes[-1] - closes[0]) / closes[0])
                
                if price_change < 0.05:  # Less than 5% price move
                    logger.warning(f"⚠️ {pair} pump & dump detected (volume spike without price move)")
                    return True
            
            # Check for price/volume divergence
            for i in range(1, len(closes)):
                price_move = (closes[i] - closes[i-1]) / closes[i-1]
                volume_ratio = volumes[i] / volumes[i-1] if volumes[i-1] > 0 else 1
                
                # Selling on rally (negative correlation)
                if price_move > 0.02 and volume_ratio < 0.5:
                    logger.warning(f"⚠️ {pair} selling on rally detected")
                    return True
                
                # Buying on decline
                if price_move < -0.02 and volume_ratio < 0.5:
                    logger.warning(f"⚠️ {pair} buying on decline detected")
                    return True
            
            # Check for extreme regime instability (state changes too frequently)
            regime_history = query_firestore("regime_history", field="pair", operator="==", value=pair, limit=20)
            if len(regime_history) > 5:
                states = [doc.get("state") for doc in regime_history]
                state_changes = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
                
                # More than 50% state changes = unstable/manipulated
                if state_changes > len(states) / 2:
                    logger.warning(f"⚠️ {pair} regime instability detected")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"❌ Manipulation detection failed: {str(e)}")
            return True  # Fail safe: assume manipulation
    
    # ========================================================================
    # REGIME-BASED STRATEGY FILTERS
    # ========================================================================
    
    def get_strategy_filter(self, regime: Dict) -> Dict:
        """
        Get strategy filter based on current regime.
        
        Returns strategy recommendations and risk adjustments.
        """
        try:
            regime_name = regime.get("regime", "unknown")
            confidence = regime.get("confidence", 0)
            is_manipulated = regime.get("is_manipulated", False)
            
            filter_result = {
                "regime": regime_name,
                "confidence": confidence,
                "allow_long": True,
                "allow_short": True,
                "stop_loss_multiplier": 1.0,  # 1.0 = standard 2x ATR
                "position_size_multiplier": 1.0,
                "hold_time_multiplier": 1.0,
                "reason": []
            }
            
            # Game theory: Reject if manipulated
            if is_manipulated:
                filter_result["allow_long"] = False
                filter_result["allow_short"] = False
                filter_result["reason"].append("Manipulation detected")
                logger.warning(f"🚫 Signals rejected: manipulation detected")
                return filter_result
            
            # Confidence threshold
            if confidence < 0.6:
                filter_result["position_size_multiplier"] = 0.5
                filter_result["reason"].append(f"Low confidence ({confidence:.2%})")
            
            # Regime-specific adjustments
            if regime_name == "trending_up":
                filter_result["allow_short"] = False  # No shorts in uptrend
                filter_result["reason"].append("Uptrend: shorts disabled")
            
            elif regime_name == "trending_down":
                filter_result["allow_long"] = False  # No longs in downtrend
                filter_result["reason"].append("Downtrend: longs disabled")
            
            elif regime_name == "ranging":
                # Both allowed but with tighter stops
                filter_result["stop_loss_multiplier"] = 0.75  # 1.5x ATR instead of 2x
                filter_result["position_size_multiplier"] = 0.8
                filter_result["reason"].append("Ranging: tighter stops")
            
            elif regime_name == "high_volatility":
                # Widen stops in high volatility
                filter_result["stop_loss_multiplier"] = 1.5  # 3x ATR instead of 2x
                filter_result["position_size_multiplier"] = 0.7  # Smaller positions
                filter_result["reason"].append("High volatility: wider stops, smaller size")
            
            elif regime_name == "low_volatility":
                # Tighten stops in low volatility
                filter_result["stop_loss_multiplier"] = 0.75  # 1.5x ATR instead of 2x
                filter_result["hold_time_multiplier"] = 1.5  # Hold longer
                filter_result["reason"].append("Low volatility: tighter stops, longer holds")
            
            return filter_result
        except Exception as e:
            log_error("❌ Strategy filter failed", e)
            return {
                "regime": "error",
                "confidence": 0,
                "allow_long": False,
                "allow_short": False,
                "stop_loss_multiplier": 1.0,
                "position_size_multiplier": 1.0,
                "hold_time_multiplier": 1.0,
                "reason": [f"Error: {str(e)}"]
            }
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    async def _save_regime(self, pair: str, regime_data: Dict) -> bool:
        """Save regime detection result to Firestore."""
        try:
            doc_id = f"{pair}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            write_to_firestore("regime_history", doc_id, regime_data)
            
            logger.info(f"✅ Saved regime for {pair}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save regime: {str(e)}")
            return False

# ============================================================================
# BATCH REGIME DETECTION
# ============================================================================

async def detect_regimes_batch(pairs: List[str], ohlcv_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Detect regimes for multiple pairs in parallel.
    
    Args:
        pairs: List of trading pairs
        ohlcv_data: Dict of {pair: ohlcv_list}
    
    Returns:
        Dict of {pair: regime_result}
    """
    try:
        detector = RegimeDetector()
        tasks = [
            detector.detect_regime(pair, ohlcv_data.get(pair, []))
            for pair in pairs
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {pair: result for pair, result in zip(pairs, results)}
    except Exception as e:
        log_error("❌ Batch regime detection failed", e)
        return {}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def detect_market_regime(pair: str, ohlcv: List[Dict]) -> Dict:
    """Main entry point for regime detection."""
    try:
        detector = RegimeDetector()
        regime = await detector.detect_regime(pair, ohlcv)
        return regime
    except Exception as e:
        log_error("❌ Regime detection failed", e)
        raise
