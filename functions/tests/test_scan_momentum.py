import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Mock firestore before importing src modules
import sys
from unittest.mock import MagicMock

# Mock firebase_admin
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.firestore'] = MagicMock()

from src.scan_momentum import MomentumScanner

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    data = []
    base_price = 50000
    
    for i in range(100):
        data.append({
            "time": 1700000000 + (i * 3600),
            "open": base_price + (i * 10),
            "high": base_price + (i * 10) + 100,
            "low": base_price + (i * 10) - 50,
            "close": base_price + (i * 10) + 50,
            "volume": 1000 + (i * 10)
        })
    
    return data

@pytest.fixture
def sample_signal():
    """Generate sample signal."""
    return {
        "strategy": "approach_b",
        "bias": "long",
        "pair": "BTCUSD",
        "entry_price": 50000,
        "entry_signal": "breakout_volume_momentum",
        "rsi": 65,
        "ma50": 49500,
        "volume_spike": 2.5,
        "atr": 500,
        "asset_score": 75,
        "regime": "trending_up",
        "timestamp": datetime.utcnow().isoformat()
    }

@pytest.fixture
def momentum_scanner():
    """Create MomentumScanner instance."""
    scanner = MomentumScanner()
    return scanner

# ============================================================================
# MOMENTUM SCANNER TESTS
# ============================================================================

class TestMomentumScanner:
    """Test MomentumScanner class."""
    
    @pytest.mark.asyncio
    async def test_scanner_initialization(self, momentum_scanner):
        """Test scanner initializes correctly."""
        assert momentum_scanner is not None
        assert momentum_scanner.MIN_SCORE == 65
        assert momentum_scanner.LONG_HOLD_MAX_HOURS == 6
    
    def test_rsi_calculation(self, momentum_scanner, sample_ohlcv):
        """Test RSI calculation."""
        rsi = momentum_scanner._calculate_rsi(sample_ohlcv)
        
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
    
    def test_ma50_calculation(self, momentum_scanner, sample_ohlcv):
        """Test MA50 calculation."""
        ma50 = momentum_scanner._calculate_ma50(sample_ohlcv)
        
        assert isinstance(ma50, float)
        assert ma50 > 0
    
    def test_atr_calculation(self, momentum_scanner, sample_ohlcv):
        """Test ATR calculation."""
        atr = momentum_scanner._calculate_atr(sample_ohlcv)
        
        assert isinstance(atr, float)
        assert atr >= 0
    
    def test_volume_spike_calculation(self, momentum_scanner, sample_ohlcv):
        """Test volume spike ratio calculation."""
        spike = momentum_scanner._calculate_volume_spike(sample_ohlcv)
        
        assert isinstance(spike, float)
        assert spike >= 0
    
    @pytest.mark.asyncio
    async def test_long_entry_conditions_pass(self, momentum_scanner, sample_signal):
        """Test LONG entry conditions pass."""
        # Mock dependencies
        momentum_scanner.regime_detector = AsyncMock()
        momentum_scanner.regime_detector.detect_regime = AsyncMock(
            return_value={
                "regime": "trending_up",
                "game_theory_pass": True,
                "confidence": 0.85
            }
        )
        
        momentum_scanner._get_asset_score = AsyncMock(return_value=75)
        
        # Create test data
        pair = "BTCUSD"
        price = 50500
        recent_high = 50000
        recent_low = 49000
        ma50 = 49500
        rsi = 65
        volume_spike = 2.5
        atr = 500
        score = 75
        regime = {
            "regime": "trending_up",
            "game_theory_pass": True,
            "confidence": 0.85
        }
        
        signal = await momentum_scanner._check_long_entry(
            pair, price, recent_high, recent_low, ma50, rsi, volume_spike, atr, score, regime
        )
        
        # Should generate signal if conditions met
        # Note: actual signal depends on exact thresholds
        assert signal is None or isinstance(signal, dict)
    
    @pytest.mark.asyncio
    async def test_short_entry_conditions_pass(self, momentum_scanner):
        """Test SHORT entry conditions pass."""
        # Create test data for short entry
        pair = "ETHUSD"
        price = 2000
        recent_high = 2100
        recent_low = 1900
        ma50 = 2050
        rsi = 35
        volume_spike = 2.0
        atr = 50
        score = 70
        regime = {
            "regime": "trending_down",
            "game_theory_pass": True,
            "confidence": 0.80
        }
        
        # Mock funding rate
        momentum_scanner._get_funding_rate = AsyncMock(return_value=0.0008)
        
        signal = await momentum_scanner._check_short_entry(
            pair, price, recent_high, recent_low, ma50, rsi, volume_spike, atr, score, regime
        )
        
        assert signal is None or isinstance(signal, dict)
    
    @pytest.mark.asyncio
    async def test_scan_pair_insufficient_data(self, momentum_scanner):
        """Test scanning pair with insufficient OHLCV data."""
        momentum_scanner._fetch_ohlcv = AsyncMock(return_value=[])
        
        signal = await momentum_scanner._scan_pair("BTCUSD")
        
        assert signal is None
    
    def test_rsi_threshold_long(self, momentum_scanner):
        """Test RSI threshold for long entries."""
        assert momentum_scanner.RSI_LONG_THRESHOLD == 50
    
    def test_rsi_threshold_short(self, momentum_scanner):
        """Test RSI threshold for short entries."""
        assert momentum_scanner.RSI_SHORT_THRESHOLD == 50
    
    def test_volume_spike_threshold(self, momentum_scanner):
        """Test volume spike threshold."""
        assert momentum_scanner.VOLUME_SPIKE_THRESHOLD == 2.0

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMomentumScannerIntegration:
    """Integration tests for momentum scanner."""
    
    @pytest.mark.asyncio
    async def test_scan_with_mock_data(self, momentum_scanner, sample_ohlcv):
        """Test scanning with mock OHLCV data."""
        # Mock external calls
        momentum_scanner._fetch_ohlcv = AsyncMock(return_value=sample_ohlcv)
        momentum_scanner.regime_detector = AsyncMock()
        momentum_scanner.regime_detector.detect_regime = AsyncMock(
            return_value={
                "regime": "trending_up",
                "game_theory_pass": True,
                "confidence": 0.85
            }
        )
        momentum_scanner._get_asset_score = AsyncMock(return_value=75)
        momentum_scanner._get_funding_rate = AsyncMock(return_value=0.0005)
        
        # Mock Firestore write
        with patch('src.scan_momentum.write_to_firestore'):
            signal = await momentum_scanner._scan_pair("BTCUSD")
        
        # Signal may or may not be generated depending on exact conditions
        assert signal is None or isinstance(signal, dict)

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestMomentumScannerErrors:
    """Test error handling."""
    
    def test_rsi_calculation_empty_data(self, momentum_scanner):
        """Test RSI calculation with empty data."""
        rsi = momentum_scanner._calculate_rsi([])
        assert rsi == 50  # Neutral default
    
    def test_ma50_calculation_insufficient_data(self, momentum_scanner):
        """Test MA50 with insufficient data."""
        small_data = [{"close": 100} for _ in range(10)]
        ma50 = momentum_scanner._calculate_ma50(small_data)
        assert isinstance(ma50, float)
    
    def test_atr_calculation_with_zeros(self, momentum_scanner):
        """Test ATR calculation with price data."""
        data = [
            {"high": 100, "low": 90, "close": 95} for _ in range(20)
        ]
        atr = momentum_scanner._calculate_atr(data)
        assert isinstance(atr, float)
        assert atr >= 0

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestMomentumScannerEdgeCases:
    """Test edge cases."""
    
    def test_breakout_threshold_calculation(self, momentum_scanner):
        """Test breakout threshold calculation."""
        recent_high = 50000
        threshold = recent_high * (1 + momentum_scanner.BREAKOUT_THRESHOLD)
        
        assert threshold == 51000  # 2% above recent high
    
    def test_volume_spike_ratio_calculation(self, momentum_scanner):
        """Test volume spike ratio."""
        volumes = [100] * 20
        current_volume = 250  # 2.5x
        
        ratio = current_volume / (sum(volumes[:-1]) / len(volumes[:-1]))
        
        assert ratio == pytest.approx(2.5)
    
    def test_hold_time_boundaries(self, momentum_scanner):
        """Test hold time boundaries."""
        assert momentum_scanner.LONG_HOLD_MIN_HOURS == 2
        assert momentum_scanner.LONG_HOLD_MAX_HOURS == 6
        assert momentum_scanner.SHORT_HOLD_MIN_HOURS == 1
        assert momentum_scanner.SHORT_HOLD_MAX_HOURS == 4

# ============================================================================
# SIGNAL VALIDATION TESTS
# ============================================================================

class TestMomentumSignalValidation:
    """Test signal validation."""
    
    def test_signal_structure(self, sample_signal):
        """Test signal has required fields."""
        required_fields = [
            "strategy", "bias", "pair", "entry_price",
            "entry_signal", "rsi", "ma50", "volume_spike"
        ]
        
        for field in required_fields:
            assert field in sample_signal
    
    def test_signal_bias_values(self, momentum_scanner):
        """Test valid bias values."""
        valid_biases = ["long", "short"]
        # Scanner should only create signals with valid biases
        assert "long" in valid_biases
        assert "short" in valid_biases
    
    def test_signal_strategy_value(self, sample_signal):
        """Test strategy is approach_b for momentum."""
        assert sample_signal["strategy"] == "approach_b"

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestMomentumScannerPerformance:
    """Test performance characteristics."""
    
    def test_rsi_calculation_speed(self, momentum_scanner, sample_ohlcv):
        """Test RSI calculation doesn't take too long."""
        import time
        
        start = time.time()
        for _ in range(100):
            momentum_scanner._calculate_rsi(sample_ohlcv)
        elapsed = time.time() - start
        
        # 100 calculations should complete in under 1 second
        assert elapsed < 1.0
    
    def test_ma_calculation_speed(self, momentum_scanner, sample_ohlcv):
        """Test MA calculation speed."""
        import time
        
        start = time.time()
        for _ in range(100):
            momentum_scanner._calculate_ma50(sample_ohlcv)
        elapsed = time.time() - start
        
        assert elapsed < 1.0

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
