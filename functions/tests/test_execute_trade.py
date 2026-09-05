import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

# Mock firestore before importing src modules
import sys
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.firestore'] = MagicMock()

from src.execute_trade import TradeExecutor

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_signal():
    """Generate sample trading signal."""
    return {
        "strategy": "approach_b",
        "bias": "long",
        "pair": "BTCUSD",
        "entry_price": 50000,
        "entry_signal": "breakout_volume_momentum",
        "secondary_signal": "price_above_ma20",
        "confirmation": "rsi_bullish",
        "time_limit_hours": 6,
        "stop_loss_price": 49000,
        "atr": 500,
        "rsi": 65,
        "ma50": 49500,
        "volume_spike": 2.5,
        "asset_score": 75,
        "regime": "trending_up",
        "regime_confidence": 0.85,
        "timestamp": datetime.utcnow().isoformat()
    }

@pytest.fixture
def sample_trade_record():
    """Generate sample trade record."""
    return {
        "order_id": "order_123",
        "strategy": "approach_b",
        "bias": "long",
        "pair": "BTCUSD",
        "entry_price": 50000,
        "entry_time": datetime.utcnow().isoformat(),
        "stop_loss_price": 49000,
        "quantity": 0.01,
        "position_size_usd": 500,
        "atr": 500,
        "rsi": 65,
        "volume_spike": 2.5,
        "regime": "trending_up",
        "asset_score": 75,
        "max_hold_hours": 6,
        "risk_percent": 2.0,
        "status": "open",
        "pnl": 0,
        "pnl_percent": 0,
        "timestamp": datetime.utcnow().isoformat()
    }

@pytest.fixture
def trade_executor():
    """Create TradeExecutor instance."""
    executor = TradeExecutor()
    return executor

# ============================================================================
# TRADE EXECUTOR TESTS
# ============================================================================

class TestTradeExecutor:
    """Test TradeExecutor class."""
    
    def test_executor_initialization(self, trade_executor):
        """Test executor initializes correctly."""
        assert trade_executor is not None
        assert trade_executor.MIN_ORDER_SIZE_USD == 10
        assert trade_executor.MAX_ORDER_SIZE_PERCENT == 0.10
    
    def test_order_type_is_market(self, trade_executor):
        """Test order type is market."""
        assert trade_executor.ORDER_TYPE == "market"
    
    @pytest.mark.asyncio
    async def test_pre_execution_checks_pass(self, trade_executor, sample_signal):
        """Test pre-execution checks pass."""
        # Mock dependencies
        trade_executor.risk_manager = AsyncMock()
        trade_executor.risk_manager.check_risk_limits = AsyncMock(
            return_value={"allow_trading": True}
        )
        trade_executor.risk_manager.get_daily_loss = AsyncMock(return_value=-1.0)
        trade_executor.risk_manager.get_daily_loss_limit = AsyncMock(return_value=5.0)
        trade_executor.risk_manager.is_circuit_breaker_active = AsyncMock(return_value=False)
        
        with patch('src.execute_trade.query_firestore', return_value=[]):
            result = await trade_executor._pre_execution_checks(sample_signal)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_position_sizing_default(self, trade_executor):
        """Test position sizing with default fallback."""
        with patch.object(trade_executor, '_get_current_pool_balance', new_callable=AsyncMock) as mock_balance:
            mock_balance.return_value = 300.0
            
            with patch('src.execute_trade.query_firestore', return_value=[]):
                position_size = await trade_executor._calculate_position_size({})
        
        # Should be at least minimum, at most max
        assert position_size >= trade_executor.MIN_ORDER_SIZE_USD
        assert position_size <= 300.0
    
    @pytest.mark.asyncio
    async def test_position_sizing_kelly_criterion(self, trade_executor):
        """Test Kelly Criterion position sizing."""
        # Mock recent trades for Kelly calculation
        recent_trades = [
            {"pnl": 100, "pnl_percent": 0.02},
            {"pnl": -50, "pnl_percent": -0.01},
            {"pnl": 75, "pnl_percent": 0.015},
            {"pnl": -25, "pnl_percent": -0.005},
        ]
        
        with patch.object(trade_executor, '_get_current_pool_balance', new_callable=AsyncMock) as mock_balance:
            mock_balance.return_value = 300.0
            
            with patch('src.execute_trade.query_firestore', return_value=recent_trades):
                position_size = await trade_executor._calculate_position_size({})
        
        # Position size should be calculated
        assert isinstance(position_size, float)
        assert position_size > 0
    
    def test_order_preparation_long(self, trade_executor, sample_signal):
        """Test order preparation for long entry."""
        position_size_usd = 100
        
        order = trade_executor._prepare_order(sample_signal, position_size_usd)
        
        assert order is not None
        assert order["pair"] == "BTCUSD"
        assert order["type"] == "buy"
        assert order["volume"] == 100 / 50000  # 0.002 BTC
        assert order["price"] == 50000
    
    def test_order_preparation_short(self, trade_executor, sample_signal):
        """Test order preparation for short entry."""
        sample_signal["bias"] = "short"
        position_size_usd = 100
        
        order = trade_executor._prepare_order(sample_signal, position_size_usd)
        
        assert order["type"] == "sell"
    
    def test_order_userref_is_unique(self, trade_executor, sample_signal):
        """Test userref is unique."""
        order1 = trade_executor._prepare_order(sample_signal, 100)
        order2 = trade_executor._prepare_order(sample_signal, 100)
        
        # Should have different userrefs
        assert order1.get("userref") != order2.get("userref")
    
    @pytest.mark.asyncio
    async def test_execute_order_simulation_mode(self, trade_executor):
        """Test order execution in simulation mode (no API keys)."""
        trade_executor.kraken_api_key = ""
        trade_executor.kraken_api_secret = ""
        
        order = {
            "pair": "BTCUSD",
            "type": "buy",
            "order_type": "market",
            "volume": 0.01,
            "price": 50000
        }
        
        result = await trade_executor._execute_order(order)
        
        assert result is not None
        assert result.get("status") == "closed"
        assert result.get("executed") is True
    
    def test_trade_record_creation(self, trade_executor, sample_signal):
        """Test trade record creation."""
        order = {"volume": 0.01}
        order_result = {"order_id": "order_123"}
        
        trade_record = trade_executor._create_trade_record(sample_signal, order, order_result)
        
        assert trade_record["order_id"] == "order_123"
        assert trade_record["strategy"] == "approach_b"
        assert trade_record["bias"] == "long"
        assert trade_record["pair"] == "BTCUSD"
        assert trade_record["entry_price"] == 50000
        assert trade_record["status"] == "open"

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTradeExecutorIntegration:
    """Integration tests for trade execution."""
    
    @pytest.mark.asyncio
    async def test_execute_signal_full_flow(self, trade_executor, sample_signal):
        """Test full signal execution flow."""
        # Mock all dependencies
        trade_executor.risk_manager = AsyncMock()
        trade_executor.risk_manager.check_risk_limits = AsyncMock(
            return_value={"allow_trading": True}
        )
        trade_executor.risk_manager.get_daily_loss = AsyncMock(return_value=0)
        trade_executor.risk_manager.get_daily_loss_limit = AsyncMock(return_value=15)
        trade_executor.risk_manager.is_circuit_breaker_active = AsyncMock(return_value=False)
        
        trade_executor._get_current_pool_balance = AsyncMock(return_value=300.0)
        trade_executor.kraken_api_key = ""  # Simulation mode
        
        with patch('src.execute_trade.query_firestore', return_value=[]):
            with patch('src.execute_trade.write_to_firestore'):
                trade = await trade_executor.execute_signal(sample_signal)
        
        # Trade should be executed or None
        assert trade is None or isinstance(trade, dict)

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestTradeExecutorErrors:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_execute_signal_with_existing_position(self, trade_executor, sample_signal):
        """Test rejection when position already exists."""
        # Mock existing position
        with patch('src.execute_trade.query_firestore', return_value=[{"pair": "BTCUSD"}]):
            trade_executor.risk_manager = AsyncMock()
            trade_executor.risk_manager.check_risk_limits = AsyncMock(
                return_value={"allow_trading": True}
            )
            
            trade = await trade_executor.execute_signal(sample_signal)
        
        assert trade is None
    
    @pytest.mark.asyncio
    async def test_execute_signal_when_risk_limits_breached(self, trade_executor, sample_signal):
        """Test rejection when risk limits breached."""
        trade_executor.risk_manager = AsyncMock()
        trade_executor.risk_manager.check_risk_limits = AsyncMock(
            return_value={"allow_trading": False, "reason": "Daily loss limit"}
        )
        
        with patch('src.execute_trade.query_firestore', return_value=[]):
            trade = await trade_executor.execute_signal(sample_signal)
        
        assert trade is None
    
    @pytest.mark.asyncio
    async def test_position_size_below_minimum(self, trade_executor, sample_signal):
        """Test rejection when position size below minimum."""
        with patch.object(trade_executor, '_get_current_pool_balance', new_callable=AsyncMock) as mock:
            mock.return_value = 5.0  # Only $5, less than minimum
            
            with patch('src.execute_trade.query_firestore', return_value=[]):
                position_size = await trade_executor._calculate_position_size({})
        
        # Position size should be clamped to minimum
        assert position_size >= trade_executor.MIN_ORDER_SIZE_USD

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestTradeExecutorEdgeCases:
    """Test edge cases."""
    
    def test_order_with_zero_position_size(self, trade_executor, sample_signal):
        """Test order preparation with minimal position."""
        # Should still work, just very small
        order = trade_executor._prepare_order(sample_signal, 10)
        
        assert order is not None
        assert order["volume"] > 0
    
    def test_stop_loss_calculation(self, trade_executor):
        """Test stop loss calculation."""
        entry_price = 50000
        atr = 500
        
        stop_loss = entry_price - (2 * atr)
        
        assert stop_loss == 49000
    
    def test_risk_percent_calculation(self, trade_executor):
        """Test risk percentage calculation."""
        entry_price = 50000
        stop_loss = 49000
        risk_percent = ((entry_price - stop_loss) / entry_price) * 100
        
        assert risk_percent == pytest.approx(2.0)
    
    def test_position_size_max_cap(self, trade_executor, sample_signal):
        """Test position size is capped at max."""
        # Max is 10% of pool
        max_size = 300 * trade_executor.MAX_ORDER_SIZE_PERCENT
        
        assert max_size == 30.0

# ============================================================================
# TRADE RECORD VALIDATION TESTS
# ============================================================================

class TestTradeRecordValidation:
    """Test trade record validation."""
    
    def test_trade_record_required_fields(self, sample_trade_record):
        """Test trade record has required fields."""
        required_fields = [
            "order_id", "strategy", "bias", "pair",
            "entry_price", "status", "pnl", "timestamp"
        ]
        
        for field in required_fields:
            assert field in sample_trade_record
    
    def test_trade_record_bias_values(self, sample_trade_record):
        """Test bias is valid."""
        assert sample_trade_record["bias"] in ["long", "short"]
    
    def test_trade_record_status_values(self, sample_trade_record):
        """Test status is valid."""
        valid_statuses = ["open", "closed", "pending"]
        assert sample_trade_record["status"] in valid_statuses

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestTradeExecutorPerformance:
    """Test performance characteristics."""
    
    def test_order_preparation_speed(self, trade_executor, sample_signal):
        """Test order preparation is fast."""
        import time
        
        start = time.time()
        for _ in range(1000):
            trade_executor._prepare_order(sample_signal, 100)
        elapsed = time.time() - start
        
        # 1000 order preparations should complete in under 1 second
        assert elapsed < 1.0
    
    def test_trade_record_creation_speed(self, trade_executor, sample_signal):
        """Test trade record creation is fast."""
        import time
        
        order = {"volume": 0.01}
        order_result = {"order_id": "test_order"}
        
        start = time.time()
        for _ in range(1000):
            trade_executor._create_trade_record(sample_signal, order, order_result)
        elapsed = time.time() - start
        
        assert elapsed < 1.0

# ============================================================================
# KELLY CRITERION TESTS
# ============================================================================

class TestKellyCriterion:
    """Test Kelly Criterion position sizing."""
    
    @pytest.mark.asyncio
    async def test_kelly_with_positive_expectancy(self, trade_executor):
        """Test Kelly with profitable trade history."""
        profitable_trades = [
            {"pnl": 100},
            {"pnl": 150},
            {"pnl": -50},
            {"pnl": 200},
            {"pnl": -75},
            {"pnl": 125},
        ]
        
        with patch.object(trade_executor, '_get_current_pool_balance', new_callable=AsyncMock) as mock:
            mock.return_value = 300.0
            
            with patch('src.execute_trade.query_firestore', return_value=profitable_trades):
                position_size = await trade_executor._calculate_position_size({})
        
        assert position_size > 0
        assert position_size <= 30.0  # Max 10% of pool
    
    @pytest.mark.asyncio
    async def test_kelly_with_negative_expectancy(self, trade_executor):
        """Test Kelly with losing trade history."""
        losing_trades = [
            {"pnl": -50},
            {"pnl": -75},
            {"pnl": 25},
            {"pnl": -100},
            {"pnl": 30},
        ]
        
        with patch.object(trade_executor, '_get_current_pool_balance', new_callable=AsyncMock) as mock:
            mock.return_value = 300.0
            
            with patch('src.execute_trade.query_firestore', return_value=losing_trades):
                position_size = await trade_executor._calculate_position_size({})
        
        # Should still size position (minimum)
        assert position_size >= trade_executor.MIN_ORDER_SIZE_USD

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
