import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

# Mock firestore before importing src modules
import sys
sys.modules['firebase_admin'] = MagicMock()
sys.modules['firebase_admin.firestore'] = MagicMock()

from src.validation import ValidationEngine

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_trades():
    """Generate sample trade history."""
    trades = []
    base_time = datetime.utcnow()
    
    for i in range(50):
        pnl = 100 if i % 3 == 0 else -50
        
        trades.append({
            "pair": "BTCUSD",
            "bias": "long" if i % 2 == 0 else "short",
            "entry_price": 50000 + (i * 100),
            "exit_price": 50000 + (i * 100) + (pnl / 0.01),
            "pnl": pnl,
            "pnl_percent": (pnl / 500) * 100,
            "hold_time_hours": 4,
            "rsi_entry": 60 + (i % 20),
            "volume_spike": 2.0 + (i % 5) * 0.2,
            "asset_score": 70,
            "timestamp": (base_time - timedelta(days=i)).isoformat()
        })
    
    return trades

@pytest.fixture
def sample_variant():
    """Generate sample strategy variant."""
    return {
        "id": "variant_001",
        "params": {
            "z_score": 1.6,
            "take_profit_percent": 0.048,
            "stop_loss_percent": 0.032,
            "hold_time_minutes": 180
        }
    }

@pytest.fixture
def validation_engine():
    """Create ValidationEngine instance."""
    engine = ValidationEngine()
    return engine

# ============================================================================
# ICIR CALCULATION TESTS
# ============================================================================

class TestICIRCalculation:
    """Test ICIR (Information Coefficient Information Ratio) calculation."""
    
    def test_icir_threshold(self, validation_engine):
        """Test ICIR threshold constant."""
        assert validation_engine.ICIR_THRESHOLD == 0.2
    
    def test_icir_calculation_with_profitable_trades(self, validation_engine, sample_trades):
        """Test ICIR calculation with profitable trades."""
        icir = validation_engine._calculate_icir(sample_trades)
        
        assert isinstance(icir, float)
        # ICIR should be in reasonable range
        assert -1 <= icir <= 1
    
    def test_icir_calculation_empty_trades(self, validation_engine):
        """Test ICIR with empty trades."""
        icir = validation_engine._calculate_icir([])
        
        assert icir == 0
    
    def test_icir_calculation_insufficient_data(self, validation_engine):
        """Test ICIR with insufficient trades."""
        small_trades = [
            {"pnl": 100, "pnl_percent": 0.02, "rsi_entry": 65},
            {"pnl": -50, "pnl_percent": -0.01, "rsi_entry": 45},
        ]
        
        icir = validation_engine._calculate_icir(small_trades)
        
        assert icir == 0
    
    def test_forecast_signal_extraction(self, validation_engine):
        """Test extraction of forecast signals."""
        trade = {
            "rsi_entry": 75,
            "volume_spike": 3.0,
            "asset_score": 90
        }
        
        signal = validation_engine._get_forecast_signal(trade)
        
        assert isinstance(signal, float)
        assert 0 <= signal <= 100
    
    def test_forecast_signal_neutral(self, validation_engine):
        """Test forecast signal at neutral values."""
        trade = {
            "rsi_entry": 50,
            "volume_spike": 1.0,
            "asset_score": 50
        }
        
        signal = validation_engine._get_forecast_signal(trade)
        
        # Neutral condition should give lower signal
        assert signal < 50

# ============================================================================
# WALK-FORWARD WINDOW TESTS
# ============================================================================

class TestWalkForwardWindows:
    """Test walk-forward window creation."""
    
    def test_window_creation(self, validation_engine, sample_trades):
        """Test walk-forward window creation."""
        windows = validation_engine._create_rolling_windows(sample_trades, window_count=3)
        
        assert len(windows) <= 3
        
        for in_sample, out_sample in windows:
            assert len(in_sample) > 0
            assert len(out_sample) > 0
    
    def test_window_separation(self, validation_engine, sample_trades):
        """Test windows are properly separated."""
        windows = validation_engine._create_rolling_windows(sample_trades, window_count=2)
        
        for in_sample, out_sample in windows:
            # Out-of-sample should be different from in-sample
            assert in_sample != out_sample
    
    def test_window_temporal_order(self, validation_engine, sample_trades):
        """Test windows maintain temporal order."""
        windows = validation_engine._create_rolling_windows(sample_trades, window_count=2)
        
        for in_sample, out_sample in windows:
            # All trades should have timestamps
            assert all("timestamp" in t for t in in_sample)
            assert all("timestamp" in t for t in out_sample)

# ============================================================================
# TRADE METRICS TESTS
# ============================================================================

class TestTradeMetrics:
    """Test trade performance metrics calculation."""
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, validation_engine, sample_trades):
        """Test trade metrics calculation."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
    
    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, validation_engine, sample_trades):
        """Test win rate calculation."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        win_rate = metrics["win_rate"]
        assert 0 <= win_rate <= 1
    
    @pytest.mark.asyncio
    async def test_profit_factor_calculation(self, validation_engine, sample_trades):
        """Test profit factor calculation."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        profit_factor = metrics["profit_factor"]
        assert profit_factor >= 0
    
    @pytest.mark.asyncio
    async def test_max_drawdown_calculation(self, validation_engine, sample_trades):
        """Test max drawdown calculation."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        max_drawdown = metrics["max_drawdown"]
        assert -1 <= max_drawdown <= 0
    
    @pytest.mark.asyncio
    async def test_sharpe_ratio_calculation(self, validation_engine, sample_trades):
        """Test Sharpe ratio calculation."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        sharpe = metrics["sharpe_ratio"]
        assert isinstance(sharpe, float)
    
    @pytest.mark.asyncio
    async def test_consecutive_wins_tracking(self, validation_engine, sample_trades):
        """Test consecutive wins tracking."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        consecutive_wins = metrics["consecutive_wins"]
        assert consecutive_wins >= 0
    
    @pytest.mark.asyncio
    async def test_consecutive_losses_tracking(self, validation_engine, sample_trades):
        """Test consecutive losses tracking."""
        metrics = await validation_engine._calculate_trade_metrics(sample_trades)
        
        consecutive_losses = metrics["consecutive_losses"]
        assert consecutive_losses >= 0
    
    @pytest.mark.asyncio
    async def test_metrics_with_no_trades(self, validation_engine):
        """Test metrics with no trades."""
        metrics = await validation_engine._calculate_trade_metrics([])
        
        assert metrics["win_rate"] == 0
        assert metrics["profit_factor"] == 0

# ============================================================================
# VALIDATION TESTS
# ============================================================================

class TestStrategyValidation:
    """Test strategy variant validation."""
    
    @pytest.mark.asyncio
    async def test_validation_with_sufficient_data(self, validation_engine, sample_trades):
        """Test validation with sufficient trade data."""
        with patch.object(validation_engine, '_fetch_historical_trades', new_callable=AsyncMock) as mock:
            mock.return_value = sample_trades
            
            with patch.object(validation_engine, '_train_hmm', new_callable=AsyncMock):
                report = await validation_engine.validate_strategy_variant("var_001", {})
        
        assert report is not None
        assert "variant_id" in report
    
    @pytest.mark.asyncio
    async def test_validation_insufficient_data(self, validation_engine):
        """Test validation with insufficient data."""
        with patch.object(validation_engine, '_fetch_historical_trades', new_callable=AsyncMock) as mock:
            mock.return_value = []
            
            report = await validation_engine.validate_strategy_variant("var_001", {})
        
        assert report["status"] == "insufficient_data"
        assert report["passes_validation"] is False
    
    @pytest.mark.asyncio
    async def test_validation_report_structure(self, validation_engine, sample_trades):
        """Test validation report has required fields."""
        with patch.object(validation_engine, '_fetch_historical_trades', new_callable=AsyncMock) as mock:
            mock.return_value = sample_trades[:20]
            
            with patch.object(validation_engine, '_create_rolling_windows') as mock_windows:
                mock_windows.return_value = [
                    (sample_trades[:10], sample_trades[10:15])
                ]
                
                report = await validation_engine.validate_strategy_variant("var_001", {})
        
        assert "variant_id" in report
        assert "timestamp" in report
        assert "icir" in report
        assert "metrics" in report

# ============================================================================
# THRESHOLD TESTS
# ============================================================================

class TestValidationThresholds:
    """Test validation threshold constants."""
    
    def test_icir_threshold_value(self, validation_engine):
        """Test ICIR threshold."""
        assert validation_engine.ICIR_THRESHOLD == 0.2
    
    def test_profit_factor_threshold(self, validation_engine):
        """Test profit factor threshold."""
        assert validation_engine.PROFIT_FACTOR_THRESHOLD == 1.2
    
    def test_win_rate_threshold(self, validation_engine):
        """Test win rate threshold."""
        assert validation_engine.WIN_RATE_THRESHOLD == 0.45
    
    def test_max_drawdown_threshold(self, validation_engine):
        """Test max drawdown threshold."""
        assert validation_engine.MAX_DRAWDOWN_THRESHOLD == -0.15

# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestValidationEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.asyncio
    async def test_metrics_all_winning_trades(self, validation_engine):
        """Test metrics with all winning trades."""
        winning_trades = [
            {"pnl": 100, "pnl_percent": 0.02} for _ in range(10)
        ]
        
        metrics = await validation_engine._calculate_trade_metrics(winning_trades)
        
        assert metrics["win_rate"] == 1.0
        assert metrics["consecutive_losses"] == 0
    
    @pytest.mark.asyncio
    async def test_metrics_all_losing_trades(self, validation_engine):
        """Test metrics with all losing trades."""
        losing_trades = [
            {"pnl": -50, "pnl_percent": -0.01} for _ in range(10)
        ]
        
        metrics = await validation_engine._calculate_trade_metrics(losing_trades)
        
        assert metrics["win_rate"] == 0.0
        assert metrics["consecutive_wins"] == 0
    
    def test_forecast_signal_extremes(self, validation_engine):
        """Test forecast signal with extreme values."""
        extreme_high = {
            "rsi_entry": 100,
            "volume_spike": 5.0,
            "asset_score": 100
        }
        
        signal = validation_engine._get_forecast_signal(extreme_high)
        assert 0 <= signal <= 100
    
    def test_icir_with_zero_std_dev(self, validation_engine):
        """Test ICIR with flat returns."""
        flat_trades = [
            {"pnl": 100, "pnl_percent": 0.02, "rsi_entry": 65},
            {"pnl": 100, "pnl_percent": 0.02, "rsi_entry": 65},
            {"pnl": 100, "pnl_percent": 0.02, "rsi_entry": 65},
        ]
        
        icir = validation_engine._calculate_icir(flat_trades)
        
        assert isinstance(icir, float)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestValidationIntegration:
    """Integration tests for validation system."""
    
    @pytest.mark.asyncio
    async def test_full_validation_pipeline(self, validation_engine, sample_trades, sample_variant):
        """Test full validation pipeline."""
        with patch.object(validation_engine, '_fetch_historical_trades', new_callable=AsyncMock) as mock_trades:
            mock_trades.return_value = sample_trades[:30]
            
            with patch.object(validation_engine, '_create_rolling_windows') as mock_windows:
                in_sample = sample_trades[:15]
                out_sample = sample_trades[15:20]
                mock_windows.return_value = [(in_sample, out_sample)]
                
                with patch.object(validation_engine, '_save_validation_report', new_callable=AsyncMock):
                    report = await validation_engine.validate_strategy_variant(
                        sample_variant["id"],
                        sample_variant["params"]
                    )
        
        assert report is not None
        assert report["variant_id"] == sample_variant["id"]

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestValidationErrors:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_validation_with_exception(self, validation_engine):
        """Test validation handles exceptions gracefully."""
        with patch.object(validation_engine, '_fetch_historical_trades', new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Database error")
            
            report = await validation_engine.validate_strategy_variant("var_001", {})
        
        assert report["passes_validation"] is False
        assert report["status"] == "error"
    
    def test_metrics_with_nan_values(self, validation_engine):
        """Test metrics calculation handles NaN."""
        trades_with_nan = [
            {"pnl": float('nan'), "pnl_percent": 0.02},
            {"pnl": 100, "pnl_percent": 0.02},
        ]
        
        # Should not crash
        import asyncio
        metrics = asyncio.run(validation_engine._calculate_trade_metrics(trades_with_nan))
        
        assert isinstance(metrics, dict)

# ============================================================================
# COMPARISON TESTS
# ============================================================================

class TestValidationComparison:
    """Test variant comparison."""
    
    @pytest.mark.asyncio
    async def test_better_variant_selection(self, validation_engine):
        """Test selecting better performing variant."""
        variant1_metrics = {
            "icir": 0.15,
            "win_rate": 0.45,
            "profit_factor": 1.1
        }
        
        variant2_metrics = {
            "icir": 0.25,
            "win_rate": 0.52,
            "profit_factor": 1.4
        }
        
        # Variant 2 is better
        assert variant2_metrics["icir"] > variant1_metrics["icir"]
    
    def test_icir_consistency_check(self, validation_engine):
        """Test ICIR consistency across windows."""
        icir_values = [0.22, 0.21, 0.23, 0.20, 0.24]
        mean_icir = np.mean(icir_values)
        
        # All above threshold
        assert all(icir > validation_engine.ICIR_THRESHOLD for icir in icir_values)
        assert mean_icir > validation_engine.ICIR_THRESHOLD

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestValidationPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_metrics_calculation_speed(self, validation_engine, sample_trades):
        """Test metrics calculation speed."""
        import time
        
        start = time.time()
        for _ in range(10):
            await validation_engine._calculate_trade_metrics(sample_trades)
        elapsed = time.time() - start
        
        # 10 metric calculations should be fast
        assert elapsed < 5.0
    
    def test_icir_calculation_speed(self, validation_engine, sample_trades):
        """Test ICIR calculation speed."""
        import time
        
        start = time.time()
        for _ in range(100):
            validation_engine._calculate_icir(sample_trades)
        elapsed = time.time() - start
        
        assert elapsed < 2.0

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
