import functions_framework
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import logging
import os

from src.scan_momentum import scan_momentum
from src.fetch_ohlcv import fetch_ohlcv
from src.generate_swing_signals import generate_swing_signals
from src.execute_trade import execute_trade
from src.check_positions import check_positions
from src.close_trade import close_trade
from src.evolution_engine import evolution_engine
from src.send_notification import send_notification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firebase (auto-detects service account from environment)
try:
    firebase_admin.initialize_app()
except ValueError:
    # App already initialized
    pass

db = firestore.client()

# ============================================================================
# CLOUD SCHEDULER TRIGGERS
# ============================================================================

@functions_framework.cloud_event
def scan_momentum_trigger(cloud_event):
    """
    Triggered every 5 minutes by Cloud Scheduler.
    Scans all Kraken assets for momentum opportunities (Approach B).
    """
    try:
        logger.info("🔍 Starting momentum scan...")
        await scan_momentum()
        logger.info("✅ Momentum scan completed")
    except Exception as e:
        logger.error(f"❌ Momentum scan failed: {str(e)}")
        raise

@functions_framework.cloud_event
def fetch_ohlcv_trigger(cloud_event):
    """
    Triggered every 60 minutes by Cloud Scheduler.
    Fetches OHLCV data from Kraken and stores in Firestore.
    """
    try:
        logger.info("📊 Fetching OHLCV data...")
        await fetch_ohlcv()
        logger.info("✅ OHLCV fetch completed")
    except Exception as e:
        logger.error(f"❌ OHLCV fetch failed: {str(e)}")
        raise

@functions_framework.cloud_event
def generate_swing_signals_trigger(cloud_event):
    """
    Triggered every 60 minutes by Cloud Scheduler.
    Calculates MA50, RSI, Volume signals (Approach C).
    """
    try:
        logger.info("📈 Generating swing signals...")
        await generate_swing_signals()
        logger.info("✅ Swing signals generated")
    except Exception as e:
        logger.error(f"❌ Swing signal generation failed: {str(e)}")
        raise

@functions_framework.cloud_event
def check_positions_trigger(cloud_event):
    """
    Triggered every 60 minutes by Cloud Scheduler.
    Checks all open positions for exit conditions.
    """
    try:
        logger.info("🔎 Checking open positions...")
        await check_positions()
        logger.info("✅ Position check completed")
    except Exception as e:
        logger.error(f"❌ Position check failed: {str(e)}")
        raise

@functions_framework.cloud_event
def evolution_engine_trigger(cloud_event):
    """
    Triggered daily at 00:05 UTC by Cloud Scheduler.
    Runs ICIR + Walk-Forward validation and parameter mutation.
    """
    try:
        logger.info("🧬 Starting evolution engine...")
        await evolution_engine()
        logger.info("✅ Evolution engine completed")
    except Exception as e:
        logger.error(f"❌ Evolution engine failed: {str(e)}")
        raise

# ============================================================================
# FIRESTORE TRIGGERS
# ============================================================================

@functions_framework.cloud_event
def execute_trade_on_signal(cloud_event):
    """
    Triggered when a new signal document is created in Firestore.
    Places a spot order on Kraken REST API.
    """
    try:
        signal_data = cloud_event.data["value"]["fields"]
        logger.info(f"🎯 Executing trade for signal: {signal_data}")
        await execute_trade(signal_data)
        logger.info("✅ Trade executed")
    except Exception as e:
        logger.error(f"❌ Trade execution failed: {str(e)}")
        raise

@functions_framework.cloud_event
def close_trade_on_update(cloud_event):
    """
    Triggered when an open_positions document is updated.
    Closes position via Kraken REST API if exit condition met.
    """
    try:
        position_data = cloud_event.data["value"]["fields"]
        logger.info(f"📉 Closing trade for position: {position_data}")
        await close_trade(position_data)
        logger.info("✅ Trade closed")
    except Exception as e:
        logger.error(f"❌ Trade closure failed: {str(e)}")
        raise

@functions_framework.cloud_event
def send_trade_notification(cloud_event):
    """
    Triggered when a new trade document is created in Firestore.
    Sends FCM/Telegram notification.
    """
    try:
        trade_data = cloud_event.data["value"]["fields"]
        logger.info(f"📢 Sending notification for trade: {trade_data}")
        await send_notification(trade_data)
        logger.info("✅ Notification sent")
    except Exception as e:
        logger.error(f"❌ Notification send failed: {str(e)}")
        raise

# ============================================================================
# HEALTH CHECK (Cold Start Prevention)
# ============================================================================

@functions_framework.http
def health_check(request):
    """
    HTTP endpoint for health check.
    Triggered every 5 minutes to prevent cold starts.
    """
    try:
        logger.info("🏥 Health check: OK")
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "V15.0"
        }, 200
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {"status": "error", "error": str(e)}, 500

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@functions_framework.http
def get_status(request):
    """
    HTTP endpoint to get current system status.
    Returns: Equity, ICIR, open positions, recent trades.
    """
    try:
        logger.info("📊 Getting system status...")
        
        # Fetch latest equity
        equity_snapshot = db.collection("equity_history").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
        latest_equity = list(equity_snapshot)
        
        # Fetch ICIR
        genetic_history = db.collection("genetic_history").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
        latest_icir = list(genetic_history)
        
        # Fetch open positions
        open_positions = db.collection("open_positions").stream()
        positions = [doc.to_dict() for doc in open_positions]
        
        # Fetch recent trades
        recent_trades = db.collection("trade_history").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
        trades = [doc.to_dict() for doc in recent_trades]
        
        status = {
            "latest_equity": latest_equity[0].to_dict() if latest_equity else None,
            "latest_icir": latest_icir[0].to_dict() if latest_icir else None,
            "open_positions": positions,
            "recent_trades": trades,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return status, 200
    except Exception as e:
        logger.error(f"❌ Status check failed: {str(e)}")
        return {"error": str(e)}, 500

@functions_framework.http
def get_dashboard_data(request):
    """
    HTTP endpoint for dashboard data.
    Returns: Equity curve, ICIR history, asset rankings.
    """
    try:
        logger.info("📈 Fetching dashboard data...")
        
        # Fetch equity history (last 100)
        equity_history = db.collection("equity_history").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(100).stream()
        equity_data = [doc.to_dict() for doc in equity_history]
        
        # Fetch ICIR history (last 50)
        icir_history = db.collection("genetic_history").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50).stream()
        icir_data = [doc.to_dict() for doc in icir_history]
        
        # Fetch asset scores
        asset_scores = db.collection("asset_scores").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
        scores = list(asset_scores)
        
        dashboard_data = {
            "equity_curve": equity_data,
            "icir_history": icir_data,
            "asset_scores": scores[0].to_dict() if scores else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return dashboard_data, 200
    except Exception as e:
        logger.error(f"❌ Dashboard data fetch failed: {str(e)}")
        return {"error": str(e)}, 500
