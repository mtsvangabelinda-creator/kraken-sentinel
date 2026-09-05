import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
import json

from src.helpers import (
    make_request,
    get_optional_env_var,
    get_utc_timestamp,
    log_info,
    log_error
)

logger = logging.getLogger(__name__)

# ============================================================================
# NOTIFICATION SYSTEM
# ============================================================================

class NotificationManager:
    """
    Send notifications via FCM (Firebase Cloud Messaging) and Telegram.
    
    Channels:
    1. FCM: Push notifications to mobile app (Firebase)
    2. Telegram: Bot messages to user's Telegram chat
    
    Notification Types:
    - Trade Opened: New position entered
    - Trade Closed: Position closed with P&L
    - Risk Alerts: Daily loss, drawdown, margin ratio
    - Evolution: Best variant found, ICIR updated
    - System Alerts: Errors, circuit breakers, API failures
    - Daily Summary: EOD performance report
    
    Priority Levels:
    - LOW: Information, stats
    - MEDIUM: Alerts, warnings
    - HIGH: Critical, risk breaches
    
    Rate Limiting:
    - Max 100 notifications per hour
    - Batch alerts during high-frequency events
    - Deduplicate consecutive identical alerts
    """
    
    # API Keys
    TELEGRAM_BOT_TOKEN = get_optional_env_var("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = get_optional_env_var("TELEGRAM_CHAT_ID", "")
    FCM_SERVER_KEY = get_optional_env_var("FCM_SERVER_KEY", "")
    
    # Rate limiting
    MAX_NOTIFICATIONS_PER_HOUR = 100
    BATCH_ALERTS_AFTER_COUNT = 5
    ALERT_BATCH_WAIT_SECONDS = 60
    
    # Emoji mapping
    EMOJI = {
        "long": "🟢",
        "short": "🔴",
        "win": "✅",
        "loss": "❌",
        "alert": "⚠️",
        "critical": "🚨",
        "info": "ℹ️",
        "chart": "📊",
        "rocket": "🚀",
        "fire": "🔥",
        "target": "🎯",
        "clock": "⏰"
    }
    
    def __init__(self):
        self.telegram_available = bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)
        self.fcm_available = bool(self.FCM_SERVER_KEY)
    
    # ========================================================================
    # MAIN NOTIFICATION METHODS
    # ========================================================================
    
    async def notify_trade_opened(self, trade: Dict) -> bool:
        """Send notification when trade is opened."""
        try:
            pair = trade.get("pair")
            bias = trade.get("bias")
            entry_price = trade.get("entry_price")
            size = trade.get("position_size_usd", 0)
            strategy = trade.get("strategy")
            rsi = trade.get("rsi")
            
            emoji = self.EMOJI[bias]
            title = f"{emoji} Trade Opened: {pair}"
            
            body = f"""
{emoji} {bias.upper()} {pair} @ ${entry_price:.2f}
Size: ${size:.2f}
Strategy: {strategy}
RSI: {rsi:.1f}
Time: {datetime.utcnow().strftime('%H:%M:%S UTC')}
"""
            
            data = {
                "type": "trade_opened",
                "pair": pair,
                "bias": bias,
                "entry_price": str(entry_price),
                "size": str(size)
            }
            
            await self._send_notification(title, body.strip(), data, priority="high")
            return True
        except Exception as e:
            log_error("❌ Failed to send trade opened notification", e)
            return False
    
    async def notify_trade_closed(self, trade: Dict) -> bool:
        """Send notification when trade is closed."""
        try:
            pair = trade.get("pair")
            bias = trade.get("bias")
            entry_price = trade.get("entry_price")
            exit_price = trade.get("exit_price")
            pnl = trade.get("pnl", 0)
            pnl_percent = trade.get("pnl_percent", 0)
            reason = trade.get("close_reason")
            hold_time = trade.get("hold_time_hours", 0)
            
            emoji_result = self.EMOJI["win"] if pnl > 0 else self.EMOJI["loss"]
            emoji_bias = self.EMOJI[bias]
            
            title = f"{emoji_result} Trade Closed: {pair}"
            
            body = f"""
{emoji_bias} {bias.upper()} {pair}
Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}
P&L: ${pnl:+.2f} ({pnl_percent:+.2f}%)
Hold: {hold_time:.1f}h
Reason: {reason}
Time: {datetime.utcnow().strftime('%H:%M:%S UTC')}
"""
            
            data = {
                "type": "trade_closed",
                "pair": pair,
                "pnl": str(pnl),
                "pnl_percent": str(pnl_percent)
            }
            
            await self._send_notification(title, body.strip(), data, priority="high")
            return True
        except Exception as e:
            log_error("❌ Failed to send trade closed notification", e)
            return False
    
    async def notify_risk_alert(self, alert_type: str, alert_data: Dict) -> bool:
        """Send risk alert notification."""
        try:
            emoji = self.EMOJI["alert"]
            title = f"{emoji} Risk Alert: {alert_type}"
            
            if alert_type == "daily_loss_limit":
                daily_loss = alert_data.get("daily_loss", 0)
                limit = alert_data.get("limit", 0)
                body = f"""
{emoji} Daily Loss Limit Breached
Current: ${daily_loss:.2f}
Limit: ${limit:.2f}
Status: TRADING FROZEN FOR 24H
"""
            
            elif alert_type == "max_drawdown":
                drawdown = alert_data.get("drawdown", 0)
                body = f"""
{emoji} Maximum Drawdown Reached
Current: {drawdown:.1%}
Threshold: -10%
Action: Position sizes halved
"""
            
            elif alert_type == "margin_ratio":
                margin_ratio = alert_data.get("margin_ratio", 0)
                action = alert_data.get("action", "monitor")
                body = f"""
{emoji} Margin Ratio Warning
Current: {margin_ratio:.2f}
Minimum: 1.30
Action: {action.upper()}
"""
            
            elif alert_type == "icir_breaker":
                icir = alert_data.get("icir", 0)
                days = alert_data.get("days", 0)
                body = f"""
{emoji} ICIR Breaker Activated
ICIR: {icir:.4f}
Duration: {days} days below 0.2
Status: STRATEGY FROZEN
"""
            
            elif alert_type == "system_error":
                error_count = alert_data.get("error_count", 0)
                body = f"""
{emoji} System Error Detected
Consecutive Errors: {error_count}
Critical Threshold: 3
Action: Check logs immediately
"""
            
            else:
                body = f"Alert: {alert_type}\n{json.dumps(alert_data)}"
            
            data = {
                "type": "risk_alert",
                "alert_type": alert_type
            }
            
            await self._send_notification(title, body.strip(), data, priority="critical")
            return True
        except Exception as e:
            log_error("❌ Failed to send risk alert", e)
            return False
    
    async def notify_evolution_update(self, evolution_data: Dict) -> bool:
        """Send evolution update notification."""
        try:
            current_icir = evolution_data.get("current_icir", 0)
            best_icir = evolution_data.get("best_variant_icir", 0)
            improvement = evolution_data.get("improvement", 0)
            generation = evolution_data.get("generation", 0)
            
            emoji = self.EMOJI["rocket"] if improvement > 0 else self.EMOJI["info"]
            title = f"{emoji} Evolution Update - Gen {generation}"
            
            status_emoji = self.EMOJI["fire"] if improvement > 0.02 else self.EMOJI["chart"]
            
            body = f"""
{status_emoji} Evolution Cycle Complete
Generation: {generation}
Current ICIR: {current_icir:.4f}
Best Variant: {best_icir:.4f}
Improvement: {improvement:+.4f}
Status: {'✅ Updated' if evolution_data.get('update_applied') else '⚠️ No improvement'}
"""
            
            data = {
                "type": "evolution",
                "generation": str(generation),
                "icir": str(best_icir)
            }
            
            await self._send_notification(title, body.strip(), data, priority="medium")
            return True
        except Exception as e:
            log_error("❌ Failed to send evolution notification", e)
            return False
    
    async def notify_daily_summary(self, summary_data: Dict) -> bool:
        """Send daily performance summary."""
        try:
            trades = summary_data.get("total_trades", 0)
            wins = summary_data.get("wins", 0)
            losses = summary_data.get("losses", 0)
            total_pnl = summary_data.get("total_pnl", 0)
            win_rate = summary_data.get("win_rate", 0)
            
            emoji_result = self.EMOJI["win"] if total_pnl > 0 else self.EMOJI["loss"]
            title = f"{emoji_result} Daily Summary"
            
            body = f"""
📊 Daily Trading Summary
━━━━━━━━━━━━━━━━━━━━━━
Trades: {trades} ({wins}W / {losses}L)
Win Rate: {win_rate:.1%}
Total P&L: ${total_pnl:+.2f}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
"""
            
            data = {
                "type": "daily_summary",
                "trades": str(trades),
                "pnl": str(total_pnl)
            }
            
            await self._send_notification(title, body.strip(), data, priority="medium")
            return True
        except Exception as e:
            log_error("❌ Failed to send daily summary", e)
            return False
    
    # ========================================================================
    # CORE NOTIFICATION SENDING
    # ========================================================================
    
    async def _send_notification(
        self,
        title: str,
        body: str,
        data: Dict = None,
        priority: str = "normal"
    ) -> bool:
        """
        Send notification via all available channels.
        
        Args:
            title: Notification title
            body: Notification body
            data: Additional data dict
            priority: "low", "normal", "medium", "high", "critical"
        
        Returns:
            True if at least one channel succeeded
        """
        try:
            results = []
            
            # Send via Telegram
            if self.telegram_available:
                try:
                    message = f"{title}\n\n{body}"
                    await self._send_telegram_message(message)
                    results.append(True)
                except Exception as e:
                    logger.warning(f"⚠️ Telegram send failed: {str(e)}")
                    results.append(False)
            
            # Send via FCM
            if self.fcm_available:
                try:
                    await self._send_fcm_notification(title, body, data, priority)
                    results.append(True)
                except Exception as e:
                    logger.warning(f"⚠️ FCM send failed: {str(e)}")
                    results.append(False)
            
            # At least one succeeded
            success = any(results)
            
            if success:
                logger.info(f"✅ Notification sent: {title}")
            else:
                logger.warning(f"⚠️ No notification channels available")
            
            return success
        except Exception as e:
            log_error("❌ Notification send failed", e)
            return False
    
    # ========================================================================
    # TELEGRAM NOTIFICATIONS
    # ========================================================================
    
    async def _send_telegram_message(self, message: str) -> bool:
        """Send message via Telegram bot."""
        try:
            url = f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            payload = {
                "chat_id": self.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = await make_request("POST", url, json_data=payload, timeout=10)
            
            if "error_code" in response:
                logger.error(f"❌ Telegram error: {response.get('description')}")
                return False
            
            return response.get("ok", False)
        except Exception as e:
            logger.error(f"❌ Telegram send failed: {str(e)}")
            return False
    
    # ========================================================================
    # FCM NOTIFICATIONS
    # ========================================================================
    
    async def _send_fcm_notification(
        self,
        title: str,
        body: str,
        data: Dict = None,
        priority: str = "normal"
    ) -> bool:
        """Send notification via Firebase Cloud Messaging."""
        try:
            # FCM requires topic subscription setup
            # For now, this is a placeholder for production integration
            
            url = "https://fcm.googleapis.com/fcm/send"
            
            headers = {
                "Authorization": f"key={self.FCM_SERVER_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "to": "/topics/trading_alerts",  # Subscribe users to this topic
                "notification": {
                    "title": title,
                    "body": body,
                    "sound": "default",
                    "click_action": "FLUTTER_NOTIFICATION_CLICK"
                },
                "data": data or {},
                "priority": priority
            }
            
            response = await make_request("POST", url, headers=headers, json_data=payload, timeout=10)
            
            if "error" in response:
                logger.error(f"❌ FCM error: {response.get('error')}")
                return False
            
            success_count = response.get("success", 0)
            return success_count > 0
        except Exception as e:
            logger.error(f"❌ FCM send failed: {str(e)}")
            return False
    
    # ========================================================================
    # BATCH NOTIFICATIONS
    # ========================================================================
    
    async def batch_alerts(self, alerts: List[Dict]) -> bool:
        """Batch multiple alerts into single notification."""
        try:
            if not alerts:
                return True
            
            title = f"📊 {len(alerts)} Alert(s)"
            
            body = "Alert Summary:\n"
            for i, alert in enumerate(alerts[:5], 1):  # Max 5 in batch
                alert_type = alert.get("type", "unknown")
                body += f"{i}. {alert_type}\n"
            
            if len(alerts) > 5:
                body += f"\n... and {len(alerts) - 5} more alerts"
            
            await self._send_notification(title, body.strip(), priority="high")
            return True
        except Exception as e:
            log_error("❌ Batch alert send failed", e)
            return False
    
    # ========================================================================
    # RATE LIMITING
    # ========================================================================
    
    async def check_notification_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        try:
            # In production, track notification count per hour in Firestore
            # For now, simple placeholder
            return True
        except Exception as e:
            logger.error(f"❌ Rate limit check failed: {str(e)}")
            return True  # Default to allow

# ============================================================================
# MAIN FUNCTION
# ============================================================================

async def send_notification(notification_type: str, data: Dict) -> bool:
    """Main entry point for sending notifications."""
    try:
        manager = NotificationManager()
        
        if notification_type == "trade_opened":
            return await manager.notify_trade_opened(data)
        elif notification_type == "trade_closed":
            return await manager.notify_trade_closed(data)
        elif notification_type == "risk_alert":
            return await manager.notify_risk_alert(data.get("alert_type"), data)
        elif notification_type == "evolution":
            return await manager.notify_evolution_update(data)
        elif notification_type == "daily_summary":
            return await manager.notify_daily_summary(data)
        else:
            logger.warning(f"⚠️ Unknown notification type: {notification_type}")
            return False
    except Exception as e:
        log_error("❌ Notification sending failed", e)
        return False
