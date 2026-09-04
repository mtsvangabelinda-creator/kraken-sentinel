"""
Kraken Sentinel V15.0
Autonomous cryptocurrency trading bot for Kraken exchange.

Strategies:
- Approach B (Momentum): High-frequency directional trades
- Approach C (Swing): Swing trading on technical patterns

Validation: Combined ICIR + Walk-Forward
Risk Management: ATR stops + Kelly Criterion
Game Theory: Embedded across all layers
"""

__version__ = "15.0"
__author__ = "Belinda"
__description__ = "Build Once, Evolve Forever"

from src.helpers import *
from src.asset_scoring import *
from src.regime_detection import *
from src.scan_momentum import *
from src.fetch_ohlcv import *
from src.generate_swing_signals import *
from src.execute_trade import *
from src.check_positions import *
from src.close_trade import *
from src.risk_manager import *
from src.validation import *
from src.evolution_engine import *
from src.llm_strategist import *
from src.margin_toggle import *
from src.send_notification import *

__all__ = [
    "scan_momentum",
    "fetch_ohlcv",
    "generate_swing_signals",
    "execute_trade",
    "check_positions",
    "close_trade",
    "evolution_engine",
    "send_notification",
    "asset_scoring",
    "regime_detection",
    "risk_manager",
    "validation",
    "llm_strategist",
    "margin_toggle",
    "helpers",
]
