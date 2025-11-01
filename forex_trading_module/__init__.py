"""
Forex Trading Module - Complete AI-powered forex trading system
"""

from .core.trading_engine import ForexTradingEngine
from .core.ai_agent import AITradingAgent
from .core.database import ForexDatabase
from .core.performance import PerformanceTracker
from .dashboard.live_dashboard import LiveForexDashboard

__version__ = "1.0.0"
__all__ = [
    "ForexTradingEngine",
    "AITradingAgent", 
    "ForexDatabase",
    "PerformanceTracker",
    "LiveForexDashboard"
]