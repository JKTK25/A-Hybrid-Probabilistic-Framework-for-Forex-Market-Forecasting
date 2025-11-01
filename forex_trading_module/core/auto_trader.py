#!/usr/bin/env python3
"""
Automated Trading Engine - Full automation
"""

import threading
import time
from datetime import datetime, timedelta
from core.trading_engine import ForexTradingEngine
from core.risk_manager import RiskManager
from core.broker_connector import BrokerConnector, TradeOrder
from core.database import ForexDatabase
import logging

class AutoTrader:
    def __init__(self, broker: BrokerConnector, risk_manager: RiskManager):
        self.broker = broker
        self.risk_manager = risk_manager
        self.engine = ForexTradingEngine()
        self.db = ForexDatabase()
        
        self.is_running = False
        self.thread = None
        self.pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
        
        # Trading parameters
        self.min_confidence = 0.75  # Minimum confidence for auto trading
        self.max_positions = 3      # Maximum concurrent positions
        self.check_interval = 300   # 5 minutes between checks
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start automated trading"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.thread.start()
            self.logger.info("🤖 Auto Trader Started")
            return True
        return False
    
    def stop(self):
        """Stop automated trading"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=10)
        self.logger.info("🛑 Auto Trader Stopped")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                self._check_and_trade()
                self._manage_positions()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _check_and_trade(self):
        """Check for trading opportunities"""
        # Check daily loss limit
        if not self.risk_manager.check_daily_loss_limit():
            self.logger.warning("Daily loss limit reached - no new trades")
            return
        
        # Check maximum positions
        current_positions = len(self.broker.get_positions())
        if current_positions >= self.max_positions:
            self.logger.info(f"Max positions reached ({current_positions})")
            return
        
        for pair in self.pairs:
            try:
                # Get AI analysis
                analysis = self.engine.ai_agent.analyze_forex_pair(pair)
                
                # Check if signal meets criteria
                if self._should_trade(analysis):
                    self._execute_trade(pair, analysis)
                    
            except Exception as e:
                self.logger.error(f"Analysis error for {pair}: {e}")
    
    def _should_trade(self, analysis):
        """Determine if we should trade based on analysis"""
        confidence = analysis.get("confidence", 0)
        recommendation = analysis.get("recommendation", "HOLD")
        
        # Must meet minimum confidence
        if confidence < self.min_confidence:
            return False
        
        # Must be BUY or SELL (not HOLD)
        if recommendation == "HOLD":
            return False
        
        # Check if we already have position in this pair
        positions = self.broker.get_positions()
        pair_symbol = analysis.get("symbol", "")
        
        for pos in positions:
            if pos.get("symbol", "").replace("=X", "") == pair_symbol.replace("=X", ""):
                return False  # Already have position
        
        return True
    
    def _execute_trade(self, pair, analysis):
        """Execute trading order"""
        try:
            price = analysis.get("price", 0)
            recommendation = analysis.get("recommendation", "HOLD")
            confidence = analysis.get("confidence", 0)
            
            # Calculate position size
            stop_loss = self.risk_manager.calculate_stop_loss(pair, price, recommendation)
            position_size = self.risk_manager.calculate_position_size(pair, price, stop_loss)
            
            # Calculate take profit
            take_profit = self.risk_manager.calculate_take_profit(pair, price, stop_loss)
            
            # Validate trade
            is_valid, checks = self.risk_manager.validate_trade(pair, recommendation, position_size, confidence)
            
            if not is_valid:
                self.logger.warning(f"Trade validation failed for {pair}: {checks}")
                return
            
            # Create and place order
            order = TradeOrder(
                symbol=pair,
                action=recommendation,
                volume=position_size,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            result = self.broker.place_order(order)
            
            if result.get("status") == "filled":
                self.logger.info(f"✅ Trade executed: {pair} {recommendation} {position_size} @ {price}")
                
                # Log to database
                self._log_trade(pair, analysis, order, result)
            else:
                self.logger.error(f"❌ Trade failed: {pair} - {result}")
                
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
    
    def _manage_positions(self):
        """Manage open positions"""
        positions = self.broker.get_positions()
        
        for position in positions:
            try:
                symbol = position.get("symbol", "")
                current_pnl = position.get("pnl", 0)
                entry_price = position.get("entry_price", 0)
                
                # Check for trailing stop or position management
                if self._should_close_position(position):
                    result = self.broker.close_position(symbol)
                    if result.get("status") == "closed":
                        self.logger.info(f"🔒 Position closed: {symbol} P&L: {current_pnl}")
                        
            except Exception as e:
                self.logger.error(f"Position management error: {e}")
    
    def _should_close_position(self, position):
        """Determine if position should be closed"""
        pnl = position.get("pnl", 0)
        entry_time = position.get("timestamp", "")
        
        # Close if loss exceeds 2% of account
        max_loss = self.risk_manager.account_balance * 0.02
        if pnl < -max_loss:
            return True
        
        # Close if profit exceeds 4% of account (take profit)
        max_profit = self.risk_manager.account_balance * 0.04
        if pnl > max_profit:
            return True
        
        # Close if position is older than 24 hours
        try:
            if entry_time:
                entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                if datetime.now() - entry_dt > timedelta(hours=24):
                    return True
        except:
            pass
        
        return False
    
    def _log_trade(self, pair, analysis, order, result):
        """Log trade to database"""
        try:
            trade_log = {
                "symbol": pair,
                "action": order.action,
                "volume": order.volume,
                "entry_price": order.entry_price,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "confidence": analysis.get("confidence", 0),
                "order_id": result.get("order_id", ""),
                "timestamp": datetime.now(),
                "auto_trade": True
            }
            
            # Save to database (extend ForexAnalysis table or create new table)
            self.logger.info(f"📝 Trade logged: {trade_log}")
            
        except Exception as e:
            self.logger.error(f"Trade logging error: {e}")
    
    def get_status(self):
        """Get auto trader status"""
        positions = self.broker.get_positions()
        account = self.broker.get_account_info()
        
        return {
            "is_running": self.is_running,
            "positions_count": len(positions),
            "account_balance": account.get("balance", 0),
            "daily_pnl": sum(pos.get("pnl", 0) for pos in positions),
            "last_check": datetime.now().isoformat(),
            "parameters": {
                "min_confidence": self.min_confidence,
                "max_positions": self.max_positions,
                "check_interval": self.check_interval
            }
        }
    
    def update_parameters(self, min_confidence=None, max_positions=None, check_interval=None):
        """Update trading parameters"""
        if min_confidence is not None:
            self.min_confidence = max(0.5, min(1.0, min_confidence))
        if max_positions is not None:
            self.max_positions = max(1, min(10, max_positions))
        if check_interval is not None:
            self.check_interval = max(60, min(3600, check_interval))
        
        self.logger.info(f"Parameters updated: conf={self.min_confidence}, pos={self.max_positions}, interval={self.check_interval}")

# Global auto trader instance
auto_trader_instance = None

def get_auto_trader():
    """Get global auto trader instance"""
    global auto_trader_instance
    if auto_trader_instance is None:
        broker = BrokerConnector("demo")
        risk_manager = RiskManager()
        auto_trader_instance = AutoTrader(broker, risk_manager)
    return auto_trader_instance