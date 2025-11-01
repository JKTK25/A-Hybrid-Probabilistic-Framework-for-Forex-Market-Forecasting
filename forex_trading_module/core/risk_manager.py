#!/usr/bin/env python3
"""
Risk Management System - Critical for live trading
"""

import numpy as np
from datetime import datetime, timedelta
from core.database import ForexDatabase

class RiskManager:
    def __init__(self, account_balance=10000, max_risk_per_trade=0.02, max_daily_loss=0.05):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_daily_loss = max_daily_loss  # 5% daily loss limit
        self.max_drawdown = 0.15  # 15% max drawdown
        self.db = ForexDatabase()
        
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, risk_amount=None):
        """Calculate position size based on risk management"""
        if risk_amount is None:
            risk_amount = self.account_balance * self.max_risk_per_trade
            
        # Calculate pip value and position size
        pip_value = abs(entry_price - stop_loss_price)
        if pip_value == 0:
            return 0
            
        # For forex pairs, calculate lot size
        if "JPY" in symbol:
            pip_size = 0.01  # JPY pairs
        else:
            pip_size = 0.0001  # Other pairs
            
        pips_at_risk = pip_value / pip_size
        lot_size = risk_amount / (pips_at_risk * 10)  # $10 per pip for standard lot
        
        return min(lot_size, 1.0)  # Max 1 standard lot
    
    def check_daily_loss_limit(self):
        """Check if daily loss limit is reached"""
        today = datetime.now().date()
        
        # Get today's trades (simulated from database)
        analyses = self.db.session.query(self.db.ForexAnalysis).filter(
            self.db.ForexAnalysis.timestamp >= today
        ).all()
        
        daily_pnl = 0
        for analysis in analyses:
            # Simulate P&L calculation
            if "BUY" in analysis.recommendation:
                daily_pnl += analysis.confidence * 100  # Simplified P&L
            elif "SELL" in analysis.recommendation:
                daily_pnl -= analysis.confidence * 100
                
        daily_loss_pct = abs(daily_pnl) / self.account_balance
        return daily_loss_pct < self.max_daily_loss
    
    def validate_trade(self, symbol, action, lot_size, confidence):
        """Validate if trade meets risk criteria"""
        checks = {
            "daily_limit": self.check_daily_loss_limit(),
            "position_size": lot_size <= 1.0,
            "confidence": confidence >= 0.6,
            "max_positions": self.get_open_positions_count() < 5
        }
        
        return all(checks.values()), checks
    
    def get_open_positions_count(self):
        """Get number of open positions (simulated)"""
        # In real implementation, this would query broker API
        return len(self.db.session.query(self.db.ForexAnalysis).filter(
            self.db.ForexAnalysis.timestamp >= datetime.now() - timedelta(hours=4)
        ).all())
    
    def calculate_stop_loss(self, symbol, entry_price, action, atr_multiplier=2.0):
        """Calculate stop loss based on ATR"""
        # Simplified ATR calculation
        atr = entry_price * 0.001  # 0.1% of price as ATR approximation
        
        if action.upper() == "BUY":
            stop_loss = entry_price - (atr * atr_multiplier)
        else:
            stop_loss = entry_price + (atr * atr_multiplier)
            
        return round(stop_loss, 5)
    
    def calculate_take_profit(self, symbol, entry_price, stop_loss, risk_reward_ratio=2.0):
        """Calculate take profit based on risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        
        if entry_price > stop_loss:  # BUY trade
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:  # SELL trade
            take_profit = entry_price - (risk * risk_reward_ratio)
            
        return round(take_profit, 5)
    
    def get_risk_report(self):
        """Generate risk management report"""
        return {
            "account_balance": self.account_balance,
            "max_risk_per_trade": f"{self.max_risk_per_trade:.1%}",
            "daily_loss_limit": f"{self.max_daily_loss:.1%}",
            "open_positions": self.get_open_positions_count(),
            "daily_limit_ok": self.check_daily_loss_limit(),
            "max_position_size": 1.0
        }