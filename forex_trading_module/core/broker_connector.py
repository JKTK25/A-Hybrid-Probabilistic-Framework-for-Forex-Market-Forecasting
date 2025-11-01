#!/usr/bin/env python3
"""
Broker Integration - For actual trade execution
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TradeOrder:
    symbol: str
    action: str  # BUY/SELL
    volume: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    order_type: str = "MARKET"

@dataclass
class Position:
    symbol: str
    volume: float
    entry_price: float
    current_price: float
    pnl: float
    timestamp: datetime

class BrokerConnector:
    """Base broker connector - can be extended for specific brokers"""
    
    def __init__(self, broker_type="demo", api_key=None, account_id=None):
        self.broker_type = broker_type
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = self._get_broker_url()
        self.positions = []  # Simulated positions
        self.balance = 10000.0  # Demo balance
        
    def _get_broker_url(self):
        """Get broker API URL based on type"""
        urls = {
            "demo": "https://demo-api.broker.com",
            "oanda": "https://api-fxtrade.oanda.com",
            "mt4": "http://localhost:8080/mt4",
            "ib": "https://api.interactivebrokers.com"
        }
        return urls.get(self.broker_type, urls["demo"])
    
    def connect(self):
        """Establish connection to broker"""
        try:
            if self.broker_type == "demo":
                return {"status": "connected", "account": "DEMO_ACCOUNT"}
            
            # Real broker connection logic here
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/account", headers=headers, timeout=10)
            
            if response.status_code == 200:
                return {"status": "connected", "account": response.json()}
            else:
                return {"status": "failed", "error": response.text}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def place_order(self, order: TradeOrder):
        """Place trading order"""
        try:
            if self.broker_type == "demo":
                # Simulate order execution
                order_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Add to simulated positions
                position = Position(
                    symbol=order.symbol,
                    volume=order.volume,
                    entry_price=order.entry_price,
                    current_price=order.entry_price,
                    pnl=0.0,
                    timestamp=datetime.now()
                )
                self.positions.append(position)
                
                return {
                    "status": "filled",
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "volume": order.volume,
                    "price": order.entry_price
                }
            
            # Real broker order placement
            order_data = {
                "symbol": order.symbol,
                "side": order.action.lower(),
                "type": order.order_type.lower(),
                "quantity": order.volume,
                "stopLoss": order.stop_loss,
                "takeProfit": order.take_profit
            }
            
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.post(
                f"{self.base_url}/orders", 
                json=order_data, 
                headers=headers,
                timeout=10
            )
            
            return response.json()
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_positions(self):
        """Get open positions"""
        if self.broker_type == "demo":
            return [
                {
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "pnl": pos.pnl,
                    "timestamp": pos.timestamp.isoformat()
                }
                for pos in self.positions
            ]
        
        # Real broker positions
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/positions", headers=headers, timeout=10)
            return response.json() if response.status_code == 200 else []
        except:
            return []
    
    def get_account_info(self):
        """Get account information"""
        if self.broker_type == "demo":
            total_pnl = sum(pos.pnl for pos in self.positions)
            return {
                "balance": self.balance + total_pnl,
                "equity": self.balance + total_pnl,
                "margin": sum(pos.volume * 1000 for pos in self.positions),
                "free_margin": self.balance - sum(pos.volume * 1000 for pos in self.positions),
                "positions_count": len(self.positions)
            }
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/account", headers=headers, timeout=10)
            return response.json() if response.status_code == 200 else {}
        except:
            return {"error": "Failed to get account info"}
    
    def close_position(self, symbol, volume=None):
        """Close position"""
        if self.broker_type == "demo":
            for i, pos in enumerate(self.positions):
                if pos.symbol == symbol:
                    if volume is None or volume >= pos.volume:
                        removed_pos = self.positions.pop(i)
                        return {"status": "closed", "symbol": symbol, "pnl": removed_pos.pnl}
            return {"status": "not_found"}
        
        # Real broker position closing
        try:
            close_data = {"symbol": symbol, "volume": volume}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.delete(
                f"{self.base_url}/positions", 
                json=close_data, 
                headers=headers,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def update_position_prices(self, current_prices: Dict[str, float]):
        """Update position prices and P&L"""
        for pos in self.positions:
            if pos.symbol in current_prices:
                pos.current_price = current_prices[pos.symbol]
                # Simple P&L calculation
                if pos.volume > 0:  # Long position
                    pos.pnl = (pos.current_price - pos.entry_price) * pos.volume * 100000
                else:  # Short position
                    pos.pnl = (pos.entry_price - pos.current_price) * abs(pos.volume) * 100000

class MT4Connector(BrokerConnector):
    """MetaTrader 4 specific connector"""
    
    def __init__(self, server_ip="localhost", port=8080):
        super().__init__("mt4")
        self.server_ip = server_ip
        self.port = port
        
    def connect(self):
        # MT4 specific connection logic
        return {"status": "connected", "platform": "MT4"}

class OandaConnector(BrokerConnector):
    """OANDA specific connector"""
    
    def __init__(self, api_key, account_id, environment="practice"):
        super().__init__("oanda", api_key, account_id)
        self.environment = environment
        self.base_url = f"https://api-fx{'practice' if environment == 'practice' else 'trade'}.oanda.com"