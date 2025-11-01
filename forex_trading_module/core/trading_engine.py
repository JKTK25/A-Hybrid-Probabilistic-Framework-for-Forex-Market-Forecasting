#!/usr/bin/env python3
"""
Forex Trading Engine - Main orchestrator for forex trading operations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.ai_agent import AITradingAgent
from core.database import forex_db
from core.performance import PerformanceTracker
import yfinance as yf
from datetime import datetime

class ForexTradingEngine:
    def __init__(self):
        self.ai_agent = AITradingAgent()
        self.performance_tracker = PerformanceTracker()
        self.major_pairs = {
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X", 
            "USD/JPY": "USDJPY=X",
            "AUD/USD": "AUDUSD=X",
            "USD/CHF": "USDCHF=X",
            "USD/CAD": "USDCAD=X"
        }
    
    def get_live_prices(self):
        """Get current forex prices"""
        live_data = {}
        
        for pair_name, symbol in self.major_pairs.items():
            try:
                ticker = yf.Ticker(symbol)
                current = ticker.history(period="1d", interval="1m")
                
                if not current.empty:
                    price = current['Close'].iloc[-1]
                    timestamp = current.index[-1]
                    
                    live_data[symbol] = {
                        'pair': pair_name,
                        'price': price,
                        'timestamp': timestamp,
                        'symbol': symbol
                    }
            except Exception as e:
                print(f"Error getting price for {pair_name}: {e}")
        
        return live_data
    
    def analyze_all_pairs(self):
        """Analyze all major forex pairs"""
        print("🚀 FOREX TRADING ENGINE")
        print("="*50)
        
        results = {}
        live_prices = self.get_live_prices()
        
        # Update performance metrics
        current_prices = {symbol: data['price'] for symbol, data in live_prices.items()}
        forex_db.update_performance(current_prices)
        
        # Analyze each pair
        for pair_name, symbol in self.major_pairs.items():
            print(f"\n🔍 Analyzing {pair_name}...")
            result = self.ai_agent.analyze_pair(symbol)
            
            if "error" not in result:
                results[symbol] = result
                self.ai_agent.print_analysis(result)
            else:
                print(f"❌ {result['error']}")
        
        return results
    
    def get_trading_signals(self):
        """Get actionable trading signals"""
        results = self.analyze_all_pairs()
        
        signals = []
        for symbol, result in results.items():
            if result['recommendation'] != 'HOLD' and result['confidence'] > 0.6:
                signals.append({
                    'symbol': symbol,
                    'pair': symbol.replace('=X', '').replace('USD', '/USD'),
                    'recommendation': result['recommendation'],
                    'confidence': result['confidence'],
                    'price': result['price']
                })
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
    
    def print_trading_summary(self):
        """Print comprehensive trading summary"""
        signals = self.get_trading_signals()
        
        print(f"\n🎯 TRADING SIGNALS SUMMARY")
        print("="*50)
        
        if signals:
            for i, signal in enumerate(signals, 1):
                emoji = "🟢" if "BUY" in signal['recommendation'] else "🔴"
                print(f"{i}. {emoji} {signal['recommendation']} {signal['pair']}")
                print(f"   Price: {signal['price']:.5f} | Confidence: {signal['confidence']:.1%}")
        else:
            print("😴 No high-confidence signals at this time")
        
        # Show performance stats
        stats = forex_db.get_accuracy_stats(days=7)
        if "error" not in stats:
            print(f"\n🏆 ENGINE PERFORMANCE (7 days):")
            print(f"   Accuracy: {stats['recommendation_accuracy']:.1%}")
            print(f"   Predictions: {stats['total_predictions']}")
            print(f"   Price Accuracy: {stats['price_prediction_accuracy']:.1%}")
        
        print(f"\n⚠️  Risk Warning: Trading involves significant risk!")
        
        return signals
    
    def run_analysis(self):
        """Run complete forex analysis"""
        return self.print_trading_summary()