#!/usr/bin/env python3
"""
Market Advisor - Entry/Exit signals with multi-timeframe analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import forex_db

class MarketAdvisor:
    def __init__(self):
        self.timeframes = {
            '1mo': {'period': '3mo', 'interval': '1d'},
            '1wk': {'period': '1mo', 'interval': '1h'}, 
            '1d': {'period': '5d', 'interval': '15m'},
            '4h': {'period': '2d', 'interval': '5m'},
            '1h': {'period': '1d', 'interval': '1m'}
        }
        
    def get_multi_timeframe_data(self, symbol):
        """Get data across multiple timeframes"""
        data = {}
        ticker = yf.Ticker(symbol)
        
        for tf, params in self.timeframes.items():
            try:
                df = ticker.history(period=params['period'], interval=params['interval'])
                if not df.empty:
                    data[tf] = df
            except:
                continue
                
        return data
    
    def calculate_entry_exit_signals(self, df, timeframe):
        """Calculate entry/exit signals for given timeframe"""
        if len(df) < 50:
            return None
            
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Moving averages
        df['SMA_20'] = close.rolling(20).mean()
        df['SMA_50'] = close.rolling(50).mean()
        df['EMA_12'] = close.ewm(span=12).mean()
        df['EMA_26'] = close.ewm(span=26).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Support/Resistance levels
        df['Support'] = low.rolling(20).min()
        df['Resistance'] = high.rolling(20).max()
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Entry signals
        entry_signals = []
        exit_signals = []
        
        # BUY Entry Conditions
        buy_score = 0
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            buy_score += 2
            entry_signals.append("Price above MAs (bullish trend)")
            
        if latest['RSI'] < 40 and prev['RSI'] >= 40:
            buy_score += 2
            entry_signals.append("RSI oversold bounce")
            
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            buy_score += 3
            entry_signals.append("MACD bullish crossover")
            
        if latest['Close'] <= latest['BB_Lower'] * 1.01:
            buy_score += 2
            entry_signals.append("Price near lower Bollinger Band")
            
        if latest['Close'] > latest['Support'] * 1.02:
            buy_score += 1
            entry_signals.append("Price above support level")
        
        # SELL Entry Conditions  
        sell_score = 0
        if latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            sell_score += 2
            entry_signals.append("Price below MAs (bearish trend)")
            
        if latest['RSI'] > 60 and prev['RSI'] <= 60:
            sell_score += 2
            entry_signals.append("RSI overbought reversal")
            
        if latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            sell_score += 3
            entry_signals.append("MACD bearish crossover")
            
        if latest['Close'] >= latest['BB_Upper'] * 0.99:
            sell_score += 2
            entry_signals.append("Price near upper Bollinger Band")
            
        if latest['Close'] < latest['Resistance'] * 0.98:
            sell_score += 1
            entry_signals.append("Price below resistance level")
        
        # Exit signals (for existing positions)
        if latest['RSI'] > 70:
            exit_signals.append("RSI overbought - consider taking profits")
            
        if latest['RSI'] < 30:
            exit_signals.append("RSI oversold - consider cutting losses")
            
        if latest['Close'] < latest['SMA_20'] and prev['Close'] >= prev['SMA_20']:
            exit_signals.append("Price broke below 20 SMA - trend change")
            
        if latest['Close'] > latest['SMA_20'] and prev['Close'] <= prev['SMA_20']:
            exit_signals.append("Price broke above 20 SMA - trend change")
        
        return {
            'timeframe': timeframe,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'entry_signals': entry_signals,
            'exit_signals': exit_signals,
            'current_price': latest['Close'],
            'rsi': latest['RSI'],
            'macd': latest['MACD'],
            'support': latest['Support'],
            'resistance': latest['Resistance'],
            'bb_position': (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) if latest['BB_Upper'] != latest['BB_Lower'] else 0.5
        }
    
    def generate_market_advice(self, symbol):
        """Generate comprehensive market entry/exit advice"""
        print(f"\n🎯 MARKET ADVISOR: {symbol.replace('=X', '')}")
        print("="*60)
        
        # Get multi-timeframe data
        tf_data = self.get_multi_timeframe_data(symbol)
        
        if not tf_data:
            return {"error": "No data available"}
        
        # Analyze each timeframe
        analyses = {}
        for tf, df in tf_data.items():
            analysis = self.calculate_entry_exit_signals(df, tf)
            if analysis:
                analyses[tf] = analysis
        
        if not analyses:
            return {"error": "Analysis failed"}
        
        # Generate overall recommendation
        total_buy_score = sum(a['buy_score'] for a in analyses.values())
        total_sell_score = sum(a['sell_score'] for a in analyses.values())
        
        # Weight longer timeframes more heavily
        weights = {'1mo': 3, '1wk': 2.5, '1d': 2, '4h': 1.5, '1h': 1}
        weighted_buy = sum(analyses[tf]['buy_score'] * weights.get(tf, 1) for tf in analyses)
        weighted_sell = sum(analyses[tf]['sell_score'] * weights.get(tf, 1) for tf in analyses)
        
        # Determine action
        if weighted_buy >= 8 and weighted_buy > weighted_sell:
            action = "STRONG BUY ENTRY"
            confidence = min(0.9, 0.6 + (weighted_buy - 8) * 0.05)
        elif weighted_buy >= 5 and weighted_buy > weighted_sell:
            action = "BUY ENTRY"
            confidence = min(0.8, 0.5 + (weighted_buy - 5) * 0.1)
        elif weighted_sell >= 8 and weighted_sell > weighted_buy:
            action = "STRONG SELL ENTRY"
            confidence = min(0.9, 0.6 + (weighted_sell - 8) * 0.05)
        elif weighted_sell >= 5 and weighted_sell > weighted_buy:
            action = "SELL ENTRY"
            confidence = min(0.8, 0.5 + (weighted_sell - 5) * 0.1)
        else:
            action = "WAIT / NO ENTRY"
            confidence = 0.3
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'timeframe_analyses': analyses,
            'weighted_buy_score': weighted_buy,
            'weighted_sell_score': weighted_sell,
            'timestamp': datetime.now()
        }
    
    def print_market_advice(self, advice):
        """Print detailed market advice"""
        if "error" in advice:
            print(f"❌ {advice['error']}")
            return
        
        action = advice['action']
        conf = advice['confidence']
        
        # Main recommendation
        if "BUY" in action:
            emoji = "🟢"
            color = "GREEN"
        elif "SELL" in action:
            emoji = "🔴" 
            color = "RED"
        else:
            emoji = "🟡"
            color = "YELLOW"
        
        print(f"\n{emoji} RECOMMENDATION: {action}")
        print(f"🎯 CONFIDENCE: {conf:.1%}")
        print(f"📊 Buy Score: {advice['weighted_buy_score']:.1f}")
        print(f"📊 Sell Score: {advice['weighted_sell_score']:.1f}")
        
        # Timeframe analysis
        print(f"\n📈 TIMEFRAME ANALYSIS:")
        print("-" * 50)
        
        for tf, analysis in advice['timeframe_analyses'].items():
            print(f"\n⏰ {tf.upper()} Timeframe:")
            print(f"   💰 Price: {analysis['current_price']:.5f}")
            print(f"   📊 RSI: {analysis['rsi']:.1f}")
            print(f"   🎯 Buy Score: {analysis['buy_score']} | Sell Score: {analysis['sell_score']}")
            
            if analysis['entry_signals']:
                print(f"   🚀 Entry Signals:")
                for signal in analysis['entry_signals'][:3]:
                    print(f"      • {signal}")
            
            if analysis['exit_signals']:
                print(f"   🚪 Exit Signals:")
                for signal in analysis['exit_signals'][:2]:
                    print(f"      • {signal}")
        
        # Trading advice
        print(f"\n💡 TRADING ADVICE:")
        print("-" * 30)
        
        if "BUY" in action:
            current_price = list(advice['timeframe_analyses'].values())[0]['current_price']
            stop_loss = current_price * 0.98
            take_profit = current_price * 1.04
            
            print(f"   🎯 Entry: Market price (~{current_price:.5f})")
            print(f"   🛑 Stop Loss: {stop_loss:.5f} (-2%)")
            print(f"   💰 Take Profit: {take_profit:.5f} (+4%)")
            print(f"   📏 Risk/Reward: 1:2")
            
        elif "SELL" in action:
            current_price = list(advice['timeframe_analyses'].values())[0]['current_price']
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.96
            
            print(f"   🎯 Entry: Market price (~{current_price:.5f})")
            print(f"   🛑 Stop Loss: {stop_loss:.5f} (+2%)")
            print(f"   💰 Take Profit: {take_profit:.5f} (-4%)")
            print(f"   📏 Risk/Reward: 1:2")
            
        else:
            print(f"   ⏳ Wait for clearer signals")
            print(f"   📊 Monitor key levels and timeframe alignment")
            print(f"   🔍 Look for breakouts or trend confirmations")
        
        print(f"\n⚠️  Risk Management:")
        print(f"   • Never risk more than 2% of account per trade")
        print(f"   • Use proper position sizing")
        print(f"   • Follow your trading plan")

def main():
    """Main market advisor function"""
    advisor = MarketAdvisor()
    
    pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
    
    for symbol in pairs:
        advice = advisor.generate_market_advice(symbol)
        advisor.print_market_advice(advice)
        print("\n" + "="*80)

if __name__ == "__main__":
    main()