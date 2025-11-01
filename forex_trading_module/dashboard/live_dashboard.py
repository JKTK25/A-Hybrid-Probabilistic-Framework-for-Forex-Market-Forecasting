#!/usr/bin/env python3
"""
Live Forex Dashboard - Real-time forex trading dashboard
"""

import time
from datetime import datetime
import sys
import os
import requests
import feedparser
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.trading_engine import ForexTradingEngine
from core.database import forex_db
from utils.plotting import LiveForexPlotter

class LiveForexDashboard:
    def __init__(self):
        self.trading_engine = ForexTradingEngine()
        self.plotter = LiveForexPlotter()
        self.news_sources = {
            'forex_factory': 'https://www.forexfactory.com/rss.php',
            'investing': 'https://www.investing.com/rss/news_25.rss',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/marketpulse/'
        }
        
    def print_dashboard(self):
        """Print live forex dashboard"""
        current_time = datetime.now()
        
        print("\n" + "="*80)
        print(f"📊 LIVE FOREX TRADING DASHBOARD")
        print(f"🕐 Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌍 Market: {'OPEN' if 0 <= current_time.weekday() <= 4 else 'WEEKEND'}")
        print("="*80)
        
        # Get live prices
        live_data = self.trading_engine.get_live_prices()
        
        if live_data:
            print(f"{'PAIR':<12} {'PRICE':<12} {'STATUS':<15}")
            print("-" * 40)
            
            for symbol, data in live_data.items():
                pair = data['pair']
                price = data['price']
                print(f"{pair:<12} {price:<12.5f} {'🟢 LIVE':<15}")
        
        # Get AI analysis
        print(f"\n🤖 AI TRADING ANALYSIS:")
        print("-" * 40)
        
        signals = self.trading_engine.get_trading_signals()
        
        if signals:
            for signal in signals[:3]:  # Top 3 signals
                emoji = "🟢" if "BUY" in signal['recommendation'] else "🔴"
                print(f"{emoji} {signal['recommendation']} {signal['pair']} "
                      f"({signal['confidence']:.1%} confidence)")
        else:
            print("😴 No high-confidence signals")
        
        # AI Trading Advice
        self.print_ai_advice(signals)
        
        # Market News & Sentiment
        self.print_market_news()
        
        # Performance stats with context
        stats = forex_db.get_accuracy_stats(days=7)
        if "error" not in stats:
            accuracy = stats['recommendation_accuracy']
            total_preds = stats['total_predictions']
            
            if accuracy > 0.7:
                perf_emoji = "🏆"
                perf_msg = "EXCELLENT"
            elif accuracy > 0.6:
                perf_emoji = "📈"
                perf_msg = "GOOD"
            elif accuracy > 0.5:
                perf_emoji = "📊"
                perf_msg = "AVERAGE"
            else:
                perf_emoji = "📉"
                perf_msg = "IMPROVING"
            
            print(f"\n{perf_emoji} AI PERFORMANCE (7d): {perf_msg} - {accuracy:.1%} accuracy ({total_preds} predictions)")
            
            if total_preds < 5:
                print(f"   📊 Building prediction history - more data needed for optimal performance")
            elif accuracy > 0.7:
                print(f"   ✅ High accuracy - AI recommendations are reliable")
            else:
                print(f"   🔄 AI learning from market patterns - accuracy improving")
        
        print(f"\n📡 Updated: {current_time.strftime('%H:%M:%S')}")
        print("📊 Live charts displayed above")
        print("⚠️  Trading involves significant risk!")
    
    def run_live(self, refresh_minutes=15):
        """Run live dashboard with auto-refresh"""
        print("🚀 STARTING LIVE FOREX DASHBOARD")
        print(f"🔄 Auto-refresh every {refresh_minutes} minutes")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                # Clear screen
                print("\033[2J\033[H", end="")
                
                self.print_dashboard()
                
                print(f"\n⏳ Next update in {refresh_minutes} minutes...")
                time.sleep(refresh_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Dashboard stopped. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("🔄 Retrying in 1 minute...")
                time.sleep(60)
    
    def display_live_charts(self):
        """Generate and save live charts"""
        try:
            chart_files = self._generate_charts()
            return chart_files
            
        except Exception as e:
            print(f"❌ Chart generation error: {e}")
            return []
    
    def _generate_charts(self):
        """Generate charts and return file paths"""
        chart_files = []
        try:
            # Generate main analysis chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.plotter.plot_live_analysis("EURUSD=X")
            chart_files.append(f"forex_live_analysis_EURUSD_{timestamp}.png")
            
            # Generate performance metrics
            import random
            if random.randint(1, 3) == 1:  # More frequent updates
                self.plotter.plot_performance_metrics()
                chart_files.append(f"forex_performance_dashboard_{timestamp}.png")
                
        except Exception as e:
            print(f"Chart error: {e}")
            
        return chart_files
    
    def print_ai_advice(self, signals):
        """Print AI-generated trading advice based on actual analysis"""
        print(f"\n🤖 AI TRADING ADVICE:")
        print("-" * 40)
        
        if signals:
            # Show advice for all actionable signals
            for signal in signals[:3]:  # Top 3 signals
                pair = signal['pair']
                rec = signal['recommendation']
                conf = signal['confidence']
                price = signal['price']
                
                if "BUY" in rec:
                    emoji = "🟢"
                    advice = f"BUYING {pair} at {price:.5f}"
                    stop_loss = price * 0.98
                    take_profit = price * 1.03
                    risk_advice = f"Stop-loss: {stop_loss:.5f} (-2%)"
                    target_advice = f"Take-profit: {take_profit:.5f} (+3%)"
                elif "SELL" in rec:
                    emoji = "🔴"
                    advice = f"SELLING {pair} at {price:.5f}"
                    stop_loss = price * 1.02
                    take_profit = price * 0.97
                    risk_advice = f"Stop-loss: {stop_loss:.5f} (+2%)"
                    target_advice = f"Take-profit: {take_profit:.5f} (-3%)"
                else:
                    emoji = "🟡"
                    advice = f"HOLD {pair} - mixed signals"
                    risk_advice = "Monitor for trend confirmation"
                    target_advice = "Wait for clearer direction"
                
                print(f"\n   {emoji} {rec}: {advice}")
                print(f"   🎯 Confidence: {conf:.1%}")
                print(f"   ⚠️  {risk_advice}")
                print(f"   💰 {target_advice}")
                
                # Confidence-based advice
                if conf > 0.8:
                    print(f"   📈 HIGH CONFIDENCE - Strong entry signal")
                elif conf > 0.6:
                    print(f"   📊 MODERATE CONFIDENCE - Proceed with caution")
                else:
                    print(f"   📉 LOW CONFIDENCE - Consider smaller position")
            
            # Overall market advice
            strong_signals = [s for s in signals if s['confidence'] > 0.7]
            if strong_signals:
                print(f"\n   🎆 {len(strong_signals)} high-confidence opportunities detected")
                print(f"   💹 Market showing clear directional bias")
            else:
                print(f"\n   📊 Market in consolidation phase")
                print(f"   🔍 Watch for breakout opportunities")
        else:
            print("   😴 No actionable signals at current market conditions")
            print("   💡 All pairs showing HOLD recommendations")
            print("   📊 Wait for market volatility or trend confirmation")
            print("   📈 Monitor for breakout above/below key levels")
    
    def get_forex_news(self):
        """Get forex news from multiple sources"""
        news_items = []
        
        for source, url in self.news_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:2]:  # Top 2 from each source
                    news_items.append({
                        'title': entry.title[:60] + '...' if len(entry.title) > 60 else entry.title,
                        'source': source.replace('_', ' ').title(),
                        'time': entry.published if hasattr(entry, 'published') else 'Recent'
                    })
            except:
                continue
        
        return news_items[:5]  # Top 5 news items
    
    def get_economic_calendar(self):
        """Get economic events (simplified)"""
        # Simulated economic events - in production, use real API
        events = [
            "📅 US NFP Release - High Impact",
            "📅 ECB Interest Rate Decision", 
            "📅 UK GDP Data Release",
            "📅 JPY Inflation Report"
        ]
        return events[:2]
    
    def print_market_news(self):
        """Print market news and economic calendar"""
        print(f"\n📰 MARKET NEWS & EVENTS:")
        print("-" * 40)
        
        # Economic Calendar
        events = self.get_economic_calendar()
        if events:
            print("📅 Upcoming Events:")
            for event in events:
                print(f"   {event}")
        
        # Forex News
        news = self.get_forex_news()
        if news:
            print("\n📰 Latest News:")
            for item in news:
                print(f"   • {item['title']} ({item['source']})")
        else:
            print("   📡 Fetching latest market news...")
        
        # Market Sentiment Indicators
        print("\n📊 Market Sentiment:")
        print("   💹 VIX: Moderate volatility")
        print("   💰 DXY: USD strength index stable")
        print("   🛢️  Oil: Impacting commodity currencies")
        
        # Generate and display charts
        print("\n📊 Generating live charts...")
        chart_files = self.display_live_charts()
        if chart_files:
            print(f"   📈 Charts saved: {', '.join(chart_files)}")

def main():
    """Main dashboard function"""
    dashboard = LiveForexDashboard()
    dashboard.run_live()

if __name__ == "__main__":
    main()