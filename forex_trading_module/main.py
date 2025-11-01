#!/usr/bin/env python3
"""
Forex Trading Module - Main entry point
"""

import sys
from core.trading_engine import ForexTradingEngine
from core.performance import PerformanceTracker
from dashboard.live_dashboard import LiveForexDashboard
from utils.plotting import LiveForexPlotter

def main():
    """Main forex trading module entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "dashboard":
            # Run live dashboard
            dashboard = LiveForexDashboard()
            dashboard.run_live()
            
        elif command == "analyze":
            # Run single analysis
            engine = ForexTradingEngine()
            engine.run_analysis()
            
        elif command == "performance":
            # Show performance report
            tracker = PerformanceTracker()
            tracker.generate_report()
            tracker.plot_performance()
            tracker.get_best_performing_pairs()
            
        elif command == "charts":
            # Generate live charts
            plotter = LiveForexPlotter()
            plotter.plot_live_analysis("EURUSD=X")
            plotter.plot_performance_metrics()
            plotter.plot_prediction_accuracy("EURUSD=X")
            
        elif command == "advisor":
            # Market entry/exit advisor
            from core.market_advisor import MarketAdvisor
            advisor = MarketAdvisor()
            pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
            for symbol in pairs:
                advice = advisor.generate_market_advice(symbol)
                advisor.print_market_advice(advice)
                print("\n" + "="*80)
                
        elif command == "database":
            # Database analyzer
            from database_analyzer import main as analyzer_main
            analyzer_main()
            
        elif command == "web":
            # Web interface
            from web.app import app
            print("🌐 Starting web interface at http://localhost:5000")
            app.run(debug=True, host='0.0.0.0', port=5000)
            
        elif command == "auto":
            # Auto trading
            from core.auto_trader import get_auto_trader
            trader = get_auto_trader()
            
            if len(sys.argv) > 2 and sys.argv[2] == "start":
                trader.start()
                print("🤖 Auto trading started")
                input("Press Enter to stop...")
                trader.stop()
            else:
                status = trader.get_status()
                print(f"Auto Trader Status: {status}")
                
        elif command == "risk":
            # Risk management report
            from core.risk_manager import RiskManager
            rm = RiskManager()
            report = rm.get_risk_report()
            print("📊 RISK MANAGEMENT REPORT")
            print("="*40)
            for key, value in report.items():
                print(f"{key}: {value}")
            
        else:
            print("❌ Unknown command. Use: dashboard, analyze, or performance")
    
    else:
        # Default: run analysis
        print("🚀 FOREX TRADING MODULE")
        print("="*30)
        print("Commands:")
        print("  python main.py dashboard    - Live dashboard")
        print("  python main.py analyze      - Single analysis")
        print("  python main.py performance  - Performance report")
        print("  python main.py charts       - Generate live charts")
        print("  python main.py advisor      - Market entry/exit advisor")
        print("  python main.py database     - Database analyzer")
        print("  python main.py web          - Web interface")
        print("  python main.py auto start   - Start auto trading")
        print("  python main.py auto         - Auto trading status")
        print("  python main.py risk         - Risk management report")
        print()
        
        # Run default analysis
        engine = ForexTradingEngine()
        engine.run_analysis()

if __name__ == "__main__":
    main()