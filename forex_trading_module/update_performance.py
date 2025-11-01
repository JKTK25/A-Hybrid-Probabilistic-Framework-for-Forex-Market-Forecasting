#!/usr/bin/env python3
"""
Update Performance Data - Populate performance metrics from existing analysis
"""

import yfinance as yf
from core.database import ForexDatabase, ForexAnalysis
from datetime import datetime, timedelta

def update_performance_data():
    """Update performance data for existing analyses"""
    db = ForexDatabase()
    
    # Get all analyses from last 30 days
    cutoff = datetime.now() - timedelta(days=30)
    analyses = db.session.query(ForexAnalysis).filter(
        ForexAnalysis.timestamp >= cutoff
    ).all()
    
    print(f"Found {len(analyses)} analyses to process...")
    
    # Get current prices for all symbols
    symbols = list(set(a.symbol for a in analyses))
    current_prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                current_prices[symbol] = data['Close'].iloc[-1]
                print(f"Got price for {symbol}: {current_prices[symbol]:.4f}")
        except:
            print(f"Failed to get price for {symbol}")
    
    # Update performance for each analysis
    db.update_performance(current_prices)
    print("Performance data updated!")
    
    # Show updated stats
    stats = db.get_accuracy_stats(days=30)
    print(f"\nUpdated Stats: {stats}")

if __name__ == "__main__":
    update_performance_data()