#!/usr/bin/env python3
"""
Fix Performance Data - Direct performance calculation
"""

import yfinance as yf
from core.database import ForexDatabase, ForexAnalysis, ForexPerformance
from datetime import datetime, timedelta

def fix_performance_data():
    """Fix performance data calculation"""
    db = ForexDatabase()
    
    # Get all analyses
    analyses = db.session.query(ForexAnalysis).all()
    print(f"Processing {len(analyses)} analyses...")
    
    # Get current prices
    symbols = list(set(a.symbol for a in analyses))
    current_prices = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                current_prices[symbol] = data['Close'].iloc[-1]
        except:
            pass
    
    added = 0
    for analysis in analyses:
        # Check if performance already exists
        existing = db.session.query(ForexPerformance).filter(
            ForexPerformance.analysis_id == analysis.id
        ).first()
        
        if existing:
            continue
            
        current_price = current_prices.get(analysis.symbol)
        if not current_price:
            continue
            
        days_elapsed = (datetime.now() - analysis.timestamp).days
        if days_elapsed < 1:
            continue
        
        # Simple predicted price (use ml_prediction if available, else use current price)
        ml_pred = analysis.ml_prediction or 0
        predicted_price = analysis.price * (1 + ml_pred)
        
        # Calculate accuracy
        price_error = abs(predicted_price - current_price)
        price_accuracy = max(0, 1 - price_error / current_price)
        
        # Check recommendation correctness
        price_change = (current_price - analysis.price) / analysis.price
        rec_correct = False
        
        if "BUY" in analysis.recommendation.upper() and price_change > 0:
            rec_correct = True
        elif "SELL" in analysis.recommendation.upper() and price_change < 0:
            rec_correct = True
        elif "HOLD" in analysis.recommendation.upper() and abs(price_change) < 0.01:
            rec_correct = True
        
        # Create performance record
        performance = ForexPerformance(
            analysis_id=analysis.id,
            symbol=analysis.symbol,
            predicted_price=predicted_price,
            actual_price=current_price,
            prediction_accuracy=price_accuracy,
            recommendation_correct=rec_correct,
            days_elapsed=days_elapsed
        )
        
        db.session.add(performance)
        added += 1
    
    db.session.commit()
    print(f"Added {added} performance records")
    
    # Check results
    total = db.session.query(ForexPerformance).count()
    print(f"Total performance records: {total}")

if __name__ == "__main__":
    fix_performance_data()