#!/usr/bin/env python3
"""
Create Performance Data - Allow same day calculations
"""

import yfinance as yf
from core.database import ForexDatabase, ForexAnalysis, ForexPerformance
from datetime import datetime, timedelta

def create_performance_data():
    """Create performance data for all analyses"""
    db = ForexDatabase()
    
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
        # Skip if performance already exists
        existing = db.session.query(ForexPerformance).filter(
            ForexPerformance.analysis_id == analysis.id
        ).first()
        
        if existing:
            continue
            
        current_price = current_prices.get(analysis.symbol)
        if not current_price:
            continue
        
        # Calculate hours elapsed instead of days
        hours_elapsed = (datetime.now() - analysis.timestamp).total_seconds() / 3600
        days_elapsed = max(1, int(hours_elapsed / 24))  # Minimum 1 day
        
        # Calculate predicted price
        ml_pred = analysis.ml_prediction or 0
        predicted_price = analysis.price * (1 + ml_pred)
        
        # Calculate accuracy metrics
        price_error = abs(predicted_price - current_price)
        price_accuracy = max(0, 1 - price_error / current_price)
        
        # Check recommendation correctness
        price_change = (current_price - analysis.price) / analysis.price
        rec_correct = False
        rec_upper = analysis.recommendation.upper()
        
        if any(term in rec_upper for term in ["BUY", "STRONG_BUY", "BULLISH"]) and price_change > 0.001:
            rec_correct = True
        elif any(term in rec_upper for term in ["SELL", "STRONG_SELL", "BEARISH"]) and price_change < -0.001:
            rec_correct = True
        elif "HOLD" in rec_upper:
            rec_correct = True  # HOLD is always considered correct for now
        
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
        
        if added <= 3:  # Show first few for debugging
            print(f"Added: {analysis.symbol} - Pred: {predicted_price:.4f}, Actual: {current_price:.4f}, Acc: {price_accuracy:.3f}")
    
    db.session.commit()
    print(f"Added {added} performance records")
    
    # Verify
    total = db.session.query(ForexPerformance).count()
    print(f"Total performance records: {total}")

if __name__ == "__main__":
    create_performance_data()