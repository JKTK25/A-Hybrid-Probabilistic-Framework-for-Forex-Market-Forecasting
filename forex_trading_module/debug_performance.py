#!/usr/bin/env python3
"""
Debug Performance Data
"""

import yfinance as yf
from core.database import ForexDatabase, ForexAnalysis, ForexPerformance
from datetime import datetime, timedelta

def debug_performance():
    """Debug performance data issues"""
    db = ForexDatabase()
    
    # Get sample analysis
    analysis = db.session.query(ForexAnalysis).first()
    if not analysis:
        print("No analysis data found")
        return
    
    print(f"Sample Analysis:")
    print(f"  ID: {analysis.id}")
    print(f"  Symbol: {analysis.symbol}")
    print(f"  Timestamp: {analysis.timestamp}")
    print(f"  Price: {analysis.price}")
    print(f"  ML Prediction: {analysis.ml_prediction}")
    print(f"  Recommendation: {analysis.recommendation}")
    
    # Check if performance exists
    existing = db.session.query(ForexPerformance).filter(
        ForexPerformance.analysis_id == analysis.id
    ).first()
    print(f"  Existing Performance: {existing}")
    
    # Get current price
    try:
        ticker = yf.Ticker(analysis.symbol)
        data = ticker.history(period="1d")
        current_price = data['Close'].iloc[-1] if not data.empty else None
        print(f"  Current Price: {current_price}")
    except Exception as e:
        print(f"  Price Error: {e}")
        current_price = None
    
    # Calculate days elapsed
    days_elapsed = (datetime.now() - analysis.timestamp).days
    print(f"  Days Elapsed: {days_elapsed}")
    
    if current_price and days_elapsed >= 1:
        # Force create performance record
        ml_pred = analysis.ml_prediction or 0
        predicted_price = analysis.price * (1 + ml_pred)
        price_error = abs(predicted_price - current_price)
        price_accuracy = max(0, 1 - price_error / current_price)
        
        price_change = (current_price - analysis.price) / analysis.price
        rec_correct = "HOLD" in analysis.recommendation.upper()
        
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
        db.session.commit()
        print("  Performance record created!")
        
        # Verify
        count = db.session.query(ForexPerformance).count()
        print(f"  Total performance records: {count}")

if __name__ == "__main__":
    debug_performance()