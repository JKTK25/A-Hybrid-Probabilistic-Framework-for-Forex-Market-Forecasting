#!/usr/bin/env python3
"""
Web Interface - User experience dashboard
"""

from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

app = Flask(__name__)

# Simple data fetcher
def get_forex_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
    except:
        pass
    return 0

@app.route('/')
def dashboard():
    """Main dashboard"""
    return """
    <html>
    <head><title>Forex Trading Dashboard</title></head>
    <body>
        <h1>🚀 Forex Trading Dashboard</h1>
        <div id="live-data"></div>
        <script>
            setInterval(() => {
                fetch('/api/live_data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('live-data').innerHTML = 
                    '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                });
            }, 5000);
        </script>
    </body>
    </html>
    """

@app.route('/api/live_data')
def get_live_data():
    """Get live forex data"""
    pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    data = []
    
    for pair in pairs:
        price = get_forex_price(pair)
        # Simple recommendation logic
        if price > 0:
            recommendation = "HOLD"
            confidence = 0.65
        else:
            recommendation = "ERROR"
            confidence = 0
            
        data.append({
            "symbol": pair.replace("=X", ""),
            "price": round(price, 5) if price > 0 else 0,
            "recommendation": recommendation,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
    
    return jsonify(data)

@app.route('/api/account')
def get_account():
    """Get account information"""
    return jsonify({
        "balance": 10000.0,
        "equity": 10000.0,
        "margin": 0,
        "free_margin": 10000.0,
        "positions_count": 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)