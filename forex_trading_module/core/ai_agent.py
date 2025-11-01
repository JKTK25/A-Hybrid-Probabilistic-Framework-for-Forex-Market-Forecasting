#!/usr/bin/env python3
"""
AI Trading Agent - Advanced forex analysis with ML and probability
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import forex_db
from core.market_advisor import MarketAdvisor

class AITradingAgent:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.market_advisor = MarketAdvisor()
        
    def calculate_indicators(self, df):
        """Calculate technical indicators with probabilities"""
        close = df['Close']
        
        # RSI with probability zones
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Buy_Prob'] = np.where(df['RSI'] < 30, 0.8, 0.2)
        df['RSI_Sell_Prob'] = np.where(df['RSI'] > 70, 0.8, 0.2)
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility
        df['Returns'] = close.pct_change()
        df['Volatility'] = df['Returns'].rolling(20).std()
        
        return df
    
    def monte_carlo_simulation(self, current_price, volatility, days=5, simulations=1000):
        """Monte Carlo price simulation"""
        results = []
        for _ in range(simulations):
            price = current_price
            for day in range(days):
                drift = random.gauss(0, volatility)
                price *= (1 + drift)
            results.append(price)
        
        results = np.array(results)
        return {
            'prob_up': np.sum(results > current_price) / simulations,
            'prob_down': np.sum(results < current_price) / simulations,
            'mean_price': np.mean(results)
        }
    
    def analyze_pair(self, symbol):
        """Complete AI analysis of forex pair with historical consistency"""
        try:
            # Get historical consistency data
            historical_data = self.get_historical_consistency(symbol)
            
            # Check previous analysis
            prev_analysis = forex_db.get_previous_analysis(symbol, hours_back=6)
            if prev_analysis:
                print(f"   📚 Previous: {prev_analysis.recommendation} ({prev_analysis.confidence:.1%})")
            print(f"   📊 Historical accuracy: {historical_data['accuracy']:.1%} (MAE: {historical_data['price_mae']:.3f})")
            print(f"   🎯 Directional accuracy: {historical_data['directional_accuracy']:.1%}")
            
            # Show quick market advice preview
            market_advice = self.market_advisor.generate_market_advice(symbol)
            if 'error' not in market_advice:
                print(f"   🎯 Market advice: {market_advice['action']} ({market_advice['confidence']:.1%})")
            
            # Get data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
            current = ticker.history(period="1d", interval="5m")
            
            if hist.empty:
                return {"error": "No data available"}
            
            # Calculate indicators
            hist = self.calculate_indicators(hist)
            latest = hist.iloc[-1]
            current_price = current['Close'].iloc[-1] if not current.empty else latest['Close']
            
            # ML prediction
            ml_prediction = 0
            if len(hist) > 50:
                X = hist[['RSI', 'MACD', 'BB_Position', 'Volatility']].fillna(0)
                y = hist['Returns'].shift(-1).fillna(0)
                if len(X) > 20:
                    X_scaled = self.scaler.fit_transform(X[:-1])
                    self.model.fit(X_scaled, y[:-1])
                    X_latest = self.scaler.transform(X.iloc[-1:])
                    ml_prediction = self.model.predict(X_latest)[0]
            
            # Monte Carlo simulation
            volatility = latest['Volatility'] if latest['Volatility'] > 0 else 0.01
            mc_results = self.monte_carlo_simulation(current_price, volatility)
            
            # Advanced probability combination with error-adjusted weighting
            accuracy_weight = historical_data['weighted_accuracy']
            directional_weight = historical_data['directional_accuracy']
            price_error_penalty = historical_data['price_mae']
            
            # Adaptive weighting based on historical performance
            technical_weight = 0.2 + (accuracy_weight - 0.5) * 0.1
            ml_weight = 0.3 + (directional_weight - 0.5) * 0.2
            mc_weight = 0.25 - price_error_penalty * 0.1
            historical_weight = 0.25 + (historical_data['consistency_score'] - 0.5) * 0.1
            
            # Normalize weights
            total_weight = technical_weight + ml_weight + mc_weight + historical_weight
            technical_weight /= total_weight
            ml_weight /= total_weight
            mc_weight /= total_weight
            historical_weight /= total_weight
            
            buy_prob = (latest['RSI_Buy_Prob'] * technical_weight + 
                       mc_results['prob_up'] * mc_weight + 
                       (0.8 if ml_prediction > 0.01 else 0.2) * ml_weight +
                       historical_data['buy_success_rate'] * historical_weight)
            
            sell_prob = (latest['RSI_Sell_Prob'] * technical_weight + 
                        mc_results['prob_down'] * mc_weight + 
                        (0.8 if ml_prediction < -0.01 else 0.2) * ml_weight +
                        historical_data['sell_success_rate'] * historical_weight)
            
            # Advanced recommendation with error-corrected confidence
            consistency_factor = historical_data['consistency_score']
            confidence_calibration = 1 - historical_data['confidence_mae']
            
            # Dynamic thresholds based on historical accuracy
            buy_threshold = 0.6 - (accuracy_weight - 0.5) * 0.1
            sell_threshold = 0.6 - (accuracy_weight - 0.5) * 0.1
            strong_threshold = 0.75 - (accuracy_weight - 0.5) * 0.05
            
            if buy_prob > buy_threshold:
                recommendation = "STRONG BUY" if buy_prob > strong_threshold else "BUY"
                base_confidence = buy_prob
            elif sell_prob > sell_threshold:
                recommendation = "STRONG SELL" if sell_prob > strong_threshold else "SELL"
                base_confidence = sell_prob
            else:
                recommendation = "HOLD"
                base_confidence = max(1 - buy_prob - sell_prob, 0.3)
            
            # Error-corrected confidence calculation
            confidence_adjustment = consistency_factor * confidence_calibration
            confidence = min(0.95, base_confidence * (0.7 + confidence_adjustment * 0.3))
            
            # Get market entry/exit advice
            market_advice = self.market_advisor.generate_market_advice(symbol)
            
            result = {
                'symbol': symbol,
                'price': current_price,
                'recommendation': recommendation,
                'confidence': confidence,
                'probabilities': {
                    'buy_probability': buy_prob,
                    'sell_probability': sell_prob,
                    'hold_probability': 1 - buy_prob - sell_prob
                },
                'analysis': {
                    'ml_prediction': ml_prediction,
                    'monte_carlo': mc_results,
                    'rsi': latest['RSI'],
                    'bb_position': latest['BB_Position'],
                    'historical_consistency': historical_data
                },
                'market_advice': market_advice,
                'risk_metrics': {
                    'volatility': volatility,
                    'expected_return': ml_prediction
                },
                'timestamp': datetime.now()
            }
            
            # Save to database
            analysis_id = forex_db.save_analysis(result)
            if analysis_id:
                print(f"   💾 Saved to database (ID: {analysis_id})")
            
            return result
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_historical_consistency(self, symbol):
        """Analyze historical performance with advanced error calculation"""
        try:
            from core.database import ForexAnalysis, ForexPerformance
            
            # Get last 60 days of analysis for better accuracy
            cutoff = datetime.now() - timedelta(days=60)
            analyses = forex_db.session.query(ForexAnalysis).filter(
                ForexAnalysis.symbol == symbol,
                ForexAnalysis.timestamp >= cutoff
            ).order_by(ForexAnalysis.timestamp).all()
            
            if len(analyses) < 10:
                return self._default_consistency_data(len(analyses))
            
            # Get performance data with error calculations
            performances = []
            price_errors = []
            confidence_errors = []
            
            for analysis in analyses:
                perf = forex_db.session.query(ForexPerformance).filter(
                    ForexPerformance.analysis_id == analysis.id
                ).first()
                
                if perf:
                    performances.append((analysis, perf))
                    
                    # Calculate price prediction error
                    price_error = abs(perf.predicted_price - perf.actual_price) / perf.actual_price
                    price_errors.append(price_error)
                    
                    # Calculate confidence calibration error
                    actual_success = 1.0 if perf.recommendation_correct else 0.0
                    confidence_error = abs(analysis.confidence - actual_success)
                    confidence_errors.append(confidence_error)
            
            if not performances:
                return self._default_consistency_data(len(analyses))
            
            # Advanced accuracy metrics
            total_correct = sum(1 for _, p in performances if p.recommendation_correct)
            accuracy = total_correct / len(performances)
            
            # Calculate Mean Absolute Error (MAE) for prices
            mae_price = np.mean(price_errors) if price_errors else 0.1
            
            # Calculate confidence calibration error
            mae_confidence = np.mean(confidence_errors) if confidence_errors else 0.3
            
            # Time-weighted accuracy (recent predictions matter more)
            time_weights = np.exp(np.linspace(-1, 0, len(performances)))
            weighted_accuracy = sum(w * (1 if p.recommendation_correct else 0) 
                                  for w, (_, p) in zip(time_weights, performances)) / sum(time_weights)
            
            # Directional accuracy (did we predict direction correctly?)
            directional_correct = 0
            directional_total = 0
            
            for analysis, perf in performances:
                if perf.days_elapsed > 0:
                    predicted_direction = 1 if analysis.ml_prediction > 0 else -1
                    actual_direction = 1 if perf.actual_price > analysis.price else -1
                    
                    if predicted_direction == actual_direction:
                        directional_correct += 1
                    directional_total += 1
            
            directional_accuracy = directional_correct / directional_total if directional_total > 0 else 0.5
            
            # Buy/Sell success rates with error analysis
            buy_data = [(a, p) for a, p in performances if 'BUY' in a.recommendation]
            sell_data = [(a, p) for a, p in performances if 'SELL' in a.recommendation]
            
            buy_success_rate = sum(1 for _, p in buy_data if p.recommendation_correct) / len(buy_data) if buy_data else 0.5
            sell_success_rate = sum(1 for _, p in sell_data if p.recommendation_correct) / len(sell_data) if sell_data else 0.5
            
            # Advanced consistency score based on multiple factors
            consistency_factors = [
                accuracy * 0.3,
                weighted_accuracy * 0.25,
                directional_accuracy * 0.2,
                (1 - mae_price) * 0.15,  # Lower price error = higher consistency
                (1 - mae_confidence) * 0.1  # Better calibration = higher consistency
            ]
            
            consistency_score = sum(consistency_factors)
            consistency_score = max(0.1, min(0.95, consistency_score))
            
            # Volatility-adjusted accuracy
            volatilities = [a.analysis_data for a, _ in performances if hasattr(a, 'analysis_data')]
            avg_volatility = 0.02  # Default volatility
            
            # Risk-adjusted returns calculation
            returns = []
            for analysis, perf in performances:
                if perf.recommendation_correct and analysis.confidence > 0.6:
                    expected_return = abs(perf.actual_price - analysis.price) / analysis.price
                    risk_adjusted = expected_return / (avg_volatility + 0.001)
                    returns.append(risk_adjusted)
            
            avg_risk_adjusted_return = np.mean(returns) if returns else 0.0
            
            return {
                'accuracy': accuracy,
                'weighted_accuracy': weighted_accuracy,
                'directional_accuracy': directional_accuracy,
                'consistency_score': consistency_score,
                'buy_success_rate': buy_success_rate,
                'sell_success_rate': sell_success_rate,
                'price_mae': mae_price,
                'confidence_mae': mae_confidence,
                'risk_adjusted_return': avg_risk_adjusted_return,
                'total_predictions': len(analyses),
                'performance_count': len(performances)
            }
            
        except Exception as e:
            print(f"Error in historical consistency: {e}")
            return self._default_consistency_data(0)
    
    def _default_consistency_data(self, prediction_count):
        """Return default consistency data when insufficient historical data"""
        return {
            'accuracy': 0.5,
            'weighted_accuracy': 0.5,
            'directional_accuracy': 0.5,
            'consistency_score': 0.5,
            'buy_success_rate': 0.5,
            'sell_success_rate': 0.5,
            'price_mae': 0.1,
            'confidence_mae': 0.3,
            'risk_adjusted_return': 0.0,
            'total_predictions': prediction_count,
            'performance_count': 0
        }
    
    def print_analysis(self, result):
        """Print analysis report"""
        if "error" in result:
            print(f"❌ {result['error']}")
            return
        
        rec = result['recommendation']
        conf = result['confidence']
        emoji = "🟢" if "BUY" in rec else "🔴" if "SELL" in rec else "🟡"
        
        print(f"\n{emoji} {rec} | Confidence: {conf:.1%}")
        print(f"💰 Price: {result['price']:.5f}")
        print(f"🎲 Buy Prob: {result['probabilities']['buy_probability']:.1%}")
        print(f"🎲 Sell Prob: {result['probabilities']['sell_probability']:.1%}")
        print(f"🤖 ML Prediction: {result['analysis']['ml_prediction']:.2%}")
        print(f"📊 RSI: {result['analysis']['rsi']:.1f}")
        hist_data = result['analysis']['historical_consistency']
        print(f"📈 Historical Accuracy: {hist_data['accuracy']:.1%}")
        print(f"🎯 Consistency Score: {hist_data['consistency_score']:.2f}")
        print(f"📊 Directional Accuracy: {hist_data['directional_accuracy']:.1%}")
        print(f"❌ Price MAE: {hist_data['price_mae']:.3f}")
        print(f"📊 Performance Count: {hist_data['performance_count']}")
        
        # Print market entry/exit advice
        if 'market_advice' in result and 'error' not in result['market_advice']:
            advice = result['market_advice']
            print(f"\n🎯 MARKET ENTRY/EXIT ADVICE:")
            print(f"   Action: {advice['action']} ({advice['confidence']:.1%} confidence)")
            
            # Show key timeframe signals
            if 'timeframe_analyses' in advice:
                for tf, analysis in list(advice['timeframe_analyses'].items())[:2]:  # Top 2 timeframes
                    print(f"   {tf.upper()}: Buy {analysis['buy_score']} | Sell {analysis['sell_score']}")