#!/usr/bin/env python3
"""
Real-time Plotting - Live forex charts and analysis visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from core.database import forex_db, ForexAnalysis

class LiveForexPlotter:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Live Forex Analysis Dashboard', fontsize=16, fontweight='bold')
        
    def plot_live_analysis(self, symbol="EURUSD=X"):
        """Plot live forex analysis with predictions"""
        try:
            # Get price data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1h")
            
            if hist.empty:
                # Fallback to daily data
                hist = ticker.history(period="30d", interval="1d")
                
            if hist.empty:
                print("No price data available for plotting")
                return
            
            # Get analysis data from database
            cutoff = datetime.now() - timedelta(days=30)  # Longer period
            analyses = forex_db.session.query(ForexAnalysis).filter(
                ForexAnalysis.symbol == symbol,
                ForexAnalysis.timestamp >= cutoff
            ).order_by(ForexAnalysis.timestamp).all()
            
            print(f"Found {len(analyses)} analyses for {symbol}")
            
            # Plot 1: Price Chart with Predictions
            ax1 = self.axes[0, 0]
            ax1.clear()
            
            # Always plot price data
            ax1.plot(hist.index, hist['Close'], 'b-', linewidth=2, label=f'Price ({len(hist)} points)')
            
            # Add prediction points if available
            if analyses:
                pred_times = [a.timestamp for a in analyses]
                pred_prices = [a.price for a in analyses]
                pred_colors = ['green' if 'BUY' in a.recommendation else 'red' if 'SELL' in a.recommendation else 'gray' for a in analyses]
                
                ax1.scatter(pred_times, pred_prices, c=pred_colors, s=100, alpha=0.7, label=f'Predictions ({len(analyses)})')
            else:
                # Add sample prediction if no data
                current_price = hist['Close'].iloc[-1]
                current_time = hist.index[-1]
                ax1.scatter([current_time], [current_price], c='blue', s=100, alpha=0.7, label='Current Price')
            
            ax1.set_title(f'{symbol.replace("=X", "")} Price & Predictions')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Confidence Over Time
            ax2 = self.axes[0, 1]
            ax2.clear()
            
            if analyses and len(analyses) > 0:
                conf_times = [a.timestamp for a in analyses]
                confidences = [a.confidence for a in analyses]
                ax2.plot(conf_times, confidences, 'o-', color='purple', linewidth=2, label=f'Confidence ({len(analyses)} points)')
                ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Threshold')
            else:
                # Show sample confidence data
                sample_times = pd.date_range(end=datetime.now(), periods=5, freq='D')
                sample_conf = [0.7, 0.6, 0.8, 0.5, 0.75]
                ax2.plot(sample_times, sample_conf, 'o-', color='purple', linewidth=2, label='Sample Confidence')
                ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Threshold')
            
            ax2.set_title('AI Confidence Levels')
            ax2.set_ylabel('Confidence')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Buy/Sell Probability Distribution
            ax3 = self.axes[1, 0]
            ax3.clear()
            
            if analyses and len(analyses) > 0:
                buy_probs = [a.buy_probability for a in analyses if a.buy_probability and a.buy_probability > 0]
                sell_probs = [a.sell_probability for a in analyses if a.sell_probability and a.sell_probability > 0]
                
                if buy_probs or sell_probs:
                    if buy_probs:
                        ax3.hist(buy_probs, alpha=0.6, color='green', label=f'Buy Probs ({len(buy_probs)})', bins=min(10, len(buy_probs)))
                    if sell_probs:
                        ax3.hist(sell_probs, alpha=0.6, color='red', label=f'Sell Probs ({len(sell_probs)})', bins=min(10, len(sell_probs)))
                else:
                    # Sample data
                    sample_probs = [0.3, 0.5, 0.7, 0.6, 0.8]
                    ax3.hist(sample_probs, alpha=0.6, color='blue', label='Sample Probabilities', bins=5)
            else:
                # Sample data when no analyses
                sample_buy = [0.6, 0.7, 0.5, 0.8]
                sample_sell = [0.4, 0.3, 0.5, 0.2]
                ax3.hist(sample_buy, alpha=0.6, color='green', label='Sample Buy', bins=4)
                ax3.hist(sample_sell, alpha=0.6, color='red', label='Sample Sell', bins=4)
            
            ax3.set_title('Probability Distribution')
            ax3.set_xlabel('Probability')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Recommendation Timeline
            ax4 = self.axes[1, 1]
            ax4.clear()
            
            if analyses and len(analyses) > 0:
                rec_times = [a.timestamp for a in analyses]
                rec_values = []
                rec_colors = []
                
                for a in analyses:
                    if 'STRONG BUY' in a.recommendation:
                        rec_values.append(2)
                        rec_colors.append('darkgreen')
                    elif 'BUY' in a.recommendation:
                        rec_values.append(1)
                        rec_colors.append('green')
                    elif 'STRONG SELL' in a.recommendation:
                        rec_values.append(-2)
                        rec_colors.append('darkred')
                    elif 'SELL' in a.recommendation:
                        rec_values.append(-1)
                        rec_colors.append('red')
                    else:
                        rec_values.append(0)
                        rec_colors.append('gray')
                
                ax4.scatter(rec_times, rec_values, c=rec_colors, s=100, alpha=0.8, label=f'Signals ({len(analyses)})')
            else:
                # Sample recommendation data
                sample_times = pd.date_range(end=datetime.now(), periods=5, freq='D')
                sample_values = [1, -1, 0, 2, -1]
                sample_colors = ['green', 'red', 'gray', 'darkgreen', 'red']
                ax4.scatter(sample_times, sample_values, c=sample_colors, s=100, alpha=0.8, label='Sample Signals')
            
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('Recommendation Timeline')
            ax4.set_ylabel('Signal Strength')
            ax4.set_ylim(-2.5, 2.5)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'forex_live_analysis_{symbol.replace("=X", "")}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Chart saved: {filename}")
            
        except Exception as e:
            print(f"Error plotting live analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def plot_performance_metrics(self):
        """Plot performance metrics across all pairs"""
        try:
            symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Forex AI Performance Dashboard', fontsize=16, fontweight='bold')
            
            # Accuracy by pair
            accuracies = []
            pair_names = []
            
            for symbol in symbols:
                stats = forex_db.get_accuracy_stats(symbol, days=30)
                if "error" not in stats:
                    accuracies.append(stats['recommendation_accuracy'])
                    pair_names.append(symbol.replace('=X', '').replace('USD', '/USD'))
            
            if accuracies:
                colors = ['green' if acc > 0.6 else 'orange' if acc > 0.4 else 'red' for acc in accuracies]
                ax1.bar(pair_names, accuracies, color=colors, alpha=0.7)
                ax1.set_title('Accuracy by Currency Pair')
                ax1.set_ylabel('Accuracy Rate')
                ax1.set_ylim(0, 1)
                ax1.grid(True, alpha=0.3)
            
            # Predictions over time
            cutoff = datetime.now() - timedelta(days=7)
            all_analyses = forex_db.session.query(ForexAnalysis).filter(
                ForexAnalysis.timestamp >= cutoff
            ).order_by(ForexAnalysis.timestamp).all()
            
            if all_analyses:
                daily_counts = {}
                for analysis in all_analyses:
                    date = analysis.timestamp.date()
                    daily_counts[date] = daily_counts.get(date, 0) + 1
                
                dates = list(daily_counts.keys())
                counts = list(daily_counts.values())
                
                ax2.plot(dates, counts, 'o-', linewidth=2, markersize=8)
                ax2.set_title('Daily Prediction Volume')
                ax2.set_ylabel('Number of Predictions')
                ax2.grid(True, alpha=0.3)
            
            # Confidence distribution
            if all_analyses:
                confidences = [a.confidence for a in all_analyses]
                ax3.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(confidences):.2f}')
                ax3.set_title('Confidence Distribution')
                ax3.set_xlabel('Confidence Level')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Recommendation distribution
            if all_analyses:
                rec_counts = {}
                for analysis in all_analyses:
                    rec = analysis.recommendation
                    rec_counts[rec] = rec_counts.get(rec, 0) + 1
                
                recs = list(rec_counts.keys())
                counts = list(rec_counts.values())
                colors = ['green' if 'BUY' in rec else 'red' if 'SELL' in rec else 'gray' for rec in recs]
                
                ax4.pie(counts, labels=recs, colors=colors, autopct='%1.1f%%', startangle=90)
                ax4.set_title('Recommendation Distribution')
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'forex_performance_dashboard_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Performance chart saved: {filename}")
            
        except Exception as e:
            print(f"Error plotting performance metrics: {e}")
    
    def plot_prediction_accuracy(self, symbol="EURUSD=X"):
        """Plot prediction vs actual price movements"""
        try:
            from core.database import ForexPerformance
            
            # Get performance data
            cutoff = datetime.now() - timedelta(days=30)
            query = forex_db.session.query(ForexAnalysis, ForexPerformance).join(
                ForexPerformance, ForexAnalysis.id == ForexPerformance.analysis_id
            ).filter(
                ForexAnalysis.symbol == symbol,
                ForexAnalysis.timestamp >= cutoff
            )
            
            results = query.all()
            
            if not results:
                print("No performance data available for plotting")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{symbol.replace("=X", "")} Prediction Accuracy Analysis', fontsize=14, fontweight='bold')
            
            # Predicted vs Actual prices
            predicted_prices = [perf.predicted_price for _, perf in results]
            actual_prices = [perf.actual_price for _, perf in results]
            
            ax1.scatter(predicted_prices, actual_prices, alpha=0.6, s=100)
            
            # Perfect prediction line
            min_price = min(min(predicted_prices), min(actual_prices))
            max_price = max(max(predicted_prices), max(actual_prices))
            ax1.plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.8, label='Perfect Prediction')
            
            ax1.set_xlabel('Predicted Price')
            ax1.set_ylabel('Actual Price')
            ax1.set_title('Predicted vs Actual Prices')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Accuracy over time
            timestamps = [analysis.timestamp for analysis, _ in results]
            accuracies = [perf.prediction_accuracy for _, perf in results]
            correct_recs = [perf.recommendation_correct for _, perf in results]
            
            ax2.plot(timestamps, accuracies, 'b-', linewidth=2, label='Price Accuracy')
            
            # Add recommendation correctness as scatter
            correct_times = [t for t, c in zip(timestamps, correct_recs) if c]
            incorrect_times = [t for t, c in zip(timestamps, correct_recs) if not c]
            
            ax2.scatter(correct_times, [1.0] * len(correct_times), color='green', s=50, alpha=0.7, label='Correct Rec')
            ax2.scatter(incorrect_times, [0.9] * len(incorrect_times), color='red', s=50, alpha=0.7, label='Wrong Rec')
            
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Accuracy Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'forex_accuracy_{symbol.replace("=X", "")}_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Accuracy chart saved: {filename}")
            
        except Exception as e:
            print(f"Error plotting prediction accuracy: {e}")

def main():
    """Main plotting function"""
    plotter = LiveForexPlotter()
    
    print("📊 Generating Forex Analysis Charts...")
    
    # Plot live analysis for EUR/USD
    plotter.plot_live_analysis("EURUSD=X")
    
    # Plot performance metrics
    plotter.plot_performance_metrics()
    
    # Plot prediction accuracy
    plotter.plot_prediction_accuracy("EURUSD=X")
    
    print("✅ Charts generated and saved!")

if __name__ == "__main__":
    main()