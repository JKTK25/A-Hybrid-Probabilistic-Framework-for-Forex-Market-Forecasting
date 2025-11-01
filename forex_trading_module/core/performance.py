#!/usr/bin/env python3
"""
Performance Tracker - Forex agent performance analysis and visualization
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import forex_db

class PerformanceTracker:
    def __init__(self):
        self.db = forex_db
        
    def generate_report(self, days=30):
        """Generate performance report"""
        print("🏆 FOREX PERFORMANCE REPORT")
        print("="*40)
        
        stats = self.db.get_accuracy_stats(days=days)
        
        if "error" in stats:
            print("❌ No performance data available")
            return
        
        print(f"📊 PERFORMANCE (Last {days} days):")
        print(f"   Total Predictions: {stats['total_predictions']}")
        print(f"   Recommendation Accuracy: {stats['recommendation_accuracy']:.1%}")
        print(f"   Price Prediction Accuracy: {stats['price_prediction_accuracy']:.1%}")
        
        # Performance grade
        accuracy = stats['recommendation_accuracy']
        if accuracy >= 0.8:
            grade = "A+ (Excellent)"
        elif accuracy >= 0.7:
            grade = "A (Very Good)"
        elif accuracy >= 0.6:
            grade = "B (Good)"
        elif accuracy >= 0.5:
            grade = "C (Average)"
        else:
            grade = "D (Needs Improvement)"
        
        print(f"🎯 PERFORMANCE GRADE: {grade}")
        
        return stats
    
    def plot_performance(self, days=30):
        """Plot performance charts"""
        try:
            # Get performance data
            cutoff = datetime.now() - timedelta(days=days)
            query = self.db.session.query(self.db.ForexAnalysis, self.db.ForexPerformance).join(
                self.db.ForexPerformance, self.db.ForexAnalysis.id == self.db.ForexPerformance.analysis_id
            ).filter(self.db.ForexAnalysis.timestamp >= cutoff)
            
            results = query.all()
            
            if not results:
                print("No data for plotting")
                return
            
            # Create DataFrame
            data = []
            for analysis, performance in results:
                data.append({
                    'timestamp': analysis.timestamp,
                    'symbol': analysis.symbol,
                    'confidence': analysis.confidence,
                    'accuracy': performance.prediction_accuracy,
                    'correct': performance.recommendation_correct
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy over time
            daily_accuracy = df.groupby(df['timestamp'].dt.date)['correct'].mean()
            ax1.plot(daily_accuracy.index, daily_accuracy.values, marker='o', linewidth=2)
            ax1.set_title('Recommendation Accuracy Over Time')
            ax1.set_ylabel('Accuracy Rate')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Confidence vs Accuracy
            ax2.scatter(df['confidence'], df['accuracy'], alpha=0.6, c=df['correct'], cmap='RdYlGn')
            ax2.set_xlabel('Confidence Level')
            ax2.set_ylabel('Price Accuracy')
            ax2.set_title('Confidence vs Accuracy')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('forex_module_performance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📈 Performance chart saved: forex_module_performance.png")
            
        except Exception as e:
            print(f"Error plotting performance: {e}")
    
    def get_best_performing_pairs(self, days=30):
        """Get best performing forex pairs"""
        symbols = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
        
        performance_data = {}
        
        for symbol in symbols:
            stats = self.db.get_accuracy_stats(symbol, days)
            if "error" not in stats:
                pair_name = symbol.replace('=X', '').replace('USD', '/USD')
                performance_data[pair_name] = {
                    'accuracy': stats['recommendation_accuracy'],
                    'predictions': stats['total_predictions']
                }
        
        # Sort by accuracy
        sorted_pairs = sorted(performance_data.items(), 
                            key=lambda x: x[1]['accuracy'], reverse=True)
        
        print(f"\n📈 BEST PERFORMING PAIRS (Last {days} days):")
        print("-" * 40)
        
        for i, (pair, data) in enumerate(sorted_pairs, 1):
            print(f"{i}. {pair}: {data['accuracy']:.1%} accuracy ({data['predictions']} predictions)")
        
        return sorted_pairs