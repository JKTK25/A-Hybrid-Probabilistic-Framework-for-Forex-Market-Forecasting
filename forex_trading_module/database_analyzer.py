#!/usr/bin/env python3
"""
Database Analyzer for Forex Trading Module
Visualizes prediction trends vs real-time data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from core.database import ForexDatabase, ForexAnalysis, ForexPerformance
from sqlalchemy import func
import warnings
warnings.filterwarnings('ignore')

class DatabaseAnalyzer:
    def __init__(self):
        self.db = ForexDatabase()
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
    def get_analysis_data(self, days=30, pair=None):
        """Get analysis data from database"""
        query = self.db.session.query(ForexAnalysis)
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            query = query.filter(ForexAnalysis.timestamp >= cutoff)
            
        if pair:
            query = query.filter(ForexAnalysis.symbol == pair)
            
        return query.order_by(ForexAnalysis.timestamp).all()
    
    def get_performance_data(self, days=30, pair=None):
        """Get performance data from database"""
        query = self.db.session.query(ForexPerformance)
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            query = query.filter(ForexPerformance.created_at >= cutoff)
            
        if pair:
            query = query.filter(ForexPerformance.symbol == pair)
            
        return query.order_by(ForexPerformance.created_at).all()
    
    def create_dataframes(self, days=30, pair=None):
        """Convert database data to pandas DataFrames"""
        # Analysis data
        analysis_data = self.get_analysis_data(days, pair)
        analysis_df = pd.DataFrame([{
            'timestamp': a.timestamp,
            'pair': a.symbol,
            'current_price': a.price,
            'predicted_price': a.ml_prediction,
            'prediction_confidence': a.confidence,
            'recommendation': a.recommendation,
            'ml_prediction': a.ml_prediction,
            'buy_probability': a.buy_probability,
            'sell_probability': a.sell_probability
        } for a in analysis_data])
        
        # Performance data
        performance_data = self.get_performance_data(days, pair)
        performance_df = pd.DataFrame([{
            'timestamp': p.created_at,
            'pair': p.symbol,
            'predicted_price': p.predicted_price,
            'actual_price': p.actual_price,
            'prediction_error': abs(p.predicted_price - p.actual_price) if p.predicted_price and p.actual_price else 0,
            'directional_accuracy': 1 if p.recommendation_correct else 0,
            'confidence_score': p.prediction_accuracy
        } for p in performance_data])
        
        return analysis_df, performance_df
    
    def show_summary_stats(self, days=30, pair=None):
        """Display summary statistics"""
        analysis_df, performance_df = self.create_dataframes(days, pair)
        
        print(f"\n{'='*60}")
        print(f"DATABASE SUMMARY - Last {days} days")
        if pair:
            print(f"Currency Pair: {pair}")
        print(f"{'='*60}")
        
        print(f"\nAnalysis Records: {len(analysis_df)}")
        print(f"Performance Records: {len(performance_df)}")
        
        if not performance_df.empty:
            avg_error = performance_df['prediction_error'].mean()
            directional_acc = performance_df['directional_accuracy'].mean() * 100
            avg_confidence = performance_df['confidence_score'].mean()
            
            print(f"\nPERFORMANCE METRICS:")
            print(f"Average Prediction Error: {avg_error:.4f}")
            print(f"Directional Accuracy: {directional_acc:.1f}%")
            print(f"Average Confidence: {avg_confidence:.3f}")
        
        if not analysis_df.empty:
            pairs = analysis_df['pair'].unique()
            print(f"\nCurrency Pairs: {', '.join(pairs)}")
            
            recommendations = analysis_df['recommendation'].value_counts()
            print(f"\nRecommendations Distribution:")
            for rec, count in recommendations.items():
                print(f"  {rec}: {count}")
    
    def plot_prediction_accuracy(self, days=30, pair=None):
        """Plot prediction accuracy over time"""
        _, performance_df = self.create_dataframes(days, pair)
        
        if performance_df.empty:
            print("No performance data available for plotting")
            return
        
        print(f"Creating plots with {len(performance_df)} records...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Prediction Accuracy Analysis - Last {days} days', fontsize=16)
            
            # Prediction Error Over Time
            axes[0,0].plot(performance_df['timestamp'], performance_df['prediction_error'], 'b-', alpha=0.7)
            axes[0,0].set_title('Prediction Error Over Time')
            axes[0,0].set_ylabel('Absolute Error')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Directional Accuracy Over Time
            axes[0,1].plot(performance_df['timestamp'], performance_df['directional_accuracy'], 'g-', alpha=0.7)
            axes[0,1].set_title('Directional Accuracy Over Time')
            axes[0,1].set_ylabel('Accuracy (1=Correct, 0=Wrong)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Error Distribution
            axes[1,0].hist(performance_df['prediction_error'], bins=20, alpha=0.7, color='orange')
            axes[1,0].set_title('Prediction Error Distribution')
            axes[1,0].set_xlabel('Absolute Error')
            axes[1,0].set_ylabel('Frequency')
            
            # Confidence vs Accuracy Scatter
            axes[1,1].scatter(performance_df['confidence_score'], performance_df['directional_accuracy'], alpha=0.6)
            axes[1,1].set_title('Confidence vs Directional Accuracy')
            axes[1,1].set_xlabel('Confidence Score')
            axes[1,1].set_ylabel('Directional Accuracy')
            
            plt.tight_layout()
            
            # Save plot first
            filename = f'prediction_accuracy_{days}days.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {filename}")
            
            # Try to display
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Plot error: {e}")
            plt.savefig('prediction_accuracy.png', dpi=150, bbox_inches='tight')
            print("Plot saved as prediction_accuracy.png")
            plt.close()
    
    def plot_price_predictions(self, days=30, pair=None):
        """Plot predicted vs actual prices"""
        analysis_df, performance_df = self.create_dataframes(days, pair)
        
        if analysis_df.empty:
            print("No analysis data available for plotting")
            return
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'Price Predictions vs Reality - Last {days} days', fontsize=16)
            
            # Current vs Predicted Prices
            axes[0].plot(analysis_df['timestamp'], analysis_df['current_price'], 'b-', label='Current Price', linewidth=2)
            axes[0].plot(analysis_df['timestamp'], analysis_df['predicted_price'], 'r--', label='Predicted Price', alpha=0.7)
            axes[0].set_title('Current vs Predicted Prices')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].tick_params(axis='x', rotation=45)
            
            # Model Predictions Comparison
            if 'ml_prediction' in analysis_df.columns:
                axes[1].plot(analysis_df['timestamp'], analysis_df['ml_prediction'], 'g-', label='ML Prediction', alpha=0.7)
                axes[1].set_title('ML Predictions Over Time')
                axes[1].set_ylabel('ML Prediction Value')
                axes[1].legend()
                axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot first
            filename = f'price_predictions_{days}days.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {filename}")
            
            # Try to display
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Plot error: {e}")
            plt.savefig('price_predictions_error.png', dpi=150, bbox_inches='tight')
            print("Plot saved as price_predictions_error.png")
            plt.close()
    
    def show_data_table(self, days=7, pair=None, table_type='analysis'):
        """Display data in table format"""
        analysis_df, performance_df = self.create_dataframes(days, pair)
        
        if table_type == 'analysis' and not analysis_df.empty:
            print(f"\nANALYSIS DATA - Last {days} days")
            print("="*80)
            display_df = analysis_df[['timestamp', 'pair', 'current_price', 'predicted_price', 'recommendation']].tail(20)
            print(display_df.to_string(index=False))
            
        elif table_type == 'performance' and not performance_df.empty:
            print(f"\nPERFORMANCE DATA - Last {days} days")
            print("="*80)
            display_df = performance_df[['timestamp', 'pair', 'predicted_price', 'actual_price', 'prediction_error', 'directional_accuracy']].tail(20)
            print(display_df.to_string(index=False))
        else:
            print(f"No {table_type} data available")
    
    def analyze_pair_performance(self, pair, days=30):
        """Detailed analysis for specific currency pair"""
        analysis_df, performance_df = self.create_dataframes(days, pair)
        
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS FOR {pair}")
        print(f"{'='*60}")
        
        if not performance_df.empty:
            error_stats = performance_df['prediction_error'].describe()
            print(f"\nPrediction Error Statistics:")
            print(f"Mean: {error_stats['mean']:.4f}")
            print(f"Std:  {error_stats['std']:.4f}")
            print(f"Min:  {error_stats['min']:.4f}")
            print(f"Max:  {error_stats['max']:.4f}")
            
            directional_acc = performance_df['directional_accuracy'].mean() * 100
            print(f"\nDirectional Accuracy: {directional_acc:.1f}%")
            
            # Recent trend
            recent_data = performance_df.tail(10)
            recent_acc = recent_data['directional_accuracy'].mean() * 100
            print(f"Recent Accuracy (last 10): {recent_acc:.1f}%")
        
        if not analysis_df.empty:
            recommendations = analysis_df['recommendation'].value_counts()
            print(f"\nRecommendations:")
            for rec, count in recommendations.items():
                print(f"  {rec}: {count}")

def main():
    analyzer = DatabaseAnalyzer()
    
    while True:
        print(f"\n{'='*60}")
        print("FOREX DATABASE ANALYZER")
        print(f"{'='*60}")
        print("1. Summary Statistics")
        print("2. Prediction Accuracy Plots")
        print("3. Price Prediction Plots")
        print("4. Analysis Data Table")
        print("5. Performance Data Table")
        print("6. Pair-Specific Analysis")
        print("7. Exit")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == '1':
            days = int(input("Days to analyze (default 30): ") or 30)
            pair = input("Currency pair (optional, e.g., EURUSD): ").strip().upper() or None
            analyzer.show_summary_stats(days, pair)
            
        elif choice == '2':
            days = int(input("Days to analyze (default 30): ") or 30)
            pair = input("Currency pair (optional): ").strip().upper() or None
            analyzer.plot_prediction_accuracy(days, pair)
            
        elif choice == '3':
            days = int(input("Days to analyze (default 30): ") or 30)
            pair = input("Currency pair (optional): ").strip().upper() or None
            analyzer.plot_price_predictions(days, pair)
            
        elif choice == '4':
            days = int(input("Days to show (default 7): ") or 7)
            pair = input("Currency pair (optional): ").strip().upper() or None
            analyzer.show_data_table(days, pair, 'analysis')
            
        elif choice == '5':
            days = int(input("Days to show (default 7): ") or 7)
            pair = input("Currency pair (optional): ").strip().upper() or None
            analyzer.show_data_table(days, pair, 'performance')
            
        elif choice == '6':
            pair = input("Currency pair (e.g., EURUSD): ").strip().upper()
            days = int(input("Days to analyze (default 30): ") or 30)
            analyzer.analyze_pair_performance(pair, days)
            
        elif choice == '7':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()