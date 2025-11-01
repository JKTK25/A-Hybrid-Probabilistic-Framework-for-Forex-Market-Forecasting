#!/usr/bin/env python3
"""
Forex Database - SQLAlchemy database for forex analysis storage
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import config

Base = declarative_base()

class ForexAnalysis(Base):
    __tablename__ = 'forex_analysis'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    price = Column(Float, nullable=False)
    recommendation = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    buy_probability = Column(Float)
    sell_probability = Column(Float)
    ml_prediction = Column(Float)
    analysis_data = Column(Text)

class ForexPerformance(Base):
    __tablename__ = 'forex_performance'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    predicted_price = Column(Float)
    actual_price = Column(Float)
    prediction_accuracy = Column(Float)
    recommendation_correct = Column(Boolean)
    days_elapsed = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

class ForexDatabase:
    def __init__(self, db_url=None):
        db_url = db_url or config.DATABASE_URL
        self.engine = create_engine(db_url, echo=config.DATABASE_ECHO)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_analysis(self, analysis_result):
        """Save analysis to database"""
        try:
            analysis = ForexAnalysis(
                symbol=analysis_result['symbol'],
                price=analysis_result['price'],
                recommendation=analysis_result['recommendation'],
                confidence=analysis_result['confidence'],
                buy_probability=analysis_result['probabilities']['buy_probability'],
                sell_probability=analysis_result['probabilities']['sell_probability'],
                ml_prediction=analysis_result['analysis']['ml_prediction'],
                analysis_data=json.dumps(analysis_result, default=str)
            )
            
            self.session.add(analysis)
            self.session.commit()
            return analysis.id
            
        except Exception as e:
            self.session.rollback()
            return None
    
    def get_previous_analysis(self, symbol, hours_back=24):
        """Get previous analysis for symbol"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return self.session.query(ForexAnalysis).filter(
            ForexAnalysis.symbol == symbol,
            ForexAnalysis.timestamp >= cutoff_time
        ).order_by(ForexAnalysis.timestamp.desc()).first()
    
    def update_performance(self, current_price_data):
        """Update performance metrics"""
        cutoff = datetime.now() - timedelta(days=7)
        old_analyses = self.session.query(ForexAnalysis).filter(
            ForexAnalysis.timestamp >= cutoff
        ).all()
        
        for analysis in old_analyses:
            existing = self.session.query(ForexPerformance).filter(
                ForexPerformance.analysis_id == analysis.id
            ).first()
            
            if existing:
                continue
            
            current_price = current_price_data.get(analysis.symbol)
            if not current_price:
                continue
            
            days_elapsed = (datetime.now() - analysis.timestamp).days
            if days_elapsed == 0:
                continue
            
            predicted_price = analysis.price * (1 + analysis.ml_prediction)
            price_accuracy = 1 - abs(predicted_price - current_price) / current_price
            
            price_change = (current_price - analysis.price) / analysis.price
            rec_correct = False
            
            if "BUY" in analysis.recommendation and price_change > 0.001:
                rec_correct = True
            elif "SELL" in analysis.recommendation and price_change < -0.001:
                rec_correct = True
            elif analysis.recommendation == "HOLD" and abs(price_change) < 0.005:
                rec_correct = True
            
            performance = ForexPerformance(
                analysis_id=analysis.id,
                symbol=analysis.symbol,
                predicted_price=predicted_price,
                actual_price=current_price,
                prediction_accuracy=price_accuracy,
                recommendation_correct=rec_correct,
                days_elapsed=days_elapsed
            )
            
            self.session.add(performance)
        
        self.session.commit()
    
    def get_accuracy_stats(self, symbol=None, days=30):
        """Get accuracy statistics"""
        cutoff = datetime.now() - timedelta(days=days)
        query = self.session.query(ForexPerformance).filter(
            ForexPerformance.created_at >= cutoff
        )
        
        if symbol:
            query = query.filter(ForexPerformance.symbol == symbol)
        
        performances = query.all()
        
        if not performances:
            return {"error": "No performance data available"}
        
        total_predictions = len(performances)
        correct_recommendations = sum(1 for p in performances if p.recommendation_correct)
        avg_price_accuracy = sum(p.prediction_accuracy for p in performances) / total_predictions
        
        return {
            "total_predictions": total_predictions,
            "recommendation_accuracy": correct_recommendations / total_predictions,
            "price_prediction_accuracy": avg_price_accuracy,
            "correct_recommendations": correct_recommendations
        }

# Global database instance
forex_db = ForexDatabase()