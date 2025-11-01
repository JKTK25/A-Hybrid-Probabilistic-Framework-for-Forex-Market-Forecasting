#!/usr/bin/env python3
"""
Configuration management for forex trading module
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', '')
    
    # DeepSeek Configuration
    DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///forex_module.db')
    DATABASE_ECHO = os.getenv('DATABASE_ECHO', 'False').lower() == 'true'
    
    # Trading
    DEFAULT_REFRESH_MINUTES = int(os.getenv('DEFAULT_REFRESH_MINUTES', '5'))
    MAX_ANALYSIS_HISTORY_DAYS = int(os.getenv('MAX_ANALYSIS_HISTORY_DAYS', '30'))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
    
    # Risk Management
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.02'))
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.04'))
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'forex_trading.log')

# Global config instance
config = Config()