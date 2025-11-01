# Forex Trading Module

A comprehensive AI-powered forex trading system with real-time analysis, prediction, and performance tracking capabilities.

## Developer
**James Kariuki**  
📧 jamexkarix583@gmail.com  
📱 0792698424

## Features

### 🤖 AI Trading Agent
- **Multi-Model Predictions**: ML, Monte Carlo simulation, and Bayesian analysis
- **Real-time Analysis**: Live forex data processing and prediction
- **Smart Recommendations**: BUY/SELL/HOLD decisions with confidence scores
- **Performance Learning**: Database-driven accuracy tracking and improvement
- **Advanced Mathematics**: Technical indicators, probability distributions, and error correction

### 📊 Live Dashboard
- **Real-time Monitoring**: 15-minute refresh intervals
- **Multi-pair Analysis**: Track multiple currency pairs simultaneously
- **Visual Charts**: Live price charts with technical indicators
- **AI Advice Display**: Current trading recommendations with reasoning

### 📈 Market Advisor
- **Multi-timeframe Analysis**: 1 month to 1 hour intervals
- **Entry/Exit Timing**: Optimal trade timing recommendations
- **Weighted Scoring**: Longer timeframes weighted more heavily
- **Risk Assessment**: Market volatility and trend analysis

### 🗄️ Database Analytics
- **Performance Tracking**: Prediction accuracy over time
- **Visual Analytics**: Charts and plots for trend analysis
- **Historical Data**: Complete analysis and performance history
- **Pair-specific Stats**: Detailed metrics for individual currency pairs

### 📊 Advanced Plotting
- **Live Charts**: Real-time price and indicator visualization
- **Performance Metrics**: Accuracy and error trend analysis
- **Prediction Comparison**: Multiple model comparison charts
- **Export Capabilities**: Save charts as PNG files

## Installation

### Prerequisites
```bash
pip install yfinance pandas numpy matplotlib seaborn sqlalchemy python-dotenv requests
```

### API Keys Required
Create a `.env` file with:
```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
FINNHUB_API_KEY=your_finnhub_api_key
DATABASE_URL=sqlite:///forex_trading.db
DATABASE_ECHO=False
```

## Usage

### Command Line Interface

#### 1. Live Dashboard
```bash
python main.py dashboard
```
- Real-time forex monitoring
- 15-minute auto-refresh
- Multi-pair analysis
- AI trading advice
- Live charts generation

#### 2. Single Analysis
```bash
python main.py analyze
```
- Analyze current market conditions
- Generate trading recommendations
- Store results in database
- Display confidence scores

#### 3. Performance Report
```bash
python main.py performance
```
- View prediction accuracy statistics
- Generate performance plots
- Show best performing pairs
- Historical accuracy trends

#### 4. Live Charts
```bash
python main.py charts
```
- Generate real-time price charts
- Technical indicator visualization
- Performance metric plots
- Prediction accuracy charts

#### 5. Market Advisor
```bash
python main.py advisor
```
- Multi-timeframe market analysis
- Entry/exit timing recommendations
- Risk assessment for major pairs
- Weighted scoring system

#### 6. Database Analyzer
```bash
python main.py database
```
Interactive menu with options:
- **Summary Statistics**: Overall performance metrics
- **Prediction Accuracy Plots**: Visual error and accuracy trends
- **Price Prediction Charts**: Predicted vs actual price comparisons
- **Analysis Data Table**: Recent analysis records
- **Performance Data Table**: Historical performance data
- **Pair-Specific Analysis**: Detailed stats for individual pairs

### Python API Usage

#### Basic Analysis
```python
from core.trading_engine import ForexTradingEngine

engine = ForexTradingEngine()
engine.run_analysis()
```

#### Custom Configuration
```python
from core.ai_agent import ForexAIAgent

agent = ForexAIAgent()
result = agent.analyze_forex_pair("EURUSD=X")
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Database Operations
```python
from core.database import ForexDatabase

db = ForexDatabase()
stats = db.get_accuracy_stats(symbol="EURUSD=X", days=30)
print(f"Accuracy: {stats['recommendation_accuracy']:.2%}")
```

#### Market Advisor
```python
from core.market_advisor import MarketAdvisor

advisor = MarketAdvisor()
advice = advisor.generate_market_advice("EURUSD=X")
advisor.print_market_advice(advice)
```

## AI Agent Mathematical Functions

### Technical Indicators

#### RSI (Relative Strength Index)
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss

Average Gain = SMA(max(price_change, 0), 14)
Average Loss = SMA(max(-price_change, 0), 14)

RSI_Buy_Prob = 0.8 if RSI < 30 else 0.2
RSI_Sell_Prob = 0.8 if RSI > 70 else 0.2
```

#### MACD (Moving Average Convergence Divergence)
```
EMA_12 = EMA(close_price, 12)
EMA_26 = EMA(close_price, 26)
MACD = EMA_12 - EMA_26
MACD_Signal = EMA(MACD, 9)

Bullish_Signal = MACD > MACD_Signal
Bearish_Signal = MACD < MACD_Signal
```

#### Bollinger Bands
```
BB_Middle = SMA(close_price, 20)
BB_Std = StdDev(close_price, 20)
BB_Upper = BB_Middle + (2 * BB_Std)
BB_Lower = BB_Middle - (2 * BB_Std)

# Safe division to prevent divide by zero
BB_Position = (current_price - BB_Lower) / (BB_Upper - BB_Lower) if BB_Upper != BB_Lower else 0.5
```

### Monte Carlo Simulation

#### Price Simulation Algorithm
```python
for simulation in range(1000):
    price = current_price
    for day in range(forecast_days):
        drift = random.gauss(0, volatility)
        price *= (1 + drift)
    results.append(price)

prob_up = count(results > current_price) / total_simulations
prob_down = count(results < current_price) / total_simulations
mean_price = average(results)
```

### Machine Learning Model

#### Random Forest Feature Engineering
```
Features = [RSI, MACD, BB_Position, Volatility]
Target = future_returns = price_change(t+1)

X_scaled = StandardScaler().fit_transform(Features)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_scaled, Target)

ml_prediction = model.predict(current_features)
```

### Advanced Probability Combination

#### Adaptive Weighting System
```
accuracy_weight = historical_accuracy
directional_weight = directional_accuracy
price_error_penalty = mean_absolute_error

technical_weight = 0.2 + (accuracy_weight - 0.5) * 0.1
ml_weight = 0.3 + (directional_weight - 0.5) * 0.2
mc_weight = 0.25 - price_error_penalty * 0.1
historical_weight = 0.25 + (consistency_score - 0.5) * 0.1

# Normalize weights
total_weight = sum(all_weights)
weights = [w / total_weight for w in weights]
```

#### Final Probability Calculation
```
buy_probability = (
    RSI_Buy_Prob * technical_weight +
    MC_prob_up * mc_weight +
    ML_bullish_signal * ml_weight +
    historical_buy_success * historical_weight
)

sell_probability = (
    RSI_Sell_Prob * technical_weight +
    MC_prob_down * mc_weight +
    ML_bearish_signal * ml_weight +
    historical_sell_success * historical_weight
)
```

### Error Correction & Learning

#### Historical Consistency Metrics
```
# Mean Absolute Error for prices
MAE_price = mean(|predicted_price - actual_price| / actual_price)

# Confidence calibration error
MAE_confidence = mean(|confidence_score - actual_success|)

# Time-weighted accuracy (recent predictions weighted more)
time_weights = exp(linspace(-1, 0, num_predictions))
weighted_accuracy = sum(weights * correct_predictions) / sum(weights)

# Directional accuracy
directional_accuracy = count(predicted_direction == actual_direction) / total
```

#### Dynamic Threshold Adjustment
```
# Adaptive thresholds based on historical performance
buy_threshold = 0.6 - (accuracy_weight - 0.5) * 0.1
sell_threshold = 0.6 - (accuracy_weight - 0.5) * 0.1
strong_threshold = 0.75 - (accuracy_weight - 0.5) * 0.05

# Error-corrected confidence
consistency_factor = consistency_score
confidence_calibration = 1 - confidence_MAE
confidence_adjustment = consistency_factor * confidence_calibration

final_confidence = min(0.95, base_confidence * (0.7 + confidence_adjustment * 0.3))
```

### Risk-Adjusted Returns
```
risk_adjusted_return = expected_return / (volatility + 0.001)
avg_volatility = rolling_std(returns, window=20)
expected_return = |actual_price - predicted_price| / predicted_price
```

### Consistency Score Calculation
```
consistency_score = (
    accuracy * 0.3 +
    weighted_accuracy * 0.25 +
    directional_accuracy * 0.2 +
    (1 - price_MAE) * 0.15 +
    (1 - confidence_MAE) * 0.1
)

consistency_score = max(0.1, min(0.95, consistency_score))
```

## Architecture

### Core Components

#### 1. AI Agent (`core/ai_agent.py`)
- **ML Predictions**: Linear regression with technical indicators
- **Monte Carlo**: Probabilistic price simulation
- **Bayesian Analysis**: Prior belief updating with new data
- **Database Learning**: Historical accuracy tracking for improvement

#### 2. Trading Engine (`core/trading_engine.py`)
- **Data Integration**: YFinance and Finnhub API support
- **Analysis Orchestration**: Coordinates AI agent and data sources
- **Result Processing**: Formats and stores analysis results

#### 3. Database System (`core/database.py`)
- **Analysis Storage**: Complete prediction and recommendation history
- **Performance Tracking**: Accuracy metrics and error calculations
- **SQLAlchemy ORM**: Robust database operations

#### 4. Market Advisor (`core/market_advisor.py`)
- **Multi-timeframe Analysis**: 1mo, 1wk, 1d, 4h, 1h intervals
- **Weighted Scoring**: Longer timeframes have higher weights
- **Entry/Exit Timing**: Optimal trade timing recommendations

##### Market Advisor Mathematical Functions

**Entry Signal Scoring:**
```
# BUY Entry Conditions
buy_score = 0
if price > SMA_20 > SMA_50: buy_score += 2
if RSI < 40 and prev_RSI >= 40: buy_score += 2
if MACD > MACD_Signal and prev_MACD <= prev_Signal: buy_score += 3
if price <= BB_Lower * 1.01: buy_score += 2
if price > Support * 1.02: buy_score += 1

# SELL Entry Conditions
sell_score = 0
if price < SMA_20 < SMA_50: sell_score += 2
if RSI > 60 and prev_RSI <= 60: sell_score += 2
if MACD < MACD_Signal and prev_MACD >= prev_Signal: sell_score += 3
if price >= BB_Upper * 0.99: sell_score += 2
if price < Resistance * 0.98: sell_score += 1
```

**Weighted Multi-Timeframe Analysis:**
```
weights = {'1mo': 3, '1wk': 2.5, '1d': 2, '4h': 1.5, '1h': 1}

weighted_buy_score = sum(score[tf] * weights[tf] for tf in timeframes)
weighted_sell_score = sum(score[tf] * weights[tf] for tf in timeframes)

# Action determination
if weighted_buy >= 8 and weighted_buy > weighted_sell:
    action = "STRONG BUY"
    confidence = min(0.9, 0.6 + (weighted_buy - 8) * 0.05)
elif weighted_buy >= 5 and weighted_buy > weighted_sell:
    action = "BUY"
    confidence = min(0.8, 0.5 + (weighted_buy - 5) * 0.1)
```

### Risk Management Mathematical Functions

#### Position Sizing Calculation
```
risk_amount = account_balance * max_risk_per_trade  # 2% default
pip_value = |entry_price - stop_loss_price|
pip_size = 0.01 if "JPY" in symbol else 0.0001
pips_at_risk = pip_value / pip_size
lot_size = risk_amount / (pips_at_risk * 10)  # $10 per pip
position_size = min(lot_size, 1.0)  # Max 1 standard lot
```

#### Stop Loss & Take Profit
```
# ATR-based stop loss
ATR = average_true_range(20_periods)
if action == "BUY":
    stop_loss = entry_price - (ATR * 2.0)
    take_profit = entry_price + (risk * risk_reward_ratio)
else:
    stop_loss = entry_price + (ATR * 2.0)
    take_profit = entry_price - (risk * risk_reward_ratio)

risk = |entry_price - stop_loss|
risk_reward_ratio = 2.0  # Default 1:2 ratio
```

#### Daily Loss Limit Check
```
daily_pnl = sum(position.pnl for position in today_positions)
daily_loss_pct = abs(daily_pnl) / account_balance
within_limit = daily_loss_pct < max_daily_loss  # 5% default
```

### Dashboard Components

#### 1. Live Dashboard (`dashboard/live_dashboard.py`)
- **Real-time Updates**: 15-minute refresh cycle
- **Multi-pair Monitoring**: Simultaneous analysis of major pairs
- **AI Integration**: Live trading advice display
- **Chart Generation**: Automated plot creation and display

#### 2. Plotting System (`utils/plotting.py`)
- **Live Charts**: Real-time price and indicator visualization
- **Performance Plots**: Accuracy and error trend analysis
- **Export Functions**: PNG file generation with timestamps

### Utility Components

#### 1. Configuration (`utils/config.py`)
- **Environment Management**: API key and setting handling
- **Database Configuration**: Connection string management
- **Default Parameters**: Trading and analysis settings

#### 2. Performance Tracker (`core/performance.py`)
- **Accuracy Calculation**: Prediction vs actual price comparison
- **Trend Analysis**: Performance improvement over time
- **Reporting**: Detailed performance statistics

## Database Schema

### ForexAnalysis Table
- `id`: Primary key
- `symbol`: Currency pair (e.g., EURUSD=X)
- `timestamp`: Analysis time
- `price`: Current price at analysis
- `recommendation`: BUY/SELL/HOLD decision
- `confidence`: Prediction confidence score
- `ml_prediction`: Machine learning prediction
- `buy_probability`: Probability of price increase
- `sell_probability`: Probability of price decrease
- `analysis_data`: Complete analysis JSON

### ForexPerformance Table
- `id`: Primary key
- `analysis_id`: Reference to analysis record
- `symbol`: Currency pair
- `predicted_price`: AI predicted price
- `actual_price`: Actual market price
- `prediction_accuracy`: Accuracy score (0-1)
- `recommendation_correct`: Boolean correctness
- `days_elapsed`: Time since prediction
- `created_at`: Performance record timestamp

## Supported Currency Pairs

- **EURUSD=X**: Euro/US Dollar
- **GBPUSD=X**: British Pound/US Dollar
- **USDJPY=X**: US Dollar/Japanese Yen
- **AUDUSD=X**: Australian Dollar/US Dollar
- **USDCHF=X**: US Dollar/Swiss Franc
- **USDCAD=X**: US Dollar/Canadian Dollar

## Performance Metrics

### Accuracy Measures
- **Directional Accuracy**: Percentage of correct BUY/SELL predictions
- **Price Prediction Error**: Mean absolute error between predicted and actual prices
- **Confidence Calibration**: Alignment between confidence scores and actual accuracy
- **Recommendation Success**: Percentage of profitable recommendations

### Learning System
- **Historical Analysis**: 60 days of data for accuracy calculation
- **Error Correction**: MAE-based prediction adjustment
- **Confidence Tuning**: Dynamic confidence score calibration
- **Model Improvement**: Continuous learning from prediction outcomes

## File Structure
```
forex_trading_module/
├── core/
│   ├── ai_agent.py          # AI trading agent
│   ├── database.py          # Database operations
│   ├── market_advisor.py    # Multi-timeframe advisor
│   ├── performance.py       # Performance tracking
│   └── trading_engine.py    # Main trading engine
├── dashboard/
│   └── live_dashboard.py    # Real-time dashboard
├── utils/
│   ├── config.py           # Configuration management
│   └── plotting.py         # Chart generation
├── main.py                 # CLI entry point
├── database_analyzer.py    # Database visualization tool
├── .env                    # Environment variables
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source. Please contact the developer for commercial use inquiries.

## Contact

**Developer**: James Kariuki  
**Email**: jamexkarix583@gmail.com  
**Phone**: 0792698424

For support, feature requests, or collaboration opportunities, please reach out via email or phone.