# USD to CAD Exchange Rate Prediction System
# This notebook combines sentiment analysis of political news with financial data
# to predict optimal USD to CAD conversion timing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import requests
import json
from textblob import TextBlob
import yfinance as yf

# Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# News API and sentiment analysis
import feedparser
from transformers import pipeline

print("=== USD to CAD Exchange Rate Prediction System ===")
print("Setting up environment...")

# Step 1: Data Collection Functions
class NewsAnalyzer:
    """Collects and analyzes political news related to US-Canada relations"""
    
    def __init__(self):
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except:
            print("Using TextBlob for sentiment analysis as backup")
            self.sentiment_analyzer = None
    
    def get_canada_us_news(self, days_back=30):
        """Fetch news related to Canada-US relations and tariffs"""
        news_data = []
        
        # RSS feeds for Canadian and US political/economic news
        feeds = [
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://www.cbc.ca/cmlink/rss-topstories',
            'https://globalnews.ca/feed/',
        ]
        
        keywords = ['canada', 'tariff', 'trade', 'usd', 'cad', 'dollar', 'trump', 'trudeau']
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Limit to recent entries
                    title = entry.title.lower()
                    summary = getattr(entry, 'summary', '').lower()
                    
                    # Check if article contains relevant keywords
                    if any(keyword in title or keyword in summary for keyword in keywords):
                        news_data.append({
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'published': getattr(entry, 'published', ''),
                            'link': getattr(entry, 'link', ''),
                            'source': feed_url
                        })
            except Exception as e:
                print(f"Error fetching from {feed_url}: {e}")
        
        return news_data
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of given text"""
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(text[:512])  # Limit text length
                return {
                    'label': result[0]['label'],
                    'score': result[0]['score']
                }
            except:
                pass
        
        # Fallback to TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return {'label': 'POSITIVE', 'score': polarity}
        elif polarity < -0.1:
            return {'label': 'NEGATIVE', 'score': abs(polarity)}
        else:
            return {'label': 'NEUTRAL', 'score': abs(polarity)}
    
    def get_sentiment_scores(self, news_data):
        """Process news data and extract sentiment scores"""
        sentiment_scores = []
        
        for article in news_data:
            text = f"{article['title']} {article['summary']}"
            sentiment = self.analyze_sentiment(text)
            
            sentiment_scores.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'title': article['title'],
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'source': article['source']
            })
        
        return pd.DataFrame(sentiment_scores)

class ExchangeRateCollector:
    """Collects historical USD/CAD exchange rate data"""
    
    def get_exchange_rate_data(self, start_date, end_date):
        """Fetch USD/CAD exchange rate data"""
        try:
            # Using Yahoo Finance for CAD=X (USD/CAD)
            ticker = yf.Ticker("USDCAD=X")
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                # Alternative: use Federal Reserve Economic Data API or manual data
                print("Yahoo Finance data not available, generating sample data...")
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Generate realistic USD/CAD data around 1.35 with some volatility
                np.random.seed(42)
                base_rate = 1.35
                returns = np.random.normal(0, 0.01, len(dates))
                prices = [base_rate]
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                data = pd.DataFrame({
                    'Open': prices,
                    'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                    'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                    'Close': prices,
                    'Volume': np.random.randint(1000000, 5000000, len(dates))
                }, index=dates)
            
            return data
        
        except Exception as e:
            print(f"Error fetching exchange rate data: {e}")
            return pd.DataFrame()

# Step 2: Feature Engineering
class FeatureEngineering:
    """Creates features for machine learning model"""
    
    def create_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_3d'] = df['Close'].pct_change(periods=3)
        df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
        
        return df
    
    def add_sentiment_features(self, exchange_df, sentiment_df):
        """Add sentiment features to exchange rate data"""
        # Aggregate daily sentiment scores
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).flatten_cols()
        
        daily_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count']
        daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
        
        # Merge with exchange rate data
        exchange_df = exchange_df.join(daily_sentiment, how='left')
        
        # Forward fill missing sentiment data
        exchange_df[['sentiment_mean', 'sentiment_std', 'sentiment_count']] = \
            exchange_df[['sentiment_mean', 'sentiment_std', 'sentiment_count']].fillna(method='ffill')
        
        return exchange_df

# Step 3: Machine Learning Model
class ExchangeRatePredictor:
    """Machine learning model for predicting exchange rates"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_columns = None
    
    def prepare_features(self, df):
        """Prepare features for machine learning"""
        feature_cols = [
            'Open', 'High', 'Low', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'RSI',
            'BB_upper', 'BB_lower', 'Volatility',
            'Price_Change', 'Price_Change_3d', 'Price_Change_7d',
            'sentiment_mean', 'sentiment_std', 'sentiment_count'
        ]
        
        # Remove rows with NaN values
        df_clean = df[feature_cols + ['Close']].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean['Close']
        
        self.feature_columns = feature_cols
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = float('-inf')
        results = {}
        
        for name, model in self.models.items():
            # Train model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate model
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'model': model
            }
            
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with R2: {best_score:.4f}")
        return results
    
    def predict_future_rates(self, last_features, days=90):
        """Predict future exchange rates"""
        predictions = []
        current_features = last_features.copy()
        
        for day in range(days):
            if self.best_model_name == 'Linear Regression':
                features_scaled = self.scaler.transform([current_features])
                pred = self.best_model.predict(features_scaled)[0]
            else:
                pred = self.best_model.predict([current_features])[0]
            
            predictions.append(pred)
            
            # Update features for next prediction (simplified approach)
            # In practice, you'd want more sophisticated feature updating
            current_features[0] = pred  # Update 'Open' with predicted close
            
        return predictions

# Step 4: Main Execution Pipeline
def main_pipeline():
    """Execute the complete pipeline"""
    print("\n=== Step 1: Data Collection ===")
    
    # Initialize components
    news_analyzer = NewsAnalyzer()
    exchange_collector = ExchangeRateCollector()
    feature_engineer = FeatureEngineering()
    predictor = ExchangeRatePredictor()
    
    # Collect news data
    print("Collecting news data...")
    news_data = news_analyzer.get_canada_us_news()
    print(f"Collected {len(news_data)} relevant news articles")
    
    # Analyze sentiment
    print("Analyzing sentiment...")
    sentiment_df = news_analyzer.get_sentiment_scores(news_data)
    print(f"Processed sentiment for {len(sentiment_df)} articles")
    
    # Display sentiment summary
    if not sentiment_df.empty:
        sentiment_summary = sentiment_df['sentiment_label'].value_counts()
        print("Sentiment Distribution:")
        print(sentiment_summary)
    
    # Collect exchange rate data
    print("\n=== Step 2: Exchange Rate Data Collection ===")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    exchange_df = exchange_collector.get_exchange_rate_data(start_date, end_date)
    print(f"Collected {len(exchange_df)} days of exchange rate data")
    
    # Feature engineering
    print("\n=== Step 3: Feature Engineering ===")
    exchange_df = feature_engineer.create_technical_indicators(exchange_df)
    
    if not sentiment_df.empty:
        exchange_df = feature_engineer.add_sentiment_features(exchange_df, sentiment_df)
    else:
        # Add dummy sentiment features if no news data available
        exchange_df['sentiment_mean'] = 0
        exchange_df['sentiment_std'] = 0
        exchange_df['sentiment_count'] = 0
    
    print("Features created successfully")
    
    # Prepare data for ML
    print("\n=== Step 4: Machine Learning Training ===")
    X, y = predictor.prepare_features(exchange_df)
    print(f"Training data shape: {X.shape}")
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Make predictions
    print("\n=== Step 5: Future Predictions ===")
    last_features = X.iloc[-1].values
    future_rates = predictor.predict_future_rates(last_features, days=90)
    
    # Create prediction dates
    last_date = exchange_df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(90)]
    
    # Display results
    print("\n=== Results Summary ===")
    current_rate = exchange_df['Close'].iloc[-1]
    avg_future_rate = np.mean(future_rates)
    
    print(f"Current USD/CAD rate: {current_rate:.4f}")
    print(f"Predicted average rate (next 3 months): {avg_future_rate:.4f}")
    
    # Find best conversion opportunities
    min_rate_idx = np.argmin(future_rates)
    max_rate_idx = np.argmax(future_rates)
    
    print(f"\nBest time to convert USD to CAD (lowest rate):")
    print(f"Date: {future_dates[min_rate_idx].strftime('%Y-%m-%d')}")
    print(f"Predicted rate: {future_rates[min_rate_idx]:.4f}")
    
    print(f"\nWorst time to convert USD to CAD (highest rate):")
    print(f"Date: {future_dates[max_rate_idx].strftime('%Y-%m-%d')}")
    print(f"Predicted rate: {future_rates[max_rate_idx]:.4f}")
    
    # Plotting
    print("\n=== Generating Visualizations ===")
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Historical rates and predictions
    plt.subplot(2, 2, 1)
    plt.plot(exchange_df.index[-60:], exchange_df['Close'].iloc[-60:], 
             label='Historical Rates', color='blue')
    plt.plot(future_dates, future_rates, 
             label='Predicted Rates', color='red', linestyle='--')
    plt.title('USD/CAD Exchange Rate Prediction')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Sentiment over time (if available)
    plt.subplot(2, 2, 2)
    if not sentiment_df.empty:
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('News Sentiment Distribution')
    else:
        plt.text(0.5, 0.5, 'No sentiment data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('News Sentiment Distribution')
    
    # Plot 3: Feature importance (for tree-based models)
    plt.subplot(2, 2, 3)
    if hasattr(predictor.best_model, 'feature_importances_'):
        importances = predictor.best_model.feature_importances_
        features = predictor.feature_columns
        indices = np.argsort(importances)[::-1][:10]
        
        plt.bar(range(len(indices)), importances[indices])
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45)
    else:
        plt.text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')
    
    # Plot 4: Prediction confidence intervals
    plt.subplot(2, 2, 4)
    # Create confidence intervals (simplified approach)
    std_dev = np.std(future_rates)
    upper_bound = np.array(future_rates) + 2 * std_dev
    lower_bound = np.array(future_rates) - 2 * std_dev
    
    plt.fill_between(future_dates, lower_bound, upper_bound, alpha=0.3, color='gray')
    plt.plot(future_dates, future_rates, color='red', linewidth=2)
    plt.title('Prediction with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'current_rate': current_rate,
        'future_predictions': list(zip(future_dates, future_rates)),
        'best_conversion_date': future_dates[min_rate_idx],
        'best_conversion_rate': future_rates[min_rate_idx],
        'sentiment_summary': sentiment_df['sentiment_label'].value_counts().to_dict() if not sentiment_df.empty else {},
        'model_performance': results
    }

# Step 5: Installation and Setup Instructions
print("""
=== INSTALLATION INSTRUCTIONS ===

Before running this notebook, install the required packages:

pip install pandas numpy matplotlib seaborn
pip install requests feedparser textblob yfinance
pip install scikit-learn transformers torch
pip install beautifulsoup4 lxml

For sentiment analysis, you may also need:
python -m textblob.download_corpora

=== USAGE INSTRUCTIONS ===

1. Run all cells in order
2. The system will automatically:
   - Collect news about US-Canada relations
   - Analyze sentiment of political/economic news
   - Fetch historical USD/CAD exchange rates
   - Train machine learning models
   - Predict future exchange rates
   - Identify optimal conversion timing

3. Results will include:
   - Current exchange rate
   - 3-month predictions
   - Best dates for USD to CAD conversion
   - Visualization of trends and predictions

=== CUSTOMIZATION OPTIONS ===

You can modify:
- News sources (add more RSS feeds)
- Prediction horizon (change from 90 days)
- Model parameters
- Feature engineering approaches
- Sentiment analysis models

""")

# Execute the main pipeline
if __name__ == "__main__":
    results = main_pipeline()
    
    print("\n=== FINAL RECOMMENDATIONS ===")
    print(f"Based on current analysis:")
    print(f"Current Rate: {results['current_rate']:.4f}")
    print(f"Recommended conversion date: {results['best_conversion_date'].strftime('%Y-%m-%d')}")
    print(f"Expected rate on that date: {results['best_conversion_rate']:.4f}")
    
    if results['best_conversion_rate'] < results['current_rate']:
        savings = (results['current_rate'] - results['best_conversion_rate']) / results['current_rate'] * 100
        print(f"Potential savings: {savings:.2f}% by waiting")
    else:
        print("Consider converting now as rates may increase")
