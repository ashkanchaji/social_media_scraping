import pandas as pd
from datetime import datetime, timedelta

def align_sentiment_and_price(sentiment_csv, price_csv, video_publish_time):
    print("Loading datasets...")
    
    # 1. Load the data
    sentiment_df = pd.read_csv(sentiment_csv)
    price_df = pd.read_csv(price_csv)
    
    # 2. Average the sentiment for the whole video
    # Convert FinBERT labels to numbers (Positive = 1, Neutral = 0, Negative = -1)
    def score_sentiment(row):
        label = str(row['sentiment']).upper()
        if label == 'POSITIVE': 
            return row['confidence']
        elif label == 'NEGATIVE': 
            return -row['confidence']
        return 0.0  # Neutral
        
    sentiment_df['numeric_score'] = sentiment_df.apply(score_sentiment, axis=1)
    average_sentiment = sentiment_df['numeric_score'].mean()
    
    print(f"Overall Video Sentiment Score: {average_sentiment:.2f}")
    
    # 3. Format the timestamps
    # Convert video publish time to a Pandas datetime object, rounded to the nearest hour
    publish_dt = pd.to_datetime(video_publish_time).floor('h')
    
    # Ensure the price dates are also timezone-aware datetime objects
    price_df['Datetime'] = pd.to_datetime(price_df['Datetime'], utc=True)
    publish_dt = publish_dt.tz_localize('UTC')
    
    # 4. Find the baseline price (Price at the time the video came out)
    baseline_row = price_df[price_df['Datetime'] == publish_dt]
    
    if baseline_row.empty:
        print("Warning: Market was closed at this exact hour, or data is missing.")
        # Fallback: Get the closest available price
        price_df = price_df.sort_values('Datetime')
        baseline_row = price_df[price_df['Datetime'] >= publish_dt].head(1)
        
    # 5. Find the Target Price (e.g., 4 hours later)
    target_dt = baseline_row['Datetime'].iloc[0] + pd.Timedelta(hours=4)
    target_row = price_df[price_df['Datetime'] >= target_dt].head(1)
    
    # 6. Build the Final Training Data Row
    baseline_price = baseline_row['Close'].iloc[0]
    target_price = target_row['Close'].iloc[0]
    price_change = target_price - baseline_price
    
    # Target: 1 if price went up, 0 if it went down
    target_label = 1 if price_change > 0 else 0 
    
    final_data = {
        "timestamp": publish_dt,
        "sentiment_score": average_sentiment,
        "baseline_price": baseline_price,
        "price_4h_later": target_price,
        "target_label": target_label
    }
    
    final_df = pd.DataFrame([final_data])
    print("\n--- FINAL ALIGNED DATASET ---")
    print(final_df.to_string())
    
    # In a real project, you would append this row to a master CSV file
    final_df.to_csv("master_training_data.csv", mode='a', header=not pd.io.common.file_exists("master_training_data.csv"), index=False)
    print("\nAdded to master training dataset!")

if __name__ == "__main__":
    # The time the video was published (15 hours ago). Adjust this to the exact timestamp!
    # Example: If current time is Aug 10, 2026, 6:00 AM, 15 hours ago was Aug 9, 2026, 3:00 PM.
    exact_publish_time = "2026-08-09 22:00:00" 
    
    align_sentiment_and_price(
        sentiment_csv="youtube_FTh9D13JVEM_transcript_sentiment.csv",
        price_csv="CL_F_market_data.csv",
        video_publish_time=exact_publish_time
    )