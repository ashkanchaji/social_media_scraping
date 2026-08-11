import yfinance as yf
import pandas as pd

def fetch_commodity_prices(ticker_symbol: str, period: str = "7d", interval: str = "1h"):
    print(f"Fetching market data for '{ticker_symbol}'...")
    
    # 1. Initialize ticker
    ticker = yf.Ticker(ticker_symbol)
    
    # 2. Download historical data (e.g., last 7 days, hourly intervals)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        print("No data found. Check your ticker symbol or time interval.")
        return None
    
    # 3. Calculate price changes (% return over each period)
    df['Price_Change_%'] = df['Close'].pct_change() * 100
    
    # Clean up columns
    df = df.reset_index()
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change_%']]
    
    # Save to CSV
    filename = f"market-data/{ticker_symbol.replace('=', '_')}_market_data.csv"
    df.to_csv(filename, index=False)
    
    print(f"Successfully fetched {len(df)} price points!")
    print("-" * 50)
    print(df.tail(5))  # Preview recent prices
    print(f"\nSaved market data to '{filename}'")
    
    return df

if __name__ == "__main__":
    # Fetch 7 days of hourly Crude Oil futures data
    oil_df = fetch_commodity_prices("CL=F", period="7d", interval="1h")