import time
from datetime import datetime
from Scweet import Scweet

print("Initializing Scweet for dynamic real-time market scraping...")

# 1. Initialize with your authenticated cookies file
s = Scweet(cookies_file="cookies.json")

# 2. Dynamically fetch current system date in YYYY-MM-DD format
TODAY_DATE = datetime.now().strftime("%Y-%m-%d")

# List of search queries to monitor
queries = [
    "Oil Iran",
    "Strait of Hormuz",
    "Crude Oil",
    "Israel Iran oil"
]

print(f"Scraping the LATEST live feed dynamically for date: {TODAY_DATE}\n")

for query in queries:
    print(f"--> Fetching real-time tweets for query: '{query}'")
    try:
        s.search(
            query,
            since=TODAY_DATE,       # Dynamically generated current date
            limit=50,
            display_type="Latest",  # Pulls directly from the 'Latest' live tab
            save=True
        )
        print(f"Done with '{query}'. Pausing 3 seconds...\n")
        time.sleep(3)
    except Exception as e:
        print(f"Error scraping '{query}': {e}\n")

print("All done! Fresh real-time CSV files are saved in your 'outputs/' folder.")