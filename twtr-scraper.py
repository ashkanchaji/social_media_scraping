import os
import re
import time
import random
import pandas as pd
import datetime
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, TimeoutException
from transformers import pipeline

# Pacing: neither DDGS nor FxTwitter guarantee immunity from rate limits --
# bursty automated traffic is what gets flagged, on any free/no-key service.
# These add small randomized delays between calls instead of firing back-to-back.
DDGS_DELAY_RANGE = (2.0, 4.0)      # seconds between search queries
FETCH_DELAY_RANGE = (0.5, 1.5)     # seconds between individual tweet fetches

MAX_RESULTS_PER_KEYWORD = 100
MAX_PAST_DAYS = 30

try:
    from xtf import Router, NotFound, RateLimited
except ImportError:
    print("Error: x-tweet-fetcher is not installed.")
    print("Run: git clone https://github.com/ythx-101/x-tweet-fetcher && cd x-tweet-fetcher && pip install .")
    exit(1)

# -----------------------------------------------------------------------------
# Discovery uses ddgs.
# The FETCH step below genuinely uses x-tweet-fetcher's own code:
# Router.fetch_tweet() is zero-dependency (fxtwitter backend)
# -----------------------------------------------------------------------------

TWEET_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/(\d+)"
)


def extract_tweet_ref(url: str):
    m = TWEET_URL_RE.search(url)
    if not m:
        return None
    return m.group(1), m.group(2)


def discover_tweet_urls(keyword: str, max_results: int = 15) -> list:
    """Discovers candidate tweet URLs via DuckDuckGo (free, no key, no login)."""
    print(f"  -> Searching for: {keyword}...")

    trusted_accounts = ["Reuters", "business", "CNBC", "BBCWorld", "CNN", "AJEnglish"]
    account_filter = " OR ".join(f"from:{a}" for a in trusted_accounts)
    query = f"site:x.com {keyword} ({account_filter})"

    found_urls = set()
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query, region="us-en", safesearch="off",
                timelimit="w", max_results=max_results * 3,
            )
    except RatelimitException:
        print("     -> DDGS rate-limited. Skipping this keyword for this run "
              "(consider spacing runs further apart, or lowering DDGS_DELAY_RANGE frequency).")
        results = []
    except TimeoutException:
        print("     -> DDGS timed out, skipping this keyword for this run.")
        results = []
    except Exception as e:
        print(f"     Error during DDGS search: {str(e).splitlines()[0]}")
        results = []
    finally:
        # Pace ourselves before the *next* keyword's search, regardless of outcome.
        time.sleep(random.uniform(*DDGS_DELAY_RANGE))

    for r in results:
        ref = extract_tweet_ref(r.get("href", ""))
        if ref:
            found_urls.add((ref[0], ref[1]))

    final_refs = list(found_urls)[:max_results]
    print(f"  -> Found {len(final_refs)} candidate posts.")
    return final_refs


def search_and_scrape_tweets(keyword: str, router: "Router", output_dir: str = "twtr-tweets/"):
    """
    Discovers (username, tweet_id) pairs via ddgs, then fetches each via
    xtf's Router.fetch_tweet() -- the repo's own zero-dependency fxtwitter backend.
    """
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.datetime.now(datetime.timezone.utc)
    cutoff = today - datetime.timedelta(days=MAX_PAST_DAYS)

    refs = discover_tweet_urls(keyword, max_results=MAX_RESULTS_PER_KEYWORD)
    if not refs:
        print(f"     -> No candidate posts found for '{keyword}'.")
        return None

    tweet_data = []
    for username, tweet_id in refs:
        time.sleep(random.uniform(*FETCH_DELAY_RANGE))  # pace individual fetches

        try:
            tw = router.fetch_tweet(username, tweet_id)
        except NotFound:
            continue
        except RateLimited:
            print(f"     -> Rate limited fetching {tweet_id}. Backing off 30s and skipping it for this run.")
            time.sleep(30)
            continue
        except Exception as e:
            print(f"     -> Skipping {tweet_id}: {type(e).__name__}: {str(e).splitlines()[0]}")
            continue

        if not tw:
            continue

        text = tw.get("text") or tw.get("full_text") or ""
        if not text.strip():
            continue

        created_raw = tw.get("created_at") or tw.get("timestamp")
        created_dt = None
        created_str = created_raw or "N/A"
        if created_raw:
            for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    created_dt = datetime.datetime.strptime(created_raw, fmt)
                    created_str = created_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
                    break
                except (ValueError, TypeError):
                    continue

        if created_dt is not None and created_dt < cutoff:
            continue

        author_field = tw.get("author")
        if isinstance(author_field, dict):
            author_val = author_field.get("screen_name", username)
        else:
            author_val = author_field or username

        tweet_data.append({
            "tweet_id": tw.get("id") or tw.get("tweet_id") or tweet_id,
            "text": text.replace("\n", " "),
            "timestamp": created_str,
            "author": author_val,
            "likes": tw.get("likes") or tw.get("like_count") or 0,
            "retweets": tw.get("retweets") or tw.get("retweet_count") or 0,
            "reply_count": tw.get("replies") or tw.get("reply_count") or 0,
            "views": tw.get("views") or tw.get("view_count") or 0,
        })

    if not tweet_data:
        print(f"     -> No recent, fetchable tweets found for '{keyword}'.")
        return None

    df = pd.DataFrame(tweet_data)
    safe_keyword = keyword.replace(" ", "_").lower()
    output_filename = os.path.join(output_dir, f"twitter_{safe_keyword}.csv")
    df.to_csv(output_filename, index=False)

    print(f"  -> Found {len(tweet_data)} reliable, high-engagement tweets from the past {MAX_PAST_DAYS * 24} hours.")
    return output_filename


def analyze_tweet_sentiment(csv_filename: str, search_keyword: str, sentiment_analyzer, output_dir: str = "sentiments/twitter/"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_filename)

    results = []
    for idx, row in df.iterrows():
        text = str(row['text'])
        if not text.strip():
            continue

        prediction = sentiment_analyzer(text, truncation=True, max_length=512)[0]

        results.append({
            "tweet_id": row['tweet_id'],
            "text": text, # Replaced text_snippet to keep the full unmodified text
            "sentiment": prediction['label'].upper(),
            "confidence": round(prediction['score'], 3),
            "search_keyword": search_keyword,
            "author": row['author'],
            "timestamp": row['timestamp'],
            "likes": row['likes'],
            "retweets": row['retweets'],
            "reply_count": row.get('reply_count', 0),
            "views": row.get('views', 0)
        })

    results_df = pd.DataFrame(results)
    base_filename = os.path.basename(csv_filename).replace(".csv", "_sentiment.csv")
    output_filename = os.path.join(output_dir, base_filename)
    results_df.to_csv(output_filename, index=False)


def main():
    keyword_file = "twtr-keywords.txt"

    if not os.path.exists(keyword_file):
        print(f"File '{keyword_file}' not found.")
        return

    with open(keyword_file, 'r') as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        print(f"No keywords found in {keyword_file}.")
        return

    print("Starting scraping for X (Twitter)...")
    router = Router()  # backend="auto"; fetch_tweet resolves to fxtwitter regardless
    analysis_queue = []

    for keyword in keywords:
        try:
            safe_keyword = keyword.replace(" ", "_").lower()
            tweet_file = os.path.join("twtr-tweets", f"twitter_{safe_keyword}.csv")
            sentiment_file = os.path.join("sentiments/twitter", f"twitter_{safe_keyword}_sentiment.csv")

            if os.path.exists(sentiment_file):
                continue

            if not os.path.exists(tweet_file):
                scraped_file = search_and_scrape_tweets(keyword, router)
                if not scraped_file:
                    continue

            analysis_queue.append((tweet_file, keyword))
        except Exception as e:
            print(f"     Error executing search/scrape for keyword '{keyword}': {str(e).splitlines()[0]}")

    print("Scraping for all keywords is done.")

    if not analysis_queue:
        print("No new tweets to analyze.")
        return

    print("Starting analyzing scraped tweet data...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    for tweet_file, keyword in analysis_queue:
        try:
            analyze_tweet_sentiment(tweet_file, keyword, sentiment_analyzer)
        except Exception as e:
            print(f"Error analyzing {tweet_file}: {str(e).splitlines()[0]}")

    print("Analyzing for all tweets is done.")


if __name__ == "__main__":
    main()