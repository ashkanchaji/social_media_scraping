import os
import re
import time
import random
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, TimeoutException
from transformers import pipeline

import dedup_utils
from pipeline_utils import setup_logging, with_retry, RateLimiter, detect_language

logger = setup_logging("twtr-scraper", log_file="twtr-scraper.log")

MAX_RESULTS_PER_KEYWORD = 100
MAX_PAST_DAYS = 30

FETCH_MAX_WORKERS = 6          # concurrent tweet fetches
FETCH_MIN_INTERVAL = 0.35      # seconds between fetch calls, enforced globally across all workers
DDGS_DELAY_RANGE = (5.0, 10.0)

# English-only filter: FinBERT (ProsusAI/finbert) is trained on English
# financial text. Running it on Japanese/Chinese/etc. text doesn't error --
# it just produces meaningless sentiment scores. Filtering here avoids
# polluting the dataset with garbage sentiment labels.
LANGUAGE_MIN_CONFIDENCE = 0.70

_fetch_rate_limiter = RateLimiter(FETCH_MIN_INTERVAL)

try:
    from xtf import Router, NotFound, RateLimited
except ImportError:
    logger.error("x-tweet-fetcher is not installed.")
    exit(1)

TWEET_URL_RE = re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/(\d+)")


def extract_tweet_ref(url: str):
    m = TWEET_URL_RE.search(url)
    if not m:
        return None
    return m.group(1), m.group(2)


def discover_tweet_urls(keyword: str, max_results: int = 15) -> list:
    logger.info(f"Searching for: {keyword}...")
    trusted_accounts = ["Reuters", "business", "CNBC", "BBCWorld", "CNN", "AJEnglish"]
    account_filter = " OR ".join(f"from:{a}" for a in trusted_accounts)
    query = f"site:x.com {keyword} ({account_filter})"

    found_urls = set()
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region="us-en", safesearch="off", timelimit="w", max_results=max_results * 3)
    except RatelimitException:
        logger.warning(f"DDGS rate-limited on '{keyword}'. Skipping this keyword for this run.")
        results = []
    except TimeoutException:
        logger.warning(f"DDGS timed out on '{keyword}'. Skipping this keyword for this run.")
        results = []
    except Exception as e:
        logger.warning(f"DDGS error on '{keyword}': {str(e).splitlines()[0]}")
        results = []
    finally:
        time.sleep(random.uniform(*DDGS_DELAY_RANGE))

    for r in results:
        ref = extract_tweet_ref(r.get("href", ""))
        if ref:
            found_urls.add((ref[0], ref[1]))

    final_refs = list(found_urls)[:max_results]
    logger.info(f"Found {len(final_refs)} candidate posts for '{keyword}'.")
    return final_refs


@with_retry(max_attempts=3, base_delay=3.0, exceptions=(RateLimited, ConnectionError, TimeoutError))
def _fetch_tweet_safe(username: str, tweet_id: str) -> dict:
    """
    Fetches one tweet. Instantiates its own Router rather than sharing one
    across threads -- x-tweet-fetcher's docs don't say whether Router
    holds thread-unsafe internal state (session/backend fallback tracking),
    so a fresh instance per call sidesteps the question entirely at low
    cost (Router() construction is just config, not a network call).
    _fetch_rate_limiter enforces the actual pacing globally across all
    worker threads so concurrency doesn't turn into a burst.
    """
    _fetch_rate_limiter.wait()
    router = Router()
    return router.fetch_tweet(username, tweet_id)


def _parse_created_at(created_raw):
    if not created_raw:
        return None, created_raw or "N/A"
    for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.datetime.strptime(created_raw, fmt)
            return dt, dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, TypeError):
            continue
    return None, created_raw


def search_and_scrape_tweets(keyword: str, lsh, hash_by_id, run_timestamp: str, output_dir: str = "twtr-tweets/"):
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.datetime.now(datetime.timezone.utc)
    cutoff = today - datetime.timedelta(days=MAX_PAST_DAYS)

    refs = discover_tweet_urls(keyword, max_results=MAX_RESULTS_PER_KEYWORD)
    if not refs:
        logger.info(f"No candidate posts found for '{keyword}'.")
        return None

    # --- Concurrent fetch stage (I/O-bound, benefits from threads despite the GIL) ---
    fetched = []
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        future_to_ref = {executor.submit(_fetch_tweet_safe, u, tid): (u, tid) for u, tid in refs}
        for future in as_completed(future_to_ref):
            username, tweet_id = future_to_ref[future]
            try:
                tw = future.result()
            except NotFound:
                continue
            except Exception as e:
                logger.warning(f"Giving up on tweet {tweet_id} after retries: {type(e).__name__}: {e}")
                continue
            if tw:
                fetched.append((username, tweet_id, tw))

    # --- Sequential processing stage (language filter, dedup DB writes) ---
    tweet_data = []
    skipped_non_english = 0

    for username, tweet_id, tw in fetched:
        text = tw.get("text") or tw.get("full_text") or ""
        if not text.strip():
            continue

        created_dt, created_str = _parse_created_at(tw.get("created_at") or tw.get("timestamp"))
        if created_dt is not None and created_dt < cutoff:
            continue

        clean_text = text.replace("\n", " ")

        lang, confidence = detect_language(clean_text)
        if lang != "en" or confidence < LANGUAGE_MIN_CONFIDENCE:
            skipped_non_english += 1
            continue

        author_field = tw.get("author")
        author_val = author_field.get("screen_name", username) if isinstance(author_field, dict) else (author_field or username)

        collected_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        record_id = f"X_{author_val}_{tweet_id}"

        duplication_info = dedup_utils.check_and_register(record_id, clean_text, lsh, hash_by_id)

        record = {
            "record_id": record_id,
            "Source": {
                "platform": "X",
                "Source_type": "Social_media",
                "Source_name": author_val,
                "Source_id": author_val,
                "url": f"https://x.com/{author_val}/status/{tweet_id}",
                "author_name": author_val,
                "author_id": author_val
            },
            "Content": {
                "Content_type": "tweet",
                "title": "N/A",
                "raw_text": clean_text,
                "Clean_text": clean_text,
                "language": lang
            },
            "time_stamps": {
                "published_at": created_str,
                "collected_at": collected_at,
                "updated_at": collected_at
            },
            "asset_mention": [{
                "mentioned_text": None,
                "canonical_name": None,
                "Symbol": None,
                "asset_class": None,
                "asset_id": None,
                "confidence": None
            }],
            "Engagement": {
                "Views": tw.get("views") or tw.get("view_count") or 0,
                "likes": tw.get("likes") or tw.get("like_count") or 0,
                "Comments": tw.get("replies") or tw.get("reply_count") or 0,
                "shares": tw.get("retweets") or tw.get("retweet_count") or 0
            },
            "media": {
                "has_media": False,
                "media_type": None,
                "media_url": None
            },
            "deduplication": duplication_info
        }
        tweet_data.append(record)

    if skipped_non_english:
        logger.info(f"Filtered out {skipped_non_english} non-English post(s) for '{keyword}'.")

    if not tweet_data:
        logger.info(f"No recent, fetchable, English-language tweets found for '{keyword}'.")
        return None

    safe_keyword = keyword.replace(" ", "_").lower()
    # Timestamped per-run filename (not one-file-per-keyword-forever): this
    # matters for continuous/day-and-night operation, since a fixed
    # keyword->filename mapping means the search for that keyword would
    # only ever run ONCE -- the old code checked "does this file already
    # exist?" and skipped scraping entirely on every later run, forever.
    output_filename = os.path.join(output_dir, f"twitter_{safe_keyword}_{run_timestamp}.json")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(tweet_data, f, indent=4, ensure_ascii=False)

    logger.info(f"Found {len(tweet_data)} reliable, English-language tweets for '{keyword}' from the past {MAX_PAST_DAYS * 24} hours.")
    return output_filename


def analyze_tweet_sentiment(json_filename: str, sentiment_analyzer, output_dir: str = "sentiments/twitter/"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for row in data:
        text = str(row['Content']['raw_text'])
        if not text.strip():
            continue

        prediction = sentiment_analyzer(text, truncation=True, max_length=512)[0]

        row['Sentiment'] = {
            "sentiment": prediction['label'].upper(),
            "confidence": round(prediction['score'], 3)
        }
        results.append(row)

    base_filename = os.path.basename(json_filename).replace(".json", "_sentiment.json")
    output_filename = os.path.join(output_dir, base_filename)

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    keyword_file = "twtr-keywords.txt"
    if not os.path.exists(keyword_file):
        logger.error(f"File '{keyword_file}' not found.")
        return

    with open(keyword_file, 'r') as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        logger.error(f"No keywords found in {keyword_file}.")
        return

    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logger.info(f"Starting scraping run {run_timestamp} for X (Twitter)...")

    lsh, hash_by_id = dedup_utils.load_lsh()  # shared cross-platform dedup index
    analysis_queue = []

    for keyword in keywords:
        try:
            scraped_file = search_and_scrape_tweets(keyword, lsh, hash_by_id, run_timestamp)
            if scraped_file:
                analysis_queue.append(scraped_file)
        except Exception as e:
            logger.error(f"Error executing search/scrape for keyword '{keyword}': {str(e).splitlines()[0]}")

    logger.info("Scraping for all keywords is done.")
    if not analysis_queue:
        logger.info("No new tweets to analyze.")
        return

    logger.info("Starting analyzing scraped tweet data...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    for tweet_file in analysis_queue:
        try:
            analyze_tweet_sentiment(tweet_file, sentiment_analyzer)
        except Exception as e:
            logger.error(f"Error analyzing {tweet_file}: {str(e).splitlines()[0]}")

    logger.info("Analyzing for all tweets is done.")


if __name__ == "__main__":
    main()
