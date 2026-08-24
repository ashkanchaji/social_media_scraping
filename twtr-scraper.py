import os
import io
import re
import time
import contextlib
import random
import json
import bisect
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from ddgs import DDGS
from ddgs.exceptions import RatelimitException, TimeoutException

import dedup_utils
from pipeline_utils import (
    setup_logging,
    with_retry,
    RateLimiter,
    detect_language,
    build_quality,
    analyze_text,
    get_scrape_mode,
    env_int,
    env_float,
    env_set,
    env_range,
)

logger = setup_logging("twtr-scraper", log_file="twtr-scraper.log")

# Every constant below is overridable from .env (see .env.example): the bare
# name applies to all scrapers, an "X_"-prefixed name applies to this one only.
MAX_RESULTS_PER_KEYWORD = env_int("MAX_RESULTS_PER_KEYWORD", 100, prefix="X")
MAX_PAST_MINUTES = env_int("MAX_PAST_MINUTES", 30 * 24 * 60, prefix="X")  # default 30 days

# Account-only mode has no keyword to narrow the search, so every trusted
# account's timeline is pulled up to this many recent tweets (xtf's
# fetch_timeline pages nitter search up to 10 pages -- this is that ceiling,
# not an arbitrary cap). ponytail: nitter search has no cursor exposed past
# that page budget, so an account posting more than this within the time
# window will miss the oldest of them; upgrade path is asking xtf for a
# cursor-based fetch_timeline if that ever matters in practice.
ACCOUNT_MODE_FETCH_LIMIT = env_int("ACCOUNT_MODE_FETCH_LIMIT", 200, prefix="X")

FETCH_MAX_WORKERS = env_int("FETCH_MAX_WORKERS", 10, prefix="X")            # concurrent tweet fetches
FETCH_MIN_INTERVAL = env_float("FETCH_MIN_INTERVAL", 0.5, prefix="X")      # seconds between fetch calls, enforced globally across all workers
DDGS_DELAY_RANGE = env_range("DDGS_DELAY_RANGE", (12.0, 18.0), prefix="X")  # Pacing delay between keyword searches
DDGS_MAX_RETRIES = env_int("DDGS_MAX_RETRIES", 3, prefix="X")              # Number of retry attempts on rate limit/timeout

LANGUAGE_MIN_CONFIDENCE = env_float("LANGUAGE_MIN_CONFIDENCE", 0.70, prefix="X")
ALLOWED_LANGUAGES = env_set("ALLOWED_LANGUAGES", {"en", "fa"}, prefix="X")

_fetch_rate_limiter = RateLimiter(FETCH_MIN_INTERVAL)

try:
    from xtf import Router, NotFound, RateLimited
    from xtf.exceptions import XtfError
except ImportError:
    logger.error("x-tweet-fetcher is not installed.")
    exit(1)


@contextlib.contextmanager
def _quiet_xtf():
    """Swallows xtf's per-instance/per-backend stderr chatter.

    xtf narrates every nitter instance and router fall-through on stderr; it
    is a vendored third-party package, so the noise is filtered here instead
    of patched there. The failure itself still surfaces as a raised XtfError,
    which this scraper logs with its own format. Logging handlers hold the
    real stderr from setup_logging, so scraper output is unaffected.
    """
    with contextlib.redirect_stderr(io.StringIO()):
        yield

TWEET_URL_RE = re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/(\d+)")
EXTERNAL_URL_RE = re.compile(r"https?://(?!twitter\.com|x\.com)\S+")


def load_trusted_accounts(filepath: str = "twtr-accounts.txt") -> list:
    """Loads trusted account handles from a file or falls back to defaults."""
    default_accounts = ["Reuters", "business", "CNBC", "BBCWorld", "CNN", "AJEnglish"]
    if not os.path.exists(filepath):
        logger.warning(f"'{filepath}' not found. Creating default file.")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(default_accounts))
        return default_accounts

    with open(filepath, "r", encoding="utf-8") as f:
        accounts = [line.strip().lstrip("@") for line in f if line.strip() and not line.startswith("#")]

    if not accounts:
        logger.warning(f"No accounts found in '{filepath}'. Using default trusted accounts.")
        return default_accounts

    return accounts


def extract_tweet_ref(url: str):
    m = TWEET_URL_RE.search(url)
    if not m:
        return None
    return m.group(1), m.group(2)


def _ddg_tweet_refs(query: str, label: str, max_results: int, allowed: set = None) -> list:
    """Runs one DuckDuckGo query and returns the (username, tweet_id) refs in it.

    Shared by keyword discovery and by the account-mode fallback, so both
    paths get the same retry/backoff and the same per-query pacing delay.

    ``allowed`` is the set of handles the caller will accept. DuckDuckGo is a
    web search, not X's own search: "from:acct" is an X operator it does not
    honour, so a query scoped that way still returns tweets by whoever else
    quoted or replied. The scoping therefore has to be enforced here on the
    parsed URLs, or untrusted accounts end up in the output.
    """
    found_urls = set()
    results = []

    for attempt in range(1, DDGS_MAX_RETRIES + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    query,
                    region="us-en",
                    safesearch="off",
                    timelimit="w",
                    max_results=max_results * 3
                )
            break
        except (RatelimitException, TimeoutException) as e:
            wait_time = random.uniform(20.0, 30.0) * attempt
            if attempt < DDGS_MAX_RETRIES:
                logger.warning(
                    f"DDGS {type(e).__name__} on '{label}' (attempt {attempt}/{DDGS_MAX_RETRIES}). "
                    f"Backing off for {wait_time:.1f}s before retry..."
                )
                time.sleep(wait_time)
            else:
                logger.warning(f"DDGS {type(e).__name__} on '{label}'. Exhausted {DDGS_MAX_RETRIES} attempts.")
                results = []
        except Exception as e:
            logger.warning(f"DDGS error on '{label}': {str(e).splitlines()[0]}")
            results = []
            break
        finally:
            time.sleep(random.uniform(*DDGS_DELAY_RANGE))

    allowed = {a.casefold() for a in allowed} if allowed else None
    dropped = 0
    for r in results:
        ref = extract_tweet_ref(r.get("href", ""))
        if not ref:
            continue
        if allowed is not None and ref[0].casefold() not in allowed:
            dropped += 1
            continue
        found_urls.add((ref[0], ref[1]))

    if dropped:
        logger.info(f"Dropped {dropped} result(s) for '{label}' from accounts outside the trusted list.")
    return list(found_urls)[:max_results]


def discover_tweet_urls(keyword: str, trusted_accounts: list, max_results: int = 15) -> list:
    """
    Searches DuckDuckGo for tweet URLs matching the keyword and trusted accounts.
    Retries on rate limits or timeouts with backoff delays instead of skipping.
    """
    logger.info(f"Searching for: {keyword}...")
    # The query carries no "from:" clauses any more: DuckDuckGo does not honour
    # that X operator, so a 100-handle OR-chain only diluted the query text and
    # returned fewer usable x.com URLs. Scoping is enforced on the results
    # instead, which measurably raises the number of trusted hits per search.
    refs = _ddg_tweet_refs(f"site:x.com {keyword}", keyword, max_results,
                           allowed=set(trusted_accounts))
    logger.info(f"Found {len(refs)} candidate posts for '{keyword}'.")
    return refs


@with_retry(max_attempts=3, base_delay=3.0, exceptions=(RateLimited, ConnectionError, TimeoutError))
def _fetch_timeline_refs(username: str, limit: int) -> list:
    _fetch_rate_limiter.wait()
    router = Router()
    with _quiet_xtf():
        timeline = router.fetch_timeline(username, limit=limit)
    return [(username, tw.tweet_id) for tw in timeline if tw.tweet_id]


def discover_account_tweet_refs(username: str, limit: int = ACCOUNT_MODE_FETCH_LIMIT) -> list:
    """
    Lists an account's recent tweet IDs (no keyword, no candidate ranking).

    Preferred path is xtf's own timeline fetch, which is complete and cheap.
    Its backends (nitter instances, browser driver) go down regularly and
    independently of DuckDuckGo, though, so a failure there falls back to the
    same DDG search keyword mode uses, with the keyword left out -- a partial
    account listing beats losing the account for the whole run.
    """
    try:
        refs = _fetch_timeline_refs(username, limit)
        if refs:
            return refs
        logger.info(f"Timeline for '@{username}' came back empty; falling back to DuckDuckGo.")
    except XtfError as e:
        logger.warning(f"Timeline fetch failed for '@{username}' ({e.code}); falling back to DuckDuckGo.")

    # Path-scoped, not "from:" -- x.com/<user>/status/... is a URL pattern the
    # web index actually matches on.
    refs = _ddg_tweet_refs(f"site:x.com/{username}", f"@{username}", limit, allowed={username})
    logger.info(f"Found {len(refs)} candidate posts for '@{username}' via DuckDuckGo.")
    return refs


def _extract_media_info(tw: dict, text: str) -> dict:
    """Extracts video URLs, photos, or news article links from tweet objects and text."""
    media_field = tw.get("media") or {}

    # 1. Video links
    video_url = None
    if isinstance(media_field, dict):
        videos = media_field.get("videos") or media_field.get("video")
        if isinstance(videos, list) and videos:
            video_url = videos[0].get("url") if isinstance(videos[0], dict) else str(videos[0])
        elif isinstance(videos, dict):
            video_url = videos.get("url")

    if not video_url:
        video_url = tw.get("video_url") or tw.get("video")

    if video_url:
        return {
            "has_media": True,
            "media_type": "video",
            "media_url": video_url
        }

    # 2. Photos / Images
    photo_url = None
    if isinstance(media_field, dict):
        photos = media_field.get("photos") or media_field.get("all")
        if isinstance(photos, list) and photos:
            photo_url = photos[0].get("url") if isinstance(photos[0], dict) else str(photos[0])

    if photo_url:
        return {
            "has_media": True,
            "media_type": "image",
            "media_url": photo_url
        }

    # 3. External news / article URLs
    urls = tw.get("urls") or tw.get("external_urls") or []
    if isinstance(urls, list) and urls:
        first_url = urls[0].get("url") or urls[0].get("expanded_url") if isinstance(urls[0], dict) else str(urls[0])
        return {
            "has_media": True,
            "media_type": "news_article",
            "media_url": first_url
        }

    ext_urls = EXTERNAL_URL_RE.findall(text)
    if ext_urls:
        return {
            "has_media": True,
            "media_type": "news_article",
            "media_url": ext_urls[0]
        }

    return {
        "has_media": False,
        "media_type": None,
        "media_url": None
    }


@with_retry(max_attempts=3, base_delay=3.0, exceptions=(RateLimited, ConnectionError, TimeoutError))
def _fetch_tweet_safe(username: str, tweet_id: str) -> dict:
    _fetch_rate_limiter.wait()
    router = Router()
    with _quiet_xtf():
        return router.fetch_tweet(username, tweet_id)


def _parse_created_at(created_raw):
    if not created_raw:
        return None, created_raw or "N/A"
    if isinstance(created_raw, datetime.datetime):
        dt = created_raw
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        else:
            dt = dt.astimezone(datetime.timezone.utc)
        return dt, dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    if isinstance(created_raw, (int, float)):
        try:
            # Accept both Unix seconds and Unix milliseconds.
            value = created_raw / 1000 if created_raw > 10_000_000_000 else created_raw
            dt = datetime.datetime.fromtimestamp(value, datetime.timezone.utc)
            return dt, dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, OSError, OverflowError):
            return None, str(created_raw)

    for fmt in (
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S UTC",
    ):
        try:
            dt = datetime.datetime.strptime(created_raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt, dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, TypeError):
            continue
    return None, created_raw


def _nested_get(data, *path):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_tweet_text(tw: dict) -> tuple:
    """Return (text, is_complete, extraction_errors, extraction_reliability).

    We only mark a tweet as truncated when the fetched object gives explicit
    truncation evidence *and* no full-text representation is available. Tweet
    length, a trailing ellipsis, and display_text_range are not truncation
    evidence by themselves: all three can occur in perfectly complete posts.
    """
    candidates = [
        ("note_tweet_result", _nested_get(tw, "note_tweet", "note_tweet_results", "result", "text")),
        ("note_tweet", _nested_get(tw, "note_tweet", "text")),
        ("note_tweet_text", tw.get("note_tweet_text")),
        ("extended_tweet", _nested_get(tw, "extended_tweet", "full_text")),
        ("full_text", tw.get("full_text")),
        ("retweeted_full_text", _nested_get(tw, "retweeted_status", "full_text")),
        ("text", tw.get("text")),
    ]
    selected_source, text = next(
        ((name, value) for name, value in candidates if isinstance(value, str) and value.strip()),
        (None, ""),
    )
    stripped = text.strip()
    if not stripped:
        return "", False, ["Tweet text is empty"], 0.0

    explicit_truncated = bool(tw.get("truncated") or tw.get("is_truncated"))
    selected_is_full = selected_source not in {None, "text"}

    # x-tweet-fetcher's unified single-tweet schema is intended to return full
    # tweet content. Therefore a normal text-only result is accepted as complete
    # unless the backend explicitly says it is truncated. If an explicit flag
    # exists but we selected one of the full/note fields, the full field wins.
    is_complete = selected_is_full or not explicit_truncated
    errors = []
    reliability = 1.0 if is_complete else None

    if not is_complete:
        errors.append("Tweet fetch explicitly reports truncated text and no full-text field was available")
        display_range = tw.get("display_text_range")
        if (
            isinstance(display_range, (list, tuple))
            and len(display_range) >= 2
            and isinstance(display_range[1], int)
            and display_range[1] > 0
        ):
            # This ratio is used only after explicit truncation was established;
            # display_text_range alone never triggers truncation.
            reliability = min(1.0, len(stripped) / display_range[1])

    return text, is_complete, errors, reliability

def _extract_author_identity(tw: dict, fallback_handle: str) -> tuple:
    """Returns (display_name, handle) with the handle normalized without @."""
    raw_author = tw.get("author")
    author = raw_author if isinstance(raw_author, dict) else {}
    user = tw.get("user") if isinstance(tw.get("user"), dict) else {}

    handle = (
        author.get("screen_name")
        or author.get("screenName")
        or author.get("username")
        or author.get("handle")
        or user.get("screen_name")
        or user.get("screenName")
        or user.get("username")
        or user.get("handle")
        or tw.get("screen_name")
        or tw.get("screenName")
        or tw.get("username")
        or fallback_handle
    )
    handle = str(handle or "").strip().lstrip("@")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,15}", handle):
        # The URL/discovery handle is usually the most trustworthy fallback.
        fallback = str(fallback_handle or "").strip().lstrip("@")
        handle = fallback if re.fullmatch(r"[A-Za-z0-9_]{1,15}", fallback) else handle

    display_name = (
        author.get("name")
        or author.get("display_name")
        or user.get("name")
        or user.get("display_name")
        or tw.get("author_name")
        or tw.get("name")
        or (raw_author if isinstance(raw_author, str) else None)
        or handle
    )
    return str(display_name or handle), handle


def _extract_updated_at(tw: dict):
    """Returns the latest explicit edit timestamp, otherwise None.

    We intentionally do not copy collection time into updated_at and do not
    use Twitter's ``editable_until`` metadata, because that is an edit window,
    not proof that the post was edited.
    """
    edit_info = tw.get("edit_info") if isinstance(tw.get("edit_info"), dict) else {}
    candidates = [
        tw.get("edited_at"),
        tw.get("last_edited_at"),
        tw.get("edit_timestamp"),
        tw.get("modified_at"),
        edit_info.get("edited_at"),
        edit_info.get("last_edited_at"),
        edit_info.get("timestamp"),
    ]
    parsed = []
    for value in candidates:
        dt, formatted = _parse_created_at(value)
        if dt is not None:
            parsed.append((dt, formatted))
    if not parsed:
        return None
    return max(parsed, key=lambda item: item[0])[1]


def _insert_sorted(records_list: list, timestamps_list: list, record: dict, dt_val: datetime.datetime):
    """
    Inserts a record into the list maintaining chronological order via binary search.
    """
    sort_key = dt_val.timestamp() if dt_val else 0.0
    idx = bisect.bisect_right(timestamps_list, sort_key)
    records_list.insert(idx, record)
    timestamps_list.insert(idx, sort_key)


def _load_existing(master_file: str) -> tuple:
    existing_records, existing_ids, existing_timestamps = [], set(), []
    if os.path.exists(master_file):
        try:
            with open(master_file, 'r', encoding='utf-8') as f:
                existing_records = json.load(f)
                for rec in existing_records:
                    rec_id = rec.get("record_id")
                    if rec_id:
                        existing_ids.add(rec_id)
                    pub_str = rec.get("time_stamps", {}).get("published_at")
                    dt_obj, _ = _parse_created_at(pub_str)
                    existing_timestamps.append(dt_obj.timestamp() if dt_obj else 0.0)
        except Exception as e:
            logger.error(f"Error reading {master_file}: {e}")
            existing_records, existing_ids, existing_timestamps = [], set(), []
    return existing_records, existing_ids, existing_timestamps


def _fetch_and_write_tweets(refs: list, master_file: str, context_terms: list,
                            lsh, hash_by_id, log_label: str) -> str:
    """
    Shared fetch/filter/write core for both keyword mode and account-only
    mode -- they differ only in how ``refs`` (username, tweet_id) pairs are
    discovered and what context_terms feed asset detection.
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=MAX_PAST_MINUTES)
    existing_records, existing_ids, existing_timestamps = _load_existing(master_file)

    new_refs = [
        (username, tweet_id) for username, tweet_id in refs
        if f"X_{username}_{tweet_id}" not in existing_ids and f"X_{username}_{tweet_id}" not in hash_by_id
    ]
    if not new_refs:
        logger.info(f"All discovered tweets for '{log_label}' have already been gathered.")
        return master_file

    fetched = []
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        future_to_ref = {executor.submit(_fetch_tweet_safe, u, tid): (u, tid) for u, tid in new_refs}
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

    added_count = 0
    skipped_unsupported_lang = 0

    for username, tweet_id, tw in fetched:
        text, text_is_complete, extraction_errors, extraction_reliability = _extract_tweet_text(tw)
        if not text.strip():
            continue

        created_dt, created_str = _parse_created_at(tw.get("created_at") or tw.get("timestamp"))
        if created_dt is not None and created_dt < cutoff:
            continue

        clean_text = text.replace("\n", " ")

        lang, confidence = detect_language(clean_text)
        if lang not in ALLOWED_LANGUAGES or confidence < LANGUAGE_MIN_CONFIDENCE:
            skipped_unsupported_lang += 1
            continue

        author_name, author_id = _extract_author_identity(tw, username)

        record_id = f"X_{author_id}_{tweet_id}"
        if record_id in existing_ids:
            continue

        duplication_info = dedup_utils.check_and_register(record_id, clean_text, lsh, hash_by_id)
        if duplication_info.get("is_duplicate"):
            continue

        media_info = _extract_media_info(tw, text)
        collected_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        updated_at = _extract_updated_at(tw)
        translated_text, sentiment, asset_mention = analyze_text(
            clean_text, lang, context_terms=context_terms)

        record = {
            "record_id": record_id,
            "Source": {
                "platform": "X",
                "Source_type": "Social_media",
                "Source_name": author_name,
                "Source_id": tweet_id,
                "url": f"https://x.com/{author_id}/status/{tweet_id}",
                "author_name": author_name,
                "author_id": author_id
            },
            "Content": {
                "Content_type": "tweet",
                "title": "N/A",
                "raw_text": clean_text,
                "Clean_text": "",
                "language": lang,
                "translated_text": translated_text
            },
            "time_stamps": {
                "published_at": created_str,
                "collected_at": collected_at,
                "updated_at": updated_at
            },
            "asset_mention": asset_mention,
            "sentiment": sentiment,
            "Engagement": {
                "Views": tw.get("views") or tw.get("view_count") or 0,
                "likes": tw.get("likes") or tw.get("like_count") or 0,
                "Comments": tw.get("replies") or tw.get("reply_count") or 0,
                "shares": tw.get("retweets") or tw.get("retweet_count") or 0
            },
            "media": media_info,
            "deduplication": duplication_info
        }
        record["quality"] = build_quality(
            record,
            complete_raw_text=text_is_complete,
            extraction_errors=extraction_errors,
            extraction_reliability=extraction_reliability,
        )

        _insert_sorted(existing_records, existing_timestamps, record, created_dt)
        existing_ids.add(record_id)
        added_count += 1

    if skipped_unsupported_lang:
        logger.info(f"Filtered out {skipped_unsupported_lang} unsupported-language post(s) for '{log_label}'.")

    if added_count > 0:
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(existing_records, f, indent=4, ensure_ascii=False)
        logger.info(f"Appended {added_count} new sorted tweets to {master_file} (Total: {len(existing_records)}).")
    else:
        logger.info(f"No new unique tweets to append for '{log_label}'.")

    return master_file


def search_and_scrape_tweets(keyword: str, trusted_accounts: list, lsh, hash_by_id, output_dir: str = "twtr-tweets/"):
    """Keyword mode: DDG-discovered tweets scoped to the trusted accounts."""
    os.makedirs(output_dir, exist_ok=True)
    safe_keyword = keyword.replace(" ", "_").lower()
    master_file = os.path.join(output_dir, f"twitter_{safe_keyword}.json")

    refs = discover_tweet_urls(keyword, trusted_accounts=trusted_accounts, max_results=MAX_RESULTS_PER_KEYWORD)
    if not refs:
        logger.info(f"No candidate posts found for '{keyword}'.")
        return None

    return _fetch_and_write_tweets(refs, master_file, context_terms=[keyword],
                                   lsh=lsh, hash_by_id=hash_by_id, log_label=keyword)


def scrape_account_tweets(username: str, lsh, hash_by_id, output_dir: str = "twtr-tweets/"):
    """
    Account-only mode: every recent tweet from one trusted account, no
    keyword filtering or candidate selection. Relevance is left entirely to
    the asset-mapping stage, the same trust model tel/tiktok/truth-scraper
    already use for their account-driven discovery.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_username = username.replace(" ", "_").lower()
    master_file = os.path.join(output_dir, f"twitter_{safe_username}.json")

    refs = discover_account_tweet_refs(username, limit=ACCOUNT_MODE_FETCH_LIMIT)
    if not refs:
        logger.info(f"No recent tweets found for '@{username}'.")
        return None

    return _fetch_and_write_tweets(refs, master_file, context_terms=[],
                                   lsh=lsh, hash_by_id=hash_by_id, log_label=f"@{username}")


def main():
    accounts_file = "twtr-accounts.txt"
    trusted_accounts = load_trusted_accounts(accounts_file)
    logger.info(f"Loaded {len(trusted_accounts)} trusted accounts from {accounts_file}.")

    mode = get_scrape_mode(logger, prefix="X")

    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logger.info(f"Starting scraping run {run_timestamp} for X (Twitter) in '{mode}' mode...")

    lsh, hash_by_id = dedup_utils.load_lsh()

    if mode == "accounts":
        for account in trusted_accounts:
            try:
                scrape_account_tweets(account, lsh, hash_by_id)
            except Exception as e:
                logger.error(f"Error scraping account '@{account}': {str(e).splitlines()[0]}")
        logger.info("Scraping for all trusted accounts is done.")
        return

    keyword_file = "twtr-keywords.txt"
    if not os.path.exists(keyword_file):
        logger.error(f"File '{keyword_file}' not found.")
        return

    with open(keyword_file, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        logger.error(f"No keywords found in {keyword_file}.")
        return

    for keyword in keywords:
        try:
            search_and_scrape_tweets(keyword, trusted_accounts, lsh, hash_by_id)
        except Exception as e:
            logger.error(f"Error executing search/scrape for keyword '{keyword}': {str(e).splitlines()[0]}")

    logger.info("Scraping for all keywords is done.")


if __name__ == "__main__":
    main()