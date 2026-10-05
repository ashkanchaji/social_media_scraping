"""
pipeline_utils.py -- shared infrastructure for the scraping pipeline:
logging, retry/backoff, a thread-safe rate limiter, language detection,
and shared social-record schema helpers.

The schema helpers are intentionally source-agnostic where possible so X,
YouTube, Telegram, and future social scrapers can emit consistent JSON.

Install core dependencies with: pip install -r requirements-core.txt
"""

import os
import re
import sys
import json
import time
import random
import logging
import tempfile
import bisect
import functools
import threading
import datetime
import warnings
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from py3langid.langid import LanguageIdentifier, MODEL_FILE

# Every scraper imports this module, so loading .env here (once) makes
# TELEGRAM_API_ID, REDDIT_CLIENT_ID, TRUTHSOCIAL_PASSWORD, etc. available to
# all of them without each script needing its own load_dotenv() call.
# Real shell exports still take precedence -- load_dotenv() never overwrites
# a variable that's already set in the environment.
load_dotenv()

# Third-party chatter that is not actionable for an operator reading a scraper
# log: transformers announces every pipeline's device and warns about long
# inputs it already truncates, marian asks for sacremoses, and Telethon logs
# routine reconnects. Set before transformers is ever imported (every import of
# it in this module is lazy), and with setdefault so an operator debugging a
# model can still export TRANSFORMERS_VERBOSITY=warning.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*sacremoses.*")
logging.getLogger("telethon").setLevel(logging.ERROR)

_identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)


# ---------------------------------------------------------------------------
# Environment-driven configuration
#
# Every tunable scraper constant is declared as ``X = env_int("X", default)``
# so an operator who has never read the code can retune a run from .env alone
# (12-factor config), while the in-code default keeps a bare checkout working
# with no .env at all. Each helper looks up "<PREFIX>_<NAME>" first and then
# the bare "<NAME>", so a value can be set once for every scraper and still be
# overridden for one of them (e.g. MAX_PAST_MINUTES=60 with YT_MAX_PAST_MINUTES=1440).
# A malformed value logs a warning and falls back to the default rather than
# killing an unattended cron run.
# ---------------------------------------------------------------------------

def _env_raw(name: str, prefix: str = ""):
    for key in ([f"{prefix}_{name}"] if prefix else []) + [name]:
        value = os.environ.get(key)
        if value is not None and value.strip():
            return key, value.strip()
    return None, None


def _env_parsed(name: str, default, prefix: str, parse):
    key, value = _env_raw(name, prefix)
    if value is None:
        return default
    try:
        return parse(value)
    except (TypeError, ValueError) as e:
        logging.getLogger(__name__).warning(
            "Invalid value for %s (%r): %s. Using default %r.", key, value, e, default
        )
        return default


def env_str(name: str, default: str = "", prefix: str = "") -> str:
    return _env_parsed(name, default, prefix, str)


def env_int(name: str, default: int, prefix: str = "") -> int:
    return _env_parsed(name, default, prefix, lambda v: int(float(v)))


def env_float(name: str, default: float, prefix: str = "") -> float:
    return _env_parsed(name, default, prefix, float)


def env_bool(name: str, default: bool, prefix: str = "") -> bool:
    def parse(value):
        lowered = value.casefold()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError("expected a boolean like true/false")
    return _env_parsed(name, default, prefix, parse)


def env_set(name: str, default: set, prefix: str = "") -> set:
    """Comma-separated list -> set of lowercased items (e.g. ALLOWED_LANGUAGES=en,fa)."""
    return _env_parsed(
        name, default, prefix,
        lambda v: {item.strip().casefold() for item in v.split(",") if item.strip()},
    )


def env_range(name: str, default: tuple, prefix: str = "") -> tuple:
    """"min,max" -> (float, float), used for the jittered pacing delays."""
    def parse(value):
        parts = [float(p) for p in value.split(",")]
        if len(parts) != 2 or parts[0] > parts[1]:
            raise ValueError("expected 'min,max' with min <= max")
        return (parts[0], parts[1])
    return _env_parsed(name, default, prefix, parse)


def get_scrape_mode(logger=None, prefix: str = "") -> str:
    """Returns "keyword" or "accounts" for this run.

    Read from SCRAPE_MODE (or <PREFIX>_SCRAPE_MODE) so a cron/bash wrapper can
    pick the discovery method with no terminal input. Only an interactive
    terminal falls back to asking; a non-interactive run with no variable set
    uses "keyword", which is the historical behaviour of every scraper that
    has keywords.
    """
    log = logger or logging.getLogger(__name__)
    key, value = _env_raw("SCRAPE_MODE", prefix)
    if value:
        mode = value.casefold()
        aliases = {"keyword": "keyword", "keywords": "keyword",
                   "accounts": "accounts", "account": "accounts", "channels": "accounts"}
        if mode in aliases:
            log.info(f"{key}={mode} set in environment; skipping interactive prompt.")
            return aliases[mode]
        log.warning(f"Unrecognised {key}={value!r}; expected 'keyword' or 'accounts'.")

    if not sys.stdin.isatty():
        log.info("SCRAPE_MODE not set and no terminal attached; defaulting to 'keyword' mode.")
        return "keyword"

    while True:
        choice = input(
            "Scrape by (1) keyword, prioritizing trusted accounts, or "
            "(2) trusted accounts only (all recent posts, no keyword)? [1/2]: "
        ).strip()
        if choice == "1":
            return "keyword"
        if choice == "2":
            return "accounts"
        print("Please enter 1 or 2.")


# Built-in starter registry. Projects can extend/override this without code
# changes by placing an ``assets-registry/asset-registry.json`` file in the repo.
# Besides aliases, entries may include ``topics`` and ``description``; those
# fields are used by the NLI relevance model for context-only inference.
_DEFAULT_ASSET_REGISTRY = [
    {"canonical_name": "Bitcoin", "Symbol": "BTC", "asset_class": "cryptocurrency", "asset_id": "crypto:BTC", "aliases": ["bitcoin", "btc", "xbt", "بیت کوین", "بیتکوین", "بیت‌کوین"], "topics": ["crypto", "cryptocurrency", "digital assets"], "description": "the Bitcoin cryptocurrency"},
    {"canonical_name": "Ethereum", "Symbol": "ETH", "asset_class": "cryptocurrency", "asset_id": "crypto:ETH", "aliases": ["ethereum", "ether", "eth", "اتریوم", "اتر"], "topics": ["crypto", "smart contracts", "digital assets"], "description": "the Ethereum cryptocurrency and smart-contract network"},
    {"canonical_name": "Solana", "Symbol": "SOL", "asset_class": "cryptocurrency", "asset_id": "crypto:SOL", "aliases": ["solana", "sol", "سولانا"], "topics": ["crypto", "blockchain", "digital assets"], "description": "the Solana cryptocurrency and blockchain"},
    {"canonical_name": "XRP", "Symbol": "XRP", "asset_class": "cryptocurrency", "asset_id": "crypto:XRP", "aliases": ["xrp", "ripple", "ریپل"], "topics": ["crypto", "payments", "digital assets"], "description": "the XRP cryptocurrency associated with Ripple payments"},
    {"canonical_name": "Gold", "Symbol": "XAU", "asset_class": "commodity", "asset_id": "commodity:XAU", "aliases": ["gold", "xau", "xauusd", "طلا", "انس طلا", "طلای جهانی"], "topics": ["precious metals", "safe haven", "bullion"], "description": "the gold precious-metal commodity"},
    {"canonical_name": "Silver", "Symbol": "XAG", "asset_class": "commodity", "asset_id": "commodity:XAG", "aliases": ["silver", "xag", "xagusd", "نقره"], "topics": ["precious metals", "bullion"], "description": "the silver precious-metal commodity"},
    {"canonical_name": "WTI Crude Oil", "Symbol": "WTI", "asset_class": "commodity", "asset_id": "commodity:WTI", "aliases": ["wti", "wti crude", "west texas intermediate"], "topics": ["oil", "crude", "petroleum", "energy markets", "oil prices", "OPEC"], "description": "West Texas Intermediate crude oil benchmark"},
    {"canonical_name": "Brent Crude Oil", "Symbol": "BRENT", "asset_class": "commodity", "asset_id": "commodity:BRENT", "aliases": ["brent", "brent crude", "brent oil", "برنت", "نفت برنت"], "topics": ["oil", "crude", "petroleum", "energy markets", "oil prices", "OPEC"], "description": "Brent crude oil benchmark"},
    {"canonical_name": "Crude Oil", "Symbol": "OIL", "asset_class": "commodity", "asset_id": "commodity:OIL", "aliases": ["crude oil", "oil prices", "oil price", "نفت", "نفت خام", "قیمت نفت"], "topics": ["oil", "petroleum", "energy markets", "refining", "OPEC", "oil supply", "oil demand"], "description": "global crude-oil prices and petroleum markets"},
    {"canonical_name": "Energy Select Sector SPDR Fund", "Symbol": "XLE", "asset_class": "ETF", "asset_id": "etf:XLE", "aliases": ["xle", "energy select sector spdr"], "topics": ["energy stocks", "oil stocks", "oil companies", "oil and gas companies", "energy companies", "integrated oil companies"], "description": "a US exchange-traded fund tracking large energy, oil and gas companies"},
    {"canonical_name": "Exxon Mobil", "Symbol": "XOM", "asset_class": "equity", "asset_id": "equity:XOM", "aliases": ["exxon", "exxon mobil", "xom"], "topics": ["oil company", "energy company", "integrated oil and gas"], "description": "Exxon Mobil, an integrated oil and gas company"},
    {"canonical_name": "Chevron", "Symbol": "CVX", "asset_class": "equity", "asset_id": "equity:CVX", "aliases": ["chevron", "cvx"], "topics": ["oil company", "energy company", "integrated oil and gas"], "description": "Chevron, an integrated oil and gas company"},
    {"canonical_name": "Shell", "Symbol": "SHEL", "asset_class": "equity", "asset_id": "equity:SHEL", "aliases": ["shell plc", "shell", "shel"], "topics": ["oil company", "energy company", "integrated oil and gas", "LNG"], "description": "Shell plc, an integrated oil, gas and LNG company"},
    {"canonical_name": "BP", "Symbol": "BP", "asset_class": "equity", "asset_id": "equity:BP", "aliases": ["bp plc", "bp"], "topics": ["oil company", "energy company", "integrated oil and gas"], "description": "BP plc, an integrated oil and gas company"},
    {"canonical_name": "S&P 500", "Symbol": "SPX", "asset_class": "index", "asset_id": "index:SPX", "aliases": ["s&p 500", "s&p500", "sp500", "spx", "اس اند پی"], "topics": ["US stocks", "large cap stocks", "US equity market"], "description": "the S&P 500 US large-cap equity index"},
    {"canonical_name": "Nasdaq Composite", "Symbol": "IXIC", "asset_class": "index", "asset_id": "index:IXIC", "aliases": ["nasdaq composite", "nasdaq", "ixic", "نزدک", "نزدَک"], "topics": ["technology stocks", "US stocks", "growth stocks"], "description": "the Nasdaq Composite equity index"},
    {"canonical_name": "Dow Jones Industrial Average", "Symbol": "DJI", "asset_class": "index", "asset_id": "index:DJI", "aliases": ["dow jones", "dow", "djia", "dji", "داوجونز", "داو جونز"], "topics": ["US stocks", "blue chip stocks"], "description": "the Dow Jones Industrial Average US equity index"},
    {"canonical_name": "US Dollar", "Symbol": "USD", "asset_class": "currency", "asset_id": "fx:USD", "aliases": ["us dollar", "u.s. dollar", "usd", "dollar index", "dxy", "دلار", "شاخص دلار"], "topics": ["foreign exchange", "FX", "dollar", "Federal Reserve"], "description": "the United States dollar currency"},
    {"canonical_name": "Euro", "Symbol": "EUR", "asset_class": "currency", "asset_id": "fx:EUR", "aliases": ["euro", "eur", "یورو"], "topics": ["foreign exchange", "FX", "ECB", "eurozone"], "description": "the euro currency"},
    {"canonical_name": "British Pound", "Symbol": "GBP", "asset_class": "currency", "asset_id": "fx:GBP", "aliases": ["british pound", "pound sterling", "sterling", "gbp", "پوند"], "topics": ["foreign exchange", "FX", "Bank of England", "UK currency"], "description": "the British pound sterling currency"},
    {"canonical_name": "Japanese Yen", "Symbol": "JPY", "asset_class": "currency", "asset_id": "fx:JPY", "aliases": ["japanese yen", "yen", "jpy", "ین"], "topics": ["foreign exchange", "FX", "Bank of Japan", "Japan currency"], "description": "the Japanese yen currency"},
    {"canonical_name": "Apple", "Symbol": "AAPL", "asset_class": "equity", "asset_id": "equity:AAPL", "aliases": ["apple inc", "apple", "aapl", "اپل"], "topics": ["iPhone", "consumer technology", "big tech"], "description": "Apple Inc. common stock"},
    {"canonical_name": "Microsoft", "Symbol": "MSFT", "asset_class": "equity", "asset_id": "equity:MSFT", "aliases": ["microsoft", "msft"], "topics": ["software", "cloud computing", "Azure", "big tech"], "description": "Microsoft Corporation common stock"},
    {"canonical_name": "NVIDIA", "Symbol": "NVDA", "asset_class": "equity", "asset_id": "equity:NVDA", "aliases": ["nvidia", "nvda", "انویدیا"], "topics": ["AI chips", "GPUs", "semiconductors", "artificial intelligence"], "description": "NVIDIA Corporation common stock"},
    {"canonical_name": "Tesla", "Symbol": "TSLA", "asset_class": "equity", "asset_id": "equity:TSLA", "aliases": ["tesla", "tsla", "تسلا"], "topics": ["electric vehicles", "EVs", "automotive"], "description": "Tesla Inc. common stock"},
    {"canonical_name": "Amazon", "Symbol": "AMZN", "asset_class": "equity", "asset_id": "equity:AMZN", "aliases": ["amazon", "amzn"], "topics": ["ecommerce", "AWS", "cloud computing", "big tech"], "description": "Amazon.com Inc. common stock"},
    {"canonical_name": "Meta Platforms", "Symbol": "META", "asset_class": "equity", "asset_id": "equity:META", "aliases": ["meta platforms", "meta", "facebook", "meta stock"], "topics": ["social media", "digital advertising", "big tech"], "description": "Meta Platforms Inc. common stock"},
    {"canonical_name": "Tether", "Symbol": "USDT", "asset_class": "cryptocurrency", "asset_id": "crypto:USDT", "aliases": ["tether", "usdt", "تتر"], "topics": ["crypto", "stablecoin", "digital assets", "dollar peg"], "description": "the Tether (USDT) dollar-pegged stablecoin"},
    {"canonical_name": "USD Coin", "Symbol": "USDC", "asset_class": "cryptocurrency", "asset_id": "crypto:USDC", "aliases": ["usd coin", "usdc"], "topics": ["crypto", "stablecoin", "digital assets", "dollar peg"], "description": "the USD Coin (USDC) dollar-pegged stablecoin"},
    {"canonical_name": "BNB", "Symbol": "BNB", "asset_class": "cryptocurrency", "asset_id": "crypto:BNB", "aliases": ["bnb", "binance coin", "بی ان بی"], "topics": ["crypto", "exchange token", "digital assets"], "description": "the BNB cryptocurrency of the BNB Chain"},
    {"canonical_name": "Cardano", "Symbol": "ADA", "asset_class": "cryptocurrency", "asset_id": "crypto:ADA", "aliases": ["cardano", "ada", "کاردانو"], "topics": ["crypto", "blockchain", "digital assets"], "description": "the Cardano cryptocurrency"},
    {"canonical_name": "Dogecoin", "Symbol": "DOGE", "asset_class": "cryptocurrency", "asset_id": "crypto:DOGE", "aliases": ["dogecoin", "doge", "دوج کوین", "دوج"], "topics": ["crypto", "meme coin", "digital assets"], "description": "the Dogecoin cryptocurrency"},
    {"canonical_name": "Toncoin", "Symbol": "TON", "asset_class": "cryptocurrency", "asset_id": "crypto:TON", "aliases": ["toncoin", "ton coin", "تون کوین"], "topics": ["crypto", "Telegram", "digital assets"], "description": "the Toncoin cryptocurrency of The Open Network"},
    {"canonical_name": "Natural Gas", "Symbol": "NG", "asset_class": "commodity", "asset_id": "commodity:NG", "aliases": ["natural gas", "henry hub", "lng", "گاز طبیعی", "گاز"], "topics": ["gas", "energy markets", "LNG", "energy prices"], "description": "natural gas and LNG markets"},
    {"canonical_name": "Iranian Rial", "Symbol": "IRR", "asset_class": "currency", "asset_id": "fx:IRR", "aliases": ["iranian rial", "irr", "toman", "ریال", "تومان"], "topics": ["foreign exchange", "FX", "Iran currency", "Iran economy"], "description": "the Iranian rial/toman currency"},
]

# Content_type each platform emits. A platform missing from this map fails the
# content-type check outright instead of silently scoring lower, and the same
# key set drives the identity/record_id checks below.
_PLATFORM_CONTENT_TYPES = {
    "X": "tweet",
    "YouTube": "video",
    "Telegram": "message",
    "Reddit": "post",
    "TikTok": "video",
    "TruthSocial": "post",
}
_KNOWN_PLATFORMS = frozenset(_PLATFORM_CONTENT_TYPES)
# Platforms whose posts carry a real title field (as opposed to body text only).
_TITLED_PLATFORMS = frozenset({"YouTube", "Reddit"})

_CASHTAG_RE = re.compile(r"(?<!\w)\$([A-Za-z][A-Za-z0-9._-]{0,14})\b")
_ASSET_REGISTRY_CACHE = {}
_ASSET_REGISTRY_LOCK = threading.Lock()
# Same parameter class as the older cross-encoder/nli-deberta-v3-large it
# replaces, but trained on a much broader NLI mixture (MNLI+FEVER+ANLI+LingNLI
# +WANLI), so entailment probabilities for the free-form hypotheses
# _asset_hypothesis builds are both sharper and better calibrated at the same
# VRAM/latency cost. It must stay a THREE-class model (entailment / neutral /
# contradiction): the zeroshot-v2.0 family is two-class (entailment /
# not_entailment) and has no contradiction label, which silently turns the
# false-positive filter below into a no-op. Swap via ASSET_NLI_MODEL if a
# machine cannot hold it -- MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli is
# the same shape, one size down.
_ASSET_NLI_MODEL_NAME = env_str("ASSET_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
_ASSET_NLI_DEVICE = env_str("ASSET_NLI_DEVICE", "auto").strip().lower()
_ASSET_CONTEXT_MAX_RESULTS = env_int("ASSET_CONTEXT_MAX_RESULTS", 0)  # 0 = keep all NLI-entailing context candidates
_ASSET_NLI_BATCH_SIZE = max(1, env_int("ASSET_NLI_BATCH_SIZE", 16))
# A literal alias match that the model actively contradicts is a false
# positive (e.g. "Shell" the noun, "gold" the colour). Direct mentions below
# this entailment probability are dropped rather than emitted with a
# misleadingly low confidence; 0 disables the filter and keeps every match.
_ASSET_DIRECT_MIN_CONFIDENCE = env_float("ASSET_DIRECT_MIN_CONFIDENCE", 0.10)
_ASSET_NLI_BUNDLE = None
_ASSET_NLI_LOCK = threading.Lock()
_ASSET_CONTEXT_CACHE = {}
_ASSET_CONTEXT_CACHE_LOCK = threading.Lock()
_ASSET_MODEL_WARNING_EMITTED = False

# Sentiment is scored in the record's OWN language -- nothing is translated
# anywhere in this pipeline, and records carry no translated_text field.
# FinBERT is English-only and finance-tuned, so it stays the English scorer;
# any other language routes to a multilingual model instead, because feeding
# FinBERT Persian would tokenize to mostly [UNK] and fabricate a label out of
# noise. Set MULTILINGUAL_SENTIMENT_MODEL="none" to score only English and
# leave every other language's sentiment null.
_SENTIMENT_MODEL_NAME = env_str("SENTIMENT_MODEL", "ProsusAI/finbert")
_MULTILINGUAL_SENTIMENT_MODEL_NAME = env_str(
    "MULTILINGUAL_SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
# Two independently-loaded pipelines keyed by model name, so a run that sees
# both English and Persian records pays for each model exactly once.
_SENTIMENT_BUNDLES = {}
_SENTIMENT_LOCK = threading.Lock()
_SENTIMENT_WARNED = set()
# cardiffnlp ships LABEL_0/1/2 rather than words; map both spellings.
_SENTIMENT_LABEL_MAP = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
    "neg": "negative", "neu": "neutral", "pos": "positive",
}


# Repo layout: the scrapers keep only code in the repo root, and every kind of
# run artifact/input lives in its own directory. Scrapers reference these
# constants rather than bare filenames so a move is a one-line change here.
LOG_DIR = "logs"
KEYWORDS_DIR = "keywords"
SOURCES_DIR = "sources"
ASSETS_REGISTRY_DIR = "assets-registry"
DEDUP_DIR = "dedup"
SESSIONS_DIR = "sessions"
ASSET_REGISTRY_PATH = os.path.join(ASSETS_REGISTRY_DIR, "asset-registry.json")


def setup_logging(name: str, log_file: str = "scraper.log") -> logging.Logger:
    """
    Console + rotating file logging. Replaces bare print() calls so a
    long-running/24-7 process has a persistent, size-capped log to check,
    not just whatever scrolled past in a terminal.

    A bare filename is placed inside ``LOG_DIR`` so every scraper's log lands
    in one directory without each scraper repeating the path.

    The handlers go on the ROOT logger, not just the scraper's own, so that
    warnings raised inside the shared modules reach the scraper's log file.
    They previously did not: ``pipeline_utils`` logs to its own module logger,
    which had no handler, so "sentiment model unavailable" -- the one message
    that explains why a whole run came back with null sentiment -- never
    appeared in the log at all. Chatty third-party loggers are pinned to
    WARNING so this does not turn the file into a transformers/telethon dump.
    """
    logger = logging.getLogger(name)
    if logger.handlers or getattr(setup_logging, "_configured", False):
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not os.path.dirname(log_file):
        log_file = os.path.join(LOG_DIR, log_file)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)      # gates library loggers, not this one
    root.addHandler(console)
    root.addHandler(file_handler)

    # This scraper and the shared modules report at INFO; a record's level is
    # checked against the logger it was emitted on, then handed to every
    # ancestor handler, so these reach the handlers above while a library
    # left at the root's WARNING does not.
    logger.setLevel(logging.INFO)
    for shared in ("pipeline_utils", "dedup_utils"):
        logging.getLogger(shared).setLevel(logging.INFO)
    for noisy in ("telethon", "transformers", "urllib3", "httpx", "asyncio", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    setup_logging._configured = True
    return logger


def with_retry(max_attempts: int = 3, base_delay: float = 2.0, exceptions=(Exception,)):
    """
    Retries a function with exponential backoff + jitter on the given
    exception types. Use narrow exception tuples -- retrying on everything
    (including e.g. NotFound) just wastes time re-requesting things that
    will never succeed.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    sleep_for = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): "
                        f"{type(e).__name__}: {e}. Retrying in {sleep_for:.1f}s"
                    )
                    time.sleep(sleep_for)
            raise last_exc
        return wrapper
    return decorator


class RateLimiter:
    """
    Thread-safe minimum-interval rate limiter. Used so that concurrent
    fetch workers still collectively respect a global pace, instead of
    each thread's own time.sleep() only limiting itself while every
    thread fires at once.
    """
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


def detect_language(text: str) -> tuple:
    """Returns (lang_code, confidence 0-1). py3langid is deterministic
    (unlike langdetect, which needs a fixed seed to be reproducible) and
    is built to handle short text reasonably well, which matters for
    tweet-length input."""
    if not text or not text.strip():
        return "unknown", 0.0
    lang, prob = _identifier.classify(text)
    return lang, prob


def is_english(text: str, min_confidence: float = 0.70) -> bool:
    lang, prob = detect_language(text)
    return lang == "en" and prob >= min_confidence


def _is_oom_error(exc) -> bool:
    """Whether an exception is a GPU out-of-memory condition."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "outofmemory" in text or "out of memory" in text or "cuda error" in text


def _load_sentiment_model(model_name: str):
    """Lazy-loads one sentiment pipeline by model name, cached per name.

    Returns None if unavailable (missing transformers/torch, a download
    failure, or the model explicitly disabled with "none") -- callers then
    store a null sentiment rather than fabricating one.
    """
    if not model_name or model_name.strip().lower() in {"", "none", "off", "disabled"}:
        return None
    cached = _SENTIMENT_BUNDLES.get(model_name)
    if cached is not None:
        return None if cached is False else cached
    with _SENTIMENT_LOCK:
        cached = _SENTIMENT_BUNDLES.get(model_name)
        if cached is not None:
            return None if cached is False else cached
        try:
            from transformers import pipeline
            try:
                _SENTIMENT_BUNDLES[model_name] = pipeline("sentiment-analysis", model=model_name)
            except Exception as e:
                # A GPU that is full (another scraper, another process, a
                # smaller card) is a resource condition, not a missing model:
                # retry pinned to CPU rather than nulling sentiment for the
                # whole run. This mirrors _demote_nli_to_cpu on the NLI path,
                # which already degrades this way instead of giving up.
                if not _is_oom_error(e):
                    raise
                logging.getLogger(__name__).warning(
                    "Sentiment model '%s' did not fit on the GPU (%s). Falling back to CPU.",
                    model_name, str(e).splitlines()[0] if str(e) else type(e).__name__,
                )
                _SENTIMENT_BUNDLES[model_name] = pipeline(
                    "sentiment-analysis", model=model_name, device=-1)
        except Exception as e:
            _SENTIMENT_BUNDLES[model_name] = False
            if model_name not in _SENTIMENT_WARNED:
                logging.getLogger(__name__).warning(
                    "Sentiment model '%s' unavailable (%s). sentiment.label/confidence will be null.",
                    model_name, str(e).splitlines()[0] if str(e) else type(e).__name__,
                )
                _SENTIMENT_WARNED.add(model_name)
            return None
    return _SENTIMENT_BUNDLES[model_name]


def _sentiment_model_for(lang: str) -> str:
    """FinBERT for English, the multilingual model for anything else.

    Nothing is translated any more, so a Persian record is scored as Persian.
    FinBERT would tokenize it to mostly [UNK] and return a label built out of
    noise, which is worse than a null.
    """
    return _SENTIMENT_MODEL_NAME if (lang or "en") == "en" else _MULTILINGUAL_SENTIMENT_MODEL_NAME


def analyze_sentiment(text: str, lang: str = "en") -> dict:
    """Scores ``text`` in its own language -- English through FinBERT, any
    other language through the multilingual model.

    Returns {"label": "positive"|"neutral"|"negative", "confidence": 0-1} or
    {"label": None, "confidence": None} for empty input or when the model for
    that language is unavailable.
    """
    text = (text or "").strip()
    if not text:
        return {"label": None, "confidence": None}
    model = _load_sentiment_model(_sentiment_model_for(lang))
    if not model:
        return {"label": None, "confidence": None}
    # These models have a 512-token input window, so scoring a long transcript
    # in one call silently judges the whole video by its opening ~380 words.
    # Chunking and averaging the per-label probabilities keeps every part of
    # the text in the verdict; a short post is a single chunk.
    chunks = _split_nli_premise(text, max_chars=1200, max_chunks=12)
    if not chunks:
        return {"label": None, "confidence": None}
    try:
        outputs = model(chunks, truncation=True, max_length=512, top_k=None)
        totals = {}
        for scores in outputs:
            for entry in scores:
                # cardiffnlp emits LABEL_0/1/2, FinBERT emits words; both are
                # mapped onto the schema's three labels.
                label = _SENTIMENT_LABEL_MAP.get(str(entry["label"]).lower())
                if label is None:
                    continue
                totals[label] = totals.get(label, 0.0) + float(entry["score"])
        if not totals:
            return {"label": None, "confidence": None}
        label, total = max(totals.items(), key=lambda kv: kv[1])
        return {"label": label, "confidence": round(total / len(outputs), 4)}
    except Exception as e:
        logging.getLogger(__name__).warning(
            "Sentiment analysis failed: %s", str(e).splitlines()[0] if str(e) else type(e).__name__
        )
        return {"label": None, "confidence": None}


def analyze_text(text: str, lang: str, asset_text: str = None, context_terms=None) -> tuple:
    """Returns (sentiment_dict, asset_mention_list) for one record.

    The single entry point every scraper uses for text analysis, so the
    per-language handling is identical everywhere. Nothing is translated: the
    record's own text is scored in its own language (FinBERT for English, the
    multilingual model otherwise) and asset extraction reads the same original
    text, matching both Latin aliases/cashtags and the registry's non-Latin
    aliases against it.

    ``asset_text`` lets a scraper widen the asset input beyond the record's
    raw_text (Reddit adds the title, TikTok the video description); it defaults
    to ``text``.
    """
    asset_text = text if asset_text is None else asset_text
    return (
        analyze_sentiment(text, lang),
        extract_asset_mentions(asset_text, context_terms=context_terms, lang=lang),
    )


def language_allowed(lang: str, confidence: float, allowed, min_confidence: float) -> bool:
    """Whether a record's detected language clears the collection filter.

    A post is dropped only when the detector is CONFIDENT it is a language
    outside ``allowed``. A low-confidence verdict is the detector saying it
    does not know -- which is the normal outcome for a three-word post, a
    headline, or a caption full of hashtags -- and dropping those threw away
    real posts from the trusted accounts the run is supposed to collect.
    """
    if lang in allowed:
        return True
    return confidence < min_confidence


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S UTC"


def parse_record_timestamp(value):
    """Parses a stored published_at string back into a datetime for sorting."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.datetime.strptime(value, TIMESTAMP_FMT).replace(tzinfo=datetime.timezone.utc)
    except (ValueError, TypeError):
        return None


class RecordFile:
    """One output JSON file, loaded on demand and kept sorted by published_at.

    Exists because keyword mode on the account-driven platforms (TikTok, Truth
    Social) walks accounts once but routes each post into the file of whichever
    keyword it matched, so a single pass has several output files open at the
    same time. Appending never overwrites prior runs: existing records are read
    back in, new ones are inserted in chronological position, and the file is
    only rewritten if something was actually added.
    """

    def __init__(self, path: str, logger=None):
        self.path = path
        self._logger = logger or logging.getLogger(__name__)
        self.records = []
        self.ids = set()
        self._sort_keys = []
        self.added = 0
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        except Exception as e:
            self._logger.error(f"Error loading {self.path}: {e}")
            self.records = []
            return
        for record in self.records:
            record_id = record.get("record_id")
            if record_id:
                self.ids.add(record_id)
            parsed = parse_record_timestamp((record.get("time_stamps") or {}).get("published_at"))
            self._sort_keys.append(parsed.timestamp() if parsed else 0.0)

    def has(self, record_id: str) -> bool:
        self._load()
        return record_id in self.ids

    def add(self, record: dict, published_dt=None):
        self._load()
        sort_key = published_dt.timestamp() if published_dt else 0.0
        index = bisect.bisect_right(self._sort_keys, sort_key)
        self.records.insert(index, record)
        self._sort_keys.insert(index, sort_key)
        record_id = record.get("record_id")
        if record_id:
            self.ids.add(record_id)
        self.added += 1

    def save(self) -> bool:
        """Writes the file if anything was added. Returns whether it wrote."""
        if not self.added:
            return False
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=4, ensure_ascii=False)
        self._logger.info(
            f"Appended {self.added} new sorted records to {self.path} (Total: {len(self.records)})."
        )
        return True


def keyword_matches(text: str, keyword: str) -> bool:
    """True when every whitespace-separated token of ``keyword`` occurs in ``text``.

    Used by the keyword mode of the account-driven platforms, which have no
    free search API and so filter the trusted accounts' own posts instead of
    querying a search endpoint. Token-wise rather than whole-phrase matching so
    "oil prices" still matches "prices of oil"; case-insensitive and script
    agnostic, so it works the same on Persian text as on English.
    """
    haystack = (text or "").casefold()
    tokens = (keyword or "").casefold().split()
    return bool(tokens) and all(token in haystack for token in tokens)


def _normalized_asset_entry(entry: dict) -> dict:
    symbol = str(entry.get("Symbol") or entry.get("symbol") or "").strip().upper() or None
    canonical_name = str(entry.get("canonical_name") or "").strip() or symbol
    asset_class = str(entry.get("asset_class") or "").strip() or None
    asset_id = str(entry.get("asset_id") or "").strip() or (f"symbol:{symbol}" if symbol else None)
    aliases = entry.get("aliases") or []
    topics = entry.get("topics") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    if isinstance(topics, str):
        topics = [topics]
    aliases = [str(a).strip() for a in aliases if str(a).strip()]
    topics = [str(t).strip() for t in topics if str(t).strip()]
    if symbol:
        aliases.append(symbol)
    if canonical_name:
        aliases.append(canonical_name)
    seen = set()
    unique_aliases = []
    for alias in aliases:
        key = alias.casefold()
        if key not in seen:
            seen.add(key)
            unique_aliases.append(alias)
    return {
        "canonical_name": canonical_name,
        "Symbol": symbol,
        "asset_class": asset_class,
        "asset_id": asset_id,
        "aliases": unique_aliases,
        "topics": topics,
        "description": str(entry.get("description") or "").strip() or None,
        "context_inference": (
            entry.get("context_inference")
            if "context_inference" in entry
            else str(asset_class or "").strip().casefold() != "equity"
        ) is not False,
    }


def load_asset_registry(filepath: str = ASSET_REGISTRY_PATH) -> list:
    """Load the built-in registry plus optional project-specific definitions.

    Custom entries can add aliases, topics and a natural-language description.
    Entries with the same ``asset_id`` replace the built-in definition.
    """
    abs_path = os.path.abspath(filepath)
    try:
        mtime = os.path.getmtime(abs_path)
    except OSError:
        mtime = None
    cache_key = (abs_path, mtime)
    with _ASSET_REGISTRY_LOCK:
        cached = _ASSET_REGISTRY_CACHE.get(cache_key)
        if cached is not None:
            return cached
        combined = [_normalized_asset_entry(x) for x in _DEFAULT_ASSET_REGISTRY]
        if mtime is not None:
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                custom = payload.get("assets", []) if isinstance(payload, dict) else payload
                custom = [_normalized_asset_entry(x) for x in (custom or []) if isinstance(x, dict)]
                by_id = {x.get("asset_id"): x for x in combined if x.get("asset_id")}
                for entry in custom:
                    if entry.get("asset_id"):
                        by_id[entry["asset_id"]] = entry
                    else:
                        combined.append(entry)
                combined = list(by_id.values()) + [x for x in combined if not x.get("asset_id")]
            except Exception as e:
                logging.getLogger(__name__).warning("Could not load %s: %s", filepath, e)
        _ASSET_REGISTRY_CACHE.clear()
        _ASSET_REGISTRY_CACHE[cache_key] = combined
        return combined


def _find_alias(text: str, alias: str):
    r"""Finds ``alias`` in ``text`` on word boundaries, in any script.

    The boundary class is \w (Unicode-aware), not [A-Za-z0-9_]: with the ASCII
    class a Persian alias like "طلا" matched inside "طلایی", because the
    following Persian letter is not an ASCII word character and therefore
    counted as a boundary.
    """
    if not text or not alias:
        return None
    pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE | re.UNICODE)
    return pattern.search(text)


_LATIN_RE = re.compile(r"[A-Za-z]")


def _is_latin_text(text: str, min_ratio: float = 0.5) -> bool:
    """Whether ``text`` is mostly Latin script.

    The NLI relevance model is English-only. Fed a Persian premise it does not
    return "unsure" -- it returns a confident verdict on tokens it never saw,
    which used to CONTRADICT perfectly real mentions and delete them. So the
    model is consulted only for text it can actually read.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if _LATIN_RE.match(c)) / len(letters) >= min_ratio


def _asset_hypothesis(entry: dict) -> str:
    name = entry.get("canonical_name") or entry.get("Symbol") or "this asset"
    symbol = entry.get("Symbol")
    desc = entry.get("description") or entry.get("asset_class") or "financial asset"
    topics = ", ".join((entry.get("topics") or [])[:8])
    sym_text = f" ({symbol})" if symbol else ""
    topic_text = f" Related market topics include {topics}." if topics else ""
    return (
        f"This information is materially about, relevant to, or capable of affecting "
        f"the financial asset {name}{sym_text}, {desc}.{topic_text}"
    )


def _load_asset_nli_model():
    """Lazy-load a compact NLI model used to estimate asset relevance.

    If the model/dependencies are unavailable, the caller receives ``None``.
    In that case direct mentions are still extracted, but confidence is left
    null and semantic/context inference is disabled rather than fabricated.
    """
    global _ASSET_NLI_BUNDLE, _ASSET_MODEL_WARNING_EMITTED
    if _ASSET_NLI_MODEL_NAME.strip().lower() in {"", "none", "off", "disabled"}:
        return None
    if _ASSET_NLI_BUNDLE is False:
        return None
    if _ASSET_NLI_BUNDLE is not None:
        return _ASSET_NLI_BUNDLE
    with _ASSET_NLI_LOCK:
        if _ASSET_NLI_BUNDLE is not None:
            return None if _ASSET_NLI_BUNDLE is False else _ASSET_NLI_BUNDLE
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(_ASSET_NLI_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(_ASSET_NLI_MODEL_NAME)
            if _ASSET_NLI_DEVICE in {"auto", "cuda"} and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            model.to(device)
            model.eval()

            id2label = {int(k): str(v).lower() for k, v in (model.config.id2label or {}).items()}
            # "not_entailment" also contains "entail", so match the prefix, not
            # the substring. A model that exposes no contradiction/neutral
            # class leaves those None rather than aliasing them onto index 0/2,
            # which would mislabel its predictions.
            entail_idx = next((i for i, label in id2label.items() if label.startswith("entail")), 1)
            contradiction_idx = next((i for i, label in id2label.items() if "contrad" in label), None)
            neutral_idx = next((i for i, label in id2label.items() if label == "neutral"), None)
            _ASSET_NLI_BUNDLE = (tokenizer, model, torch, device, entail_idx, contradiction_idx, neutral_idx)
        except Exception as e:
            _ASSET_NLI_BUNDLE = False
            if not _ASSET_MODEL_WARNING_EMITTED:
                logging.getLogger(__name__).warning(
                    "Asset NLI model unavailable (%s). Direct mentions will be kept with confidence=null; "
                    "semantic/context-only asset inference is disabled until transformers/torch and the model are available.",
                    str(e).splitlines()[0] if str(e) else type(e).__name__,
                )
                _ASSET_MODEL_WARNING_EMITTED = True
            return None
    return _ASSET_NLI_BUNDLE


def _demote_nli_to_cpu(reason: str):
    """Moves the loaded NLI model to CPU for the rest of this process.

    VRAM is shared: a second scraper started by hand (or a long batch) can
    exhaust it mid-run, and an uncaught CUDA OOM here loses every record of
    the keyword/account being processed, not just one asset score. Same
    trade-off the Whisper circuit breaker already makes -- slower, but the
    run keeps producing records.
    """
    global _ASSET_NLI_BUNDLE
    tokenizer, model, torch, device, *rest = _ASSET_NLI_BUNDLE
    if device.type == "cpu":
        return device
    cpu = torch.device("cpu")
    model.to(cpu)
    torch.cuda.empty_cache()
    _ASSET_NLI_BUNDLE = (tokenizer, model, torch, cpu, *rest)
    logging.getLogger(__name__).warning(
        "Asset NLI model moved to CPU after CUDA OOM (%s). Scoring continues on CPU.", reason
    )
    return cpu


def _split_nli_premise(text: str, max_chars: int = 1400, max_chunks: int = 12) -> list:
    """Split long content into bounded semantic chunks for NLI scoring.

    X posts normally remain one chunk. Long YouTube transcripts are evaluated
    across several chunks so an asset discussed later in the video is not lost
    merely because a transformer input has a finite token window.
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], []
    current_len = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and current_len + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
            if len(chunks) >= max_chunks:
                break
        if len(sentence) > max_chars:
            # Extremely long subtitle segments are cut on character boundaries;
            # this is only an input-window safeguard, not a relevance heuristic.
            for i in range(0, len(sentence), max_chars):
                if len(chunks) >= max_chunks:
                    break
                piece = sentence[i:i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue
        current.append(sentence)
        current_len += len(sentence) + 1
    if current and len(chunks) < max_chunks:
        chunks.append(" ".join(current))
    return chunks[:max_chunks]


def _nli_asset_scores(premise: str, entries: list) -> dict:
    """Return model-derived entailment probabilities and predicted NLI labels.

    For long content, each asset keeps the strongest entailment evidence found
    in any semantic chunk. This avoids the first-512-token bias that would
    otherwise make YouTube asset extraction much less reliable than X.
    """
    chunks = _split_nli_premise(premise)
    if not chunks or not entries:
        return {}
    bundle = _load_asset_nli_model()
    if not bundle:
        return {}
    tokenizer, model, torch, device, entail_idx, contradiction_idx, neutral_idx = bundle
    hypotheses = [_asset_hypothesis(e) for e in entries]
    out = {}

    pairs = []
    pair_meta = []
    for chunk in chunks:
        for entry, hypothesis in zip(entries, hypotheses):
            pairs.append((chunk, hypothesis))
            pair_meta.append(entry)

    with torch.no_grad():
        for start in range(0, len(pairs), _ASSET_NLI_BATCH_SIZE):
            batch_pairs = pairs[start:start + _ASSET_NLI_BATCH_SIZE]
            batch_entries = pair_meta[start:start + _ASSET_NLI_BATCH_SIZE]
            encoded = tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            try:
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
            except torch.cuda.OutOfMemoryError as e:
                device = _demote_nli_to_cpu(str(e).splitlines()[0])
                logits = model(**{k: v.to(device) for k, v in encoded.items()}).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            pred = probs.argmax(dim=-1).tolist()
            for entry, row, pred_idx in zip(batch_entries, probs, pred):
                if pred_idx == entail_idx:
                    label = "entailment"
                elif contradiction_idx is not None and pred_idx == contradiction_idx:
                    label = "contradiction"
                elif neutral_idx is not None and pred_idx == neutral_idx:
                    label = "neutral"
                else:
                    label = "not_entailment"
                key = entry.get("asset_id") or entry.get("Symbol") or entry.get("canonical_name")
                candidate = {"entailment": float(row[entail_idx].item()), "label": label}
                current = out.get(key)
                if current is None or candidate["entailment"] > current["entailment"]:
                    out[key] = candidate
    return out

def _context_candidates(context_terms, registry: list) -> list:
    context_text = " ; ".join(str(t).strip() for t in context_terms if str(t).strip())
    if not context_text:
        return []
    registry_sig = tuple((e.get("asset_id"), e.get("canonical_name"), tuple(e.get("topics") or [])) for e in registry)
    cache_key = (context_text.casefold(), registry_sig)
    with _ASSET_CONTEXT_CACHE_LOCK:
        cached = _ASSET_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached

    eligible = [e for e in registry if e.get("context_inference", True)]
    scores = _nli_asset_scores(f"Search topic: {context_text}", eligible)
    ranked = []
    for entry in eligible:
        key = entry.get("asset_id") or entry.get("Symbol") or entry.get("canonical_name")
        score = scores.get(key)
        # NLI itself decides whether the topic entails relevance; there is no
        # hand-written confidence percentage or numeric relevance threshold.
        if score and score["label"] == "entailment":
            ranked.append((score["entailment"], entry, score))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = ranked if _ASSET_CONTEXT_MAX_RESULTS <= 0 else ranked[:_ASSET_CONTEXT_MAX_RESULTS]
    result = [(entry, score_info) for _, entry, score_info in selected]
    with _ASSET_CONTEXT_CACHE_LOCK:
        _ASSET_CONTEXT_CACHE[cache_key] = result
    return result


def extract_asset_mentions(text: str, context_terms=None, lang: str = "en",
                           registry_path: str = ASSET_REGISTRY_PATH) -> list:
    """Extract direct and contextual financial-asset relevance.

    Confidence is produced by a Natural Language Inference model, not by
    hard-coded per-rule percentages. Direct literal/cashtag mentions are always
    preserved; the model then judges whether the content is materially about or
    could affect that asset. Context-only assets are considered when the search
    topic itself entails an asset relationship and the actual record text also
    entails that relationship. Their confidence is the geometric mean of the
    two independent entailment probabilities, so a strong keyword alone cannot
    manufacture a high-confidence asset association.
    """
    text = text or ""
    if context_terms is None:
        context_terms = []
    elif isinstance(context_terms, str):
        context_terms = [context_terms]

    registry = load_asset_registry(registry_path)
    # The NLI relevance model reads English only. For a non-Latin record it is
    # skipped entirely: direct alias/cashtag matches are facts that stand on
    # their own and are emitted with a null confidence, rather than being
    # filtered by a model that cannot read the premise. This is what made a
    # Persian post saying "USDT" come back with no asset_mention at all.
    model_readable = _is_latin_text(text) and (lang or "en") not in {"fa", "ar", "ru", "zh", "ja", "ko", "he"}
    symbol_map = {e.get("Symbol", "").upper(): e for e in registry if e.get("Symbol")}
    direct = {}
    unknown = {}

    for match in _CASHTAG_RE.finditer(text):
        symbol = match.group(1).upper()
        entry = symbol_map.get(symbol)
        if entry:
            direct[entry.get("asset_id") or symbol] = (entry, match.group(0))
        else:
            unknown[symbol] = {
                "mentioned_text": match.group(0),
                "canonical_name": None,
                "Symbol": symbol,
                "asset_class": None,
                "asset_id": None,
                "confidence": None,
            }

    for entry in registry:
        best = None
        for alias in sorted(entry.get("aliases", []), key=len, reverse=True):
            m = _find_alias(text, alias)
            if m:
                best = m.group(0)
                break
        if best:
            direct[entry.get("asset_id") or entry.get("Symbol") or entry.get("canonical_name")] = (entry, best)

    context = _context_candidates(context_terms, registry) if model_readable else []
    candidate_map = {key: pair for key, pair in direct.items()}
    for entry, query_score in context:
        key = entry.get("asset_id") or entry.get("Symbol") or entry.get("canonical_name")
        candidate_map.setdefault(key, (entry, None))

    entries = [pair[0] for pair in candidate_map.values()]
    text_scores = _nli_asset_scores(text, entries) if model_readable else {}
    query_score_by_key = {
        entry.get("asset_id") or entry.get("Symbol") or entry.get("canonical_name"): score
        for entry, score in context
    }

    results = []
    for key, (entry, surface) in candidate_map.items():
        text_score = text_scores.get(key)
        confidence = None
        if surface is not None:
            # A literal mention is a fact; relevance confidence comes only from
            # the model's entailment probability for the actual record text.
            confidence = text_score["entailment"] if text_score else None
            # A word can match an alias without the content being about the
            # asset ("Shell" the noun, "gold" the colour). When the model
            # actively contradicts the relevance claim and its entailment
            # probability is negligible, that is a false positive, not a
            # low-confidence hit -- emitting it would pollute downstream
            # asset joins with mentions the model already rejected.
            # Deliberately requires an explicit "contradiction": a merely
            # neutral verdict means the model is unsure, and dropping those
            # would discard real mentions the model simply could not confirm.
            if (
                text_score
                and text_score["label"] == "contradiction"
                and confidence < _ASSET_DIRECT_MIN_CONFIDENCE
            ):
                continue
        else:
            # Context-only inference is emitted only if BOTH the search topic
            # and the record text are independently classified as entailment.
            query_score = query_score_by_key.get(key)
            if not text_score or not query_score:
                continue
            if text_score["label"] != "entailment" or query_score["label"] != "entailment":
                continue
            confidence = (text_score["entailment"] * query_score["entailment"]) ** 0.5

        results.append({
            "mentioned_text": surface,
            "canonical_name": entry.get("canonical_name"),
            "Symbol": entry.get("Symbol"),
            "asset_class": entry.get("asset_class"),
            "asset_id": entry.get("asset_id"),
            "confidence": round(float(confidence), 4) if confidence is not None else None,
        })

    results.extend(unknown.values())
    if results:
        return sorted(results, key=lambda x: (x["confidence"] is None, -(x["confidence"] or 0.0), str(x.get("Symbol") or "")))

    return [{
        "mentioned_text": None,
        "canonical_name": None,
        "Symbol": None,
        "asset_class": None,
        "asset_id": None,
        "confidence": None,
    }]


def _valid_source_id(platform: str, value) -> bool:
    value = str(value or "").strip()
    if platform == "X":
        return bool(re.fullmatch(r"\d+", value))
    if platform == "YouTube":
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{11}", value))
    if platform == "Telegram":
        return bool(re.fullmatch(r"\d+", value))
    if platform == "Reddit":
        # Reddit submission IDs are base36, without the "t3_" type prefix.
        return bool(re.fullmatch(r"[a-z0-9]{4,13}", value))
    if platform == "TikTok":
        return bool(re.fullmatch(r"\d{6,25}", value))
    if platform == "TruthSocial":
        return bool(re.fullmatch(r"\d+", value))
    return bool(value and value.lower() not in {"n/a", "unknown", "none", "null"})


def _valid_author_id(platform: str, value) -> bool:
    value = str(value or "").strip().lstrip("@")
    if platform == "X":
        return bool(re.fullmatch(r"[A-Za-z0-9_]{1,15}", value))
    if platform == "YouTube":
        # Canonical YouTube channel IDs are UC + 22 URL-safe characters.
        return bool(re.fullmatch(r"UC[A-Za-z0-9_-]{22}", value))
    if platform == "Telegram":
        # Public Telegram usernames are 5-32 alphanumeric/underscore characters.
        return bool(re.fullmatch(r"[A-Za-z0-9_]{5,32}", value))
    if platform == "Reddit":
        # The subreddit is the trusted source, so it plays the author role
        # here the same way a channel does for Telegram/YouTube.
        return bool(re.fullmatch(r"[A-Za-z0-9_]{3,21}", value))
    if platform == "TikTok":
        # TikTok usernames allow dots as well as letters/digits/underscores.
        return bool(re.fullmatch(r"[A-Za-z0-9_.]{1,24}", value))
    if platform == "TruthSocial":
        return bool(re.fullmatch(r"[A-Za-z0-9_]{1,30}", value))
    return bool(value and value.lower() not in {"n/a", "unknown", "none", "null"})


def _valid_source_url(platform: str, url, author_id=None, source_id=None) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        parsed = urlparse(url.strip())
    except Exception:
        return False
    host = parsed.netloc.lower().split(":", 1)[0]

    if platform == "X":
        if host not in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}:
            return False
        m = re.fullmatch(r"/([A-Za-z0-9_]{1,15})/status/(\d+)/?", parsed.path)
        if not m:
            return False
        if author_id and m.group(1).casefold() != str(author_id).lstrip("@").casefold():
            return False
        if source_id and m.group(2) != str(source_id):
            return False
        return True

    if platform == "YouTube":
        if host in {"youtu.be", "www.youtu.be"}:
            found_id = parsed.path.strip("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            found_id = parse_qs(parsed.query).get("v", [None])[0]
            if not found_id and parsed.path.startswith(("/shorts/", "/live/")):
                parts = parsed.path.strip("/").split("/")
                found_id = parts[1] if len(parts) > 1 else None
        else:
            return False
        return bool(found_id and (not source_id or found_id == str(source_id)))

    if platform == "Reddit":
        if host not in {"reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com"}:
            return False
        m = re.match(r"/r/([A-Za-z0-9_]{3,21})/comments/([a-z0-9]{4,13})(?:/|$)", parsed.path)
        if not m:
            return False
        if author_id and m.group(1).casefold() != str(author_id).lstrip("@").casefold():
            return False
        if source_id and m.group(2) != str(source_id):
            return False
        return True

    if platform == "TikTok":
        if host not in {"tiktok.com", "www.tiktok.com", "m.tiktok.com"}:
            return False
        m = re.fullmatch(r"/@([A-Za-z0-9_.]{1,24})/video/(\d{6,25})/?", parsed.path)
        if not m:
            return False
        if author_id and m.group(1).casefold() != str(author_id).lstrip("@").casefold():
            return False
        if source_id and m.group(2) != str(source_id):
            return False
        return True

    if platform == "TruthSocial":
        if host not in {"truthsocial.com", "www.truthsocial.com"}:
            return False
        m = re.fullmatch(r"/@([A-Za-z0-9_]{1,30})/(?:posts/)?(\d+)/?", parsed.path)
        if not m:
            return False
        if author_id and m.group(1).casefold() != str(author_id).lstrip("@").casefold():
            return False
        if source_id and m.group(2) != str(source_id):
            return False
        return True

    if platform == "Telegram":
        if host not in {"t.me", "www.t.me"}:
            return False
        m = re.fullmatch(r"/([A-Za-z0-9_]{5,32})/(\d+)/?", parsed.path)
        if not m:
            return False
        if author_id and m.group(1).casefold() != str(author_id).lstrip("@").casefold():
            return False
        if source_id and m.group(2) != str(source_id):
            return False
        return True

    return parsed.scheme in {"http", "https"} and bool(host)


def _parse_quality_timestamp(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or value == "N/A":
        return None
    raw = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S UTC", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"):
        try:
            dt = datetime.datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc)
        except (ValueError, TypeError):
            pass
    try:
        dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except (ValueError, TypeError):
        return None


def _record_id_consistent(record: dict) -> bool:
    source = record.get("Source") or {}
    platform = source.get("platform")
    record_id = str(record.get("record_id") or "")
    author_id = str(source.get("author_id") or "").lstrip("@")
    source_id = str(source.get("Source_id") or "")
    if platform in _KNOWN_PLATFORMS:
        return record_id == f"{platform}_{author_id}_{source_id}"
    return bool(record_id)


def _engagement_checks(record: dict) -> list:
    engagement = record.get("Engagement")
    if not isinstance(engagement, dict):
        return [("valid_engagement", False)]
    checks = []
    for field in ("Views", "likes", "Comments", "shares"):
        value = engagement.get(field)
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
        checks.append((f"valid_engagement_{field}", ok))
    return checks


def _media_checks(record: dict) -> list:
    media = record.get("media")
    source = record.get("Source") or {}
    if not isinstance(media, dict):
        return [("valid_media", False)]
    has_media = media.get("has_media")
    checks = [("valid_media_has_media", isinstance(has_media, bool))]
    if has_media is True:
        checks.append(("valid_media_type", bool(media.get("media_type"))))
        url = media.get("media_url")
        checks.append(("valid_media_url", isinstance(url, str) and url.startswith(("http://", "https://"))))
        if source.get("platform") == "YouTube":
            checks.append(("consistent_youtube_media_type", media.get("media_type") == "video"))
            checks.append(("consistent_youtube_media_url", media.get("media_url") == source.get("url")))
        if source.get("platform") == "Telegram":
            checks.append(("consistent_telegram_media_url", media.get("media_url") == source.get("url")))
        if source.get("platform") == "TikTok":
            checks.append(("consistent_tiktok_media_type", media.get("media_type") == "video"))
            checks.append(("consistent_tiktok_media_url", media.get("media_url") == source.get("url")))
    elif has_media is False:
        checks.append(("valid_media_type", media.get("media_type") is None))
        checks.append(("valid_media_url", media.get("media_url") is None))
    else:
        checks.extend([("valid_media_type", False), ("valid_media_url", False)])
    return checks


def _dedup_checks(record: dict) -> list:
    dedup = record.get("deduplication")
    if not isinstance(dedup, dict):
        return [("valid_deduplication", False)]
    content_hash = dedup.get("content_hash")
    is_duplicate = dedup.get("is_duplicate")
    group_id = dedup.get("duplicate_group_id")
    original = dedup.get("original_record_id")
    record_id = record.get("record_id")
    return [
        ("valid_content_hash", isinstance(content_hash, str) and bool(re.fullmatch(r"[0-9a-fA-F]{32}", content_hash))),
        ("valid_duplicate_flag", isinstance(is_duplicate, bool)),
        ("valid_duplicate_group_id", bool(group_id)),
        ("valid_original_record_id", (is_duplicate is True and bool(original)) or (is_duplicate is False and original is None)),
        ("consistent_duplicate_group", (is_duplicate is True and bool(group_id)) or (is_duplicate is False and group_id == record_id)),
    ]


def _asset_checks(record: dict) -> list:
    mentions = record.get("asset_mention")
    if not isinstance(mentions, list) or not mentions:
        return [("valid_asset_mention", False)]
    # Canonical null placeholder means "no asset identified" and is valid.
    if len(mentions) == 1 and isinstance(mentions[0], dict) and all(
        mentions[0].get(k) is None
        for k in ("mentioned_text", "canonical_name", "Symbol", "asset_class", "asset_id", "confidence")
    ):
        return [("valid_asset_mention", True)]

    checks = []
    for idx, item in enumerate(mentions):
        if not isinstance(item, dict):
            checks.append((f"valid_asset_mention_{idx}", False))
            continue
        resolved = bool(item.get("canonical_name") and item.get("Symbol") and item.get("asset_class") and item.get("asset_id"))
        confidence = item.get("confidence")
        # ``confidence is None`` is the documented degraded mode of
        # _load_asset_nli_model (model/deps unavailable), not a defect in this
        # record -- an environment condition must not mark every asset-bearing
        # record incomplete. A present value still has to be in range.
        confidence_ok = confidence is None or (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0.0 <= float(confidence) <= 1.0
        )
        # An unresolved cashtag is still a valid observed mention, but the
        # resolution/confidence checks expose that it is not a complete asset mapping.
        checks.append((f"asset_{idx}_observed", bool(item.get("mentioned_text") or resolved)))
        checks.append((f"asset_{idx}_resolved", resolved))
        checks.append((f"asset_{idx}_confidence", confidence_ok if resolved else confidence is None))
    return checks


def _sentiment_checks(record: dict) -> list:
    sentiment = record.get("sentiment")
    if not isinstance(sentiment, dict):
        return [("valid_sentiment", False)]
    label = sentiment.get("label")
    confidence = sentiment.get("confidence")
    # label=None/confidence=None is the documented degraded mode of
    # analyze_sentiment (model unavailable), the same rule already applied to
    # asset confidence -- an environment condition, not a defect in this record.
    if label is None and confidence is None:
        return [("valid_sentiment", True)]
    return [(
        "valid_sentiment",
        label in {"positive", "neutral", "negative"}
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0.0 <= float(confidence) <= 1.0,
    )]


def _sentiment_category_score(record: dict, checks: list) -> float:
    """Mirrors _asset_category_score: the boolean check and the model's own
    confidence value have equal standing, and a null (model-unavailable)
    sentiment stays a perfect score in this dimension."""
    sentiment = record.get("sentiment") if isinstance(record.get("sentiment"), dict) else {}
    confidence = sentiment.get("confidence")
    values = [100.0 if passed else 0.0 for _, passed in checks]
    if sentiment.get("label") is not None and isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        values.append(max(0.0, min(1.0, float(confidence))) * 100.0)
    return sum(values) / len(values) if values else 100.0


def _category_score(checks: list) -> float:
    if not checks:
        return 100.0
    return 100.0 * sum(1 for _, passed in checks if passed) / len(checks)


def _extraction_category_score(checks: list, reliability=None) -> float:
    """Score extraction quality from boolean checks plus optional reliability.

    Reliability is folded into this dimension rather than appended as a
    dimension of its own, and a platform that does not report it counts as
    fully reliable. Both parts are needed for the score to mean the same thing
    everywhere: otherwise a platform that reports reliability is scored over a
    different number of values than one that does not, and two records with
    identical defects get different scores purely because of which scraper
    produced them.
    """
    values = [100.0 if passed else 0.0 for _, passed in checks]
    try:
        rel = 1.0 if reliability is None else max(0.0, min(1.0, float(reliability)))
    except (TypeError, ValueError):
        rel = 1.0
    values.append(rel * 100.0)
    return sum(values) / len(values) if values else 100.0


def _asset_category_score(record: dict, checks: list) -> float:
    """Score asset-field quality from structure plus model certainty.

    Boolean schema/resolution checks and model-generated confidence values have
    equal standing inside the asset dimension. A canonical all-null placeholder
    (no asset identified) remains a perfect asset-field state.
    """
    mentions = record.get("asset_mention")
    if (
        isinstance(mentions, list)
        and len(mentions) == 1
        and isinstance(mentions[0], dict)
        and all(mentions[0].get(k) is None for k in ("mentioned_text", "canonical_name", "Symbol", "asset_class", "asset_id", "confidence"))
    ):
        return 100.0

    values = [100.0 if passed else 0.0 for _, passed in checks]
    if isinstance(mentions, list):
        for item in mentions:
            if not isinstance(item, dict):
                continue
            resolved = bool(item.get("canonical_name") and item.get("Symbol") and item.get("asset_class") and item.get("asset_id"))
            confidence = item.get("confidence")
            if resolved and isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                values.append(max(0.0, min(1.0, float(confidence))) * 100.0)
    return sum(values) / len(values) if values else 0.0


def build_quality(
    record: dict,
    complete_raw_text: bool = True,
    extraction_errors=None,
    extraction_reliability=None,
) -> dict:
    """Evaluate record quality on a 0-100 scale without arbitrary field weights.

    Validation is grouped into independent quality dimensions (identity,
    content, timestamps, engagement/media, deduplication, and asset mapping).
    Every check inside a dimension has equal standing. The overall score is the
    harmonic mean of the dimension scores, which prevents one weak dimension
    from being hidden by many unrelated good fields. A perfect record is 100.

    ``updated_at=None`` is explicitly valid. It means no edit timestamp was
    available, which is different from missing required publication/collection
    timestamps.
    """
    extraction_errors = [str(e) for e in (extraction_errors or []) if str(e).strip()]
    source = record.get("Source") or {}
    content = record.get("Content") or {}
    stamps = record.get("time_stamps") or {}
    platform = source.get("platform")

    published = _parse_quality_timestamp(stamps.get("published_at"))
    collected = _parse_quality_timestamp(stamps.get("collected_at"))
    updated_raw = stamps.get("updated_at")
    updated = _parse_quality_timestamp(updated_raw) if updated_raw is not None else None
    updated_valid = updated_raw is None or (
        updated is not None
        and (published is None or updated >= published)
        and (collected is None or updated <= collected)
    )

    identity = [
        ("valid_platform", platform in _KNOWN_PLATFORMS),
        ("valid_source_type", source.get("Source_type") == "Social_media"),
        ("valid_source_id", _valid_source_id(platform, source.get("Source_id"))),
        ("valid_author_id", _valid_author_id(platform, source.get("author_id"))),
        ("valid_author_name", bool(str(source.get("author_name") or "").strip()) and str(source.get("author_name")).strip().lower() not in {"n/a", "unknown"}),
        ("valid_source_name", bool(str(source.get("Source_name") or "").strip()) and str(source.get("Source_name")).strip().lower() not in {"n/a", "unknown"}),
        ("consistent_source_author_name", source.get("Source_name") == source.get("author_name")),
        ("valid_url", _valid_source_url(platform, source.get("url"), source.get("author_id"), source.get("Source_id"))),
        ("consistent_record_id", _record_id_consistent(record)),
    ]

    expected_type = _PLATFORM_CONTENT_TYPES.get(platform)
    content_checks = [
        ("complete_raw_text", bool(complete_raw_text and isinstance(content.get("raw_text"), str) and content.get("raw_text").strip())),
        ("valid_content_type", bool(expected_type and content.get("Content_type") == expected_type)),
        ("valid_language", bool(content.get("language") and content.get("language") not in {"unknown", "N/A"})),
    ]
    if platform in _TITLED_PLATFORMS:
        content_checks.append(("valid_title", bool(str(content.get("title") or "").strip()) and content.get("title") != "N/A"))
    # No translated_text check: the pipeline does not translate, so a record
    # carries only its original text in its own language.

    timestamp_checks = [
        ("valid_published_at", published is not None),
        ("valid_collected_at", collected is not None),
        ("valid_timestamp_order", published is not None and collected is not None and published <= collected),
        ("valid_updated_at", updated_valid),
    ]

    asset_checks = _asset_checks(record)
    sentiment_checks = _sentiment_checks(record)
    extraction_checks = [
        ("extraction_has_text", isinstance(content.get("raw_text"), str) and bool(content.get("raw_text").strip())),
        ("extraction_complete", bool(complete_raw_text)),
        ("no_extraction_errors", not extraction_errors),
    ]
    categories = {
        "identity": identity,
        "content": content_checks,
        "timestamps": timestamp_checks,
        "extraction": extraction_checks,
        "engagement_media": _engagement_checks(record) + _media_checks(record),
        "deduplication": _dedup_checks(record),
        "assets": asset_checks,
        "sentiment": sentiment_checks,
    }

    scores = []
    for name, checks in categories.items():
        if name == "assets":
            scores.append(_asset_category_score(record, checks))
        elif name == "sentiment":
            scores.append(_sentiment_category_score(record, checks))
        elif name == "extraction":
            scores.append(_extraction_category_score(checks, extraction_reliability))
        else:
            scores.append(_category_score(checks))

    if any(score <= 0 for score in scores):
        quality_score = 0.0
    else:
        quality_score = len(scores) / sum(1.0 / score for score in scores)
    quality_score = round(quality_score, 2)
    if quality_score == 100.0:
        quality_score = 100

    all_checks = [check for checks in categories.values() for check in checks]
    missing_fields = [name for name, passed in all_checks if not passed]
    is_complete = not missing_fields and not extraction_errors

    return {
        "is_complete": is_complete,
        "missing_fields": missing_fields,
        "extraction_errors": extraction_errors,
        "quality_score": quality_score,
    }


# ---------------------------------------------------------------------------
# Whisper audio transcription
#
# Shared by every scraper whose platform serves video/audio (YouTube, TikTok):
# the model is expensive to load and the circuit breaker below is only
# meaningful if all of them count failures against the same state, so this
# lives here rather than being duplicated per scraper.
# ---------------------------------------------------------------------------

# "base.en" is English-only and will hallucinate fluent-sounding but
# meaningless English text when given non-English audio (it has no notion
# of any other language, so it just pattern-matches sounds to English
# words). Using a multilingual model + task="translate" (below) instead
# lets Whisper auto-detect the spoken language and translate it to English
# properly, matching the quality of YouTube's own auto-translate captions.
# "small" is a reasonable speed/quality balance for an RTX 3070; bump to
# "medium" if translation quality on non-English sources still looks weak.
# Whisper's task for the audio fallback: "translate" emits an English
# transcript of non-English speech, "transcribe" keeps the spoken language.
# This is the ONE remaining place a non-English source can become English, and
# it is deliberate: for a video with no captions the transcript IS the record's
# only text, so there is no original being replaced. Set WHISPER_TASK=transcribe
# to keep spoken Persian as Persian -- sentiment then routes to the
# multilingual model like any other non-English record.
WHISPER_TASK = env_str("WHISPER_TASK", "translate").strip().lower()
if WHISPER_TASK not in {"translate", "transcribe"}:
    WHISPER_TASK = "translate"

WHISPER_MODEL_SIZE = env_str("WHISPER_MODEL_SIZE", "small")

_whisper_model = None
_model_lock = threading.Lock()

# Circuit breaker: if the local Whisper backend is broken (e.g. a CUDA-enabled
# ctranslate2 wheel that still tries to dlopen libcublas even in CPU mode),
# don't keep burning minutes downloading audio for every video only to fail
# at transcribe time. After a few consecutive library-load failures, disable
# Whisper for the rest of the run and fall back to subtitle-only mode.
_WHISPER_FAILURE_LIMIT = 3
_whisper_failure_count = 0
_whisper_disabled = False
_whisper_state_lock = threading.Lock()

# Substrings that indicate an environment/library problem (not a per-video
# problem) -- e.g. "Library libcublas.so.12 is not found or cannot be loaded".
_ENV_FAILURE_MARKERS = ("libcublas", "libcudnn", "cannot be loaded", "cuda")


def _whisper_logger(logger):
    return logger if logger is not None else logging.getLogger(__name__)


def get_whisper_model(logger=None):
    """Lazily loads and caches the faster-whisper model."""
    global _whisper_model
    log = _whisper_logger(logger)
    with _model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                log.info(f"Initializing faster-whisper model ({WHISPER_MODEL_SIZE})... (this may take a moment on first run)")
                try:
                    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
                    log.info(f"faster-whisper model ({WHISPER_MODEL_SIZE}) successfully loaded and ready (GPU/CUDA).")
                except Exception as gpu_err:
                    log.warning(
                        f"Could not initialize Whisper on GPU ({str(gpu_err).splitlines()[0]}). "
                        "Falling back to CPU (int8)."
                    )
                    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
                    log.info(f"faster-whisper model ({WHISPER_MODEL_SIZE}) successfully loaded and ready (CPU fallback).")
            except ImportError:
                log.warning("faster-whisper is not installed. Falling back strictly to subtitle scraping.")
                _whisper_model = False
    return _whisper_model


def _record_whisper_env_failure(error_text: str, logger=None) -> bool:
    """Tracks consecutive environment-level Whisper failures (missing CUDA
    libs, etc). Returns True once the failure limit is hit and Whisper has
    been disabled for the rest of this run."""
    global _whisper_failure_count, _whisper_disabled
    lowered = error_text.lower()
    if not any(marker in lowered for marker in _ENV_FAILURE_MARKERS):
        return False

    with _whisper_state_lock:
        if _whisper_disabled:
            return True
        _whisper_failure_count += 1
        if _whisper_failure_count >= _WHISPER_FAILURE_LIMIT:
            _whisper_disabled = True
            _whisper_logger(logger).error(
                f"Whisper failed {_whisper_failure_count} times in a row with an "
                f"environment/library error ('{error_text}'). Disabling audio "
                "transcription for the rest of this run and falling back to "
                "subtitle-only mode. Fix: use CPU transcription or follow "
                "faster-whisper's GPU setup for compatible CUDA/cuDNN libraries "
                "and library paths before the next run (see README.md)."
            )
            return True
    return False


@with_retry(max_attempts=2, base_delay=3.0, exceptions=(Exception,))
def _download_audio(media_url: str, out_template: str, extra_opts: dict = None) -> None:
    import yt_dlp

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'socket_timeout': 25,
        'retries': 3,
        'fragment_retries': 3,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
    }
    ydl_opts.update(extra_opts or {})
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([media_url])


def transcribe_audio(media_url: str, media_id: str, duration: int = 0,
                     max_duration: int = 0, extra_opts: dict = None, logger=None) -> str:
    """
    Downloads a post's audio track and returns an English transcript, or None.

    ``max_duration`` (seconds, 0 disables) skips anything longer, so a 24/7
    live broadcast can never stall a run on an unbounded download.
    """
    log = _whisper_logger(logger)
    if max_duration and duration and duration > max_duration:
        log.info(f"Skipping audio download for {media_id}: duration ({duration}s) exceeds {max_duration}s cap.")
        return None

    model = get_whisper_model(logger)
    if not model or _whisper_disabled:
        return None

    log.info(f"Downloading lightweight audio for {media_id}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, f"{media_id}.%(ext)s")

        try:
            _download_audio(media_url, out_template, extra_opts)

            audio_file = os.path.join(tmpdir, f"{media_id}.mp3")
            if not os.path.exists(audio_file):
                files = os.listdir(tmpdir)
                if not files:
                    return None
                audio_file = os.path.join(tmpdir, files[0])

            log.info(f"Transcribing audio for {media_id} with Whisper...")
            start_t = time.time()
            # task="translate" makes Whisper auto-detect the spoken language
            # and translate it directly to English, instead of transcribing
            # verbatim in whatever language is detected. Combined with a
            # multilingual model (not "*.en"), this is what actually handles
            # non-English source audio correctly.
            segments, info = model.transcribe(audio_file, beam_size=2, task=WHISPER_TASK)
            transcript_text = " ".join([seg.text.strip() for seg in segments])
            elapsed = time.time() - start_t
            log.info(
                f"Finished transcribing {media_id} in {elapsed:.1f}s "
                f"({len(transcript_text.split())} words, detected source language: {info.language})."
            )
            return transcript_text if transcript_text.strip() else None

        except Exception as e:
            err_text = str(e).splitlines()[0] if str(e) else type(e).__name__
            log.warning(f"Audio transcription failed for {media_id}: {err_text}")
            _record_whisper_env_failure(err_text, logger)
            return None
