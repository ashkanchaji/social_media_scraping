"""
pipeline_utils.py -- shared infrastructure for the scraping pipeline:
logging, retry/backoff, a thread-safe rate limiter for concurrent fetches,
and language detection (used to filter out non-English content before it
reaches FinBERT, which is an English-only financial sentiment model).

Install: pip install py3langid --break-system-packages
"""

import time
import random
import logging
import functools
import threading
from logging.handlers import RotatingFileHandler

from py3langid.langid import LanguageIdentifier, MODEL_FILE

_identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


def setup_logging(name: str, log_file: str = "scraper.log") -> logging.Logger:
    """
    Console + rotating file logging. Replaces bare print() calls so a
    long-running/24-7 process has a persistent, size-capped log to check,
    not just whatever scrolled past in a terminal.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers if called twice
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

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
