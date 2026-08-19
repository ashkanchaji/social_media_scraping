"""
pipeline_utils.py -- shared infrastructure for the scraping pipeline:
logging, retry/backoff, a thread-safe rate limiter, language detection,
and shared social-record schema helpers.

The schema helpers are intentionally source-agnostic where possible so X,
YouTube, Telegram, and future social scrapers can emit consistent JSON.

Install: pip install py3langid --break-system-packages
"""

import os
import re
import json
import time
import random
import logging
import functools
import threading
import datetime
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse, parse_qs

from py3langid.langid import LanguageIdentifier, MODEL_FILE

_identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)


# Built-in starter registry. Projects can extend/override this without code
# changes by placing an ``asset-registry.json`` file beside the scraper.
# Besides aliases, entries may include ``topics`` and ``description``; those
# fields are used by the NLI relevance model for context-only inference.
_DEFAULT_ASSET_REGISTRY = [
    {"canonical_name": "Bitcoin", "Symbol": "BTC", "asset_class": "cryptocurrency", "asset_id": "crypto:BTC", "aliases": ["bitcoin", "btc", "xbt"], "topics": ["crypto", "cryptocurrency", "digital assets"], "description": "the Bitcoin cryptocurrency"},
    {"canonical_name": "Ethereum", "Symbol": "ETH", "asset_class": "cryptocurrency", "asset_id": "crypto:ETH", "aliases": ["ethereum", "ether", "eth"], "topics": ["crypto", "smart contracts", "digital assets"], "description": "the Ethereum cryptocurrency and smart-contract network"},
    {"canonical_name": "Solana", "Symbol": "SOL", "asset_class": "cryptocurrency", "asset_id": "crypto:SOL", "aliases": ["solana", "sol"], "topics": ["crypto", "blockchain", "digital assets"], "description": "the Solana cryptocurrency and blockchain"},
    {"canonical_name": "XRP", "Symbol": "XRP", "asset_class": "cryptocurrency", "asset_id": "crypto:XRP", "aliases": ["xrp", "ripple"], "topics": ["crypto", "payments", "digital assets"], "description": "the XRP cryptocurrency associated with Ripple payments"},
    {"canonical_name": "Gold", "Symbol": "XAU", "asset_class": "commodity", "asset_id": "commodity:XAU", "aliases": ["gold", "xau", "xauusd"], "topics": ["precious metals", "safe haven", "bullion"], "description": "the gold precious-metal commodity"},
    {"canonical_name": "Silver", "Symbol": "XAG", "asset_class": "commodity", "asset_id": "commodity:XAG", "aliases": ["silver", "xag", "xagusd"], "topics": ["precious metals", "bullion"], "description": "the silver precious-metal commodity"},
    {"canonical_name": "WTI Crude Oil", "Symbol": "WTI", "asset_class": "commodity", "asset_id": "commodity:WTI", "aliases": ["wti", "wti crude", "west texas intermediate"], "topics": ["oil", "crude", "petroleum", "energy markets", "oil prices", "OPEC"], "description": "West Texas Intermediate crude oil benchmark"},
    {"canonical_name": "Brent Crude Oil", "Symbol": "BRENT", "asset_class": "commodity", "asset_id": "commodity:BRENT", "aliases": ["brent", "brent crude", "brent oil"], "topics": ["oil", "crude", "petroleum", "energy markets", "oil prices", "OPEC"], "description": "Brent crude oil benchmark"},
    {"canonical_name": "Crude Oil", "Symbol": "OIL", "asset_class": "commodity", "asset_id": "commodity:OIL", "aliases": ["crude oil", "oil prices", "oil price"], "topics": ["oil", "petroleum", "energy markets", "refining", "OPEC", "oil supply", "oil demand"], "description": "global crude-oil prices and petroleum markets"},
    {"canonical_name": "Energy Select Sector SPDR Fund", "Symbol": "XLE", "asset_class": "ETF", "asset_id": "etf:XLE", "aliases": ["xle", "energy select sector spdr"], "topics": ["energy stocks", "oil stocks", "oil companies", "oil and gas companies", "energy companies", "integrated oil companies"], "description": "a US exchange-traded fund tracking large energy, oil and gas companies"},
    {"canonical_name": "Exxon Mobil", "Symbol": "XOM", "asset_class": "equity", "asset_id": "equity:XOM", "aliases": ["exxon", "exxon mobil", "xom"], "topics": ["oil company", "energy company", "integrated oil and gas"], "description": "Exxon Mobil, an integrated oil and gas company"},
    {"canonical_name": "Chevron", "Symbol": "CVX", "asset_class": "equity", "asset_id": "equity:CVX", "aliases": ["chevron", "cvx"], "topics": ["oil company", "energy company", "integrated oil and gas"], "description": "Chevron, an integrated oil and gas company"},
    {"canonical_name": "Shell", "Symbol": "SHEL", "asset_class": "equity", "asset_id": "equity:SHEL", "aliases": ["shell plc", "shell", "shel"], "topics": ["oil company", "energy company", "integrated oil and gas", "LNG"], "description": "Shell plc, an integrated oil, gas and LNG company"},
    {"canonical_name": "BP", "Symbol": "BP", "asset_class": "equity", "asset_id": "equity:BP", "aliases": ["bp plc", "bp"], "topics": ["oil company", "energy company", "integrated oil and gas"], "description": "BP plc, an integrated oil and gas company"},
    {"canonical_name": "S&P 500", "Symbol": "SPX", "asset_class": "index", "asset_id": "index:SPX", "aliases": ["s&p 500", "s&p500", "sp500", "spx"], "topics": ["US stocks", "large cap stocks", "US equity market"], "description": "the S&P 500 US large-cap equity index"},
    {"canonical_name": "Nasdaq Composite", "Symbol": "IXIC", "asset_class": "index", "asset_id": "index:IXIC", "aliases": ["nasdaq composite", "nasdaq", "ixic"], "topics": ["technology stocks", "US stocks", "growth stocks"], "description": "the Nasdaq Composite equity index"},
    {"canonical_name": "Dow Jones Industrial Average", "Symbol": "DJI", "asset_class": "index", "asset_id": "index:DJI", "aliases": ["dow jones", "dow", "djia", "dji"], "topics": ["US stocks", "blue chip stocks"], "description": "the Dow Jones Industrial Average US equity index"},
    {"canonical_name": "US Dollar", "Symbol": "USD", "asset_class": "currency", "asset_id": "fx:USD", "aliases": ["us dollar", "u.s. dollar", "usd", "dollar index", "dxy"], "topics": ["foreign exchange", "FX", "dollar", "Federal Reserve"], "description": "the United States dollar currency"},
    {"canonical_name": "Euro", "Symbol": "EUR", "asset_class": "currency", "asset_id": "fx:EUR", "aliases": ["euro", "eur"], "topics": ["foreign exchange", "FX", "ECB", "eurozone"], "description": "the euro currency"},
    {"canonical_name": "British Pound", "Symbol": "GBP", "asset_class": "currency", "asset_id": "fx:GBP", "aliases": ["british pound", "pound sterling", "sterling", "gbp"], "topics": ["foreign exchange", "FX", "Bank of England", "UK currency"], "description": "the British pound sterling currency"},
    {"canonical_name": "Japanese Yen", "Symbol": "JPY", "asset_class": "currency", "asset_id": "fx:JPY", "aliases": ["japanese yen", "yen", "jpy"], "topics": ["foreign exchange", "FX", "Bank of Japan", "Japan currency"], "description": "the Japanese yen currency"},
    {"canonical_name": "Apple", "Symbol": "AAPL", "asset_class": "equity", "asset_id": "equity:AAPL", "aliases": ["apple inc", "apple", "aapl"], "topics": ["iPhone", "consumer technology", "big tech"], "description": "Apple Inc. common stock"},
    {"canonical_name": "Microsoft", "Symbol": "MSFT", "asset_class": "equity", "asset_id": "equity:MSFT", "aliases": ["microsoft", "msft"], "topics": ["software", "cloud computing", "Azure", "big tech"], "description": "Microsoft Corporation common stock"},
    {"canonical_name": "NVIDIA", "Symbol": "NVDA", "asset_class": "equity", "asset_id": "equity:NVDA", "aliases": ["nvidia", "nvda"], "topics": ["AI chips", "GPUs", "semiconductors", "artificial intelligence"], "description": "NVIDIA Corporation common stock"},
    {"canonical_name": "Tesla", "Symbol": "TSLA", "asset_class": "equity", "asset_id": "equity:TSLA", "aliases": ["tesla", "tsla"], "topics": ["electric vehicles", "EVs", "automotive"], "description": "Tesla Inc. common stock"},
    {"canonical_name": "Amazon", "Symbol": "AMZN", "asset_class": "equity", "asset_id": "equity:AMZN", "aliases": ["amazon", "amzn"], "topics": ["ecommerce", "AWS", "cloud computing", "big tech"], "description": "Amazon.com Inc. common stock"},
    {"canonical_name": "Meta Platforms", "Symbol": "META", "asset_class": "equity", "asset_id": "equity:META", "aliases": ["meta platforms", "meta", "facebook", "meta stock"], "topics": ["social media", "digital advertising", "big tech"], "description": "Meta Platforms Inc. common stock"},
]

_CASHTAG_RE = re.compile(r"(?<!\w)\$([A-Za-z][A-Za-z0-9._-]{0,14})\b")
_ASSET_REGISTRY_CACHE = {}
_ASSET_REGISTRY_LOCK = threading.Lock()
_ASSET_NLI_MODEL_NAME = os.environ.get("ASSET_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
_ASSET_NLI_DEVICE = os.environ.get("ASSET_NLI_DEVICE", "auto").strip().lower()
_ASSET_CONTEXT_MAX_RESULTS = int(os.environ.get("ASSET_CONTEXT_MAX_RESULTS", "0"))  # 0 = keep all NLI-entailing context candidates
_ASSET_NLI_BATCH_SIZE = max(1, int(os.environ.get("ASSET_NLI_BATCH_SIZE", "16")))
_ASSET_NLI_BUNDLE = None
_ASSET_NLI_LOCK = threading.Lock()
_ASSET_CONTEXT_CACHE = {}
_ASSET_CONTEXT_CACHE_LOCK = threading.Lock()
_ASSET_MODEL_WARNING_EMITTED = False


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


def load_asset_registry(filepath: str = "asset-registry.json") -> list:
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
    if not text or not alias:
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
    return pattern.search(text)


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
            entail_idx = next((i for i, label in id2label.items() if "entail" in label), 1)
            contradiction_idx = next((i for i, label in id2label.items() if "contrad" in label), 0)
            neutral_idx = next((i for i, label in id2label.items() if "neutral" in label), 2)
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
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            pred = probs.argmax(dim=-1).tolist()
            for entry, row, pred_idx in zip(batch_entries, probs, pred):
                if pred_idx == entail_idx:
                    label = "entailment"
                elif pred_idx == contradiction_idx:
                    label = "contradiction"
                elif pred_idx == neutral_idx:
                    label = "neutral"
                else:
                    label = str(pred_idx)
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


def extract_asset_mentions(text: str, context_terms=None, registry_path: str = "asset-registry.json") -> list:
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

    context = _context_candidates(context_terms, registry)
    candidate_map = {key: pair for key, pair in direct.items()}
    for entry, query_score in context:
        key = entry.get("asset_id") or entry.get("Symbol") or entry.get("canonical_name")
        candidate_map.setdefault(key, (entry, None))

    entries = [pair[0] for pair in candidate_map.values()]
    text_scores = _nli_asset_scores(text, entries)
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
    if platform == "X":
        return record_id == f"X_{author_id}_{source_id}"
    if platform == "YouTube":
        return record_id == f"YouTube_{author_id}_{source_id}"
    if platform == "Telegram":
        return record_id == f"Telegram_{author_id}_{source_id}"
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
        confidence_ok = isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and 0.0 <= float(confidence) <= 1.0
        # An unresolved cashtag is still a valid observed mention, but the
        # resolution/confidence checks expose that it is not a complete asset mapping.
        checks.append((f"asset_{idx}_observed", bool(item.get("mentioned_text") or resolved)))
        checks.append((f"asset_{idx}_resolved", resolved))
        checks.append((f"asset_{idx}_confidence", confidence_ok if resolved else confidence is None))
    return checks


def _category_score(checks: list) -> float:
    if not checks:
        return 100.0
    return 100.0 * sum(1 for _, passed in checks if passed) / len(checks)


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
        ("valid_platform", platform in {"X", "YouTube", "Telegram"}),
        ("valid_source_type", source.get("Source_type") == "Social_media"),
        ("valid_source_id", _valid_source_id(platform, source.get("Source_id"))),
        ("valid_author_id", _valid_author_id(platform, source.get("author_id"))),
        ("valid_author_name", bool(str(source.get("author_name") or "").strip()) and str(source.get("author_name")).strip().lower() not in {"n/a", "unknown"}),
        ("valid_source_name", bool(str(source.get("Source_name") or "").strip()) and str(source.get("Source_name")).strip().lower() not in {"n/a", "unknown"}),
        ("consistent_source_author_name", source.get("Source_name") == source.get("author_name")),
        ("valid_url", _valid_source_url(platform, source.get("url"), source.get("author_id"), source.get("Source_id"))),
        ("consistent_record_id", _record_id_consistent(record)),
    ]

    expected_type = {"X": "tweet", "YouTube": "video", "Telegram": "message"}.get(platform)
    content_checks = [
        ("complete_raw_text", bool(complete_raw_text and isinstance(content.get("raw_text"), str) and content.get("raw_text").strip())),
        ("valid_content_type", bool(expected_type and content.get("Content_type") == expected_type)),
        ("valid_language", bool(content.get("language") and content.get("language") not in {"unknown", "N/A"})),
    ]
    if platform == "YouTube":
        content_checks.append(("valid_title", bool(str(content.get("title") or "").strip()) and content.get("title") != "N/A"))

    timestamp_checks = [
        ("valid_published_at", published is not None),
        ("valid_collected_at", collected is not None),
        ("valid_timestamp_order", published is not None and collected is not None and published <= collected),
        ("valid_updated_at", updated_valid),
    ]

    asset_checks = _asset_checks(record)
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
    }

    scores = [
        _asset_category_score(record, checks) if name == "assets" else _category_score(checks)
        for name, checks in categories.items()
    ]
    if extraction_reliability is not None:
        try:
            rel = max(0.0, min(1.0, float(extraction_reliability)))
            scores.append(rel * 100.0)
        except (TypeError, ValueError):
            pass

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

