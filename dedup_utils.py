"""
dedup_utils.py -- shared cross-source duplicate detection.

Used by twtr-scraper.py, yt-scraper.py, and (later) any Telegram scraper so
that the same story covered across platforms gets linked into one
duplicate_group_id, not just treated as unrelated records.

Two layers of matching:
  1. Exact content hash (MD5 of normalized text) -- catches identical text
     reposted verbatim (e.g. a wire headline copy-pasted as a tweet).
  2. Near-duplicate via MinHash + LSH -- catches paraphrased coverage of the
     same event across sources, which exact hashing will always miss.

Backed by SQLite (not a flat JSON file) so it's safe for multiple scraper
processes to read/write concurrently, e.g. running twtr-scraper.py and
yt-scraper.py at the same time, or a future 24/7 daemon.

Install: pip install datasketch numpy --break-system-packages
"""

import os
import re
import json
import hashlib
import sqlite3
import numpy as np
from datasketch import MinHash, MinHashLSH

DB_PATH = "dedup-index.db"
NUM_PERM = 128
# Jaccard similarity threshold for "near-duplicate". Lower = catches more
# loosely related coverage; higher = only near-identical phrasing. 0.5-0.6
# is a reasonable starting point for short-form (tweet-length) financial
# news text -- tune based on false-positive/negative rate you observe.
JACCARD_THRESHOLD = 0.55
SHINGLE_SIZE = 4  # words per shingle


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # strip URLs (differ even for identical stories)
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation/emoji
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for shingle in _shingles(_normalize_text(text)):
        m.update(shingle.encode("utf8"))
    return m


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")  # lets multiple processes read while one writes
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dedup_records (
            record_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            duplicate_group_id TEXT NOT NULL,
            minhash TEXT NOT NULL
        )
    """)
    return conn


def load_lsh():
    """
    Rebuilds an in-memory LSH index from the SQLite table. Call this once
    at the start of a run (and periodically again in a long-running/24-7
    process, via reload_lsh(), to pick up records inserted by other
    concurrently running scraper processes).
    Returns (lsh, hash_by_id) -- pass both into check_and_register().
    """
    conn = _get_conn()
    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
    hash_by_id = {}

    for record_id, content_hash, group_id, minhash_json in conn.execute(
        "SELECT record_id, content_hash, duplicate_group_id, minhash FROM dedup_records"
    ):
        m = MinHash(num_perm=NUM_PERM)
        m.hashvalues = np.array(json.loads(minhash_json), dtype=np.uint64)
        lsh.insert(record_id, m)
        hash_by_id[record_id] = {"content_hash": content_hash, "duplicate_group_id": group_id}

    conn.close()
    return lsh, hash_by_id


def check_and_register(record_id: str, text: str, lsh: MinHashLSH, hash_by_id: dict) -> dict:
    content_hash = hashlib.md5(_normalize_text(text).encode("utf-8")).hexdigest()
    m = _make_minhash(text)

    exact_match_id = next(
        (rid for rid, info in hash_by_id.items() if info["content_hash"] == content_hash),
        None,
    )

    if exact_match_id:
        group_id = hash_by_id[exact_match_id]["duplicate_group_id"] or exact_match_id
        result = {
            "content_hash": content_hash,
            "is_duplicate": True,
            "duplicate_group_id": group_id,
            "original_record_id": exact_match_id,
        }
    else:
        near_matches = lsh.query(m)
        if near_matches:
            original_id = near_matches[0]
            group_id = hash_by_id[original_id]["duplicate_group_id"] or original_id
            result = {
                "content_hash": content_hash,
                "is_duplicate": True,
                "duplicate_group_id": group_id,
                "original_record_id": original_id,
            }
        else:
            result = {
                "content_hash": content_hash,
                "is_duplicate": False,
                "duplicate_group_id": record_id,
                "original_record_id": None,
            }

    # Register only if the key is not already in the index
    if record_id not in hash_by_id:
        try:
            lsh.insert(record_id, m)
        except ValueError:
            pass

    hash_by_id[record_id] = {"content_hash": content_hash, "duplicate_group_id": result["duplicate_group_id"]}

    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO dedup_records (record_id, content_hash, duplicate_group_id, minhash) VALUES (?, ?, ?, ?)",
        (record_id, content_hash, result["duplicate_group_id"], json.dumps(m.hashvalues.tolist())),
    )
    conn.commit()
    conn.close()

    return result