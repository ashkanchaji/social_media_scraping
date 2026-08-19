# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not make any changes until you have 95% confidence in what you need to build. Ask me follow-up questions until you reach that confidence.

## What this is

A set of independent scrapers that collect financial/energy-market news coverage from social platforms (X/Twitter, YouTube, Telegram) into a shared JSON record schema, for downstream sentiment/NLP processing (e.g. FinBERT). Each platform scraper is a standalone script driven by keyword/account config files, sharing two local modules for logging, dedup, and quality scoring.

## Running the scrapers

```bash
pip install -r requirements.txt --break-system-packages
python twtr-scraper.py     # X/Twitter — no CLI args
python yt-scraper.py       # YouTube — no CLI args
python tel-scraper.py      # Telegram — no CLI args, needs TELEGRAM_API_ID/TELEGRAM_API_HASH
```

There is no test suite, build step, or linter configured in this repo. `.venv` and `x-tweet-fetcher/` (a vendored git clone, its own project with its own tests) are not part of this project's code and should not be treated as such.

Configuration is entirely via constants at the top of each script and plain-text input files in the repo root — not CLI flags:

- `twtr-keywords.txt` / `yt-keywords.txt` — required, one search keyword per line, script exits if missing.
- `twtr-accounts.txt` / `yt-channels.txt` — optional, auto-created with sane defaults (Reuters, Bloomberg, CNBC, BBC, CNN, Al Jazeera) if missing. `#`-prefixed lines are comments.
- `tel-channels.txt` — **required** for `tel-scraper.py`, one public channel username per line (`@` optional), `#`-prefixed lines are comments. Deliberately has no auto-created defaults, unlike the X/YouTube account lists: Telegram usernames for news outlets aren't reliably guessable, and a wrong guess would silently scrape the wrong channel instead of failing.
- `asset-registry.json` (optional, gitignored) — overrides/extends the built-in asset registry in `pipeline_utils.py` without code changes; see `asset-registry.example.json` for the shape (`aliases`, `topics`, `description`, `context_inference`).

Each run is resumable/interruptible: results are written incrementally per keyword, not batched at the end.

`tel-scraper.py` additionally requires `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` env vars (create an application at https://my.telegram.org). The first run does an interactive phone-number + login-code sign-in and writes a `<TELEGRAM_SESSION_NAME>.session` file; every later run — including unattended cron runs — reuses it without prompting. That session file is a live account credential and is gitignored along with `*.session-journal`.

## Architecture

### Shared modules (`pipeline_utils.py`, `dedup_utils.py`)

Every platform scraper imports both and follows the same pipeline shape: search/discover → fetch → filter → dedup → score quality → write. Do not duplicate this logic into a new scraper — extend the shared modules instead so behavior (schema, quality scoring, dedup) stays identical across platforms.

`pipeline_utils.py`:
- `setup_logging` — console + rotating file logger (5MB × 3 backups) used by every scraper.
- `with_retry`, `RateLimiter` — retry/backoff decorator and a thread-safe global rate limiter, used to pace concurrent fetch workers against upstream rate limits.
- `detect_language` / `is_english` — language-detection gate (py3langid) applied before saving any record.
- `load_asset_registry` / `extract_asset_mentions` — asset tagging. Loads the built-in registry (or `asset-registry.json` if present) and, via a shared NLI model (`cross-encoder/nli-deberta-v3-base` by default, `ASSET_NLI_MODEL` env override), scores which financial assets a record's text is actually about. Direct alias/cashtag mentions are always kept; context-only inference requires both the search topic and the record text to entail the asset, and equities require explicit `context_inference: true` in the registry entry to be inferred from sector language alone.
- `build_quality` — computes the record's `quality.quality_score` (0–100) by validating identity/URL consistency, content, timestamps, engagement/media, dedup, and asset-mapping fields, rather than a hand-weighted heuristic.

`dedup_utils.py` — cross-source duplicate detection backed by SQLite (`dedup-index.db`, gitignored), safe for concurrent access by multiple scraper processes:
1. Exact-match layer: MD5 of normalized text.
2. Near-duplicate layer: MinHash + LSH over word shingles (`JACCARD_THRESHOLD = 0.55`), catching paraphrased coverage of the same story across platforms.

This index is shared across all scrapers specifically so the same underlying story reported on X, YouTube, and Telegram isn't double-counted — always route new records for dedup through `dedup_utils`, never implement a scraper-local dedup check.

### Record schema

All scrapers emit the same JSON record shape (see the `record_id`, `Source`, `Content`, `time_stamps`, `asset_mention`, `Engagement`, `media`, `deduplication`, `quality` fields documented in `twtr-scraper.md` / `yt-scraper.md`). `record_id` is `<Platform>_<author_or_channel_id>_<platform_native_id>`. `Content.Clean_text` is intentionally left empty by every scraper — populated by a separate downstream cleaning step, not by these scripts. `time_stamps.updated_at` must stay `null` unless the platform actually exposes an edit/modification timestamp; never substitute collection time for it. Telegram is currently the only platform that does expose one (`message.edit_date`), so it is the only scraper that ever populates that field.

`build_quality` validates identity fields per-platform (`_valid_source_id`, `_valid_author_id`, `_valid_source_url`, `_record_id_consistent`, `_media_checks` in `pipeline_utils.py`). Adding a new platform means adding a branch to each of those plus the `valid_platform` set and the `expected_type` map — a platform with no branch falls through to weak generic checks and silently scores lower. Current `Content_type` per platform: X `tweet`, YouTube `video`, Telegram `message`.

Output is written per-keyword to `twtr-tweets/twitter_<keyword>.json` and `yt-transcripts/youtube_<keyword>.json`, and per-channel to `tel-transcripts/telegram_<channel>.json`. Each run appends new non-duplicate records to the existing file and re-sorts the whole file chronologically by `published_at` — it never overwrites prior runs.

### Platform-specific pipelines

- **`twtr-scraper.py`**: X has no free search API, so discovery goes through DuckDuckGo (`ddgs`) using `site:x.com <keyword> (from:acct1 OR from:acct2 ...)` queries scoped to the trusted-account list, then full tweet content is fetched via the vendored `x-tweet-fetcher` package (imported as `xtf`). Both the DDG search path and the X fetch path are unofficial/unauthenticated and carry real exposure to rate-limiting/blocking — this is why the script has retry/backoff and per-keyword pacing delays baked in rather than being optional.
- **`yt-scraper.py`**: Discovers candidate videos via `yt-dlp` search filtered to trusted channels, then extracts spoken content with a two-tier strategy: YouTube's own captions first (`youtube-transcript-api`, fast/free), falling back to local Whisper transcription (`faster-whisper`) only when no captions exist. Whisper uses a multilingual model + `task="translate"` (not an English-only model) so non-English audio is still handled correctly. Requires `ffmpeg` on `PATH` for the Whisper fallback's audio postprocessing. GPU (CUDA) is used automatically if available and falls back to CPU after repeated library-load failures via a circuit breaker (`_WHISPER_FAILURE_LIMIT`) — this is a graceful perf degradation, not an error condition to "fix" by making GPU mandatory.
- **`tel-scraper.py`**: Unlike the other two there is no keyword search step — it walks each trusted channel's recent history directly and keeps every message inside a `TIME_WINDOW_MINUTES` window (default 60), so relevance filtering is left entirely to the asset-mapping stage. Auth is Telethon MTProto as a *user account*, not the Bot API, because a bot can only read channels it was added to as admin. Channels are processed **sequentially, not through a `ThreadPoolExecutor`** like X/YouTube: one MTProto session gains nothing from parallel requests and concurrent reads raise the flood-ban risk for the account — don't "optimize" this into a thread pool. Flood control is handled by `_with_flood_retry`, which sleeps for the exact duration Telegram dictates (`FloodWaitError.seconds`) rather than the fixed backoff schedule `pipeline_utils.with_retry` provides; short waits under `FLOOD_SLEEP_THRESHOLD` are absorbed by Telethon itself.
  - `REFRESH_EDITED_MESSAGES` (default `True`) re-scans the whole window so edits to already-saved posts refresh the stored record in place (text, `updated_at`, assets, and quality are recomputed; `published_at` and therefore sort position never change). Setting it `False` switches to a cheaper append-only mode that passes the newest stored message ID as Telethon's `min_id` and never sees an edit.
  - Channels without a public username are skipped — a stable `t.me/<username>/<id>` URL can't be built for them, and the schema's URL/identity checks require one.

### x-tweet-fetcher

`x-tweet-fetcher/` is a vendored clone of a separate upstream project (its own git repo, README, tests, SKILL.md) that provides the `xtf` Python package used by `twtr-scraper.py` to fetch tweet content without X's paid API (via FxTwitter for single tweets, a Nitter instance for timelines/search, and a browser driver for lists/articles). Treat it as a third-party dependency, not project code — don't modify it as part of scraper feature work; if it needs a fix, that's a separate concern from this repo's scrapers.

## Data sensitivity

`twtr-keywords.txt`, `twtr-accounts.txt`, `yt-keywords.txt`, `yt-channels.txt`, `tel-channels.txt`, and all `*-scraper.log` files are gitignored — they can reveal what topics/entities are being monitored. `dedup-index.db` and the `twtr-tweets/` / `yt-transcripts/` / `tel-transcripts/` output directories are also gitignored since they're regenerable run artifacts, not source.

`*.session` / `*.session-journal` (Telethon login sessions) are gitignored and must stay that way — a session file is a live authenticated credential for the Telegram account that created it, and anyone holding it can act as that account. `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` are read from the environment for the same reason and must never be hardcoded into `tel-scraper.py`.
