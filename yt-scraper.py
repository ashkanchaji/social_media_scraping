import os
import re
import time
import random
import json
import bisect
import tempfile
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

import dedup_utils
from pipeline_utils import (
    setup_logging,
    with_retry,
    RateLimiter,
    detect_language,
    extract_asset_mentions,
    build_quality,
)

# Silence huggingface hub warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = setup_logging("yt-scraper", log_file="yt-scraper.log")

MAX_RESULTS_PER_KEYWORD = 30
MAX_PAST_DAYS = 30
MAX_VIDEO_DURATION_SECONDS = 1200  # 20 minutes max (skips 24/7 live broadcasts)

# "base.en" is English-only and will hallucinate fluent-sounding but
# meaningless English text when given non-English audio (it has no notion
# of any other language, so it just pattern-matches sounds to English
# words). Using a multilingual model + task="translate" (below) instead
# lets Whisper auto-detect the spoken language and translate it to English
# properly, matching the quality of YouTube's own auto-translate captions.
# "small" is a reasonable speed/quality balance for an RTX 3070; bump to
# "medium" if translation quality on non-English sources still looks weak.
WHISPER_MODEL_SIZE = "small"

FETCH_MAX_WORKERS = 4              # Concurrent audio/metadata workers
FETCH_MIN_INTERVAL = 1.0           # Seconds between video calls enforced globally
SEARCH_DELAY_RANGE = (8.0, 14.0)   # Pacing delay between keyword searches
SEARCH_MAX_RETRIES = 3

LANGUAGE_MIN_CONFIDENCE = 0.70

_fetch_rate_limiter = RateLimiter(FETCH_MIN_INTERVAL)

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


def get_whisper_model():
    """Lazily loads and caches the faster-whisper model."""
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Initializing faster-whisper model ({WHISPER_MODEL_SIZE})... (this may take a moment on first run)")
                try:
                    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16")
                    logger.info(f"faster-whisper model ({WHISPER_MODEL_SIZE}) successfully loaded and ready (GPU/CUDA).")
                except Exception as gpu_err:
                    logger.warning(
                        f"Could not initialize Whisper on GPU ({str(gpu_err).splitlines()[0]}). "
                        "Falling back to CPU (int8)."
                    )
                    _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
                    logger.info(f"faster-whisper model ({WHISPER_MODEL_SIZE}) successfully loaded and ready (CPU fallback).")
            except ImportError:
                logger.warning("faster-whisper is not installed. Falling back strictly to subtitle scraping.")
                _whisper_model = False
    return _whisper_model


def _record_whisper_env_failure(error_text: str) -> bool:
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
            logger.error(
                f"Whisper failed {_whisper_failure_count} times in a row with an "
                f"environment/library error ('{error_text}'). Disabling audio "
                "transcription for the rest of this run and falling back to "
                "subtitle-only mode. Fix: reinstall a CPU-only ctranslate2 build, "
                "or run `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
                "--break-system-packages` and point LD_LIBRARY_PATH at the "
                "installed nvidia/*/lib directories before the next run."
            )
            return True
    return False


def load_trusted_channels(filepath: str = "yt-channels.txt") -> list:
    default_channels = ["Reuters", "Bloomberg", "CNBC", "BBC", "CNN", "Al Jazeera"]
    if not os.path.exists(filepath):
        logger.warning(f"'{filepath}' not found. Creating default file.")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(default_channels))
        return default_channels

    with open(filepath, "r", encoding="utf-8") as f:
        channels = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    return channels if channels else default_channels


def extract_video_id(video_url: str) -> str:
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/live/" in video_url:
        return video_url.split("youtube.com/live/")[1].split("?")[0]
    return video_url.strip()


@with_retry(max_attempts=3, base_delay=2.0, exceptions=(Exception,))
def fetch_video_metadata(video_url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'socket_timeout': 15,
        'retries': 3,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    raw_timestamp = info.get('timestamp')
    if raw_timestamp:
        dt = datetime.datetime.fromtimestamp(raw_timestamp, datetime.timezone.utc)
        upload_datetime = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    else:
        dt = None
        upload_datetime = info.get('upload_date', 'N/A')

    modified_timestamp = info.get('modified_timestamp')
    if modified_timestamp:
        try:
            modified_dt = datetime.datetime.fromtimestamp(modified_timestamp, datetime.timezone.utc)
            updated_at = modified_dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except (ValueError, TypeError, OSError, OverflowError):
            updated_at = None
    else:
        updated_at = None

    # yt-dlp exposes both uploader_id and channel_id. For a stable author
    # identifier we want YouTube's canonical channel ID (normally UC...).
    channel_id = info.get('channel_id')
    if not channel_id:
        for channel_url in (info.get('channel_url'), info.get('uploader_url')):
            if not channel_url:
                continue
            match = re.search(r"youtube\.com/channel/(UC[A-Za-z0-9_-]{22})", channel_url)
            if match:
                channel_id = match.group(1)
                break

    return {
        "title": info.get('title', 'N/A'),
        "channel": info.get('channel') or info.get('uploader') or 'N/A',
        "channel_id": channel_id,
        "channel_handle": info.get('uploader_id'),
        "upload_datetime": upload_datetime,
        "updated_at": updated_at,
        "created_dt": dt,
        "duration": info.get('duration') or 0,
        "is_live": info.get('is_live', False),
        "view_count": info.get('view_count', 0) or 0,
        "like_count": info.get('like_count', 0) or 0,
        "comment_count": info.get('comment_count', 0) or 0
    }


def search_youtube_videos(keyword: str, trusted_channels: list, max_results: int = 15) -> list:
    logger.info(f"Searching live feed for: {keyword}...")
    today = datetime.datetime.now(datetime.timezone.utc)
    past_date = today - datetime.timedelta(days=MAX_PAST_DAYS)
    date_filter = f"after:{past_date.strftime('%Y-%m-%d')}"

    search_query = f"ytsearch30:{keyword} news {date_filter}"
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'extract_flat': True,
        'socket_timeout': 15,
        'retries': 3,
    }

    entries = []
    for attempt in range(1, SEARCH_MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get('entries', []) or []
            break
        except Exception as e:
            wait_time = random.uniform(10.0, 20.0) * attempt
            if attempt < SEARCH_MAX_RETRIES:
                logger.warning(f"yt-dlp search failed for '{keyword}' ({e}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"yt-dlp search exhausted attempts for '{keyword}': {e}")
                entries = []
        finally:
            time.sleep(random.uniform(*SEARCH_DELAY_RANGE))

    trusted_lower = [c.lower() for c in trusted_channels]
    valid_entries = []
    for entry in entries:
        uploader = (entry.get('uploader') or entry.get('channel') or '').lower()
        title = (entry.get('title') or '').lower()
        if trusted_channels:
            if any(tc in uploader or tc in title for tc in trusted_lower):
                valid_entries.append(entry.get('id'))
        else:
            valid_entries.append(entry.get('id'))

    if not valid_entries:
        valid_entries = [entry.get('id') for entry in entries if entry.get('id')]

    logger.info(f"Found {len(valid_entries[:max_results])} candidate video entries for '{keyword}'.")
    return valid_entries[:max_results]


@with_retry(max_attempts=2, base_delay=3.0, exceptions=(Exception,))
def _download_audio(video_url: str, video_id: str, out_template: str) -> None:
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'socket_timeout': 25,
        'retries': 3,
        'fragment_retries': 3,
        # Forcing the android client sidesteps YouTube's current
        # signature/PO-token checks that were causing blanket 403s on the
        # web client's adaptive audio streams.
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'http_headers': {
            'User-Agent': 'com.google.android.youtube/19.09.37 (Linux; U; Android 14) gzip'
        },
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def _transcribe_audio_stream(video_url: str, video_id: str, duration: int = 0) -> str:
    if duration and duration > MAX_VIDEO_DURATION_SECONDS:
        logger.info(f"Skipping audio download for {video_id}: duration ({duration}s) exceeds {MAX_VIDEO_DURATION_SECONDS}s cap.")
        return None

    model = get_whisper_model()
    if not model or _whisper_disabled:
        return None

    logger.info(f"Downloading lightweight audio for {video_id}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = os.path.join(tmpdir, f"{video_id}.%(ext)s")

        try:
            _download_audio(video_url, video_id, out_template)

            audio_file = os.path.join(tmpdir, f"{video_id}.mp3")
            if not os.path.exists(audio_file):
                files = os.listdir(tmpdir)
                if not files:
                    return None
                audio_file = os.path.join(tmpdir, files[0])

            logger.info(f"Transcribing audio for {video_id} with Whisper...")
            start_t = time.time()
            # task="translate" makes Whisper auto-detect the spoken language
            # and translate it directly to English, instead of transcribing
            # verbatim in whatever language is detected. Combined with a
            # multilingual model (not "*.en"), this is what actually handles
            # non-English source audio correctly.
            segments, info = model.transcribe(audio_file, beam_size=2, task="translate")
            transcript_text = " ".join([seg.text.strip() for seg in segments])
            elapsed = time.time() - start_t
            logger.info(
                f"Finished transcribing {video_id} in {elapsed:.1f}s "
                f"({len(transcript_text.split())} words, detected source language: {info.language})."
            )
            return transcript_text if transcript_text.strip() else None

        except Exception as e:
            err_text = str(e).splitlines()[0] if str(e) else type(e).__name__
            logger.warning(f"Audio transcription failed for {video_id}: {err_text}")
            _record_whisper_env_failure(err_text)
            return None


def extract_video_text(video_url: str, duration: int = 0) -> str:
    video_id = extract_video_id(video_url)
    ytt_api = YouTubeTranscriptApi()

    # Tier 1: Instant Subtitle API
    try:
        transcript_list = ytt_api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-CA'])
            transcript_data = transcript.fetch().to_raw_data()
        except Exception:
            translatable = next((t for t in transcript_list if t.is_translatable), None)
            if translatable:
                transcript_data = translatable.translate('en').fetch().to_raw_data()
            else:
                transcript_data = None

        if transcript_data:
            return " ".join([item['text'].replace('\n', ' ') for item in transcript_data])
    except Exception as e:
        logger.debug(f"Subtitle fetch failed for {video_id}: {str(e).splitlines()[0] if str(e) else type(e).__name__}")

    # Tier 2: Whisper Audio Fallback
    logger.info(f"Subtitles unavailable for {video_id}. Falling back to audio Whisper transcription...")
    return _transcribe_audio_stream(video_url, video_id, duration=duration)


def _process_single_video(video_id: str) -> tuple:
    _fetch_rate_limiter.wait()
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        meta = fetch_video_metadata(video_url)
        if meta.get('is_live'):
            logger.info(f"Skipping {video_id}: Active live stream.")
            return video_id, video_url, None, None

        text = extract_video_text(video_url, duration=meta.get('duration', 0))
        return video_id, video_url, meta, text
    except Exception as e:
        logger.warning(f"Failed processing video {video_id}: {e}")
        return video_id, video_url, None, None


def _insert_sorted(records_list: list, timestamps_list: list, record: dict, dt_val: datetime.datetime):
    sort_key = dt_val.timestamp() if dt_val else 0.0
    idx = bisect.bisect_right(timestamps_list, sort_key)
    records_list.insert(idx, record)
    timestamps_list.insert(idx, sort_key)


def search_and_scrape_youtube(keyword: str, trusted_channels: list, lsh, hash_by_id, output_dir: str = "yt-transcripts/"):
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.datetime.now(datetime.timezone.utc)
    cutoff = today - datetime.timedelta(days=MAX_PAST_DAYS)

    safe_keyword = keyword.replace(" ", "_").lower()
    master_file = os.path.join(output_dir, f"youtube_{safe_keyword}.json")

    existing_records = []
    existing_ids = set()
    existing_timestamps = []

    if os.path.exists(master_file):
        try:
            with open(master_file, 'r', encoding='utf-8') as f:
                existing_records = json.load(f)
                for rec in existing_records:
                    rec_id = rec.get("record_id")
                    if rec_id:
                        existing_ids.add(rec_id)
                    pub_str = rec.get("time_stamps", {}).get("published_at")
                    try:
                        dt_obj = datetime.datetime.strptime(pub_str, "%Y-%m-%d %H:%M:%S UTC")
                    except Exception:
                        dt_obj = None
                    existing_timestamps.append(dt_obj.timestamp() if dt_obj else 0.0)
        except Exception as e:
            logger.error(f"Error loading {master_file}: {e}")
            existing_records, existing_ids, existing_timestamps = [], set(), []

    candidate_ids = search_youtube_videos(keyword, trusted_channels, max_results=MAX_RESULTS_PER_KEYWORD)
    if not candidate_ids:
        logger.info(f"No video candidates found for '{keyword}'.")
        return None

    # Match on the exact "_{video_id}" suffix rather than a raw substring
    # check -- a plain `vid in existing_id` could false-positive if the
    # video ID happens to appear inside another record's channel_id.
    new_candidate_ids = [
        vid for vid in candidate_ids
        if not any(existing_id.endswith(f"_{vid}") for existing_id in existing_ids)
    ]
    if not new_candidate_ids:
        logger.info(f"All discovered videos for '{keyword}' are already scraped.")
        return master_file

    fetched_results = []
    with ThreadPoolExecutor(max_workers=FETCH_MAX_WORKERS) as executor:
        future_to_id = {executor.submit(_process_single_video, vid): vid for vid in new_candidate_ids}
        for future in as_completed(future_to_id):
            vid, vurl, meta, text = future.result()
            if meta and text:
                fetched_results.append((vid, vurl, meta, text))

    added_count = 0
    skipped_non_english = 0

    for vid, vurl, meta, text in fetched_results:
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            continue

        created_dt = meta.get("created_dt")
        if created_dt is not None and created_dt < cutoff:
            continue

        lang, confidence = detect_language(clean_text)
        if lang != "en" or confidence < LANGUAGE_MIN_CONFIDENCE:
            skipped_non_english += 1
            continue

        channel_id = meta.get("channel_id")
        channel_name = meta.get("channel") or "Unknown"
        record_id = f"YouTube_{channel_id or 'unknown'}_{vid}"

        if record_id in existing_ids:
            continue

        duplication_info = dedup_utils.check_and_register(record_id, clean_text, lsh, hash_by_id)
        if duplication_info.get("is_duplicate"):
            continue

        collected_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

        asset_evidence_text = (str(meta.get("title") or "") + "\n" + clean_text).strip()

        record = {
            "record_id": record_id,
            "Source": {
                "platform": "YouTube",
                "Source_type": "Social_media",
                "Source_name": channel_name,
                "Source_id": vid,
                "url": vurl,
                "author_name": channel_name,
                "author_id": channel_id
            },
            "Content": {
                "Content_type": "video",
                "title": meta.get("title", "N/A"),
                "raw_text": clean_text,
                "Clean_text": "",
                "language": lang
            },
            "time_stamps": {
                "published_at": meta.get("upload_datetime", "N/A"),
                "collected_at": collected_at,
                "updated_at": meta.get("updated_at")
            },
            "asset_mention": extract_asset_mentions(asset_evidence_text, context_terms=[keyword]),
            "Engagement": {
                "Views": meta.get("view_count", 0),
                "likes": meta.get("like_count", 0),
                "Comments": meta.get("comment_count", 0),
                "shares": 0
            },
            "media": {
                "has_media": True,
                "media_type": "video",
                "media_url": vurl
            },
            "deduplication": duplication_info
        }
        record["quality"] = build_quality(record, complete_raw_text=True, extraction_errors=[])

        _insert_sorted(existing_records, existing_timestamps, record, created_dt)
        existing_ids.add(record_id)
        added_count += 1

    if skipped_non_english:
        logger.info(f"Filtered out {skipped_non_english} non-English video(s) for '{keyword}'.")

    if added_count > 0:
        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(existing_records, f, indent=4, ensure_ascii=False)
        logger.info(f"Appended {added_count} new sorted videos to {master_file} (Total: {len(existing_records)}).")
    else:
        logger.info(f"No new unique videos to append for '{keyword}'.")

    return master_file


def main():
    keyword_file = "yt-keywords.txt"
    channels_file = "yt-channels.txt"

    if not os.path.exists(keyword_file):
        logger.error(f"File '{keyword_file}' not found.")
        return

    with open(keyword_file, 'r', encoding='utf-8') as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        logger.error(f"No keywords found in '{keyword_file}'.")
        return

    trusted_channels = load_trusted_channels(channels_file)
    logger.info(f"Loaded {len(trusted_channels)} trusted channels from {channels_file}.")

    # Pre-load Whisper model once at startup
    get_whisper_model()

    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logger.info(f"Starting scraping run {run_timestamp} for YouTube videos...")

    lsh, hash_by_id = dedup_utils.load_lsh()

    for keyword in keywords:
        try:
            search_and_scrape_youtube(keyword, trusted_channels, lsh, hash_by_id)
        except Exception as e:
            logger.error(f"Error executing search/scrape for keyword '{keyword}': {str(e).splitlines()[0]}")

    logger.info("Scraping for all YouTube keywords is done.")


if __name__ == "__main__":
    main()