import os
import json
import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import yt_dlp

import dedup_utils
from pipeline_utils import setup_logging, with_retry, detect_language

logger = setup_logging("yt-scraper", log_file="yt-scraper.log")

MAX_RESULTS_PER_KEYWORD = 30
MAX_PAST_DAYS = 30
LANGUAGE_MIN_CONFIDENCE = 0.70  # safety-net check even after translation


def extract_video_id(video_url: str) -> str:
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/live/" in video_url:
        return video_url.split("youtube.com/live/")[1].split("?")[0]
    return video_url.strip()


@with_retry(max_attempts=3, base_delay=3.0, exceptions=(Exception,))
def fetch_video_metadata(video_url: str) -> dict:
    ydl_opts = {'quiet': True, 'skip_download': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    raw_timestamp = info.get('timestamp')
    if raw_timestamp:
        upload_datetime = datetime.datetime.fromtimestamp(raw_timestamp, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    else:
        upload_datetime = info.get('upload_date', 'N/A')

    return {
        "title": info.get('title', 'N/A'),
        "channel": info.get('uploader', 'N/A'),
        "channel_id": info.get('uploader_id', 'N/A'),
        "upload_datetime": upload_datetime,
        "view_count": info.get('view_count', 0) or 0,
        "like_count": info.get('like_count', 0) or 0,
        "comment_count": info.get('comment_count', 0) or 0
    }


def search_youtube_videos(keyword: str, max_results: int = 15) -> list:
    logger.info(f"Searching live feed for: {keyword}...")
    trusted_agencies = "(Reuters OR Bloomberg OR CNBC OR BBC OR CNN OR Al Jazeera)"
    today = datetime.datetime.now(datetime.timezone.utc)
    past_date = today - datetime.timedelta(days=MAX_PAST_DAYS)
    date_filter = f"after:{past_date.strftime('%Y-%m-%d')}"

    search_query = f"ytsearch30:{keyword} {trusted_agencies} {date_filter}"
    ydl_opts = {'quiet': True, 'skip_download': True, 'no_warnings': True, 'extract_flat': True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)
            entries = info.get('entries', [])
        except Exception as e:
            logger.warning(f"Error during yt-dlp search for '{keyword}': {str(e).splitlines()[0]}")
            entries = []

    allowed_formats = []
    for i in range(MAX_PAST_DAYS + 1):
        d = today - datetime.timedelta(days=i)
        allowed_formats.extend([d.strftime('%Y-%m-%d'), d.strftime('%Y%m%d')])

    valid_videos = []
    for entry in entries:
        video_url = f"https://www.youtube.com/watch?v={entry.get('id')}"
        try:
            meta = fetch_video_metadata(video_url)
            upload_str = str(meta.get('upload_datetime', ''))

            if any(allowed in upload_str for allowed in allowed_formats):
                valid_videos.append({'url': video_url, 'view_count': meta.get('view_count', 0)})
        except Exception:
            continue

    valid_videos.sort(key=lambda x: x['view_count'], reverse=True)
    final_urls = [v['url'] for v in valid_videos[:max_results]]
    logger.info(f"Found {len(final_urls)} reliable, high-engagement videos for '{keyword}' from the past {MAX_PAST_DAYS * 24} hours.")
    return final_urls


def scrape_youtube_transcript_text(video_url: str) -> str:
    """
    Fetches a video's transcript, translated to English if needed.

    Note: this used to accept a native Farsi ('fa') transcript as-is
    without translating it, on the theory that Iran-related coverage in
    Farsi was still useful raw signal. But everything downstream (FinBERT)
    is English-only -- a raw Farsi transcript doesn't get scored
    meaningfully, it just silently produces garbage sentiment. Now ANY
    non-English transcript goes through translation, same as before for
    every other language; only native English is used directly.
    """
    video_id = extract_video_id(video_url)
    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.list(video_id)
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-CA'])
            transcript_data = transcript.fetch().to_raw_data()
        except Exception:
            translatable = None
            for t in transcript_list:
                if t.is_translatable:
                    translatable = t
                    break
            if translatable:
                transcript_data = translatable.translate('en').fetch().to_raw_data()
            else:
                logger.info(f"Skipping {video_id}: No translatable transcript available.")
                return None
    except Exception as e:
        logger.info(f"Skipping {video_id}: Transcripts unavailable ({type(e).__name__}).")
        return None

    return " ".join([item['text'].replace('\n', ' ') for item in transcript_data])


def search_and_scrape_youtube(keyword: str, lsh, hash_by_id, run_timestamp: str, output_dir: str = "yt-transcripts/"):
    os.makedirs(output_dir, exist_ok=True)
    found_urls = search_youtube_videos(keyword, max_results=MAX_RESULTS_PER_KEYWORD)
    if not found_urls:
        return None

    video_data = []
    skipped_non_english = 0

    for url in found_urls:
        video_id = extract_video_id(url)
        full_text = scrape_youtube_transcript_text(url)
        if not full_text or not full_text.strip():
            continue

        # Safety net: even after attempted translation, verify the result
        # is actually English before it reaches FinBERT.
        lang, confidence = detect_language(full_text)
        if lang != "en" or confidence < LANGUAGE_MIN_CONFIDENCE:
            skipped_non_english += 1
            continue

        try:
            meta = fetch_video_metadata(url)
            collected_at = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            record_id = f"YouTube_{meta.get('channel_id', 'Unknown')}_{video_id}"

            duplication_info = dedup_utils.check_and_register(record_id, full_text, lsh, hash_by_id)

            record = {
                "record_id": record_id,
                "Source": {
                    "platform": "YouTube",
                    "Source_type": "Social_media",
                    "Source_name": meta.get('channel', 'N/A'),
                    "Source_id": meta.get('channel_id', 'N/A'),
                    "url": url,
                    "author_name": meta.get('channel', 'N/A'),
                    "author_id": meta.get('channel_id', 'N/A')
                },
                "Content": {
                    "Content_type": "video",
                    "title": meta.get('title', 'N/A'),
                    "raw_text": full_text,
                    "Clean_text": full_text,
                    "language": lang
                },
                "time_stamps": {
                    "published_at": meta.get('upload_datetime', 'N/A'),
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
                    "Views": meta.get('view_count') or 0,
                    "likes": meta.get('like_count') or 0,
                    "Comments": meta.get('comment_count') or 0,
                    "shares": 0
                },
                "media": {
                    "has_media": True,
                    "media_type": "video",
                    "media_url": url
                },
                "deduplication": duplication_info
            }
            video_data.append(record)
        except Exception:
            continue

    if skipped_non_english:
        logger.info(f"Filtered out {skipped_non_english} non-English video(s) for '{keyword}'.")

    if not video_data:
        return None

    safe_keyword = keyword.replace(" ", "_").lower()
    # Timestamped per-run filename -- see twtr-scraper.py for why this
    # matters for continuous/day-and-night operation.
    output_filename = os.path.join(output_dir, f"youtube_{safe_keyword}_{run_timestamp}.json")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(video_data, f, indent=4, ensure_ascii=False)

    return output_filename


def analyze_youtube_sentiment(json_filename: str, sentiment_analyzer, output_dir: str = "sentiments/youtube/"):
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
    keyword_file = "yt-keywords.txt"
    if not os.path.exists(keyword_file):
        logger.error(f"File '{keyword_file}' not found.")
        return

    with open(keyword_file, 'r') as f:
        keywords = [line.strip() for line in f if line.strip()]

    if not keywords:
        logger.error("No keywords found in yt-keywords.txt.")
        return

    run_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logger.info(f"Starting scraping run {run_timestamp} for videos...")

    lsh, hash_by_id = dedup_utils.load_lsh()  # shared cross-platform dedup index
    analysis_queue = []

    for keyword in keywords:
        try:
            scraped_file = search_and_scrape_youtube(keyword, lsh, hash_by_id, run_timestamp)
            if scraped_file:
                analysis_queue.append(scraped_file)
        except Exception as e:
            logger.error(f"Error executing search/scrape for keyword '{keyword}': {str(e).splitlines()[0]}")

    logger.info("Scraping for all videos is done.")

    if not analysis_queue:
        logger.info("No new videos to analyze.")
        return

    logger.info("Starting analyzing scraped data...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    for video_file in analysis_queue:
        try:
            analyze_youtube_sentiment(video_file, sentiment_analyzer)
        except Exception as e:
            logger.error(f"Error analyzing {video_file}: {str(e).splitlines()[0]}")

    logger.info("Analyzing for all videos is done.")


if __name__ == "__main__":
    main()
