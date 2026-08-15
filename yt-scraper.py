import os
import pandas as pd
import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import yt_dlp

MAX_RESULTS_PER_KEYWORD = 15  # Increased limit for better yield
MAX_PAST_DAYS = 3  # Loosened to 3 days to ensure transcripts have had time to generate

def extract_video_id(video_url: str) -> str:
    """Extracts the unique YouTube video ID from standard URL formats."""
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        return video_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/live/" in video_url:
        return video_url.split("youtube.com/live/")[1].split("?")[0]
    return video_url.strip()


def fetch_video_metadata(video_url: str) -> dict:
    """Extracts metadata metrics for a YouTube video using yt-dlp."""
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        
    raw_timestamp = info.get('timestamp')
    if raw_timestamp:
        upload_datetime = datetime.datetime.fromtimestamp(raw_timestamp, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    else:
        upload_datetime = info.get('upload_date', 'N/A')
        
    return {
        "title": info.get('title', 'N/A'),
        "upload_datetime": upload_datetime,
        "view_count": info.get('view_count', 0) or 0,
        "like_count": info.get('like_count', 'N/A'),
        "comment_count": info.get('comment_count', 'N/A')
    }


def search_youtube_videos(keyword: str, max_results: int = 15) -> list:
    """
    Searches YouTube for recent news from reputable agencies.
    Loosened filter to 3 days to allow YouTube time to generate auto-transcripts.
    """
    print(f"  -> Searching live feed for: {keyword}...")
    
    trusted_agencies = "(Reuters OR Bloomberg OR CNBC OR BBC OR CNN OR Al Jazeera)"
    
    today = datetime.datetime.now(datetime.timezone.utc)
    past_date = today - datetime.timedelta(days=MAX_PAST_DAYS)
    date_filter = f"after:{past_date.strftime('%Y-%m-%d')}"
    
    search_query = f"ytsearch30:{keyword} {trusted_agencies} {date_filter}"
    
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'no_warnings': True,
        'extract_flat': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_query, download=False)
            entries = info.get('entries', [])
        except Exception as e:
            print(f"     Error during yt-dlp search: {str(e).splitlines()[0]}")
            entries = []

    # Dynamically generate allowed date strings for the max past days
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
                valid_videos.append({
                    'url': video_url,
                    'view_count': meta.get('view_count', 0)
                })
        except Exception:
            continue

    valid_videos.sort(key=lambda x: x['view_count'], reverse=True)
    
    final_urls = [v['url'] for v in valid_videos[:max_results]]
    print(f"  -> Found {len(final_urls)} reliable, high-engagement videos from the past {MAX_PAST_DAYS * 24} hours.")
    
    return final_urls


def scrape_youtube_transcript(video_url: str, output_dir: str = "yt-transcripts/"):
    """
    Extracts transcript timing and text for a given YouTube URL.
    Intelligently matches dialects and auto-translates foreign languages.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_id = extract_video_id(video_url)

    ytt_api = YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.list(video_id)

        try:
            # 1. Attempt to find native English or Persian
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'en-CA', 'fa'])
            transcript_data = transcript.fetch().to_raw_data()
        except Exception:
            # 2. Fallback: Find ANY translatable transcript and translate it to English
            translatable = None
            for t in transcript_list:
                if t.is_translatable:
                    translatable = t
                    break

            if translatable:
                transcript_data = translatable.translate('en').fetch().to_raw_data()
            else:
                print(f"     -> Skipping {video_id}: No translatable transcript available.")
                return None

    except Exception as e:
        error_name = type(e).__name__
        print(f"     -> Skipping {video_id}: Transcripts unavailable ({error_name}).")
        return None

    df = pd.DataFrame(transcript_data)
    df['text'] = df['text'].str.replace('\n', ' ')
    df = df[['text', 'start', 'duration']]

    output_filename = os.path.join(output_dir, f"youtube_{video_id}_transcript.csv")
    df.to_csv(output_filename, index=False)

    return output_filename

def analyze_transcript_sentiment(csv_filename: str, metadata: dict, search_keyword: str, sentiment_analyzer, output_dir: str = "sentiments/youtube/"):
    """
    Analyzes sentiment of transcript chunks using FinBERT.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_filename)
    
    chunk_size = 5
    chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk_text = " ".join(df['text'].iloc[i:i+chunk_size].astype(str))
        chunks.append(chunk_text)
        
    results = []
    
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        truncated_chunk = chunk[:1500] 
        prediction = sentiment_analyzer(truncated_chunk)[0]
        
        results.append({
            "chunk_index": idx,
            "text_snippet": truncated_chunk,
            "sentiment": prediction['label'].upper(),
            "confidence": round(prediction['score'], 3)
        })
        
    results_df = pd.DataFrame(results)
    
    results_df['search_keyword'] = search_keyword
    results_df['title'] = metadata.get('title', 'N/A')
    results_df['upload_datetime'] = metadata.get('upload_datetime', 'N/A')
    results_df['view_count'] = metadata.get('view_count', 'N/A')
    results_df['like_count'] = metadata.get('like_count', 'N/A')
    results_df['comment_count'] = metadata.get('comment_count', 'N/A')
        
    base_filename = os.path.basename(csv_filename).replace(".csv", "_sentiment.csv")
    output_filename = os.path.join(output_dir, base_filename)
    results_df.to_csv(output_filename, index=False)


def main():
    keyword_file = "yt-keywords.txt"
    
    if not os.path.exists(keyword_file):
        print(f"File '{keyword_file}' not found.")
        return
        
    with open(keyword_file, 'r') as f:
        keywords = [line.strip() for line in f if line.strip()]
        
    if not keywords:
        print("No keywords found in yt-keywords.txt.")
        return

    print("Starting scraping for videos...")
    analysis_queue = []
    
    for keyword in keywords:
        try:
            found_urls = search_youtube_videos(keyword, max_results=MAX_RESULTS_PER_KEYWORD)
            
            for url in found_urls:
                video_id = extract_video_id(url)
                transcript_file = os.path.join("yt-transcripts", f"youtube_{video_id}_transcript.csv")
                sentiment_file = os.path.join("sentiments/youtube", f"youtube_{video_id}_transcript_sentiment.csv")
                
                if os.path.exists(sentiment_file):
                    continue
                
                if not os.path.exists(transcript_file):
                    scraped_file = scrape_youtube_transcript(url)
                    if not scraped_file:
                        continue 
                    
                analysis_queue.append((url, transcript_file, keyword))
        except Exception as e:
            print(f"     Error executing search/scrape for keyword '{keyword}': {str(e).splitlines()[0]}")
            
    print("Scraping for all videos is done.")
    
    if not analysis_queue:
        print("No new videos to analyze.")
        return
        
    print("Starting analyzing scraped data...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    for url, transcript_file, keyword in analysis_queue:
        try:
            metadata = fetch_video_metadata(url)
            analyze_transcript_sentiment(transcript_file, metadata, keyword, sentiment_analyzer)
        except Exception as e:
            print(f"Error analyzing {transcript_file}: {str(e).splitlines()[0]}")
            
    print("Analyzing for all videos is done.")

if __name__ == "__main__":
    main()