import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

def scrape_youtube_transcript(video_url: str):
    # Extract the Video ID from standard YouTube URLs
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_url:
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/live/" in video_url:
        video_id = video_url.split("youtube.com/live/")[1].split("?")[0]
    else:
        video_id = video_url

    print(f"Fetching transcript for Video ID: {video_id}...")
    
    try:
        # 1. Initialize the API object (New in recent versions)
        ytt_api = YouTubeTranscriptApi()
        
        # 2. Fetch transcript text using the updated method
        transcript_data = ytt_api.fetch(video_id, languages=['en', 'fa'])  # Fetching Persian and English transcripts
        
        # Convert into a structured Pandas DataFrame
        df = pd.DataFrame(transcript_data)
        
        # Clean up text formatting
        df['text'] = df['text'].str.replace('\n', ' ')
        
        # Combine all transcript text into one full statement for LLM / Sentiment analysis
        full_text = " ".join(df['text'])
        
        print(f"\nSuccessfully extracted {len(df)} transcript segments!")
        print("-" * 50)
        print("PREVIEW OF FULL TEXT:")
        print(full_text[:300] + "...\n")
        
        # Save to CSV
        output_filename = f"youtube_{video_id}_transcript.csv"
        df.to_csv(output_filename, index=False)
        print(f"Saved transcript data to '{output_filename}'")
        
        return df

    except Exception as e:
        print(f"An error occurred: {e}")

def analyze_transcript_sentiment(csv_filename):
    print(f"Loading transcript from {csv_filename}...")
    
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_filename}. Make sure the file exists!")
        return
    
    # 1. Load the FinBERT model
    print("Loading FinBERT model (this might take a minute to download on the first run)...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    # 2. Group transcript text into chunks
    # We combine every 5 lines into one chunk to get better context for the AI
    chunk_size = 5
    chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk_text = " ".join(df['text'].iloc[i:i+chunk_size].astype(str))
        chunks.append(chunk_text)
        
    print(f"Divided video into {len(chunks)} chunks for analysis. Processing now...\n")
    
    results = []
    
    # 3. Analyze each chunk
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        # Truncate text slightly to ensure we don't exceed the model's token limit
        truncated_chunk = chunk[:1500] 
        
        # Get sentiment prediction
        prediction = sentiment_analyzer(truncated_chunk)[0]
        
        results.append({
            "chunk_index": idx,
            "text_snippet": truncated_chunk[:80] + "...",
            "sentiment": prediction['label'],
            "confidence": round(prediction['score'], 3)
        })
        
    # 4. Convert results to a structured table and save
    results_df = pd.DataFrame(results)
    output_filename = csv_filename.replace(".csv", "_sentiment.csv")
    results_df.to_csv(output_filename, index=False)
    
    print("-" * 50)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("-" * 50)
    # Preview the first 5 rows
    print(results_df.head(5))
    print(f"\nSaved sentiment data to '{output_filename}'")


if __name__ == "__main__":
    # Test with any public news or political video URL
    # Example: A news report or speech
    test_video_url = "https://www.youtube.com/live/FTh9D13JVEM?si=7Ruq4CG0PjAdmJEe"  # Replace with your video URL
    scrape_youtube_transcript(test_video_url)

    target_csv = "youtube_FTh9D13JVEM_transcript.csv" 
    analyze_transcript_sentiment(target_csv)